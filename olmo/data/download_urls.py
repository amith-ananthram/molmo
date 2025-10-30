import dataclasses
import hashlib
import io
import logging
import random
import multiprocessing
import os
import time
import warnings
from collections import defaultdict, Counter
from os import rename, makedirs
from os.path import join, exists
from typing import Union, Dict

import PIL.Image
import datasets
import numpy as np
import requests
import urllib3
from PIL import ImageFile
from urllib3.exceptions import MaxRetryError
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from tqdm import tqdm

from olmo.data.dataset import DATA_HOME
from olmo.data.model_preprocessor import setup_pil

if "PIXMO_IMAGE_DIR" in os.environ:
    PIXMO_IMAGES = os.environ["PIXMO_IMAGE_DIR"]
elif DATA_HOME is not None:
    PIXMO_IMAGES = join(DATA_HOME, "pixmo_images")
else:
    PIXMO_IMAGES = None
"""Where to save downloaded images"""


PIL.Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclasses.dataclass
class DownloadError:
    url: str
    exception: Exception


@dataclasses.dataclass
class ImageError:
    url: str
    exception: Exception = None
    downloaded: bool = False


def compute_hash(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(string).hexdigest()


def extract_error_detail(exception):
    """Extract the most specific error information from nested exceptions."""
    error_str = str(exception)

    # Check for chained exceptions (explicit with __cause__)
    if hasattr(exception, "__cause__") and exception.__cause__ is not None:
        return f"{type(exception.__cause__).__name__}: {str(exception.__cause__)}"

    # Check for context exceptions (implicit chaining)
    if hasattr(exception, "__context__") and exception.__context__ is not None:
        return f"{type(exception.__context__).__name__}: {str(exception.__context__)}"

    # Extract "Caused by" information from the error string
    if "Caused by" in error_str:
        try:
            # Extract everything after "Caused by"
            caused_by_part = error_str.split("Caused by", 1)[1].strip()
            # Remove outer parentheses if present
            if caused_by_part.startswith("(") and caused_by_part.endswith(")"):
                caused_by_part = caused_by_part[1:-1]
            return caused_by_part
        except:
            pass

    # For 429 errors specifically, try to extract the response error
    if "429" in error_str and "ResponseError" in error_str:
        try:
            # Extract ResponseError part
            start = error_str.find("ResponseError")
            if start != -1:
                # Find the matching closing parenthesis
                paren_count = 0
                end = start
                for i, char in enumerate(error_str[start:]):
                    if char == "(":
                        paren_count += 1
                    elif char == ")":
                        paren_count -= 1
                        if paren_count == 0:
                            end = start + i + 1
                            break
                return error_str[start:end]
        except:
            pass

    # Fallback to the original exception string
    return f"{type(exception).__name__}: {str(exception)}"


def _download_images(args):
    url, image_sha, check_sha, cache_only, kwargs, clear_cache = args
    image_id = compute_hash(url)
    cache_file = join(PIXMO_IMAGES, image_id)

    # Create and configure session
    session = requests.Session()
    retries = Retry(
        # total=5,
        total=2,
        backoff_factor=2,
        status_forcelist=[429, 502, 503, 504],
        respect_retry_after_header=False,
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    cached = exists(cache_file)
    downloaded = False
    if cached:
        with open(cache_file, "rb") as f:
            image_bytes = f.read()
    else:
        if cache_only:
            return DownloadError(url, ValueError("Not in cache"))
        else:
            response = None
            try:
                # Small delay to be respectful to servers and avoid rate limiting
                # time.sleep(0.1)
                payload = {
                    "api_key": None,
                    "url": url,
                    "binary_target": True,
                }
                response = session.get(
                    "https://api.scraperapi.com/", params=payload, **kwargs
                )
                response.raise_for_status()
                image_bytes = response.content
                downloaded = True
            except Exception as e:
                # Write response to file so we know the URL failed and won't try it again
                with open(cache_file, "w") as f:
                    f.write(str(e))
                if response is not None:
                    return DownloadError(url, str(response.status_code))
                else:
                    return DownloadError(url, str(extract_error_detail(e)))

            # Else write the file bytes even though we have not confirmed the result is an image
            # Write to a tmp file and rename to ensure we don't only partially write an image if
            # we crash mid-write
            with open(cache_file + ".tmp", "wb") as f:
                f.write(image_bytes)
            rename(cache_file + ".tmp", cache_file)

    if check_sha:
        downloaded_hash = compute_hash(image_bytes)
        assert image_sha is not None
        if downloaded_hash != image_sha:
            return ImageError(url, ValueError("Mismatched image hash"), downloaded)
    else:
        # Else make sure we actually got an image, and it can be parsed by PIL
        try:
            # Avoid annoying palette transparency warnings filling up the logs
            with warnings.catch_warnings(record=True) as w:
                img = PIL.Image.open(io.BytesIO(image_bytes))
                if min(img.size) == 0:
                    raise ValueError("Zero dimensional image")
        except Exception as e:
            if clear_cache:
                os.remove(cache_file)
            return ImageError(url, e, downloaded)

    return url, cache_file


def download_pixmo_urls(
    data: datasets.Dataset,
    n_processes,
    check_sha,
    request_kwargs=None,
    cache_only=False,
    verify=True,
    retry_failures=False,
) -> Dict[str, str]:
    """Download urls from a PixMo dataset, return a map of urls->filename"""
    if check_sha:
        urls_and_shas = list(dict(zip(data["image_url"], data["image_sha256"])).items())
    else:
        urls_and_shas = [(url, None) for url in list(set(data["image_url"]))]

    # Randomize order so resuming is more convenient, speed is more predictable,
    # and to distribute requests across different domains
    urls_and_shas.sort(key=lambda x: x[0])
    np.random.RandomState(58713).shuffle(urls_and_shas)

    logging.info(f"Getting files for {len(urls_and_shas)} image URLs")
    makedirs(PIXMO_IMAGES, exist_ok=True)
    if request_kwargs is None:
        request_kwargs = dict(timeout=70)
    if not verify:
        request_kwargs["verify"] = False
        urllib3.disable_warnings()

    images = []
    to_save = [
        (url, image_sha, check_sha, cache_only, request_kwargs, retry_failures)
        for url, image_sha in urls_and_shas
    ]
    pbar = tqdm(total=len(to_save), desc=f"{0}/{len(to_save)}", leave=False)
    (
        image_error,
        image_err_types,
        download_err,
        download_err_types,
        success,
        last_success,
    ) = 0, Counter(), 0, Counter(), 0, 0

    if n_processes != 1:

        def _iter():
            with multiprocessing.Pool(
                processes=n_processes, initializer=setup_pil
            ) as pool:
                for val in pool.imap_unordered(_download_images, to_save):
                    yield val
    else:
        setup_pil()

        def _iter():
            for val in to_save:
                yield _download_images(val)

    found_urls = {}
    bad_download_urls = []
    for val_num, val in enumerate(_iter()):
        if isinstance(val, ImageError):
            image_error += 1
            image_err_types[str(val.exception)] += 1
            if val.downloaded:
                bad_download_urls.append(val.url)
        elif isinstance(val, DownloadError):
            download_err += 1
            download_err_types[str(val.exception)] += 1
        else:
            url, filename = val
            found_urls[url] = filename
            success += 1
        pbar.update(1)
        pbar.set_description(
            f"dl_er={download_err} file_err={image_error}",
            refresh=False,
        )
        # if val_num % 100 == 0:
        #     # print(f"image_err_types: {image_err_types}")
        #     print(f"download_err_types: {download_err_types}")
        #     print(
        #         f"bad_download_urls: {random.sample(bad_download_urls, min(3, len(bad_download_urls)))}"
        #     )
    pbar.close()
    logging.info(
        f"Got images for {len(found_urls)}/{len(urls_and_shas)} ({len(found_urls) / len(urls_and_shas) * 100:0.2f}%) image URLs"
    )
    with open(join(PIXMO_IMAGES, "bad_download_urls.txt"), "a") as f:
        for url in bad_download_urls:
            f.write(url + "\n")
    return found_urls


def filter_and_group_data(
    data: datasets.Dataset, url_to_path: Dict, check_sha: bool
) -> datasets.Dataset:
    """
    Groups a pixmo datasets so each row contains all annotation for one image, and add
    images path using `url_to_path`, removing rows that do not exist in `url_to_path`
    """
    grouped_by_url = defaultdict(list)
    for example in data:
        if example["image_url"] not in url_to_path:
            continue
        grouped_by_url[example["image_url"]].append(example)

    grouped_examples = []
    for image_url, examples in grouped_by_url.items():
        grouped = dict(
            image_url=image_url,
            image=url_to_path[image_url],
        )
        if "image_sha256" in examples[0] and not check_sha:
            assert all(
                examples[0]["image_sha256"] == ex["image_sha256"] for ex in examples
            )
            grouped["original_sha256"] = examples[0]["image_sha256"]
        annotations = defaultdict(list)
        for ex in examples:
            for k, v in ex.items():
                if k not in ["image_url", "image_sha256"]:
                    annotations[k].append(v)
        grouped.update(annotations)
        grouped_examples.append(grouped)
    return datasets.Dataset.from_list(grouped_examples)
