"""
LoRA (Low-Rank Adaptation) utilities for Molmo model fine-tuning.
"""

from typing import List, Optional, Union, Dict, Any
import logging
import torch
import torch.nn as nn

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from .config import TrainConfig
from .model import Molmo

__all__ = [
    "wrap_model_with_lora",
    "get_default_lora_target_modules",
    "create_lora_config",
    "PEFT_AVAILABLE",
]

log = logging.getLogger(__name__)


def get_default_lora_target_modules() -> List[str]:
    """
    Get default target modules for LoRA adaptation in Molmo.

    Based on the actual Molmo model structure:
    - OLMoSequentialBlock: att_proj (fused QKV), ff_proj, attn_out
    - OLMoLlamaBlock: q_proj, k_proj, v_proj, ff_proj1, ff_proj2, attn_out
    - OLMoEBlock (MoE): att_proj, ffn, attn_out

    Returns:
        List of module names to apply LoRA to.
    """
    return [
        # Sequential block attention and MLP
        "att_proj",  # Fused QKV projection in OLMoSequentialBlock
        "ff_proj",  # Feed-forward projection in OLMoSequentialBlock
        "attn_out",  # Attention output projection (shared across block types)
        # Llama-style block separate projections
        "q_proj",  # Query projection in OLMoLlamaBlock
        "k_proj",  # Key projection in OLMoLlamaBlock
        "v_proj",  # Value projection in OLMoLlamaBlock
        "ff_proj1",  # First FF projection in OLMoLlamaBlock
        "ff_proj2",  # Second FF projection in OLMoLlamaBlock
        # Vision components (optional - can be disabled for LLM-only fine-tuning)
        "vision_backbone.image_projector",  # Vision-language connector
    ]


def create_lora_config(
    cfg: TrainConfig,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
) -> "LoraConfig":
    """
    Create a LoRA configuration from training config.

    Args:
        cfg: Training configuration containing LoRA parameters
        target_modules: Override target modules. If None, uses cfg.lora_target_modules
                       or defaults.

    Returns:
        LoraConfig object for PEFT
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is not available. Install it with: pip install peft>=0.7.0"
        )

    if target_modules is None:
        target_modules = cfg.lora_target_modules
        if target_modules is None:
            target_modules = get_default_lora_target_modules()

    log.info(f"Creating LoRA config with target modules: {target_modules}")

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        modules_to_save=modules_to_save,
        task_type=TaskType.CAUSAL_LM,  # Molmo is a causal language model
        inference_mode=False,  # We're training, not inferencing
    )


def wrap_model_with_lora(
    model: Molmo,
    cfg: TrainConfig,
    prepare_for_kbit: bool = False,
    keep_trainable_modules: Optional[List[str]] = None,
) -> Union[PeftModel, Molmo]:
    """
    Wrap a Molmo model with LoRA adapters.

    Args:
        model: The Molmo model to wrap
        cfg: Training configuration containing LoRA settings
        prepare_for_kbit: Whether to prepare model for k-bit training

    Returns:
        LoRA-wrapped model if LoRA is enabled, otherwise original model
    """
    if not cfg.use_lora:
        log.info("LoRA not enabled, returning original model")
        return model

    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is not available. Install it with: pip install peft>=0.7.0"
        )

    log.info("Wrapping model with LoRA adapters...")

    # Prepare model for k-bit training if requested
    if prepare_for_kbit:
        log.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    # Create LoRA configuration
    lora_config = create_lora_config(cfg, modules_to_save=keep_trainable_modules)

    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in peft_model.parameters())

    log.info(
        f"LoRA applied successfully. "
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return peft_model


def get_lora_state_dict(model: Union[PeftModel, Molmo]) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA adapter state dict from a PEFT model.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Dictionary containing only LoRA adapter weights
    """
    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PEFT model")

    return model.state_dict()


def save_lora_adapters(model: Union[PeftModel, Molmo], save_path: str) -> None:
    """
    Save only the LoRA adapter weights.

    Args:
        model: PEFT model with LoRA adapters
        save_path: Path to save the adapter weights
    """
    if not isinstance(model, PeftModel):
        raise ValueError("Model is not a PEFT model")

    log.info(f"Saving LoRA adapters to {save_path}")
    model.save_pretrained(save_path)


def load_lora_adapters(
    base_model: Molmo, adapter_path: str, is_trainable: bool = True
) -> PeftModel:
    """
    Load LoRA adapters onto a base model.

    Args:
        base_model: Base Molmo model
        adapter_path: Path to LoRA adapter weights
        is_trainable: Whether to load adapters in trainable mode

    Returns:
        Model with loaded LoRA adapters
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is not available. Install it with: pip install peft>=0.7.0"
        )

    log.info(f"Loading LoRA adapters from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )

    return model


def print_lora_parameters(model: Union[PeftModel, Molmo]) -> None:
    """
    Print information about LoRA parameters in the model.

    Args:
        model: Model to analyze (PEFT model or regular model)
    """
    if isinstance(model, PeftModel):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"=== LoRA Model Parameter Summary ===")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

        print(f"\n=== LoRA Modules ===")
        for name, module in model.named_modules():
            if hasattr(module, "lora_A"):
                # assert the parameter is trainable
                assert module.lora_A[list(module.lora_A.keys())[0]].weight.requires_grad
                assert module.lora_B[list(module.lora_B.keys())[0]].weight.requires_grad

                rank = module.lora_A[list(module.lora_A.keys())[0]].weight.shape[0]
                print(f"  {name}: rank={rank}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"=== Regular Model Parameter Summary ===")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print("No LoRA adapters found.")
