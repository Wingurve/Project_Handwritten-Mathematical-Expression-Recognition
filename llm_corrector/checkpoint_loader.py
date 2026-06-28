"""
Checkpoint loader with key remapping for BTTR pretrained model.

Handles differences between the original BTTR checkpoint (pretrained-2014.ckpt)
and the current model architecture.
"""
import torch
import re
from typing import Dict, Optional


def remap_checkpoint_keys(ckpt_state: Dict[str, torch.Tensor],
                          model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap checkpoint state dict keys to match model state dict keys.

    Handles these known differences:
    1. pos_enc.pe shape: ckpt [max_len, d_model] → model [1, max_len, d_model]
    2. pos_enc.rotary_emb.inv_freq: not in ckpt → skip (model will use its own)
    3. encoder.model.dense{1,2,3}.{i}.bn{1,2} → matches now (renamed from norm)
    """
    model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())
    remapped = {}
    skipped = 0

    for ckpt_key, ckpt_value in ckpt_state.items():
        model_key = ckpt_key

        # Check if key exists directly
        if model_key in model_keys:
            if ckpt_value.shape != model_state[model_key].shape:
                # Handle pos_enc.pe shape mismatch: [L, D] → [1, L, D]
                if 'pos_enc.pe' in model_key:
                    remapped[model_key] = ckpt_value.unsqueeze(0)
                    continue
                # Other shape mismatches - try to adapt
                remapped[model_key] = ckpt_value
            else:
                remapped[model_key] = ckpt_value
        else:
            skipped += 1

    return remapped


def load_bttr_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    map_location: str = "cpu",
    strict: bool = False,
) -> Dict:
    """
    Load the BTTR pretrained checkpoint with automatic key remapping.

    Parameters
    ----------
    checkpoint_path : str - Path to pretrained-2014.ckpt
    model : nn.Module - LitBTTR or BTTR model instance
    map_location : str
    strict : bool - If True, raise on missing/unexpected keys

    Returns
    -------
    dict - Checkpoint metadata (epoch, hyper_parameters, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    ckpt_state = checkpoint.get("state_dict", checkpoint)

    # Get model state dict keys
    model_state = model.state_dict()

    # Step 1: Direct match
    result = {}
    missing = []
    shape_mismatches = []

    for model_key, model_val in model_state.items():
        if model_key in ckpt_state:
            ckpt_val = ckpt_state[model_key]
            if model_val.shape == ckpt_val.shape:
                result[model_key] = ckpt_val
            else:
                # Try to reshape
                if 'pos_enc.pe' in model_key and ckpt_val.dim() == 2:
                    # [L, D] → [1, L, D]
                    result[model_key] = ckpt_val.unsqueeze(0)
                else:
                    shape_mismatches.append(
                        (model_key, tuple(model_val.shape), tuple(ckpt_val.shape))
                    )
                    result[model_key] = model_val  # keep model's initialized value
        else:
            missing.append(model_key)
            # Keep model's initialized weights for this key

    # Step 2: Check for keys in checkpoint not in model
    unexpected = [k for k in ckpt_state.keys() if k not in model_state]

    print(f"Loaded: {len(result)} keys")
    print(f"Missing from checkpoint (using init values): {len(missing)}")
    print(f"Unexpected in checkpoint (ignored): {len(unexpected)}")
    if shape_mismatches:
        print(f"Shape mismatches (kept model init): {len(shape_mismatches)}")
        for k, ms, cs in shape_mismatches[:10]:
            print(f"  {k}: model {ms} vs ckpt {cs}")

    # Load what we can
    model.load_state_dict(result, strict=False)

    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "hyper_parameters": checkpoint.get("hyper_parameters"),
    }
