#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "peft",
#     "accelerate",
#     "safetensors",
# ]
# ///
"""
Merge LoRA weights from gpt_oss_20b into the base model (openai/gpt-oss-20b).

Usage:
    # Simple usage (uses openai/gpt-oss-20b as base by default)
    python merge_gpt_oss_20b.py
    
    # Or with all options:
    python merge_gpt_oss_20b.py \
        --base openai/gpt-oss-20b \
        --out /path/to/output \
        --dtype bf16 \
        --device cuda \
        --max_shard_size 5GB
"""
import argparse
import json
import os
import shutil
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

# Default paths for gpt_oss_20b
BASE_MODEL = "openai/gpt-oss-20b"
LORA_DIR = "/home/ashish_sarvam_ai/models/tinker/gpt_oss_20b"
DEFAULT_OUT_DIR = "/home/ashish_sarvam_ai/models/gpt_oss_20b_merged"


def merge_lora(
    base_model: str,
    lora_dir: str,
    out_dir: str,
    dtype: str = "bf16",
    device: str = "cuda",
    max_shard_size: str = "5GB",
):
    os.makedirs(out_dir, exist_ok=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    print(f"[1/5] Loading tokenizer from base: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    def _load_on(device_map):
        print(f"[2/5] Loading base model on {device_map} with dtype={dtype}")
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    # Try GPU first (single GPU), fallback to CPU if OOM
    try:
        if device == "cuda" and torch.cuda.is_available():
            base = _load_on("cuda")  # force single GPU
        else:
            base = _load_on({"": "cpu"})
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                "[!] CUDA OOM while loading base model. Falling back to CPU load/merge."
            )
            torch.cuda.empty_cache()
            base = _load_on({"": "cpu"})
        else:
            raise

    print(f"[3/5] Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(base, lora_dir)

    print("[4/5] Merging LoRA into base (merge_and_unload)")
    merged = model.merge_and_unload()

    # Important: ensure we save a plain HF model
    merged.config.use_cache = True

    print(f"[5/5] Saving merged model to: {out_dir}")

    # Custom save to bypass MXFP4 weight conversion issues
    _save_model_manual(merged, tok, out_dir, max_shard_size)

    print("✅ Done. Merged model saved to:", out_dir)


def _parse_size(size_str: str) -> int:
    """Parse size string like '5GB' to bytes."""
    size_str = size_str.upper().strip()
    if size_str.endswith("GB"):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    else:
        return int(size_str)


def _save_model_manual(model, tokenizer, out_dir: str, max_shard_size: str = "5GB"):
    """
    Manually save model state dict using safetensors, bypassing transformers'
    weight conversion logic that fails with MXFP4 dequantized models.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get state dict (move to CPU if needed)
    print("  -> Collecting state dict...")
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.detach().cpu()
    for name, buf in model.named_buffers():
        state_dict[name] = buf.detach().cpu()

    # Calculate shard sizes
    max_bytes = _parse_size(max_shard_size)

    # Group tensors into shards
    shards = []
    current_shard = {}
    current_size = 0

    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Save shards
    weight_map = {}
    total_size = 0

    if len(shards) == 1:
        # Single file
        print("  -> Saving model.safetensors...")
        save_file(shards[0], os.path.join(out_dir, "model.safetensors"))
        for name, tensor in shards[0].items():
            weight_map[name] = "model.safetensors"
            total_size += tensor.numel() * tensor.element_size()
    else:
        # Multiple shards
        for i, shard in enumerate(shards, 1):
            shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            print(f"  -> Saving {shard_name}...")
            save_file(shard, os.path.join(out_dir, shard_name))
            for name, tensor in shard.items():
                weight_map[name] = shard_name
                total_size += tensor.numel() * tensor.element_size()

        # Save index file
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    # Save config
    print("  -> Saving config.json...")
    model.config.save_pretrained(out_dir)

    # Save tokenizer
    print("  -> Saving tokenizer...")
    tokenizer.save_pretrained(out_dir)

    # Copy generation_config if exists
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Merge gpt_oss_20b LoRA weights into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple usage (uses openai/gpt-oss-20b as base)
    python merge_gpt_oss_20b.py
    
    # Custom output directory
    python merge_gpt_oss_20b.py --out /custom/output/path
    
    # Use CPU if GPU OOM
    python merge_gpt_oss_20b.py --device cpu
    
    # Use fp16 instead of bf16
    python merge_gpt_oss_20b.py --dtype fp16
        """,
    )
    ap.add_argument(
        "--base",
        default=BASE_MODEL,
        help=f"HF repo ID or local path of the base model (default: {BASE_MODEL})",
    )
    ap.add_argument(
        "--lora",
        default=LORA_DIR,
        help=f"Path to LoRA adapter dir (default: {LORA_DIR})",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output dir for merged model (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model loading (default: bf16)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for merging (default: cuda)",
    )
    ap.add_argument(
        "--max_shard_size",
        default="5GB",
        help="Max shard size for saved model (default: 5GB)",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("gpt_oss_20b LoRA Merge Script")
    print("=" * 60)
    print(f"Base model:  {args.base}")
    print(f"LoRA dir:    {args.lora}")
    print(f"Output dir:  {args.out}")
    print(f"Dtype:       {args.dtype}")
    print(f"Device:      {args.device}")
    print("=" * 60)

    merge_lora(
        base_model=args.base,
        lora_dir=args.lora,
        out_dir=args.out,
        dtype=args.dtype,
        device=args.device,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
