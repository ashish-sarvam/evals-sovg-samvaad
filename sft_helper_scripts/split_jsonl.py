#!/usr/bin/env python3
"""
Script to split combined_non_samvaad_unrolled.jsonl into train/val.

- Val: fixed 500 samples
- Train: remaining samples

Output:
- non_samvaad_train.jsonl
- non_samvaad_val.jsonl

    Usage:
        python split_and_merge.py
"""

import random
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Input file
INPUT_FILE = "/home/sft/data/sft-text-data/non-thinking/v1/samvaad/0202/combined_final_0202_fixed.jsonl"

# Output files
OUTPUT_DIR = Path("/home/sft/data/sft-text-data/non-thinking/v1/samvaad/0202")
OUTPUT_TRAIN = OUTPUT_DIR / "combined_final_0202_train.jsonl"
OUTPUT_VAL = OUTPUT_DIR / "combined_final_0202_val.jsonl"

# Fixed val size
VAL_SIZE = 500

# Random seed
RANDOM_SEED = 123


# =============================================================================
# Functions
# =============================================================================


def load_jsonl(filepath: str) -> list[str]:
    """Load JSONL file and return list of lines."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def write_jsonl(filepath: Path, lines: list[str]):
    """Write lines to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    print("=" * 70)
    print("SPLIT SCRIPT")
    print("=" * 70)

    # Set random seed
    random.seed(RANDOM_SEED)

    # Step 1: Load input file
    print(f"\n1. Loading: {INPUT_FILE}")
    all_lines = load_jsonl(INPUT_FILE)
    print(f"   Loaded {len(all_lines)} lines")

    # Step 2: Shuffle
    print(f"\n2. Shuffling...")
    random.shuffle(all_lines)

    # Step 3: Split (fixed 500 for val)
    print(f"\n3. Splitting (val={VAL_SIZE} fixed, train=rest)")

    val_lines = all_lines[:VAL_SIZE]
    train_lines = all_lines[VAL_SIZE:]

    print(f"   Val: {len(val_lines)} lines")
    print(f"   Train: {len(train_lines)} lines")

    # Step 4: Write output
    print(f"\n4. Writing output files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUTPUT_TRAIN, train_lines)
    print(f"   Written: {OUTPUT_TRAIN} ({len(train_lines)} lines)")

    write_jsonl(OUTPUT_VAL, val_lines)
    print(f"   Written: {OUTPUT_VAL} ({len(val_lines)} lines)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input: {len(all_lines)} lines")
    print(f"  Train: {len(train_lines)} lines")
    print(f"  Val: {len(val_lines)} lines")
    print(f"  Total: {len(train_lines) + len(val_lines)} lines")
    print("=" * 70)


if __name__ == "__main__":
    main()
