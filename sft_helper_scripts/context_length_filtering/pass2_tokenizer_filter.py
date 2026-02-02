#!/usr/bin/env python3
"""
Pass 2: Tokenizer-based filtering on samples that failed character filter.

Takes samples >30k chars and checks if they're actually ≤32k tokens.
Some long-char samples may still fit within token limit.

Usage:
    python pass2_tokenizer_filter.py
"""

import json
import time
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "/home/ashish_sarvam_ai/filter_32k/pass1_over_30k_chars.jsonl"
OUTPUT_PASSED = "/home/ashish_sarvam_ai/filter_32k/pass2_under_32k_tokens.jsonl"
OUTPUT_FAILED = "/home/ashish_sarvam_ai/filter_32k/pass2_over_32k_tokens.jsonl"

TOKENIZER_PATH = "/home/sft/checkpoints/hf_sft_checkpoints/sft-sov-30b-128k-2901/step_25500"

MAX_TOKENS = 32768
NUM_WORKERS = min(32, cpu_count())
LOG_EVERY = 1000

# =============================================================================
# Global tokenizer (initialized per worker)
# =============================================================================
_tokenizer = None


def init_worker(tokenizer_path):
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def count_tokens_for_line(line: str) -> tuple[str, int]:
    """Count tokens for a single line. Returns (line, num_tokens)."""
    global _tokenizer

    line = line.strip()
    if not line:
        return (line, -1)

    try:
        data = json.loads(line)
        messages = data.get("messages", [])
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = _tokenizer.encode(text, add_special_tokens=False)
        return (line, len(tokens))
    except Exception as e:
        return (line, -1)


def main():
    print("=" * 70)
    print("PASS 2: TOKENIZER-BASED FILTERING")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")
    print(f"Max tokens: {MAX_TOKENS:,}")
    print(f"Workers: {NUM_WORKERS}")

    start_time = time.time()

    # Read input file
    print(f"\n1. Reading input file...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]
    print(f"   Loaded {len(all_lines):,} samples to verify")

    if len(all_lines) == 0:
        print("   No samples to process!")
        return

    # Process in parallel
    print(f"\n2. Processing with {NUM_WORKERS} workers...")

    total = 0
    kept = 0
    removed = 0
    token_lengths = []

    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(TOKENIZER_PATH,)) as pool:
        with (
            open(OUTPUT_PASSED, "w", encoding="utf-8") as fout_passed,
            open(OUTPUT_FAILED, "w", encoding="utf-8") as fout_failed,
        ):
            for line, num_tokens in pool.imap(count_tokens_for_line, all_lines, chunksize=50):
                total += 1
                token_lengths.append(num_tokens)

                if 0 < num_tokens <= MAX_TOKENS:
                    fout_passed.write(line + "\n")
                    kept += 1
                else:
                    fout_failed.write(line + "\n")
                    removed += 1

                if total % LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed
                    print(f"   Processed {total:,} | Passed: {kept:,} | Failed: {removed:,} | Rate: {rate:.0f}/s")

    elapsed = time.time() - start_time

    # Statistics
    print("\n" + "=" * 70)
    print("PASS 2 SUMMARY")
    print("=" * 70)
    print(f"  Total verified: {total:,}")
    print(f"  Passed (≤{MAX_TOKENS:,} tokens): {kept:,} ({100 * kept / total:.1f}%)")
    print(f"  Failed (>{MAX_TOKENS:,} tokens): {removed:,} ({100 * removed / total:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({total / elapsed:.0f} samples/s)")
    print(f"\n  Output files:")
    print(f"    Passed: {OUTPUT_PASSED}")
    print(f"    Failed: {OUTPUT_FAILED}")

    if token_lengths:
        valid_lengths = [t for t in token_lengths if t > 0]
        if valid_lengths:
            print(f"\n  Token length stats (verified samples):")
            print(f"    Min: {min(valid_lengths):,}")
            print(f"    Max: {max(valid_lengths):,}")
            print(f"    Avg: {sum(valid_lengths) / len(valid_lengths):,.0f}")

    print("=" * 70)
    print("\nPASS 2 COMPLETE!")
    print("Next: Run merge_results.py to combine pass1 + pass2 results")


if __name__ == "__main__":
    main()
