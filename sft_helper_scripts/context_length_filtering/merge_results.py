#!/usr/bin/env python3
"""
Merge Pass 1 and Pass 2 results into final filtered file.

Combines:
- pass1_under_30k_chars.jsonl (definitely kept from char filter)
- pass2_under_32k_tokens.jsonl (passed tokenizer verification)

Usage:
    python merge_results.py
"""

import time

# =============================================================================
# Configuration
# =============================================================================

PASS1_PASSED = "/home/ashish_sarvam_ai/filter_32k/pass1_under_30k_chars.jsonl"
PASS2_PASSED = "/home/ashish_sarvam_ai/filter_32k/pass2_under_32k_tokens.jsonl"

OUTPUT_FINAL = "/home/ashish_sarvam_ai/filter_32k/combined_final_0202_train_filtered_32k.jsonl"

LOG_EVERY = 500000


def count_lines(filepath):
    """Count lines in a file."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main():
    print("=" * 70)
    print("MERGE PASS 1 + PASS 2 RESULTS")
    print("=" * 70)

    start_time = time.time()

    # Count input lines
    print("\n1. Counting input files...")
    pass1_count = count_lines(PASS1_PASSED)
    pass2_count = count_lines(PASS2_PASSED)
    print(f"   Pass 1 (char filter passed): {pass1_count:,}")
    print(f"   Pass 2 (tokenizer filter passed): {pass2_count:,}")
    print(f"   Expected total: {pass1_count + pass2_count:,}")

    # Merge files
    print(f"\n2. Merging to: {OUTPUT_FINAL}")
    total = 0

    with open(OUTPUT_FINAL, "w", encoding="utf-8") as fout:
        # Write pass 1 results
        print("   Writing pass 1 results...")
        with open(PASS1_PASSED, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
                total += 1
                if total % LOG_EVERY == 0:
                    print(f"      Written {total:,} lines...")

        # Write pass 2 results
        print("   Writing pass 2 results...")
        with open(PASS2_PASSED, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
                total += 1
                if total % LOG_EVERY == 0:
                    print(f"      Written {total:,} lines...")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"  Pass 1 samples: {pass1_count:,}")
    print(f"  Pass 2 samples: {pass2_count:,}")
    print(f"  Total merged: {total:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  Final output: {OUTPUT_FINAL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
