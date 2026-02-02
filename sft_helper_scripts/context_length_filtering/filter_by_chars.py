#!/usr/bin/env python3
"""
Ultra-fast character-based filtering (approximation).

Rough heuristic: ~3.5-4 tokens per character for this tokenizer.
Use this for quick filtering, then optionally verify with exact tokenization.

Usage:
    python filter_by_chars.py
"""

import json
import time

# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "/home/sft/data/sft-text-data/non-thinking/v1/samvaad/0202/combined_final_0202_train.jsonl"
OUTPUT_PASSED = "/home/ashish_sarvam_ai/filter_32k/pass1_under_30k_chars.jsonl"  # Definitely keep
OUTPUT_FAILED = "/home/ashish_sarvam_ai/filter_32k/pass1_over_30k_chars.jsonl"   # Need tokenizer check

# Conservative threshold: 30k chars
# Samples ≤30k chars → definitely keep (pass1_under_30k_chars.jsonl)
# Samples >30k chars → need Pass 2 tokenizer check (pass1_over_30k_chars.jsonl)
MAX_CHARS = 30000
LOG_EVERY = 100000


def main():
    print("=" * 70)
    print("FAST CHARACTER-BASED FILTER (APPROXIMATION)")
    print("=" * 70)
    print(
        f"Max chars: {MAX_CHARS:,} (conservative threshold, errs on caution)"
    )

    start_time = time.time()
    total = 0
    kept = 0
    removed = 0
    char_lengths = []

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as fin,
        open(OUTPUT_PASSED, "w", encoding="utf-8") as fout_passed,
        open(OUTPUT_FAILED, "w", encoding="utf-8") as fout_failed,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                # Estimate total text length
                total_chars = sum(len(m.get("content", "")) for m in messages)
                char_lengths.append(total_chars)

                if total_chars <= MAX_CHARS:
                    fout_passed.write(line + "\n")
                    kept += 1
                else:
                    fout_failed.write(line + "\n")
                    removed += 1
            except:
                removed += 1

            if total % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed
                print(
                    f"   Processed {total:,} | Passed: {kept:,} | To verify: {removed:,} | Rate: {rate:,.0f}/s"
                )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PASS 1 SUMMARY")
    print("=" * 70)
    print(f"  Total: {total:,}")
    print(f"  Passed (≤{MAX_CHARS:,} chars): {kept:,} ({100 * kept / total:.1f}%)")
    print(f"  Need verification (>{MAX_CHARS:,} chars): {removed:,} ({100 * removed / total:.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({total / elapsed:,.0f} samples/s)")
    print(f"\n  Output files:")
    print(f"    Passed: {OUTPUT_PASSED}")
    print(f"    To verify: {OUTPUT_FAILED}")

    if char_lengths:
        print(f"\n  Char length stats:")
        print(f"    Min: {min(char_lengths):,}")
        print(f"    Max: {max(char_lengths):,}")
        print(f"    Avg: {sum(char_lengths) / len(char_lengths):,.0f}")

    print("=" * 70)
    print("\nPASS 1 COMPLETE!")
    print(f"Next: Run pass2_tokenizer_filter.py on {OUTPUT_FAILED}")
    print("Then: Merge results with merge_results.py")


if __name__ == "__main__":
    main()
