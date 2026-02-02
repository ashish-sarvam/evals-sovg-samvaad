"""Unroll SFT pairs at every user turn.

This script takes SFT pairs and creates multiple sub-pairs by breaking them
at each user turn boundary. This increases training data and teaches the model
to respond appropriately at different stages of a conversation.

Example:
    Original pair: U1, A1, U2, A2, T1, A3, T2, A4, U3, A5

    Unrolled pairs:
    1. U1, A1
    2. U1, A1, U2, A2, T1, A3, T2, A4
    3. U1, A1, U2, A2, T1, A3, T2, A4, U3, A5

All messages include 'enable_thinking': true.

Usage:
    python unroll_data.py
"""

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration - Files to process
# =============================================================================

OUTPUT_DIR = Path(
    "/home/sft/data/sft-text-data/non-thinking/v1/samvaad/0102/non_samvaad"
)

files_to_unroll = [
    "/home/sft/data/sft-text-data/non-thinking/v1/chat/source_files/chat_hindi_translated.jsonl",
]


# =============================================================================
# Message Processing
# =============================================================================


def find_user_turn_boundaries(messages: list[dict[str, Any]]) -> list[int]:
    """
    Find the indices of all user messages in the conversation.

    Returns:
        List of indices where user messages appear
    """
    user_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_indices.append(i)
    return user_indices


def find_assistant_end_after_user(
    messages: list[dict[str, Any]],
    user_idx: int,
) -> int:
    """
    Find the end index of the assistant response sequence after a user message.

    After a user message, we may have:
    - Assistant message (possibly with tool_calls)
    - Tool response(s)
    - More assistant messages (if iterating on tools)
    - Eventually ending with an assistant message

    We want to include all messages until we hit the next user message
    or end of conversation.

    Returns:
        The index (exclusive) where we should slice the messages
    """
    idx = user_idx + 1

    while idx < len(messages):
        current_role = messages[idx].get("role", "")

        # If we hit another user message, stop before it
        if current_role == "user":
            # But we need to make sure we end on an assistant message
            # Go back to find the last assistant message
            end_idx = idx
            while (
                end_idx > user_idx + 1
                and messages[end_idx - 1].get("role") != "assistant"
            ):
                end_idx -= 1
            return end_idx

        idx += 1

    # We reached the end - make sure we end on assistant
    end_idx = len(messages)
    while (
        end_idx > user_idx + 1
        and messages[end_idx - 1].get("role") != "assistant"
    ):
        end_idx -= 1

    return end_idx


# =============================================================================
# Verification
# =============================================================================


def verify_pair_ends_on_assistant(pair: dict[str, Any]) -> tuple[bool, str]:
    """Verify that a pair ends on an assistant turn."""
    messages = pair.get("messages", [])

    if not messages:
        return False, "Empty messages list"

    last_message = messages[-1]
    last_role = last_message.get("role", "")

    if last_role != "assistant":
        return (
            False,
            f"Last message role is '{last_role}', expected 'assistant'",
        )

    return True, ""


def verify_pair_has_user_message(pair: dict[str, Any]) -> tuple[bool, str]:
    """Verify that a pair has at least one user message."""
    messages = pair.get("messages", [])

    has_user = any(m.get("role") == "user" for m in messages)

    if not has_user:
        return False, "No user message found in pair"

    return True, ""


def verify_pair(pair: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Run all verification checks on a pair.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    valid, error = verify_pair_ends_on_assistant(pair)
    if not valid:
        errors.append(error)

    valid, error = verify_pair_has_user_message(pair)
    if not valid:
        errors.append(error)

    return len(errors) == 0, errors


# =============================================================================
# Unrolling Logic
# =============================================================================


def unroll_single_pair(
    pair: dict[str, Any],
    original_idx: int,
) -> list[dict[str, Any]]:
    """
    Unroll a single SFT pair into multiple pairs at user turn boundaries.

    Args:
        pair: The original SFT pair
        original_idx: Index of this pair in the original file (for tracking)

    Returns:
        List of unrolled pairs
    """
    messages = pair.get("messages", [])

    if not messages:
        return []

    # Find all user turn boundaries
    user_indices = find_user_turn_boundaries(messages)

    if not user_indices:
        return []

    unrolled_pairs = []

    for turn_idx, user_idx in enumerate(user_indices):
        # Find where this user turn's response ends
        end_idx = find_assistant_end_after_user(messages, user_idx)

        # Slice messages from start to end_idx
        sub_messages = messages[:end_idx]

        # Add enable_thinking: true to all messages
        sub_messages = [
            {**msg, "enable_thinking": True} for msg in sub_messages
        ]

        # Create the new pair - keep original metadata
        new_pair = {
            "num_messages": len(sub_messages),
            "messages": sub_messages,
            "unroll_metadata": {
                "original_pair_idx": original_idx,
                "user_turn_idx": turn_idx,
                "total_user_turns": len(user_indices),
                "is_final_turn": (turn_idx == len(user_indices) - 1),
                "message_range": f"0:{end_idx}",
            },
        }

        # Copy over any existing metadata from original pair
        for key in ["common_metadata", "tools", "num_tools", "language"]:
            if key in pair:
                new_pair[key] = pair[key]

        unrolled_pairs.append(new_pair)

    return unrolled_pairs


def unroll_jsonl_file(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Unroll all pairs in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file

    Returns:
        Metrics dictionary
    """
    all_unrolled = []
    original_count = 0

    # Verification tracking
    verified_count = 0
    failed_verification_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                pair = json.loads(line)
                original_count += 1

                # Progress logging every 5000 pairs
                if original_count % 5000 == 0:
                    print(
                        f"  Processed {original_count} pairs, generated {len(all_unrolled)} unrolled pairs..."
                    )

                unrolled = unroll_single_pair(pair, i)

                for up in unrolled:
                    # Verify the pair
                    is_valid, errors = verify_pair(up)

                    if not is_valid:
                        failed_verification_count += 1
                        # Skip invalid pairs
                        continue

                    verified_count += 1
                    all_unrolled.append(up)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {i + 1}: {e}")
                continue

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_unrolled:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    metrics = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_pairs": original_count,
        "unrolled_pairs": len(all_unrolled),
        "expansion_ratio": len(all_unrolled) / original_count
        if original_count > 0
        else 0,
        "verification": {
            "passed": verified_count,
            "failed": failed_verification_count,
            "pass_rate": verified_count
            / (verified_count + failed_verification_count)
            if (verified_count + failed_verification_count) > 0
            else 0,
        },
    }

    return metrics


# =============================================================================
# Sample File Creation
# =============================================================================


def create_sample_file(
    input_path: Path, output_dir: Path, num_samples: int = 5
) -> Path:
    """
    Create a sample file with first N pairs from the input file for inspection.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save sample file
        num_samples: Number of samples to extract

    Returns:
        Path to the sample file
    """
    sample_filename = f"{input_path.stem}_sample_{num_samples}.json"
    sample_path = output_dir / sample_filename

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                samples.append({"line_number": i + 1, "data": pair})
            except json.JSONDecodeError:
                continue

    # Write as pretty-printed JSON for easy inspection
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return sample_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Process all files in the array."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    total_original = 0
    total_unrolled = 0
    low_expansion_files = []

    print("=" * 70)
    print("Unrolling Files")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files to process: {len(files_to_unroll)}")
    print("=" * 70)

    for filepath in files_to_unroll:
        input_path = Path(filepath)

        if not input_path.exists():
            print(f"\n[SKIP] File not found: {filepath}")
            continue

        # Create output filename: original_name_unrolled.jsonl
        output_filename = f"{input_path.stem}_unrolled.jsonl"
        output_path = OUTPUT_DIR / output_filename

        print(f"\n[Processing] {input_path.name}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        metrics = unroll_jsonl_file(input_path, output_path)
        all_metrics.append(metrics)

        total_original += metrics["original_pairs"]
        total_unrolled += metrics["unrolled_pairs"]

        print(f"  Original: {metrics['original_pairs']:,}")
        print(f"  Unrolled: {metrics['unrolled_pairs']:,}")
        print(f"  Expansion: {metrics['expansion_ratio']:.2f}x")
        print(
            f"  Verification: {metrics['verification']['passed']:,} passed, {metrics['verification']['failed']:,} failed"
        )

        # If expansion < 1x, create sample file for inspection
        if metrics["expansion_ratio"] < 1.0:
            sample_path = create_sample_file(
                input_path, OUTPUT_DIR, num_samples=5
            )
            metrics["sample_file"] = str(sample_path)
            low_expansion_files.append(
                {
                    "file": input_path.name,
                    "expansion_ratio": metrics["expansion_ratio"],
                    "sample_file": str(sample_path),
                }
            )
            print(f"  [LOW EXPANSION] Created sample file: {sample_path.name}")

    # Write summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(all_metrics)}")
    print(f"Total original pairs: {total_original:,}")
    print(f"Total unrolled pairs: {total_unrolled:,}")
    print(
        f"Overall expansion: {total_unrolled / total_original:.2f}x"
        if total_original > 0
        else "N/A"
    )

    if low_expansion_files:
        print(
            f"\n[WARNING] {len(low_expansion_files)} files with expansion < 1x:"
        )
        for lef in low_expansion_files:
            print(f"  - {lef['file']}: {lef['expansion_ratio']:.2f}x")

    # Save summary JSON
    summary = {
        "output_dir": str(OUTPUT_DIR),
        "total_files": len(all_metrics),
        "total_original_pairs": total_original,
        "total_unrolled_pairs": total_unrolled,
        "overall_expansion_ratio": total_unrolled / total_original
        if total_original > 0
        else 0,
        "low_expansion_files": low_expansion_files,
        "files": all_metrics,
    }

    summary_path = OUTPUT_DIR / "unroll_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
