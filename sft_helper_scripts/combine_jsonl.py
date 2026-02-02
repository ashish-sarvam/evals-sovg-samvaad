#!/usr/bin/env python3
"""Script to combine multiple JSONL files into one."""

import argparse
import json
from pathlib import Path

import yaml

# Default config file path (same directory as script)
DEFAULT_CONFIG = Path(__file__).parent / "combine_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def combine_jsonl_files(
    input_files: list[str], output_path: Path
) -> tuple[int, dict]:
    """
    Combine multiple JSONL files into one.

    Args:
        input_files: List of paths to JSONL files to combine
        output_path: Path to save the combined file

    Returns:
        Tuple of (total lines written, file stats dict)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    file_stats = {}

    with open(output_path, "w", encoding="utf-8") as outfile:
        for filepath in input_files:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"Warning: File not found, skipping: {filepath}")
                file_stats[str(filepath)] = {"lines": 0, "status": "not_found"}
                continue

            print(f"Processing: {filepath}")
            file_lines = 0

            with open(filepath, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        # Validate JSON
                        try:
                            json.loads(line)
                            outfile.write(line + "\n")
                            file_lines += 1
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON line skipped: {e}")

            print(f"  -> {file_lines} lines")
            file_stats[str(filepath)] = {"lines": file_lines, "status": "ok"}
            total_lines += file_lines

    return total_lines, file_stats


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple JSONL files into one."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        exit(1)

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    files_to_combine = config["files_to_combine"]
    output_dir = Path(config["output_dir"])
    output_filename = config["output_filename"]

    output_path = output_dir / output_filename
    summary_path = output_dir / "combined_summary.json"

    # Check all files exist before proceeding
    missing_files = []
    for filepath in files_to_combine:
        if not Path(filepath).exists():
            missing_files.append(filepath)

    if missing_files:
        print("ERROR: The following files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\n{len(missing_files)} file(s) missing. Aborting.")
        exit(1)

    print(f"Combining {len(files_to_combine)} files...")
    print(f"Output: {output_path}")
    print("-" * 50)

    total, file_stats = combine_jsonl_files(files_to_combine, output_path)

    print("-" * 50)
    print(f"Total lines written: {total}")
    print(f"Output saved to: {output_path}")

    # Generate summary JSON
    summary = {
        "output_file": str(output_path),
        "total_lines": total,
        "num_files": len(files_to_combine),
        "files": [],
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'File':<60} {'Lines':>10} {'%':>8}")
    print("-" * 80)

    for filepath in files_to_combine:
        stats = file_stats.get(filepath, {"lines": 0, "status": "unknown"})
        lines = stats["lines"]
        percentage = (lines / total * 100) if total > 0 else 0

        # Short name for display
        short_name = Path(filepath).name

        print(f"{short_name:<60} {lines:>10} {percentage:>7.2f}%")

        summary["files"].append(
            {
                "path": filepath,
                "name": short_name,
                "lines": lines,
                "percentage": round(percentage, 2),
                "status": stats["status"],
            }
        )

    print("-" * 80)
    print(f"{'TOTAL':<60} {total:>10} {'100.00%':>8}")

    # Write summary JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
