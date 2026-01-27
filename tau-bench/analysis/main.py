#!/usr/bin/env python3
# Copyright Sierra
"""
Main entry point for tau-bench analysis tools.

Usage:
    python -m analysis.main --file1 results/model1.json --file2 results/model2.json
    python -m analysis.main --file1 results/model1.json --file2 results/model2.json --show-traj
    python -m analysis.main --file1 results/model1.json --file2 results/model2.json -o comparison.json
"""

import os
import json
import argparse
from datetime import datetime

from .comparison import compare_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare two tau-bench result files and analyze failures"
    )
    parser.add_argument(
        "--file1",
        "-f1",
        type=str,
        required=True,
        help="Path to first results file (model1/baseline)",
    )
    parser.add_argument(
        "--file2",
        "-f2",
        type=str,
        required=True,
        help="Path to second results file (model2/comparison)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save comparison report JSON",
    )
    parser.add_argument(
        "--show-traj",
        action="store_true",
        help="Show trajectory summaries for failed tasks",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all tasks including both-pass",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include tasks that are only in one file (default: only compare common tasks)",
    )

    args = parser.parse_args()

    # Run comparison
    report = compare_results(
        file1_path=args.file1,
        file2_path=args.file2,
        show_trajectory=args.show_traj,
        show_all=args.show_all,
        only_common=not args.include_missing,
    )

    # Default output directory
    output_dir = "results/comparisons"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename from input files if not provided
    if args.output:
        output_path = args.output
    else:
        # Extract model names from filenames
        name1 = os.path.basename(args.file1).replace(".json", "").split("_")[1]
        name2 = os.path.basename(args.file2).replace(".json", "").split("_")[1]
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_path = f"{output_dir}/{name1}_vs_{name2}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved to {output_path}")


if __name__ == "__main__":
    main()

