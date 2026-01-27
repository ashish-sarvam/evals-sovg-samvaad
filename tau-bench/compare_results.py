#!/usr/bin/env python3
# Copyright Sierra
"""
Compare two tau-bench result files and analyze failures.

This is a wrapper script that calls the analysis module.

Usage:
    python compare_results.py --file1 results/model1.json --file2 results/model2.json
    python compare_results.py --file1 results/model1.json --file2 results/model2.json --show-traj
    python compare_results.py --file1 results/model1.json --file2 results/model2.json -o comparison.json
"""

from analysis.main import main

if __name__ == "__main__":
    main()
