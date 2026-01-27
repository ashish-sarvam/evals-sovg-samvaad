#!/usr/bin/env python3
"""
LLM-based comparison of tau-bench results.

Uses GPT-5-chat (Azure by default) to analyze failures with concrete labels.

Example usage:
    python llm_compare.py \
        --file1 results/model1_results.json \
        --file2 results/model2_results.json \
        --output results/comparisons/analysis.json

For heuristic analysis (no LLM):
    python llm_compare.py --file1 ... --file2 ... --no-llm
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    from analysis.llm_comparison import main
    main()


