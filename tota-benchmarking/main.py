#!/usr/bin/env python
"""
Main entry point for running the tota benchmarking tool.
"""
import asyncio
import argparse
import logging
from tota_benchmarking import run_benchmark, run_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from common libraries
for lib_logger in ["openai", "httpx", "asyncio", "urllib3", "requests", "azure"]:
    logging.getLogger(lib_logger).setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TOTA Benchmarking Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Path to JSONL file with pre-generated responses for evaluation only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.eval_only:
            asyncio.run(
                run_evaluation(args.eval_only, args.config, args.max_samples)
            )
        else:
            asyncio.run(run_benchmark(args.config, args.max_samples))
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        logger.exception(e)
