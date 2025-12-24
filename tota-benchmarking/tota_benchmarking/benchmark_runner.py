"""
Main benchmark orchestration module.
"""

import asyncio
import datetime
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .config_handler import load_config
from .data_processing import load_benchmark_dataset, process_sample
from .prompt_converters import get_adapter
from .reporting import (
    print_benchmark_summary,
    process_benchmark_results,
    save_benchmark_results,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Disable logging for specific loggers that might be too verbose
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)


async def run_benchmark(
    config_path: str = "config.json", max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a benchmark with all configured models on the dataset.

    Args:
        config_path: Path to the configuration file.
        max_samples: Maximum number of samples to process (None for all).

    Returns:
        Dictionary with benchmark results.
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        dataset_name = config["dataset_name"]
        revision = config["revision"]
        models_config = config["models_config"]
        llm_config = config.get("llm_config", {"temperature": 0.1, "max_tokens": 512})
        max_concurrent_tasks = config.get("max_concurrent_tasks", 5)

        # Load dataset
        logger.info(f"Loading dataset {dataset_name} (revision: {revision})")
        dataset = load_benchmark_dataset(
            dataset_name, revision, token=os.getenv("HF_READ_TOKEN")
        )
        if "system_prompt" in dataset[0]:
            logger.info(
                "**IMPORTANT** Dataset contains system_prompt, using system_prompt directly."
            )

        # Initialize results structure
        benchmark_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_samples": (
                len(dataset) if not max_samples else min(max_samples, len(dataset))
            ),
            "models": {},
            "samples": {},
        }

        # Process all models in parallel
        model_tasks = []
        for model_config in models_config:
            # Create a separate semaphore for each model
            model_rate_limit = model_config.get("rate_limit", max_concurrent_tasks)
            model_semaphore = asyncio.Semaphore(model_rate_limit)

            if "system_prompt" in dataset[0]:
                is_structured = model_config.get("structured", False)
                if is_structured:
                    logger.info(
                        f"**IMPORTANT** Using structured prompt for {model_config['name']}."
                    )
                else:
                    logger.info(
                        f"**IMPORTANT** Using unstructured prompt for {model_config['name']}."
                    )

            task = process_model(
                model_config,
                dataset,
                benchmark_results,
                llm_config,
                model_semaphore,
                max_samples,
            )
            model_tasks.append(task)

        # Wait for all models to complete
        await asyncio.gather(*model_tasks)

        # Process and save final results
        final_results = process_benchmark_results(benchmark_results)
        output_path = save_benchmark_results(final_results)

        # Print summary
        print_benchmark_summary(final_results)
        logger.info(f"Benchmark results saved to {output_path}")

        return final_results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        logger.exception(e)
        raise


async def process_model(
    model_config: Dict[str, Any],
    dataset: Any,
    benchmark_results: Dict[str, Any],
    llm_config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_samples: Optional[int] = None,
) -> None:
    """
    Process a single model against all samples in the dataset.

    Args:
        model_config: Configuration for the model.
        dataset: The dataset to benchmark against.
        benchmark_results: Dictionary to store results.
        llm_config: Configuration for the LLM.
        semaphore: Semaphore for concurrency control specific to this model.
        max_samples: Maximum number of samples to process.
    """
    model_name = model_config["name"]
    model_url = model_config.get("url", "")
    model_provider = model_config["provider"]
    base_model_type = model_config["base_model_type"]
    model_path = model_config["path"]
    model_structured = model_config.get("structured", False)

    logger.info(f"Running benchmark for {model_name}...")

    # Initialize model results
    benchmark_results["models"][model_name] = {
        "total_evaluated": 0,
        "total_failures": 0,
        "total_adequate": 0,
    }

    # Get the appropriate adapter for this model
    system_prompt_template_name, conversation_adapter = get_adapter(model_name)

    # Create tasks for all samples
    tasks = []
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        task = process_sample(
            sample,
            idx,
            model_name,
            model_path,
            model_url,
            model_provider,
            base_model_type,
            llm_config,
            model_config,
            conversation_adapter,
            system_prompt_template_name,
            model_structured,
            semaphore,
        )
        tasks.append(task)

    # Process all tasks with tqdm progress tracking
    with tqdm(total=len(tasks), desc=f"Model {model_name}", unit="sample") as pbar:
        completed = 0
        for result in asyncio.as_completed(tasks):
            result_data = await result
            if result_data:
                process_sample_result(result_data, model_name, benchmark_results)

            # Update progress bar
            completed += 1
            pbar.update(1)


def process_sample_result(
    result: Dict[str, Any], model_name: str, benchmark_results: Dict[str, Any]
) -> None:
    """
    Process a single sample result and update the benchmark results.

    Args:
        result: Result from processing a sample.
        model_name: Name of the model.
        benchmark_results: Dictionary to store results.
    """
    benchmark_results["models"][model_name]["total_evaluated"] += 1

    sample_id = str(result["idx"])
    if sample_id not in benchmark_results["samples"]:
        benchmark_results["samples"][sample_id] = {
            "sample_id": result["idx"],
            "input_messages": result["updated_conversation_thread"],
            "golden_response": result["golden_response"],
            "model_results": {},
        }

    # Add model results to this sample
    benchmark_results["samples"][sample_id]["model_results"][model_name] = {
        "model_response": result["model_response"],
        "evaluation_results": result["evaluation_results"],
        "has_failed": result["has_failed"],
        "is_adequate": result["is_adequate"],
        "inference_time": result["inference_time"],
    }

    if result["has_failed"]:
        benchmark_results["models"][model_name]["total_failures"] += 1

    if result["is_adequate"]:
        benchmark_results["models"][model_name]["total_adequate"] += 1
