import asyncio
import datetime
import json
import logging
from typing import Any, Dict, List, Optional

import tqdm
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from .benchmark_runner import process_sample_result
from .evaluator import evaluate_response, process_evaluation_results
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


def get_models_from_samples(samples: List[Dict[str, Any]]) -> List[str]:
    models = set()
    for sample in samples:
        for model_name in sample["model_responses"].keys():
            models.add(model_name)
    return list(models)


async def process_single_model(
    golden_response,
    model_response,
    model_name,
    id,
    system_prompt,
    sample_semaphore,
    p_bar,
):
    async with sample_semaphore:
        evaluator_results = await evaluate_response(
            model_response, golden_response, system_prompt
        )
        p_bar.update(1)
    processed_results = process_evaluation_results(evaluator_results)
    cleaned_model_response = evaluator_results.get("cleaned_model_response", "")
    cleaned_golden_response = evaluator_results.get("cleaned_golden_response", "")
    return {
        "idx": id,
        "model_name": model_name,
        "has_failed": processed_results.get("has_failed", True),
        "model_response": cleaned_model_response,
        "golden_response": cleaned_golden_response,
        "evaluation_results": processed_results.get("evaluation_results", {}),
    }


async def process_single_sample(sample, benchmark_results, sample_semaphore, p_bar):
    sample_id = str(sample["id"])
    benchmark_results["samples"][sample_id] = {
        "id": sample["id"],
        "input": sample["input"],
        "golden_response": sample["golden_response"],
        "model_results": {},
    }

    tasks = []
    # Use position=1 for nested progress bar to avoid overlap
    for model_name, model_response in sample["model_responses"].items():
        tasks.append(
            process_single_model(
                sample["golden_response"],
                model_response,
                model_name,
                sample["id"],
                sample["input"][0]["content"],
                sample_semaphore,
                p_bar,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error processing sample {sample_id}: {result}")
            logger.exception(result)
        else:
            process_sample_result(result, result["model_name"], benchmark_results)

    return benchmark_results


async def run_evaluation(
    samples_path: str,
    config: str,
    max_samples: Optional[int] = None,
):
    config = json.load(open(config, "r"))
    sample_semaphore = asyncio.Semaphore(config.get("max_concurrent_tasks", 20))

    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples is not None:
        samples = samples[:max_samples]

    benchmark_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_samples": (
            len(samples) if max_samples is None else min(max_samples, len(samples))
        ),
        "models": {},
        "samples": {},
    }
    p_bar = tqdm(
        total=len(samples) * len(samples[0]["model_responses"]),
        desc="Processing samples",
    )

    # Initialize models once before the loop
    models = get_models_from_samples(samples)
    benchmark_results["models"] = {
        model_name: {
            "total_evaluated": 0,
            "total_failures": 0,
            "total_adequate": 0,
        }
        for model_name in models
    }

    tasks = []
    for sample in samples:
        """
        sample structure
        id: int
        input: List[Dict[str, str]]
        golden_response: str
        model_responses: Dict[str, str]
        """
        tasks.append(
            process_single_sample(sample, benchmark_results, sample_semaphore, p_bar)
        )

    for future in asyncio.as_completed(tasks):
        try:
            await future
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            logger.exception(e)
        finally:
            p_bar.update(1)

    p_bar.close()

    final_results = process_benchmark_results(benchmark_results)
    output_path = save_benchmark_results(final_results)

    print_benchmark_summary(final_results)
    logger.info(f"Benchmark results saved to {output_path}")

    return benchmark_results
