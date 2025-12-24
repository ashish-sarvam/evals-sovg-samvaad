"""
Module for processing benchmark results and generating reports.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from .config_handler import get_results_directory
from .schema import (
    GeneralEvaluationResult,
    LineMatchStatus,
    SemanticSimilarityEvaluationResult,
    ToolCallsDetailedStatus,
    VariableUpdateStatus,
)

# Configure logging
logger = logging.getLogger(__name__)


def process_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process benchmark results to create detailed summary and category comparisons.

    Args:
        results: Raw benchmark results

    Returns:
        Processed results with summary and category comparisons
    """
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_samples": results["total_samples"],
        "models_summary": {},
        "category_comparison": {},
        "latency_analysis": {},
    }

    # Find all unique category keys across all samples and models
    category_keys = find_all_category_keys(results)

    # Order the samples by sample_id (which is the index)
    results["samples"] = order_samples_by_id(results["samples"])

    # Initialize the category comparison structure
    initialize_category_comparison(summary, category_keys, results["models"])

    # Fill in the models_summary and category_comparison
    process_model_summaries(summary, results, category_keys)

    # Combine summary and benchmark_results
    final_results = {
        "summary": summary,
        "models": results["models"],
        "samples": results["samples"],
    }

    return final_results


def find_all_category_keys(results: Dict[str, Any]) -> Set[str]:
    """
    Find all unique evaluation category keys across all samples and models.

    Args:
        results: Raw benchmark results

    Returns:
        Set of all category keys
    """
    category_keys = set()
    for sample_id, sample_data in results["samples"].items():
        for model_name, model_result in sample_data["model_results"].items():
            eval_result = model_result.get("evaluation_results")
            if eval_result and isinstance(eval_result, dict):
                for key in eval_result.keys():
                    category_keys.add(key)
    return category_keys


def order_samples_by_id(samples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Order the samples by their sample ID (which is the index).

    Args:
        samples: Dictionary of samples keyed by sample ID

    Returns:
        Ordered dictionary of samples
    """
    ordered_samples = {}
    for sample_id in sorted(samples.keys(), key=lambda x: int(x)):
        ordered_samples[sample_id] = samples[sample_id]
    return ordered_samples


def initialize_category_comparison(
    summary: Dict[str, Any], category_keys: Set[str], models: Dict[str, Any]
) -> None:
    """
    Initialize the category comparison structure in the summary.

    Args:
        summary: Summary dictionary to be populated
        category_keys: Set of all category keys
        models: Dictionary of model data
    """
    for category in category_keys:
        summary["category_comparison"][category] = {}
        for model_name in models:
            summary["category_comparison"][category][model_name] = {
                "success": 0,
                "failure": 0,
                "success_rate": 0.0,
            }


def process_model_summaries(
    summary: Dict[str, Any], results: Dict[str, Any], category_keys: Set[str]
) -> None:
    """
    Process model summaries and fill in the category comparison.

    Args:
        summary: Summary dictionary to be populated
        results: Raw benchmark results
        category_keys: Set of all category keys
    """
    for model_name, model_data in results["models"].items():
        total_evaluated = model_data["total_evaluated"]
        total_failures = model_data["total_failures"]
        failure_rate = (
            (total_failures / total_evaluated * 100) if total_evaluated > 0 else 0
        )
        success_rate = 100 - failure_rate

        # Add to summary
        summary["models_summary"][model_name] = {
            "total_evaluated": total_evaluated,
            "total_failures": total_failures,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
        }

        # Compute category stats for this model
        category_stats = {}
        for category in category_keys:
            category_stats[category] = {
                "success": 0,
                "failure": 0,
                "total_evaluated": 0,
            }

        # Count success/failure by category
        all_latencies = []
        for sample_id, sample_data in results["samples"].items():
            if model_name in sample_data["model_results"]:
                model_result = sample_data["model_results"][model_name]
                eval_result = model_result.get("evaluation_results")
                inference_time = model_result.get("inference_time")
                if inference_time:
                    all_latencies.append(inference_time)

                for category in category_keys:
                    if (
                        eval_result
                        and isinstance(eval_result, dict)
                        and category in eval_result
                    ):
                        result_value = eval_result[category]

                        # Skip samples where both keys are missing
                        if (
                            isinstance(result_value, str)
                            and result_value
                            == GeneralEvaluationResult.BOTH_KEYS_MISSING.value
                        ):
                            continue

                        # Count this sample for this category
                        category_stats[category]["total_evaluated"] += 1

                        # Determine if this is a failure based on the type
                        is_failure = False
                        if isinstance(result_value, str):
                            # String results - only CORRECT, MATCH, and PARTIAL_MATCH are successes
                            is_failure = (
                                result_value != GeneralEvaluationResult.CORRECT.value
                                and result_value
                                != SemanticSimilarityEvaluationResult.MATCH.value
                                and result_value
                                != SemanticSimilarityEvaluationResult.PARTIAL_MATCH.value
                            )
                        elif (
                            isinstance(result_value, dict) and "status" in result_value
                        ):
                            # Complex results with status field
                            is_failure = (
                                result_value["status"]
                                == ToolCallsDetailedStatus.ERROR.value
                                or result_value["status"]
                                == VariableUpdateStatus.ERROR.value
                                or result_value["status"]
                                == LineMatchStatus.NO_MATCH.value
                                or result_value["status"]
                                == LineMatchStatus.PARTIAL_MATCH.value
                            )
                        else:
                            is_failure = True  # Unknown format, count as failure

                        if is_failure:
                            category_stats[category]["failure"] += 1
                        else:
                            category_stats[category]["success"] += 1
                    elif eval_result and not isinstance(eval_result, dict):
                        # For non-dict errors like KEY_MISSING, count as failure for all categories
                        category_stats[category]["total_evaluated"] += 1
                        category_stats[category]["failure"] += 1

        # Update the category comparison
        for category, stats in category_stats.items():
            total = stats["total_evaluated"]
            success_rate = (stats["success"] / total) * 100 if total > 0 else 0

            summary["category_comparison"][category][model_name] = {
                "success": stats["success"],
                "failure": stats["failure"],
                "total_evaluated": total,
                "success_rate": success_rate,
            }

        summary["latency_analysis"][model_name] = {
            "mean_latency": (
                sum(all_latencies) / len(all_latencies) if all_latencies else None
            ),
            "min_latency": min(all_latencies) if all_latencies else None,
            "max_latency": max(all_latencies) if all_latencies else None,
        }


def save_benchmark_results(results: Dict[str, Any]) -> str:
    """
    Save benchmark results to a JSON file.

    Args:
        results: The processed benchmark results

    Returns:
        Path to the saved file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = get_results_directory()
    output_file = results_dir / f"benchmark_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return str(output_file)


def print_benchmark_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of benchmark results to the console.

    Args:
        results: The processed benchmark results
    """
    summary = results["summary"]

    print("\nBenchmark Results Summary:")
    print("=" * 80)

    # Enhanced summary with detailed statistics for each model
    for model_name, model_summary in summary["models_summary"].items():
        total_evaluated = model_summary["total_evaluated"]
        total_failures = model_summary["total_failures"]
        success_rate = model_summary["success_rate"]
        failure_rate = model_summary["failure_rate"]

        print(f"Model: {model_name}")
        print(f"  Samples Evaluated: {total_evaluated}")
        print(
            f"  Success Rate: {success_rate:.2f}% "
            f"({total_evaluated - total_failures}/{total_evaluated})"
        )
        print(
            f"  Failure Rate: {failure_rate:.2f}% ({total_failures}/{total_evaluated})"
        )

        # Detailed error analysis by error type
        if total_failures > 0:
            error_types = analyze_model_errors(results, model_name)

            header = "| {:<20} | {:<20} | {:>10} | {:>15} |".format(
                "Key", "Error Type", "Count", "% of Failures"
            )
            print("-" * len(header))
            print(header)
            for key, errors in sorted(error_types.items()):
                print("-" * len(header))
                for error_type, count in sorted(errors.items()):
                    if error_type == "both_keys_missing" or error_type == "correct":
                        percentage = ""
                    else:
                        percentage = (count / total_failures) * 100
                        percentage = f"{percentage:<6.2f}%"
                    row = "| {:<20} | {:<20} | {:>10} | {:>15} |".format(
                        key, error_type, count, percentage
                    )
                    print(row)

            print("-" * 80)

    print_category_comparison(summary)


def analyze_model_errors(
    results: Dict[str, Any], model_name: str
) -> Dict[str, Dict[str, int]]:
    """
    Analyze errors for a specific model across all samples.

    Args:
        results: The processed benchmark results
        model_name: The name of the model to analyze

    Returns:
        Dictionary of error types and counts by category
    """
    error_types = {}
    total_failures = results["summary"]["models_summary"][model_name]["total_failures"]

    if total_failures == 0:
        return error_types

    # Analyze failures for this model across all samples
    for sample_id, sample_data in results["samples"].items():
        if model_name in sample_data["model_results"]:
            model_result = sample_data["model_results"][model_name]
            eval_result = model_result.get("evaluation_results")

            if not eval_result:
                continue  # Skip samples without evaluation results

            if isinstance(eval_result, dict):
                for key, error_type in eval_result.items():
                    if key == "overall":  # Handle the overall key specially
                        if "general" not in error_types:
                            error_types["general"] = {}
                        if error_type not in error_types["general"]:
                            error_types["general"][error_type] = 0
                        error_types["general"][error_type] += 1
                        continue

                    # Check if the result indicates an error
                    is_error = False
                    if isinstance(error_type, str):
                        # Simple string error types
                        is_error = (
                            error_type != GeneralEvaluationResult.CORRECT.value
                            and error_type
                            != SemanticSimilarityEvaluationResult.MATCH.value
                            and error_type
                            != SemanticSimilarityEvaluationResult.PARTIAL_MATCH.value
                        )
                    elif isinstance(error_type, dict) and "status" in error_type:
                        # Complex error types with status field
                        is_error = (
                            error_type["status"] == ToolCallsDetailedStatus.ERROR.value
                            or error_type["status"] == VariableUpdateStatus.ERROR.value
                            or error_type["status"] == LineMatchStatus.NO_MATCH.value
                            or error_type["status"]
                            == LineMatchStatus.PARTIAL_MATCH.value
                        )

                    if is_error:
                        if key not in error_types:
                            error_types[key] = {}
                        error_category = (
                            "complex_error"
                            if isinstance(error_type, dict)
                            else error_type
                        )
                        if error_category not in error_types[key]:
                            error_types[key][error_category] = 0
                        error_types[key][error_category] += 1
                    else:
                        # correct
                        if key not in error_types:
                            error_types[key] = {}
                        if "correct" not in error_types[key]:
                            error_types[key]["correct"] = 0
                        error_types[key]["correct"] += 1
            else:
                # Handle KEY_MISSING or other non-dict results
                if "general" not in error_types:
                    error_types["general"] = {}
                error_type = str(eval_result)
                if error_type not in error_types["general"]:
                    error_types["general"][error_type] = 0
                error_types["general"][error_type] += 1

    return error_types


def print_category_comparison(summary: Dict[str, Any]) -> None:
    """
    Print a comparison of model performance by category.

    Args:
        summary: The summary section of processed benchmark results
    """
    print("\nCategory Performance Across Models:")
    print("=" * 80)

    model_names = list(summary["models_summary"].keys())
    for category, models_data in summary["category_comparison"].items():
        print(f"\nCategory: {category}")
        print("-" * 90)

        # Create a table header with model names
        header = "| {:<50} |".format("Model")
        for stat in ["Success Rate", "Success/Total"]:
            header += " {:<15} |".format(stat)
        print(header)
        print("-" * len(header))

        # Sort models by success rate for this category
        sorted_models = sorted(
            model_names, key=lambda m: models_data[m]["success_rate"], reverse=True
        )

        # Print each model's performance for this category
        for model_name in sorted_models:
            model_stats = models_data[model_name]
            total = model_stats["success"] + model_stats["failure"]
            success_rate = model_stats["success_rate"]

            row = "| {:<50} |".format(model_name[:50])
            row += " {:<15} |".format(f"{success_rate:.2f}%")
            row += " {:<15} |".format(f"{model_stats['success']}/{total}")
            print(row)

        print("-" * len(header))
