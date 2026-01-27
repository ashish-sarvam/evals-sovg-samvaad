# Copyright Sierra
"""
LLM-based comparison of tau-bench results.

Uses GPT-5-chat to analyze failures with concrete labels.
"""

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

from .comparison import (
    load_results,
    is_success,
    get_comparison_status,
    ComparisonStatus,
    TaskComparison,
    filter_trajectory,
)
from .llm_failure_analysis import (
    analyze_failure_async,
    analyze_failure_heuristic,
    FailureAnalysisResult,
    FAILURE_LABELS,
)

# Load environment variables
load_dotenv()


async def analyze_task_failures(
    comp: TaskComparison,
    model: str = "gpt-5-chat",
    api_base: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Analyze failures for a single task comparison using LLM.
    
    Args:
        comp: TaskComparison object
        model: LLM model to use
        api_base: API base URL
        use_llm: Whether to use LLM (True) or heuristic (False)
    
    Returns:
        Dictionary with analysis results
    """
    info = comp.file1_info or comp.file2_info
    
    result = {
        "task_id": comp.task_id,
        "status": comp.status.value,
        "model1_reward": comp.file1_reward,
        "model2_reward": comp.file2_reward,
        "instruction": info.get("task", {}).get("instruction", "") if info else "",
        "ground_truth_actions": info.get("task", {}).get("actions", []) if info else [],
    }
    
    # Analyze Model 1 if failed
    if comp.file1_reward < 1.0 and comp.file1_traj and info:
        if use_llm:
            analysis1 = await analyze_failure_async(comp.file1_traj, info, model, api_base)
        else:
            analysis1 = analyze_failure_heuristic(comp.file1_traj, info)
        
        result["model1_analysis"] = {
            "label": analysis1.label,
            "description": analysis1.description,
            "failing_turn_index": analysis1.failing_turn_index,
            "failing_turn_summary": analysis1.failing_turn_summary,
            "detailed_explanation": analysis1.detailed_explanation,
        }
        result["model1_trajectory"] = filter_trajectory(
            comp.file1_traj, 
            analysis1.failing_turn_index
        )
    else:
        result["model1_analysis"] = {
            "label": "success" if comp.file1_reward >= 1.0 else "no_trajectory",
            "description": "Task completed successfully" if comp.file1_reward >= 1.0 else "No trajectory",
        }
        result["model1_trajectory"] = filter_trajectory(comp.file1_traj) if comp.file1_traj else []
    
    # Analyze Model 2 if failed
    if comp.file2_reward < 1.0 and comp.file2_traj and info:
        if use_llm:
            analysis2 = await analyze_failure_async(comp.file2_traj, info, model, api_base)
        else:
            analysis2 = analyze_failure_heuristic(comp.file2_traj, info)
        
        result["model2_analysis"] = {
            "label": analysis2.label,
            "description": analysis2.description,
            "failing_turn_index": analysis2.failing_turn_index,
            "failing_turn_summary": analysis2.failing_turn_summary,
            "detailed_explanation": analysis2.detailed_explanation,
        }
        result["model2_trajectory"] = filter_trajectory(
            comp.file2_traj,
            analysis2.failing_turn_index
        )
    else:
        result["model2_analysis"] = {
            "label": "success" if comp.file2_reward >= 1.0 else "no_trajectory",
            "description": "Task completed successfully" if comp.file2_reward >= 1.0 else "No trajectory",
        }
        result["model2_trajectory"] = filter_trajectory(comp.file2_traj) if comp.file2_traj else []
    
    return result


async def compare_with_llm_analysis(
    file1_path: str,
    file2_path: str,
    model: str = "gpt-5-chat",
    api_base: Optional[str] = None,
    max_concurrency: int = 5,
    only_common: bool = True,
    use_llm: bool = True,
    analyze_all: bool = False,
) -> Dict[str, Any]:
    """
    Compare two result files with LLM-based failure analysis.
    
    Args:
        file1_path: Path to first results file (model1)
        file2_path: Path to second results file (model2)
        model: LLM model for analysis
        api_base: API base URL
        max_concurrency: Max concurrent LLM calls
        only_common: Only compare tasks present in both files
        use_llm: Use LLM analysis (True) or heuristic (False)
        analyze_all: If True, analyze all tasks; if False, only analyze failures
    
    Returns:
        Comparison report with LLM analysis
    """
    results1 = load_results(file1_path)
    results2 = load_results(file2_path)
    
    # Get task IDs
    file1_only = set(results1.keys()) - set(results2.keys())
    file2_only = set(results2.keys()) - set(results1.keys())
    common_ids = set(results1.keys()) & set(results2.keys())
    
    if only_common:
        all_task_ids = sorted(common_ids)
    else:
        all_task_ids = sorted(set(results1.keys()) | set(results2.keys()))
    
    # Build comparisons
    comparisons: List[TaskComparison] = []
    for task_id in all_task_ids:
        r1 = results1.get(task_id)
        r2 = results2.get(task_id)
        
        reward1 = r1["reward"] if r1 else 0.0
        reward2 = r2["reward"] if r2 else 0.0
        
        status = get_comparison_status(reward1, reward2)
        
        comp = TaskComparison(
            task_id=task_id,
            status=status,
            file1_reward=reward1,
            file2_reward=reward2,
            file1_info=r1.get("info") if r1 else None,
            file2_info=r2.get("info") if r2 else None,
            file1_traj=r1.get("traj") if r1 else None,
            file2_traj=r2.get("traj") if r2 else None,
        )
        comparisons.append(comp)
    
    # Group by status
    by_status = defaultdict(list)
    for comp in comparisons:
        by_status[comp.status].append(comp)
    
    # Decide which tasks to analyze
    if analyze_all:
        tasks_to_analyze = comparisons
    else:
        # Only analyze failures
        tasks_to_analyze = (
            by_status[ComparisonStatus.BOTH_FAIL] +
            by_status[ComparisonStatus.MODEL1_ONLY_FAIL] +
            by_status[ComparisonStatus.MODEL2_ONLY_FAIL]
        )
    
    # Print progress
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    
    print("=" * 80)
    print("τ-BENCH LLM-BASED COMPARISON")
    print("=" * 80)
    print(f"\nModel 1: {file1_name}")
    print(f"Model 2: {file2_name}")
    print(f"\nCommon tasks: {len(common_ids)}")
    print(f"Tasks to analyze: {len(tasks_to_analyze)}")
    print(f"Using LLM: {use_llm} (model: {model})")
    print()
    
    # Run analysis with semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def analyze_with_progress(comp: TaskComparison, idx: int):
        async with semaphore:
            print(f"  Analyzing task {comp.task_id} ({idx + 1}/{len(tasks_to_analyze)})...", end="", flush=True)
            result = await analyze_task_failures(comp, model, api_base, use_llm)
            status_emoji = {
                "both_pass": "✅",
                "both_fail": "❌",
                "model1_only_fail": "🟠",
                "model2_only_fail": "🔵",
            }.get(result["status"], "?")
            print(f" {status_emoji}")
            return result
    
    print("Analyzing failures...")
    analyzed_results = await asyncio.gather(*[
        analyze_with_progress(comp, i) 
        for i, comp in enumerate(tasks_to_analyze)
    ])
    
    # For tasks not analyzed, create simple entries
    analyzed_task_ids = {r["task_id"] for r in analyzed_results}
    for comp in comparisons:
        if comp.task_id not in analyzed_task_ids:
            # Create simple entry for passing tasks
            info = comp.file1_info or comp.file2_info
            simple_entry = {
                "task_id": comp.task_id,
                "status": comp.status.value,
                "model1_reward": comp.file1_reward,
                "model2_reward": comp.file2_reward,
                "instruction": info.get("task", {}).get("instruction", "") if info else "",
                "ground_truth_actions": info.get("task", {}).get("actions", []) if info else [],
                "model1_analysis": {"label": "success", "description": "Task completed successfully"},
                "model2_analysis": {"label": "success", "description": "Task completed successfully"},
                "model1_trajectory": [],
                "model2_trajectory": [],
            }
            analyzed_results.append(simple_entry)
    
    # Sort by task_id
    analyzed_results.sort(key=lambda x: x["task_id"])
    
    # Count failure labels
    model1_labels = defaultdict(int)
    model2_labels = defaultdict(int)
    
    for r in analyzed_results:
        if r["model1_reward"] < 1.0:
            model1_labels[r["model1_analysis"]["label"]] += 1
        if r["model2_reward"] < 1.0:
            model2_labels[r["model2_analysis"]["label"]] += 1
    
    # Calculate pass rates
    file1_passes = len(by_status[ComparisonStatus.BOTH_PASS]) + len(by_status[ComparisonStatus.MODEL2_ONLY_FAIL])
    file2_passes = len(by_status[ComparisonStatus.BOTH_PASS]) + len(by_status[ComparisonStatus.MODEL1_ONLY_FAIL])
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Status':<35} {'Count':>10} {'%':>10}")
    print("-" * 60)
    
    for status in ComparisonStatus:
        count = len(by_status[status])
        pct = (count / len(comparisons) * 100) if comparisons else 0
        status_label = {
            ComparisonStatus.BOTH_PASS: "✅ Both Pass",
            ComparisonStatus.BOTH_FAIL: "❌ Both Fail",
            ComparisonStatus.MODEL2_ONLY_FAIL: "🔵 Model2 Only Fail",
            ComparisonStatus.MODEL1_ONLY_FAIL: "🟠 Model1 Only Fail",
        }.get(status, status.value)
        print(f"{status_label:<35} {count:>10} {pct:>9.1f}%")
    
    print("-" * 60)
    print(f"\nModel 1 Pass Rate: {file1_passes}/{len(comparisons)} ({file1_passes / len(comparisons) * 100:.1f}%)")
    print(f"Model 2 Pass Rate: {file2_passes}/{len(comparisons)} ({file2_passes / len(comparisons) * 100:.1f}%)")
    
    # Print failure label distribution
    print("\n" + "=" * 80)
    print("FAILURE LABEL DISTRIBUTION")
    print("=" * 80)
    
    if model1_labels:
        print(f"\nModel 1 Failures:")
        for label, count in sorted(model1_labels.items(), key=lambda x: -x[1]):
            print(f"  {label:<30} {count:>5}")
    
    if model2_labels:
        print(f"\nModel 2 Failures:")
        for label, count in sorted(model2_labels.items(), key=lambda x: -x[1]):
            print(f"  {label:<30} {count:>5}")
    
    # Build report
    report = {
        "metadata": {
            "file1": file1_path,
            "file2": file2_path,
            "model1_name": file1_name,
            "model2_name": file2_name,
            "analysis_model": model,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "tasks_in_file1": len(results1),
            "tasks_in_file2": len(results2),
            "common_tasks": len(common_ids),
            "total_tasks_compared": len(comparisons),
            "both_pass": len(by_status[ComparisonStatus.BOTH_PASS]),
            "both_fail": len(by_status[ComparisonStatus.BOTH_FAIL]),
            "model1_only_fail": len(by_status[ComparisonStatus.MODEL1_ONLY_FAIL]),
            "model2_only_fail": len(by_status[ComparisonStatus.MODEL2_ONLY_FAIL]),
            "model1_pass_rate": file1_passes / len(comparisons) if comparisons else 0,
            "model2_pass_rate": file2_passes / len(comparisons) if comparisons else 0,
        },
        "failure_label_distribution": {
            "model1": dict(model1_labels),
            "model2": dict(model2_labels),
        },
        "failure_labels_legend": FAILURE_LABELS,
        "comparisons": analyzed_results,
    }
    
    return report


def main():
    """Main entry point for LLM-based comparison."""
    parser = argparse.ArgumentParser(
        description="Compare tau-bench results with LLM-based failure analysis"
    )
    parser.add_argument("--file1", type=str, required=True, help="Path to first results file (model1)")
    parser.add_argument("--file2", type=str, required=True, help="Path to second results file (model2)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-chat",
        help="Azure deployment name for analysis (default: gpt-5-chat)",
    )
    parser.add_argument("--api-base", type=str, help="API base URL (default: from AZURE_API_BASE env var)")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls (default: 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristic analysis instead of LLM",
    )
    parser.add_argument(
        "--analyze-all",
        action="store_true",
        help="Analyze all tasks including passing ones",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Compare all tasks (not just common ones)",
    )
    
    args = parser.parse_args()
    
    # Run comparison
    report = asyncio.run(compare_with_llm_analysis(
        file1_path=args.file1,
        file2_path=args.file2,
        model=args.model,
        api_base=args.api_base,
        max_concurrency=args.max_concurrency,
        only_common=not args.all_tasks,
        use_llm=not args.no_llm,
        analyze_all=args.analyze_all,
    ))
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path
        os.makedirs("results/comparisons", exist_ok=True)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        name1 = os.path.basename(args.file1).replace(".json", "").split("_")[1][:10]
        name2 = os.path.basename(args.file2).replace(".json", "").split("_")[1][:10]
        output_path = f"results/comparisons/llm_analysis_{name1}_vs_{name2}_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved to: {output_path}")


if __name__ == "__main__":
    main()

