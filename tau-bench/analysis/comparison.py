# Copyright Sierra
"""
Comparison utilities for tau-bench results.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .failure_analysis import (
    analyze_failure_type,
    FAILURE_TYPE_LEGEND,
)


class ComparisonStatus(Enum):
    BOTH_PASS = "both_pass"
    BOTH_FAIL = "both_fail"
    MODEL2_ONLY_FAIL = "model2_only_fail"  # file1 passes, file2 fails
    MODEL1_ONLY_FAIL = "model1_only_fail"  # file1 fails, file2 passes


@dataclass
class TaskComparison:
    task_id: int
    status: ComparisonStatus
    file1_reward: float
    file2_reward: float
    file1_info: Optional[Dict[str, Any]] = None
    file2_info: Optional[Dict[str, Any]] = None
    file1_traj: Optional[List[Dict[str, Any]]] = None
    file2_traj: Optional[List[Dict[str, Any]]] = None


def load_results(file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load results from a JSON file and index by task_id."""
    with open(file_path, "r") as f:
        results = json.load(f)
    return {r["task_id"]: r for r in results}


def is_success(reward: float) -> bool:
    """Check if a reward indicates success (reward ~= 1.0)."""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def get_comparison_status(reward1: float, reward2: float) -> ComparisonStatus:
    """Determine the comparison status based on rewards."""
    pass1 = is_success(reward1)
    pass2 = is_success(reward2)

    if pass1 and pass2:
        return ComparisonStatus.BOTH_PASS
    elif not pass1 and not pass2:
        return ComparisonStatus.BOTH_FAIL
    elif pass1 and not pass2:
        return ComparisonStatus.MODEL2_ONLY_FAIL  # model1 passes, model2 fails
    else:
        return ComparisonStatus.MODEL1_ONLY_FAIL  # model1 fails, model2 passes


def format_trajectory_summary(
    traj: List[Dict[str, Any]], max_messages: int = 10
) -> str:
    """Format a trajectory summary for display."""
    if not traj:
        return "  (empty trajectory)"

    lines = []
    msg_count = 0
    for msg in traj:
        if msg.get("role") == "system":
            continue
        msg_count += 1
        if msg_count > max_messages:
            lines.append(f"  ... ({len(traj) - max_messages} more messages)")
            break

        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                lines.append(
                    f"  [{role}] Tool: {func.get('name')}({func.get('arguments', '')[:50]}...)"
                )
        elif content:
            content_preview = content[:100].replace("\n", " ")
            if len(content) > 100:
                content_preview += "..."
            lines.append(f"  [{role}] {content_preview}")

    return "\n".join(lines) if lines else "  (no messages)"


def filter_trajectory(traj: List[Dict[str, Any]], failing_turn_idx: int = -1) -> List[Dict[str, Any]]:
    """Filter trajectory to remove system message and add metadata."""
    if not traj:
        return []
    
    filtered = []
    for i, msg in enumerate(traj):
        if msg.get("role") == "system":
            continue
        traj_msg = {
            "turn": i,
            "role": msg.get("role"),
            "content": msg.get("content"),
            "is_failing_turn": i == failing_turn_idx,
        }
        if msg.get("tool_calls"):
            traj_msg["tool_calls"] = msg.get("tool_calls")
        filtered.append(traj_msg)
    return filtered


def build_comparison_entry(comp: TaskComparison) -> Dict[str, Any]:
    """Build a complete comparison entry with both model trajectories."""
    # Get info from whichever file has it
    info = comp.file1_info or comp.file2_info
    
    result = {
        "task_id": comp.task_id,
        "status": comp.status.value,
        "model1_reward": comp.file1_reward,
        "model2_reward": comp.file2_reward,
        "instruction": "",
        "ground_truth_actions": [],
    }
    
    if info:
        result["instruction"] = info.get("task", {}).get("instruction", "")
        result["ground_truth_actions"] = info.get("task", {}).get("actions", [])
    
    # Analyze Model 1
    if comp.file1_traj and info:
        ftype, fdesc, failing_idx, _ = analyze_failure_type(comp.file1_traj, info)
        result["model1_failure_type"] = ftype if comp.file1_reward < 1.0 else "success"
        result["model1_failure_description"] = fdesc if comp.file1_reward < 1.0 else "Task completed successfully"
        result["model1_trajectory"] = filter_trajectory(comp.file1_traj, failing_idx if comp.file1_reward < 1.0 else -1)
    else:
        result["model1_failure_type"] = "no_trajectory"
        result["model1_failure_description"] = "No trajectory recorded"
        result["model1_trajectory"] = []
    
    # Analyze Model 2
    if comp.file2_traj and info:
        ftype, fdesc, failing_idx, _ = analyze_failure_type(comp.file2_traj, info)
        result["model2_failure_type"] = ftype if comp.file2_reward < 1.0 else "success"
        result["model2_failure_description"] = fdesc if comp.file2_reward < 1.0 else "Task completed successfully"
        result["model2_trajectory"] = filter_trajectory(comp.file2_traj, failing_idx if comp.file2_reward < 1.0 else -1)
    else:
        result["model2_failure_type"] = "no_trajectory"
        result["model2_failure_description"] = "No trajectory recorded"
        result["model2_trajectory"] = []
    
    return result


def build_failure_details(
    comp: TaskComparison, failed_file: int
) -> Dict[str, Any]:
    """Build detailed failure info including trajectory and failing turn."""
    traj = comp.file2_traj if failed_file == 2 else comp.file1_traj
    info = comp.file2_info if failed_file == 2 else comp.file1_info

    result = {
        "task_id": comp.task_id,
        "file1_reward": comp.file1_reward,
        "file2_reward": comp.file2_reward,
    }

    if info:
        result["instruction"] = info.get("task", {}).get("instruction", "")
        result["ground_truth_actions"] = info.get("task", {}).get("actions", [])

    if traj and info:
        ftype, fdesc, failing_turn_idx, failing_turn = analyze_failure_type(
            traj, info
        )
        result["failure_type"] = ftype
        result["failure_description"] = fdesc
        result["failing_turn_index"] = failing_turn_idx
        result["failing_turn"] = failing_turn
        result["trajectory"] = filter_trajectory(traj, failing_turn_idx)
    else:
        result["failure_type"] = "no_trajectory"
        result["failure_description"] = "No trajectory recorded"
        result["trajectory"] = []

    return result


def count_failure_types(
    comps: List[TaskComparison], failed_file: int
) -> Dict[str, int]:
    """Count failure types for a list of comparisons."""
    counts = defaultdict(int)
    for comp in comps:
        traj = comp.file2_traj if failed_file == 2 else comp.file1_traj
        info = comp.file2_info if failed_file == 2 else comp.file1_info
        if traj and info:
            ftype, _, _, _ = analyze_failure_type(traj, info)
            counts[ftype] += 1
        else:
            counts["no_trajectory"] += 1
    return dict(counts)


def compare_results(
    file1_path: str,
    file2_path: str,
    show_trajectory: bool = False,
    show_all: bool = False,
    only_common: bool = True,
) -> Dict[str, Any]:
    """
    Compare two result files and generate a comparison report.

    Args:
        file1_path: Path to first results file (model1)
        file2_path: Path to second results file (model2)
        show_trajectory: Whether to show trajectory summaries
        show_all: Whether to show all tasks (including both pass)
        only_common: If True, only compare tasks present in both files

    Returns:
        Comparison report dictionary
    """
    results1 = load_results(file1_path)
    results2 = load_results(file2_path)

    # Get task IDs - either intersection (common) or union (all)
    file1_only = set(results1.keys()) - set(results2.keys())
    file2_only = set(results2.keys()) - set(results1.keys())
    common_ids = set(results1.keys()) & set(results2.keys())
    
    if only_common:
        all_task_ids = sorted(common_ids)
    else:
        all_task_ids = sorted(set(results1.keys()) | set(results2.keys()))

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

    # Print summary
    file1_name = file1_path.split("/")[-1]
    file2_name = file2_path.split("/")[-1]

    print("=" * 80)
    print("τ-BENCH RESULTS COMPARISON")
    print("=" * 80)
    print(f"\nModel 1 (file1): {file1_name}")
    print(f"Model 2 (file2): {file2_name}")
    print(f"\nTasks in file1: {len(results1)}")
    print(f"Tasks in file2: {len(results2)}")
    print(f"Common tasks: {len(common_ids)}")
    if file1_only:
        print(f"Only in file1: {len(file1_only)} (task IDs: {sorted(file1_only)[:5]}{'...' if len(file1_only) > 5 else ''})")
    if file2_only:
        print(f"Only in file2: {len(file2_only)} (task IDs: {sorted(file2_only)[:5]}{'...' if len(file2_only) > 5 else ''})")
    print(f"\nTotal tasks compared: {len(comparisons)}")
    print()

    # Summary table
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Status':<35} {'Count':>10} {'%':>10}")
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

    # Calculate pass rates
    file1_passes = len(by_status[ComparisonStatus.BOTH_PASS]) + len(
        by_status[ComparisonStatus.MODEL2_ONLY_FAIL]
    )
    file2_passes = len(by_status[ComparisonStatus.BOTH_PASS]) + len(
        by_status[ComparisonStatus.MODEL1_ONLY_FAIL]
    )

    print(
        f"\nModel 1: {file1_passes}/{len(comparisons)} ({file1_passes / len(comparisons) * 100:.1f}%) passed"
    )
    print(
        f"Model 2: {file2_passes}/{len(comparisons)} ({file2_passes / len(comparisons) * 100:.1f}%) passed"
    )

    # Detailed analysis of failures
    print("\n" + "=" * 80)
    print("DETAILED FAILURE ANALYSIS")
    print("=" * 80)

    # Analyze failures where only Model2 failed (Model1 passed)
    if by_status[ComparisonStatus.MODEL2_ONLY_FAIL]:
        print(
            f"\n🔵 MODEL2_ONLY_FAIL: Model1 PASS → Model2 FAIL ({len(by_status[ComparisonStatus.MODEL2_ONLY_FAIL])} tasks)"
        )
        print("-" * 60)

        failure_types = defaultdict(list)
        for comp in by_status[ComparisonStatus.MODEL2_ONLY_FAIL]:
            if comp.file2_traj and comp.file2_info:
                ftype, fdesc, _, _ = analyze_failure_type(
                    comp.file2_traj, comp.file2_info
                )
                failure_types[ftype].append((comp.task_id, fdesc))

        for ftype, tasks in sorted(
            failure_types.items(), key=lambda x: -len(x[1])
        ):
            print(f"\n  {ftype}: {len(tasks)} tasks")
            for task_id, desc in tasks[:5]:
                print(f"    - Task {task_id}: {desc[:60]}...")
            if len(tasks) > 5:
                print(f"    ... and {len(tasks) - 5} more")

    # Analyze failures where only Model1 failed (Model2 passed)
    if by_status[ComparisonStatus.MODEL1_ONLY_FAIL]:
        print(
            f"\n🟠 MODEL1_ONLY_FAIL: Model1 FAIL → Model2 PASS ({len(by_status[ComparisonStatus.MODEL1_ONLY_FAIL])} tasks)"
        )
        print("-" * 60)

        failure_types = defaultdict(list)
        for comp in by_status[ComparisonStatus.MODEL1_ONLY_FAIL]:
            if comp.file1_traj and comp.file1_info:
                ftype, fdesc, _, _ = analyze_failure_type(
                    comp.file1_traj, comp.file1_info
                )
                failure_types[ftype].append((comp.task_id, fdesc))

        for ftype, tasks in sorted(
            failure_types.items(), key=lambda x: -len(x[1])
        ):
            print(f"\n  {ftype}: {len(tasks)} tasks")
            for task_id, desc in tasks[:5]:
                print(f"    - Task {task_id}: {desc[:60]}...")
            if len(tasks) > 5:
                print(f"    ... and {len(tasks) - 5} more")

    # Analyze both-fail cases
    if by_status[ComparisonStatus.BOTH_FAIL]:
        print(
            f"\n❌ BOTH FAIL ({len(by_status[ComparisonStatus.BOTH_FAIL])} tasks)"
        )
        print("-" * 60)

        failure_types_1 = defaultdict(list)
        failure_types_2 = defaultdict(list)

        for comp in by_status[ComparisonStatus.BOTH_FAIL]:
            if comp.file1_traj and comp.file1_info:
                ftype, fdesc, _, _ = analyze_failure_type(
                    comp.file1_traj, comp.file1_info
                )
                failure_types_1[ftype].append((comp.task_id, fdesc))
            if comp.file2_traj and comp.file2_info:
                ftype, fdesc, _, _ = analyze_failure_type(
                    comp.file2_traj, comp.file2_info
                )
                failure_types_2[ftype].append((comp.task_id, fdesc))

        print(f"\n  Failure types in Model 1:")
        for ftype, tasks in sorted(
            failure_types_1.items(), key=lambda x: -len(x[1])
        ):
            print(f"    - {ftype}: {len(tasks)} tasks")

        print(f"\n  Failure types in Model 2:")
        for ftype, tasks in sorted(
            failure_types_2.items(), key=lambda x: -len(x[1])
        ):
            print(f"    - {ftype}: {len(tasks)} tasks")

    # Show trajectory comparisons if requested
    if show_trajectory:
        print("\n" + "=" * 80)
        print("TRAJECTORY COMPARISONS")
        print("=" * 80)

        # Show model2_only_fail cases
        for comp in by_status[ComparisonStatus.MODEL2_ONLY_FAIL][:3]:
            print(f"\n--- Task {comp.task_id} (Model2 Only Fail) ---")
            instruction = (
                comp.file1_info.get("task", {}).get("instruction", "")
                if comp.file1_info
                else ""
            )
            print(f"Instruction: {instruction[:200]}...")
            print(f"\nModel 1 (PASS):")
            print(format_trajectory_summary(comp.file1_traj))
            print(f"\nModel 2 (FAIL):")
            print(format_trajectory_summary(comp.file2_traj))

        # Show both-fail cases
        for comp in by_status[ComparisonStatus.BOTH_FAIL][:3]:
            print(f"\n--- Task {comp.task_id} (Both Fail) ---")
            instruction = (
                comp.file1_info.get("task", {}).get("instruction", "")
                if comp.file1_info
                else ""
            )
            print(f"Instruction: {instruction[:200]}...")
            print(f"\nModel 1:")
            print(format_trajectory_summary(comp.file1_traj))
            print(f"\nModel 2:")
            print(format_trajectory_summary(comp.file2_traj))

    # Build failure analysis summary
    model2_only_fail_types = count_failure_types(
        by_status[ComparisonStatus.MODEL2_ONLY_FAIL], failed_file=2
    )
    model1_only_fail_types = count_failure_types(
        by_status[ComparisonStatus.MODEL1_ONLY_FAIL], failed_file=1
    )
    both_fail_file1_types = count_failure_types(
        by_status[ComparisonStatus.BOTH_FAIL], failed_file=1
    )
    both_fail_file2_types = count_failure_types(
        by_status[ComparisonStatus.BOTH_FAIL], failed_file=2
    )

    # Build all comparisons with full trajectory data
    all_comparisons = [build_comparison_entry(comp) for comp in comparisons]

    # Build output report
    report = {
        "file1": file1_path,
        "file2": file2_path,
        "model1_name": file1_name,
        "model2_name": file2_name,
        "summary": {
            "tasks_in_file1": len(results1),
            "tasks_in_file2": len(results2),
            "common_tasks": len(common_ids),
            "file1_only_tasks": sorted(file1_only),
            "file2_only_tasks": sorted(file2_only),
            "total_tasks_compared": len(comparisons),
            "both_pass": len(by_status[ComparisonStatus.BOTH_PASS]),
            "both_fail": len(by_status[ComparisonStatus.BOTH_FAIL]),
            "model2_only_fail": len(by_status[ComparisonStatus.MODEL2_ONLY_FAIL]),
            "model1_only_fail": len(by_status[ComparisonStatus.MODEL1_ONLY_FAIL]),
            "model1_pass_rate": file1_passes / len(comparisons)
            if comparisons
            else 0,
            "model2_pass_rate": file2_passes / len(comparisons)
            if comparisons
            else 0,
        },
        "failure_analysis": {
            "model2_only_fail": {
                "description": f"Tasks where {file1_name} (model1) passed but {file2_name} (model2) failed",
                "count": len(by_status[ComparisonStatus.MODEL2_ONLY_FAIL]),
                "failure_type_distribution": model2_only_fail_types,
            },
            "model1_only_fail": {
                "description": f"Tasks where {file1_name} (model1) failed but {file2_name} (model2) passed",
                "count": len(by_status[ComparisonStatus.MODEL1_ONLY_FAIL]),
                "failure_type_distribution": model1_only_fail_types,
            },
            "both_fail": {
                "description": "Tasks where both models failed",
                "count": len(by_status[ComparisonStatus.BOTH_FAIL]),
                "model1_failure_types": both_fail_file1_types,
                "model2_failure_types": both_fail_file2_types,
            },
            "failure_type_legend": FAILURE_TYPE_LEGEND,
        },
        # All comparisons with full data for each task
        "comparisons": all_comparisons,
    }

    return report

