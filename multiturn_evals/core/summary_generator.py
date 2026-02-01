"""
Hierarchical summary generator for evaluation results.

Creates summaries at multiple levels:
1. Per user: language summary
2. Per agent: user + language summary  
3. Per task: agent + user + language summary
4. Overall: all tasks summary
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_result_files(results_dir: Path) -> list[dict]:
    """Load all result JSON files from directory tree."""
    results = []
    for json_file in results_dir.rglob("*.json"):
        # Skip summary files
        if "summary" in json_file.name:
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
                # Only include files with conversation data
                if "conversation" in data and "task" in data:
                    data["_file_path"] = str(json_file)
                    results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def calculate_stats(results: list[dict]) -> dict:
    """Calculate statistics from a list of results."""
    if not results:
        return {"count": 0, "error": "No results"}
    
    total = len(results)
    with_verification = [r for r in results if r.get("verification")]
    
    # Count statuses
    statuses = defaultdict(int)
    for r in with_verification:
        status = r["verification"].get("overall_status", "unknown")
        statuses[status] += 1
    
    # Aggregate checks
    checks = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in with_verification:
        for check in r["verification"].get("results", []):
            name = check.get("check_name", "unknown")
            checks[name]["total"] += 1
            if check.get("passed"):
                checks[name]["passed"] += 1
    
    # Calculate pass rates
    check_results = {}
    for name, data in checks.items():
        check_results[name] = {
            "passed": data["passed"],
            "total": data["total"],
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
        }
    
    total_checks = sum(c["total"] for c in check_results.values())
    total_passed = sum(c["passed"] for c in check_results.values())
    
    return {
        "count": total,
        "verified": len(with_verification),
        "statuses": dict(statuses),
        "checks": check_results,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "overall_pass_rate": total_passed / total_checks if total_checks > 0 else 0,
    }


def generate_user_language_summary(results: list[dict], output_dir: Path) -> dict:
    """Generate summary for a single user across languages.
    
    Saved as: {task}/{agent}/{user}/{model}/user_summary.json
    """
    if not results:
        return {}
    
    # Group by language
    by_language = defaultdict(list)
    for r in results:
        lang = r.get("language_code", "unknown")
        by_language[lang].append(r)
    
    summary = {
        "task": results[0].get("task"),
        "agent": results[0].get("agent"),
        "user": results[0].get("user"),
        "models": results[0].get("models", {}),
        "timestamp": datetime.now().isoformat(),
        "total_languages": len(by_language),
        "overall": calculate_stats(results),
        "by_language": {
            lang: calculate_stats(lang_results)
            for lang, lang_results in sorted(by_language.items())
        },
    }
    
    # Save summary in the user/model directory
    summary_file = output_dir / "user_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def generate_agent_summary(results: list[dict], output_dir: Path) -> dict:
    """Generate summary for an agent across all users and languages.
    
    Saved as: {task}/{agent}/agent_summary.json
    """
    if not results:
        return {}
    
    # Group by user
    by_user = defaultdict(list)
    for r in results:
        user = r.get("user", "unknown")
        by_user[user].append(r)
    
    # Generate per-user summaries
    for user, user_results in by_user.items():
        # Find the model directory (e.g., gemini)
        model_dirs = list((output_dir / user).glob("*"))
        for model_dir in model_dirs:
            if model_dir.is_dir():
                model_results = [r for r in user_results 
                               if r.get("models", {}).get("agent_provider") in model_dir.name 
                               or model_dir.name in str(r.get("_file_path", ""))]
                if model_results:
                    generate_user_language_summary(model_results, model_dir)
    
    summary = {
        "task": results[0].get("task"),
        "agent": results[0].get("agent"),
        "models": results[0].get("models", {}),
        "timestamp": datetime.now().isoformat(),
        "total_users": len(by_user),
        "overall": calculate_stats(results),
        "by_user": {
            user: calculate_stats(user_results)
            for user, user_results in sorted(by_user.items())
        },
    }
    
    # Save summary in the agent directory
    summary_file = output_dir / "agent_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def generate_task_summary(results: list[dict], output_dir: Path) -> dict:
    """Generate summary for a task across all agents, users, and languages.
    
    Saved as: {task}/task_summary.json
    """
    if not results:
        return {}
    
    # Group by agent
    by_agent = defaultdict(list)
    for r in results:
        agent = r.get("agent", "unknown")
        by_agent[agent].append(r)
    
    # Generate per-agent summaries
    for agent, agent_results in by_agent.items():
        agent_dir = output_dir / agent
        if agent_dir.exists():
            generate_agent_summary(agent_results, agent_dir)
    
    summary = {
        "task": results[0].get("task"),
        "models": results[0].get("models", {}),
        "timestamp": datetime.now().isoformat(),
        "total_agents": len(by_agent),
        "overall": calculate_stats(results),
        "by_agent": {
            agent: calculate_stats(agent_results)
            for agent, agent_results in sorted(by_agent.items())
        },
    }
    
    # Save summary in the task directory
    summary_file = output_dir / "task_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def generate_overall_summary(results: list[dict], output_dir: Path) -> dict:
    """Generate overall summary across all tasks.
    
    Saved as: overall_summary.json
    """
    if not results:
        return {}
    
    # Group by task
    by_task = defaultdict(list)
    for r in results:
        task = r.get("task", "unknown")
        by_task[task].append(r)
    
    # Generate per-task summaries
    for task, task_results in by_task.items():
        task_dir = output_dir / task
        if task_dir.exists():
            generate_task_summary(task_results, task_dir)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(by_task),
        "total_evaluations": len(results),
        "overall": calculate_stats(results),
        "by_task": {
            task: calculate_stats(task_results)
            for task, task_results in sorted(by_task.items())
        },
    }
    
    # Save summary in the root results directory
    summary_file = output_dir / "overall_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def generate_all_summaries(results_dir: Path) -> dict:
    """Generate all hierarchical summaries from results directory."""
    print(f"\n{'='*60}")
    print("GENERATING HIERARCHICAL SUMMARIES")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    
    # Load all results
    results = load_result_files(results_dir)
    print(f"Loaded {len(results)} evaluation results")
    
    if not results:
        print("No results found!")
        return {}
    
    # Generate overall summary (which cascades to task -> agent -> user summaries)
    overall = generate_overall_summary(results, results_dir)
    
    print(f"\n{'─'*60}")
    print("SUMMARY STATISTICS")
    print(f"{'─'*60}")
    print(f"Total evaluations: {overall.get('total_evaluations', 0)}")
    print(f"Total tasks: {overall.get('total_tasks', 0)}")
    
    if overall.get("overall"):
        stats = overall["overall"]
        print(f"Overall pass rate: {stats.get('overall_pass_rate', 0):.1%}")
        print(f"Total checks: {stats.get('total_passed', 0)}/{stats.get('total_checks', 0)}")
    
    print(f"\nBy task:")
    for task, stats in overall.get("by_task", {}).items():
        rate = stats.get("overall_pass_rate", 0)
        print(f"  {task}: {rate:.1%} ({stats.get('count', 0)} evals)")
    
    print(f"\n{'='*60}")
    print(f"Summaries saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    return overall


def print_detailed_summary(results_dir: Path):
    """Print a detailed summary of all results."""
    results = load_result_files(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Group hierarchically
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        task = r.get("task", "unknown")
        agent = r.get("agent", "unknown")
        user = r.get("user", "unknown")
        hierarchy[task][agent][user].append(r)
    
    print(f"\n{'='*70}")
    print("DETAILED EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    total_pass = 0
    total_checks = 0
    
    for task in sorted(hierarchy.keys()):
        print(f"\n{'─'*70}")
        print(f"TASK: {task}")
        print(f"{'─'*70}")
        
        for agent in sorted(hierarchy[task].keys()):
            print(f"\n  AGENT: {agent}")
            
            for user in sorted(hierarchy[task][agent].keys()):
                user_results = hierarchy[task][agent][user]
                stats = calculate_stats(user_results)
                
                passed = stats.get("total_passed", 0)
                checks = stats.get("total_checks", 0)
                rate = stats.get("overall_pass_rate", 0)
                
                total_pass += passed
                total_checks += checks
                
                status_icon = "✓" if rate >= 0.8 else ("~" if rate >= 0.5 else "✗")
                print(f"    {status_icon} {user}: {passed}/{checks} ({rate:.0%}) [{len(user_results)} langs]")
                
                # Show per-check breakdown
                for check_name, check_data in stats.get("checks", {}).items():
                    check_rate = check_data.get("pass_rate", 0)
                    check_icon = "✓" if check_rate >= 0.8 else ("~" if check_rate >= 0.5 else "✗")
                    print(f"        {check_icon} {check_name}: {check_data['passed']}/{check_data['total']}")
    
    print(f"\n{'='*70}")
    print("OVERALL")
    print(f"{'='*70}")
    overall_rate = total_pass / total_checks if total_checks > 0 else 0
    print(f"Total: {total_pass}/{total_checks} ({overall_rate:.0%})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent.parent / "results" / "comprehensive"
    
    generate_all_summaries(results_dir)
    print_detailed_summary(results_dir)
