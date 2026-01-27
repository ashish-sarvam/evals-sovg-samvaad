"""
Command-line interface for running evaluations.

Usage:
    poetry run python -m multilingual_evals --task <task_name> [options]

Examples:
    poetry run python -m multilingual_evals --task multilingual --agent dcs
    poetry run python -m multilingual_evals --task multilingual --agent dcs --user correcting
    poetry run python -m multilingual_evals --task multilingual --agent dcs --user cooperative,correcting
    poetry run python -m multilingual_evals --task english_user --agent dcs --languages hi-en -v
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multilingual_evals.config import RESULTS_DIR
from multilingual_evals.languages import Language, get_language
from multilingual_evals.models import (
    create_agent_model,
    create_user_model,
    create_verifier_model,
)
from multilingual_evals.runner import ConversationRunner
from multilingual_evals.tasks import (
    get_task,
    list_tasks,
    list_users,
    list_agents,
    BaseTask,
)
from agents import get_agent

# Thread lock for synchronized printing
_print_lock = threading.Lock()


def sync_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)


def run_single_evaluation(
    language: Language,
    agent_module,
    task: BaseTask,
    output_dir: Path,
    verifier_model_type: str | None,
    verbose: bool,
    lang_index: int = 0,
    total_langs: int = 1,
    model_provider: str | None = None,
    agent_temperature: float | None = None,
) -> dict:
    """Run evaluation for a single language."""
    import time

    start_time = time.time()

    sync_print(
        f"\n[{lang_index}/{total_langs}] 🌐 {language.name} ({language.code}) - Starting..."
    )

    # Create fresh models for thread safety
    agent_model = create_agent_model(provider=model_provider, temperature=agent_temperature)
    user_model = create_user_model()
    verifier_model = (
        create_verifier_model(verifier_model_type) if verifier_model_type else None
    )

    runner = ConversationRunner(
        agent_model=agent_model,
        user_model=user_model,
        task=task,
        verbose=verbose,
    )

    result = {
        "task": task.config.name,
        "agent": task.agent,
        "user": task.user,
        "model": model_provider or "default",
        "language_code": language.code,
        "language_name": language.name,
        "timestamp": datetime.now().isoformat(),
        "conversation": [],
        "verification": None,
        "error": None,
    }

    try:
        # Run conversation
        conv_start = time.time()
        conversation = runner.run(agent_module, language)
        conv_time = time.time() - conv_start
        result["conversation"] = conversation

        # Verify if enabled
        verification_summary = ""
        if verifier_model:
            verify_start = time.time()
            verification = task.verify(conversation, language, verifier_model)
            verify_time = time.time() - verify_start
            result["verification"] = verification
            status = verification.get("overall_status", "?")
            passed = verification.get("passed", 0)
            total = verification.get("total", 0)
            status_icon = (
                "✓" if status == "pass" else ("✗" if status == "fail" else "~")
            )
            verification_summary = f" | {status_icon} {status.upper()} ({passed}/{total}) [{verify_time:.1f}s]"

            # Add score info if present
            if verification.get("scores"):
                for score_name, score_data in verification["scores"].items():
                    score_val = score_data.get("score", 0)
                    verification_summary += f" | 📊 {score_val}/100"

        total_time = time.time() - start_time
        sync_print(
            f"[{lang_index}/{total_langs}] ✓ {language.name}: {len(conversation)} msgs [{conv_time:.1f}s]{verification_summary} (total: {total_time:.1f}s)"
        )

    except Exception as e:
        result["error"] = str(e)
        sync_print(f"[{lang_index}/{total_langs}] ✗ {language.name}: ERROR - {e}")

    # Save result
    filename = f"{task.config.name}_{language.code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def run_evaluation(
    task: BaseTask,
    output_dir: Path,
    skip_verification: bool,
    verbose: bool,
    parallel: int,
    model_provider: str | None = None,
    agent_temperature: float | None = None,
):
    """Run evaluation for all languages in task."""

    agent_name = task.agent
    user_name = task.user

    print("=" * 60)
    print(f"TASK: {task.config.name}")
    print(f"Agent: {agent_name} | User: {user_name}")
    # Determine model name for output path
    model_name = model_provider or "default"
    
    print(f"Model: {model_name}")
    print(f"Description: {task.config.description}")
    print(f"Languages: {len(task.config.languages)}")
    print(f"Verification: {'off' if skip_verification else 'on'}")
    print(f"Parallel: {parallel}")
    print("=" * 60)

    # Setup - include agent, user, and model in output path
    task_output_dir = output_dir / task.config.name / agent_name / user_name / model_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    agent_module = get_agent(task.config.agent_name)
    verifier_model_type = None if skip_verification else task.config.verifier_model
    languages = task.config.languages

    print(f"\nAgent: {agent_module.AGENT_NAME}")
    print(f"Output: {task_output_dir}")

    total_langs = len(languages)
    print(f"\n{'─' * 60}")
    print(f"Starting evaluation of {total_langs} languages...")
    print(f"{'─' * 60}")

    import time

    eval_start = time.time()
    results = []

    if parallel > 1:
        print(f"[Running {parallel} languages in parallel]\n")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_single_evaluation,
                    lang,
                    agent_module,
                    task,
                    task_output_dir,
                    verifier_model_type,
                    verbose,
                    idx + 1,
                    total_langs,
                    model_provider,
                    agent_temperature,
                ): (idx, lang)
                for idx, lang in enumerate(languages)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    idx, lang = futures[future]
                    sync_print(f"[{idx + 1}/{total_langs}] ✗ {lang.name}: FAILED - {e}")
                    results.append(
                        {
                            "task": task.config.name,
                            "language_code": lang.code,
                            "language_name": lang.name,
                            "error": str(e),
                        }
                    )
    else:
        print("[Running sequentially]\n")
        for idx, language in enumerate(languages):
            result = run_single_evaluation(
                language,
                agent_module,
                task,
                task_output_dir,
                verifier_model_type,
                verbose,
                idx + 1,
                total_langs,
                model_provider,
                agent_temperature,
            )
            results.append(result)

    eval_time = time.time() - eval_start
    print(f"\n{'─' * 60}")
    print(f"Evaluation complete! Total time: {eval_time:.1f}s")
    print(f"{'─' * 60}")

    # Generate summary
    print("\nGenerating summary...")
    summary = generate_summary(task, results)
    summary["total_time_seconds"] = round(eval_time, 1)
    summary_file = (
        task_output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nSummary saved to: {summary_file}")
    return summary


def generate_summary(task: BaseTask, results: list[dict]) -> dict:
    """Generate evaluation summary."""
    successful = [r for r in results if not r.get("error")]

    # Track pass/fail for each check across all languages
    checks: dict[str, dict] = (
        {}
    )  # check_name -> {passed: int, total: int, failures: []}
    scores: dict[str, dict] = {}  # score_name -> {values: [], expected: str}
    statuses = {"pass": 0, "fail": 0, "partial": 0, "error": 0}

    for r in results:
        if r.get("verification"):
            v = r["verification"]
            statuses[v.get("overall_status", "error")] += 1

            # Track boolean checks
            for check in v.get("results", []):
                name = check["check_name"]
                if name not in checks:
                    checks[name] = {"passed": 0, "total": 0, "failures": []}
                checks[name]["total"] += 1
                if check.get("passed", False):
                    checks[name]["passed"] += 1
                else:
                    failure = {
                        "language": r["language_code"],
                        "reason": check.get("reason", ""),
                    }
                    if check.get("snippet"):
                        failure["snippet"] = check["snippet"]
                    checks[name]["failures"].append(failure)

            # Track numeric scores
            for score_name, score_data in v.get("scores", {}).items():
                if score_name not in scores:
                    scores[score_name] = {
                        "values": [],
                        "expected": score_data.get("expected", ""),
                    }
                scores[score_name]["values"].append(
                    {
                        "language": r["language_code"],
                        "score": score_data.get("score", 0),
                        "reason": score_data.get("reason", ""),
                        "examples": score_data.get("examples", []),
                    }
                )

    # Calculate pass rates
    check_results = {
        name: {
            "passed": data["passed"],
            "total": data["total"],
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
            "failures": data["failures"],
        }
        for name, data in checks.items()
    }

    # Calculate score averages
    score_results = {}
    for score_name, score_data in scores.items():
        values = [v["score"] for v in score_data["values"]]
        score_results[score_name] = {
            "expected": score_data["expected"],
            "average": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "by_language": score_data["values"],
        }

    total_checks = sum(c["total"] for c in check_results.values())
    total_passed = sum(c["passed"] for c in check_results.values())
    overall_pass_rate = total_passed / total_checks if total_checks > 0 else 0

    # Get model from first result
    model = results[0].get("model", "default") if results else "default"
    
    summary = {
        "task": task.config.name,
        "agent": task.agent,
        "user": task.user,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total_languages": len(results),
        "successful_conversations": len(successful),
        "statuses": statuses,
        "checks": check_results,
        "overall_pass_rate": overall_pass_rate,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "by_language": {
            r["language_code"]: {
                "status": (
                    r.get("verification", {}).get("overall_status", "no_verification")
                    if r.get("verification")
                    else ("error" if r.get("error") else "no_verification")
                ),
                "passed": (
                    r.get("verification", {}).get("passed", 0)
                    if r.get("verification")
                    else 0
                ),
                "total": (
                    r.get("verification", {}).get("total", 0)
                    if r.get("verification")
                    else 0
                ),
                "summary": (
                    r.get("verification", {}).get("summary", "")
                    if r.get("verification")
                    else ""
                ),
            }
            for r in results
        },
    }

    # Add scores if present
    if score_results:
        summary["scores"] = score_results

    return summary


def print_summary(summary: dict):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print(
        f"SUMMARY: {summary['task']} / {summary.get('agent', 'dcs')} / {summary['user']}"
    )
    print("=" * 60)
    print(
        f"Languages: {summary['total_languages']} | Conversations: {summary['successful_conversations']}"
    )
    print(
        f"Pass: {summary['statuses']['pass']} | Partial: {summary['statuses']['partial']} | Fail: {summary['statuses']['fail']}"
    )

    if summary.get("checks"):
        print("\nChecks:")
        for name, data in summary["checks"].items():
            status = (
                "✓"
                if data["pass_rate"] == 1.0
                else ("✗" if data["pass_rate"] == 0 else "~")
            )
            print(
                f"  {status} {name}: {data['passed']}/{data['total']} ({data['pass_rate']:.0%})"
            )
            # Show failures
            if data["failures"]:
                for f in data["failures"][:3]:  # Show max 3 failures
                    reason = (
                        f["reason"][:50] + "..."
                        if len(f["reason"]) > 50
                        else f["reason"]
                    )
                    print(f"      └─ {f['language']}: {reason}")
                    if f.get("snippet"):
                        snippet = (
                            f["snippet"][:60] + "..."
                            if len(f["snippet"]) > 60
                            else f["snippet"]
                        )
                        print(f'         "{snippet}"')
        print(
            f"\nOverall: {summary['total_passed']}/{summary['total_checks']} ({summary['overall_pass_rate']:.0%})"
        )

    # Show numeric scores (e.g., colloquial_score)
    if summary.get("scores"):
        print("\nScores:")
        for score_name, score_data in summary["scores"].items():
            expected = score_data.get("expected", "")
            avg = score_data.get("average", 0)
            min_val = score_data.get("min", 0)
            max_val = score_data.get("max", 0)

            # Determine if score matches expectation (3-tier: 67-100 rural, 34-66 mixed, 0-33 urban)
            if expected == "rural":
                status = "✓" if avg >= 67 else ("~" if avg >= 34 else "✗")
            elif expected == "urban":
                status = "✓" if avg <= 33 else ("~" if avg <= 66 else "✗")
            else:
                status = "~"

            print(
                f"  {status} {score_name}: avg={avg:.0f} (min={min_val:.0f}, max={max_val:.0f})"
            )
            print(
                f"      Expected: {expected} | {'✓ Matches' if status == '✓' else '✗ Does not match' if status == '✗' else '~ Partial match'}"
            )

            # Show per-language breakdown
            print("      By language:")
            for lang_data in score_data.get("by_language", [])[:5]:  # Show max 5
                print(f"        {lang_data['language']}: {lang_data['score']}")

    print("\nBy Language:")
    for code, data in summary["by_language"].items():
        status_icon = (
            "✓"
            if data["status"] == "pass"
            else ("✗" if data["status"] == "fail" else "~")
        )
        print(f"  {status_icon} {code}: {data['passed']}/{data['total']} checks")
    print("=" * 60)


def show_available_tasks():
    """Show available tasks, agents, and users."""
    print("\nAvailable tasks:")
    print("=" * 50)
    for task_name in list_tasks():
        agents = list_agents(task_name)
        print(f"\n  --task {task_name}")
        for agent in agents:
            users = list_users(task_name, agent)
            print(f"    --agent {agent}")
            print(f"      Users: {', '.join(users)}")
    print("\n" + "=" * 50)
    print("\nExamples:")
    print("  # Single task, single user")
    print(
        "  python -m multilingual_evals --task multilingual --agent dcs --user cooperative"
    )
    print("")
    print("  # Single task, multiple users")
    print(
        "  python -m multilingual_evals --task multilingual --agent dcs --user cooperative,correcting"
    )
    print("")
    print("  # Single task, all users")
    print("  python -m multilingual_evals --task multilingual --agent dcs")
    print("")
    print("  # Multiple tasks, all users")
    print("  python -m multilingual_evals --task multilingual,english_user --agent dcs")
    print("")
    print("  # All tasks, all users")
    print("  python -m multilingual_evals --task all --agent dcs")
    print()


def handle_conversationality_task(args, output_dir: Path):
    """Handle conversationality task with its special modes."""
    from agents import get_agent
    from multilingual_evals.tasks.conversationality import ConversationalityTask
    from multilingual_evals.tasks.conversationality.generator import BlueprintGenerator
    from multilingual_evals.tasks.conversationality.evaluator import PairwiseEvaluator
    import json

    if not args.agent:
        print("Error: --agent is required for conversationality task")
        return

    agent_module = get_agent(args.agent)
    conv_output_dir = output_dir / "conversationality" / args.agent

    if args.mode == "generate-blueprints":
        # Generate blueprints using GPT 5.2 chat
        print(f"\n{'=' * 60}")
        print(f"GENERATING BLUEPRINTS FOR AGENT: {args.agent}")
        print(f"{'=' * 60}")

        generator = BlueprintGenerator()
        blueprints = generator.generate_and_save(agent_module, args.agent)
        generator.print_summary(blueprints)

    elif args.mode == "generate":
        # Generate trajectories with specified model(s)
        print(f"\n{'=' * 60}")
        print(f"GENERATING TRAJECTORIES FOR AGENT: {args.agent}")
        print(f"{'=' * 60}")

        # Load blueprints
        blueprints = ConversationalityTask.load_blueprints(args.agent)
        if not blueprints:
            print(f"Error: No blueprints found for agent '{args.agent}'")
            print("Run with --mode generate-blueprints first")
            return

        # Filter blueprints if --user is specified
        if args.user:
            user_names = [u.strip() for u in args.user.split(",")]
            blueprints = [bp for bp in blueprints if bp.name in user_names]
            if not blueprints:
                print(f"Error: No matching blueprints found for: {user_names}")
                print(f"Available: {[bp.name for bp in ConversationalityTask.load_blueprints(args.agent)]}")
                return

        print(f"Running {len(blueprints)} blueprints: {[bp.name for bp in blueprints]}")

        # Determine which model(s) to run
        models_to_run = [args.model] if args.model else ["tinker"]

        for model_name in models_to_run:
            print(f"\n--- Running with model: {model_name} ---")

            # Create model based on provider
            if model_name == "tinker":
                agent_model = create_agent_model("tinker", temperature=args.temperature)
            elif model_name == "azure":
                agent_model = create_agent_model("azure", temperature=args.temperature)
            elif model_name == "lepton":
                agent_model = create_agent_model("lepton", temperature=args.temperature)
            else:
                agent_model = create_agent_model("openai", temperature=args.temperature)

            user_model = create_user_model()

            for bp_idx, bp in enumerate(blueprints):
                print(f"\n  [{bp_idx + 1}/{len(blueprints)}] Blueprint: {bp.name}")
                print(f"      Tags: {', '.join(bp.tags)}")
                print(f"      Challenge: {bp.challenge_start} | Difficulty: {bp.difficulty}")
                
                task = ConversationalityTask(
                    agent=args.agent,
                    blueprint=bp,
                    blueprint_name=bp.name,
                )

                # Run for each language (or subset)
                languages = task.config.languages
                if args.languages:
                    from multilingual_evals.languages import get_language
                    languages = [get_language(c.strip()) for c in args.languages.split(",")]
                    languages = [l for l in languages if l]

                for lang_idx, lang in enumerate(languages):
                    print(f"      [{lang_idx + 1}/{len(languages)}] {lang.name}: ", end="", flush=True)

                    runner = ConversationRunner(
                        agent_model=agent_model,
                        user_model=user_model,
                        task=task,
                        verbose=args.verbose,
                    )

                    conversation = runner.run(agent_module, lang)

                    # Save trajectory
                    traj_dir = conv_output_dir / "trajectories" / model_name / bp.name
                    traj_dir.mkdir(parents=True, exist_ok=True)
                    traj_file = traj_dir / f"{lang.code}.json"

                    result = {
                        "model": model_name,
                        "agent": args.agent,
                        "blueprint": bp.name,
                        "language": lang.code,
                        "conversation": conversation,
                        "timestamp": datetime.now().isoformat(),
                    }

                    with open(traj_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    print(f" -> {len(conversation)} msgs")

        print(f"\nTrajectories saved to: {conv_output_dir / 'trajectories'}")

    elif args.mode == "evaluate":
        # Run pairwise evaluation
        print(f"\n{'=' * 60}")
        print(f"RUNNING PAIRWISE EVALUATION")
        print(f"{'=' * 60}")

        evaluator = PairwiseEvaluator()
        traj_dir = conv_output_dir / "trajectories"

        if not traj_dir.exists():
            print("Error: No trajectories found. Run with --mode generate first")
            return

        # Find available models
        available_models = [d.name for d in traj_dir.iterdir() if d.is_dir()]
        print(f"Available models: {', '.join(available_models)}")

        if "tinker" not in available_models:
            print("Error: Tinker trajectories required for comparison")
            return

        baseline_models = [m for m in available_models if m != "tinker"]
        if not baseline_models:
            print("Error: Need at least one baseline model to compare against Tinker")
            return

        # Load blueprints for metadata
        blueprints_file = Path(__file__).parent / "tasks" / "conversationality" / "users" / f"{args.agent}_generated.json"
        if blueprints_file.exists():
            with open(blueprints_file) as f:
                blueprints_data = json.load(f)
                blueprints_map = {bp["name"]: bp for bp in blueprints_data.get("blueprints", [])}
        else:
            blueprints_map = {}

        # Compare Tinker vs each baseline
        all_evaluations = []
        
        # Prepare output directory
        eval_dir = conv_output_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)

        for baseline in baseline_models:
            print(f"\n--- Comparing Tinker vs {baseline} ---")

            tinker_dir = traj_dir / "tinker"
            baseline_dir = traj_dir / baseline

            # Find common blueprints
            tinker_blueprints = {d.name for d in tinker_dir.iterdir() if d.is_dir()}
            baseline_blueprints = {d.name for d in baseline_dir.iterdir() if d.is_dir()}
            common_blueprints = tinker_blueprints & baseline_blueprints

            # Filter blueprints if --user is specified
            if args.user:
                user_filter = {u.strip() for u in args.user.split(",")}
                common_blueprints = common_blueprints & user_filter
                if not common_blueprints:
                    print(f"  Warning: No matching blueprints for filter: {user_filter}")
                    continue
            
            # Collect all evaluation tasks
            eval_tasks = []
            for bp_name in common_blueprints:
                tinker_langs = {f.stem for f in (tinker_dir / bp_name).glob("*.json")}
                baseline_langs = {f.stem for f in (baseline_dir / bp_name).glob("*.json")}
                common_langs = tinker_langs & baseline_langs

                for lang_code in common_langs:
                    eval_tasks.append({
                        "bp_name": bp_name,
                        "lang_code": lang_code,
                        "tinker_file": tinker_dir / bp_name / f"{lang_code}.json",
                        "baseline_file": baseline_dir / bp_name / f"{lang_code}.json",
                    })

            print(f"  Total evaluations: {len(eval_tasks)}")
            
            # Create directory structure for per-trajectory saves
            comparison_dir = eval_dir / f"tinker_vs_{baseline}"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            baseline_evaluations = []
            
            def run_single_eval(task_info):
                """Run a single evaluation and save to individual file."""
                bp_name = task_info["bp_name"]
                lang_code = task_info["lang_code"]
                
                with open(task_info["tinker_file"]) as f:
                    tinker_traj = json.load(f)
                with open(task_info["baseline_file"]) as f:
                    baseline_traj = json.load(f)
                
                bp_meta = blueprints_map.get(bp_name, {"name": bp_name, "tags": [], "expected_agent_behavior": []})
                
                # Create fresh evaluator for thread safety
                eval_instance = PairwiseEvaluator()
                result = eval_instance.evaluate(
                    trajectory_a=tinker_traj["conversation"],
                    trajectory_b=baseline_traj["conversation"],
                    model_a="tinker",
                    model_b=baseline,
                    blueprint=bp_meta,
                )
                result["language"] = lang_code
                result["blueprint"] = bp_name
                result["comparison"] = f"tinker_vs_{baseline}"
                result["timestamp"] = datetime.now().isoformat()
                
                # Save individual evaluation file
                bp_eval_dir = comparison_dir / bp_name
                bp_eval_dir.mkdir(parents=True, exist_ok=True)
                eval_file = bp_eval_dir / f"{lang_code}.json"
                with open(eval_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                return result
            
            # Run evaluations in parallel (default 10 workers for evals)
            parallel = args.parallel if args.parallel > 0 else 10
            
            if parallel > 1:
                print(f"  Running {parallel} evaluations in parallel...")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=parallel) as executor:
                    future_to_task = {executor.submit(run_single_eval, task): task for task in eval_tasks}
                    
                    for idx, future in enumerate(as_completed(future_to_task)):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            baseline_evaluations.append(result)
                            all_evaluations.append(result)
                            
                            # Print result
                            winner = result.get("overall_preference", {}).get("winner", "?")
                            sync_print(f"  [{idx+1}/{len(eval_tasks)}] {task['bp_name']}/{task['lang_code']}: {winner}")
                                
                        except Exception as e:
                            sync_print(f"  [{idx+1}/{len(eval_tasks)}] {task['bp_name']}/{task['lang_code']}: ERROR - {e}")
            else:
                # Sequential execution
                for idx, task in enumerate(eval_tasks):
                    print(f"  [{idx+1}/{len(eval_tasks)}] {task['bp_name']}/{task['lang_code']}...", end=" ", flush=True)
                    
                    try:
                        result = run_single_eval(task)
                        baseline_evaluations.append(result)
                        all_evaluations.append(result)
                        
                        winner = result.get("overall_preference", {}).get("winner", "?")
                        print(f"Winner: {winner}")
                            
                    except Exception as e:
                        print(f"ERROR - {e}")
            
            # Save summary file for this comparison
            summary_file = comparison_dir / "summary.json"
            summary = {
                "comparison": f"tinker_vs_{baseline}",
                "agent": args.agent,
                "timestamp": datetime.now().isoformat(),
                "total_comparisons": len(baseline_evaluations),
                "status": "completed",
                "blueprints_evaluated": list(set(e.get("blueprint") for e in baseline_evaluations)),
                "quick_results": [
                    {
                        "blueprint": e.get("blueprint"),
                        "language": e.get("language"),
                        "winner": e.get("overall_preference", {}).get("winner", "?"),
                    }
                    for e in baseline_evaluations
                ],
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\nResults saved to: {comparison_dir}/")

        # Calculate and print aggregate
        aggregate = evaluator.calculate_aggregate(all_evaluations)
        print(f"\n{'=' * 60}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 60}")
        print(f"Total comparisons: {aggregate['total_comparisons']}")
        print(f"\nOverall Preference:")
        print(f"  Tinker wins: {aggregate['wins']['a']}")
        print(f"  Baseline wins: {aggregate['wins']['b']}")
        print(f"  Ties: {aggregate['wins']['tie']}")

        if aggregate.get("criteria_scores"):
            print("\nScored Criteria (avg):")
            for criterion, scores in aggregate["criteria_scores"].items():
                diff = scores['avg_a'] - scores['avg_b']
                winner = "Tinker+" if diff > 0 else ("Baseline+" if diff < 0 else "Tie")
                print(f"  {criterion}: Tinker={scores['avg_a']:.1f} vs Baseline={scores['avg_b']:.1f} [{winner}]")
        
        if aggregate.get("boolean_criteria"):
            print("\nBoolean Criteria (pass rate %):")
            for criterion, data in aggregate["boolean_criteria"].items():
                print(f"  {criterion}: Tinker={data['pass_rate_a']:.0f}% vs Baseline={data['pass_rate_b']:.0f}%")
        
        # Save aggregate summary
        summary_file = conv_output_dir / "evaluations" / "aggregate_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "agent": args.agent,
                "aggregate": aggregate,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nAggregate summary saved to: {summary_file}")

    elif args.mode == "results":
        # Show results
        eval_dir = conv_output_dir / "evaluations"
        if not eval_dir.exists():
            print("No evaluation results found. Run with --mode evaluate first")
            return

        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")

        for eval_file in eval_dir.glob("*.json"):
            print(f"\n--- {eval_file.stem} ---")
            with open(eval_file) as f:
                data = json.load(f)
            
            # Handle both old format (list) and new format (dict with evaluations key)
            if isinstance(data, list):
                evaluations = data
            else:
                evaluations = data.get("evaluations", [])
                print(f"Agent: {data.get('agent', 'unknown')}")
                print(f"Timestamp: {data.get('timestamp', 'unknown')}")

            evaluator = PairwiseEvaluator()
            aggregate = evaluator.calculate_aggregate(evaluations)

            print(f"Comparisons: {aggregate['total_comparisons']}")
            print(f"Tinker wins: {aggregate['wins']['a']} | Baseline wins: {aggregate['wins']['b']} | Ties: {aggregate['wins']['tie']}")
            
            # Show per-criteria breakdown
            if aggregate.get("criteria_scores"):
                print("\nScored Criteria (avg):")
                for criterion, scores in aggregate["criteria_scores"].items():
                    diff = scores['avg_a'] - scores['avg_b']
                    winner = "Tinker" if diff > 0 else ("Baseline" if diff < 0 else "Tie")
                    print(f"  {criterion}: Tinker={scores['avg_a']:.1f} vs Baseline={scores['avg_b']:.1f} [{winner}]")
            
            if aggregate.get("boolean_criteria"):
                print("\nBoolean Criteria (pass rate %):")
                for criterion, data in aggregate["boolean_criteria"].items():
                    print(f"  {criterion}: Tinker={data['pass_rate_a']:.0f}% vs Baseline={data['pass_rate_b']:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", help="Task(s) to run: single task, comma-separated, or 'all'"
    )
    parser.add_argument("--agent", help="Agent to test (required)")
    parser.add_argument(
        "--user",
        help="User type(s), comma-separated. If not specified, runs all users.",
    )
    parser.add_argument("--languages", help="Override languages (comma-separated)")
    parser.add_argument("--max-turns", type=int, help="Max conversation turns")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--parallel", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["run", "generate-blueprints", "generate", "evaluate", "results"],
        default="run",
        help="Mode for conversationality task: generate-blueprints, generate, evaluate, results",
    )
    parser.add_argument(
        "--model",
        help="Agent model provider: tinker, azure, lepton, or openai (default: from config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for agent model (e.g., 0.1, 0.7, 1.0)",
    )

    args = parser.parse_args()

    # If no task or agent specified, show available options
    if not args.task or not args.agent:
        show_available_tasks()
        if args.task and not args.agent:
            print("Error: --agent is required.")
        return

    # Parse tasks
    available_tasks = list_tasks()
    if args.task.lower() == "all":
        tasks_to_run = available_tasks
    else:
        tasks_to_run = [t.strip() for t in args.task.split(",")]
        # Validate tasks
        for t in tasks_to_run:
            if t not in available_tasks:
                print(f"Error: Unknown task '{t}'")
                print(f"Available tasks: {', '.join(available_tasks)}")
                return

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Handle conversationality task specially
    if "conversationality" in tasks_to_run and args.mode != "run":
        handle_conversationality_task(args, output_dir)
        # Remove from tasks_to_run if handled
        tasks_to_run = [t for t in tasks_to_run if t != "conversationality"]
        if not tasks_to_run:
            return

    # Run each task
    for task_name in tasks_to_run:
        print(f"\n{'*' * 70}")
        print(f"* TASK: {task_name}")
        print(f"{'*' * 70}")

        # Validate agent for this task
        available_agents = list_agents(task_name)
        if args.agent not in available_agents:
            print(f"Error: Agent '{args.agent}' not available for task '{task_name}'")
            print(f"Available agents: {', '.join(available_agents)}")
            continue

        # Get users to run for this task
        available_users = list_users(task_name, args.agent)
        if args.user:
            users = [u.strip() for u in args.user.split(",")]
            # Filter to only valid users for this task
            valid_users = [u for u in users if u in available_users]
            if not valid_users:
                print(
                    f"No valid users for task '{task_name}'. Available: {', '.join(available_users)}"
                )
                continue
            users = valid_users
        else:
            users = available_users
            print(f"Running all users: {', '.join(users)}")

        # Collect summaries for aggregate
        all_summaries = []

        # Run for each user
        for user in users:
            print(f"\n{'#' * 60}")
            print(f"# TASK: {task_name} | AGENT: {args.agent} | USER: {user}")
            print(f"{'#' * 60}")

            task = get_task(task_name, agent=args.agent, user=user)

            # Apply overrides
            if args.languages:
                langs = [get_language(c.strip()) for c in args.languages.split(",")]
                task.config.languages = [lang for lang in langs if lang]

            if args.max_turns:
                task.config.max_turns = args.max_turns

            summary = run_evaluation(
                task=task,
                output_dir=output_dir,
                skip_verification=args.skip_verification,
                verbose=args.verbose,
                parallel=args.parallel,
                model_provider=args.model,
                agent_temperature=args.temperature,
            )
            all_summaries.append(summary)

        # Create aggregate summary if multiple users
        if len(users) > 1:
            print_aggregate_summary(
                task_name, args.agent, users, all_summaries, output_dir
            )


def print_aggregate_summary(
    task_name: str,
    agent_name: str,
    users: list[str],
    summaries: list[dict],
    output_dir: Path,
):
    """Print and save aggregate summary across all users."""
    print("\n" + "=" * 70)
    print(f"AGGREGATE SUMMARY: {task_name} / {agent_name}")
    print("=" * 70)

    # Aggregate stats
    total_conversations = sum(s.get("successful_conversations", 0) for s in summaries)
    total_checks = sum(s.get("total_checks", 0) for s in summaries)
    total_passed = sum(s.get("total_passed", 0) for s in summaries)
    overall_pass_rate = total_passed / total_checks if total_checks > 0 else 0

    print(f"\nUsers: {', '.join(users)}")
    print(f"Total conversations: {total_conversations}")
    print(f"Overall pass rate: {total_passed}/{total_checks} ({overall_pass_rate:.0%})")

    # Check if there are colloquial scores to show scale
    has_colloquial = any(s.get("scores", {}).get("colloquial_score") for s in summaries)
    if has_colloquial:
        print(
            "\nColloquial Score Scale: 0-33 = urban (high code-mixing) | 34-66 = mixed | 67-100 = rural (low code-mixing)"
        )
        print("  Rural users should score ≥67 | Urban users should score ≤33")

    # Per-user breakdown
    print("\nBy User:")
    for summary in summaries:
        user = summary.get("user", "?")
        passed = summary.get("total_passed", 0)
        total = summary.get("total_checks", 0)
        rate = passed / total if total > 0 else 0
        status = "✓" if rate >= 0.8 else ("~" if rate >= 0.5 else "✗")
        print(f"  {status} {user}: {passed}/{total} ({rate:.0%})")

        # Show scores if present (with clear explanation)
        if summary.get("scores"):
            for score_name, score_data in summary["scores"].items():
                avg = score_data.get("average", 0)
                expected = score_data.get("expected", "")

                # Determine if matches expectation (3-tier scale)
                if expected == "rural":
                    target = "≥67"
                    matches = "✓" if avg >= 67 else "✗"
                elif expected == "urban":
                    target = "≤33"
                    matches = "✓" if avg <= 33 else "✗"
                else:
                    target = "N/A"
                    matches = "~"

                print(
                    f"      📊 {score_name}: {avg:.0f} (expected {expected} {target}) {matches}"
                )

    # Aggregate check results
    print("\nBy Check (aggregated):")
    aggregated_checks = {}
    for summary in summaries:
        for check_name, check_data in summary.get("checks", {}).items():
            if check_name not in aggregated_checks:
                aggregated_checks[check_name] = {"passed": 0, "total": 0}
            aggregated_checks[check_name]["passed"] += check_data.get("passed", 0)
            aggregated_checks[check_name]["total"] += check_data.get("total", 0)

    for check_name, data in aggregated_checks.items():
        rate = data["passed"] / data["total"] if data["total"] > 0 else 0
        status = "✓" if rate == 1.0 else ("✗" if rate == 0 else "~")
        print(f"  {status} {check_name}: {data['passed']}/{data['total']} ({rate:.0%})")

    print("=" * 70)

    # Save aggregate summary
    aggregate = {
        "task": task_name,
        "agent": agent_name,
        "users": users,
        "timestamp": datetime.now().isoformat(),
        "total_conversations": total_conversations,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "by_user": {
            s["user"]: {
                "passed": s.get("total_passed", 0),
                "total": s.get("total_checks", 0),
            }
            for s in summaries
        },
        "by_check": aggregated_checks,
    }

    # Aggregate scores if present (e.g., colloquial_score)
    aggregated_scores = {}
    for summary in summaries:
        user = summary["user"]
        for score_name, score_data in summary.get("scores", {}).items():
            if score_name not in aggregated_scores:
                aggregated_scores[score_name] = {
                    "scale": "0 = urban/casual, 100 = rural/formal",
                    "by_user": [],
                }

            # Determine expected based on user type
            expected = score_data.get("expected", "")
            avg = score_data.get("average", 0)

            # Check if score matches expectation (3-tier scale)
            if expected == "rural":
                matches = avg >= 67
                target = "≥67"
            elif expected == "urban":
                matches = avg <= 33
                target = "≤33"
            else:
                matches = True
                target = "N/A"

            aggregated_scores[score_name]["by_user"].append(
                {
                    "user": user,
                    "expected_tone": expected,
                    "target_score": target,
                    "avg_across_languages": round(avg, 1),
                    "matches_expectation": matches,
                }
            )
    if aggregated_scores:
        aggregate["scores"] = aggregated_scores

    aggregate_file = (
        output_dir
        / task_name
        / agent_name
        / f"aggregate_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    aggregate_file.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate_file, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"\nAggregate summary saved to: {aggregate_file}")


if __name__ == "__main__":
    main()
