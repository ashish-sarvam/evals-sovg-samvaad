#!/usr/bin/env python3
"""
Simple eval runner that reads config from YAML and runs evals.

Parallelizes ALL task/agent/user/language combinations in a single CLI call.

Usage:
    python run_evals.py --config eval_config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and validate YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_sections = ["model", "settings", "evals"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    required_model_fields = ["provider", "temperature", "max_tokens"]
    for field in required_model_fields:
        if field not in config["model"]:
            raise ValueError(f"Missing required model field: {field}")

    required_settings_fields = [
        "max_turns",
        "parallel_workers",
        "output_dir",
        "skip_verification",
        "verbose",
    ]
    for field in required_settings_fields:
        if field not in config["settings"]:
            raise ValueError(f"Missing required settings field: {field}")

    if not config["evals"]:
        raise ValueError("At least one eval must be specified")

    # Validate each eval entry has required fields including languages
    for i, eval_entry in enumerate(config["evals"]):
        for field in ["task", "agent", "users", "languages"]:
            if field not in eval_entry:
                raise ValueError(f"Missing required field '{field}' in eval entry {i}")

    return config


def expand_combinations(evals: list[dict]) -> list[tuple[str, str, str, str]]:
    """Expand eval entries into (task, agent, user, language) combinations."""
    combinations = []
    for eval_entry in evals:
        task = eval_entry["task"]
        agent = eval_entry["agent"]
        users = eval_entry["users"]
        languages = eval_entry["languages"]
        
        for user in users:
            for lang in languages:
                combinations.append((task, agent, user, lang))
    
    return combinations


def build_unified_command(
    combinations: list[tuple[str, str, str, str]],
    model_config: dict,
    settings: dict,
) -> list[str]:
    """Build a single CLI command for all combinations."""
    # Extract unique values
    tasks = sorted(set(c[0] for c in combinations))
    agents = sorted(set(c[1] for c in combinations))
    users = sorted(set(c[2] for c in combinations))
    languages = sorted(set(c[3] for c in combinations))

    cmd = [
        sys.executable,
        "-m",
        "core",
        "--task",
        ",".join(tasks),
        "--agent",
        ",".join(agents),
        "--user",
        ",".join(users),
        "--languages",
        ",".join(languages),
        "--model",
        model_config["provider"],
        "--temperature",
        str(model_config["temperature"]),
        "--max-turns",
        str(settings["max_turns"]),
        "--output-dir",
        settings["output_dir"],
        "-p",
        str(settings["parallel_workers"]),
    ]

    if settings["skip_verification"]:
        cmd.append("--skip-verification")

    if settings["verbose"]:
        cmd.append("-v")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run evals from YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_config = config["model"]
    settings = config["settings"]
    evals = config["evals"]

    # Expand all (task, agent, user, language) combinations
    combinations = expand_combinations(evals)
    
    # Count unique task/agent/user combos
    unique_combos = set((t, a, u) for t, a, u, _ in combinations)
    unique_langs = set(l for _, _, _, l in combinations)

    print(f"\n{'='*60}")
    print("EVAL RUNNER")
    print(f"{'='*60}")
    print(f"Model: {model_config['provider']} (temp={model_config['temperature']})")
    print(f"Languages: {len(unique_langs)} ({', '.join(sorted(unique_langs))})")
    print(f"Total combinations: {len(combinations)}")
    print(f"  ({len(unique_combos)} task/agent/user × languages)")
    print(f"Parallel workers: {settings['parallel_workers']}")
    print(f"Output dir: {settings['output_dir']}")
    print(f"{'='*60}")
    
    # Group by task for display
    by_task = {}
    for t, a, u, l in combinations:
        if t not in by_task:
            by_task[t] = set()
        by_task[t].add((a, u))
    
    print("\nCombinations by task:")
    for task in sorted(by_task.keys()):
        combos = by_task[task]
        print(f"  {task}: {len(combos)} agent/user combos × {len(unique_langs)} langs")
    print()

    # Build single unified command
    cmd = build_unified_command(combinations, model_config, settings)

    if args.dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        print(f"\n{'='*60}")
        print("DRY RUN COMPLETE")
        print(f"{'='*60}\n")
        return

    print(f"{'='*60}")
    print("RUNNING ALL COMBINATIONS IN PARALLEL")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if result.returncode == 0:
        print(f"  [PASS] All {len(combinations)} combinations completed")
    else:
        print(f"  [FAIL] Some combinations failed (exit code: {result.returncode})")

    print(f"{'='*60}\n")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
