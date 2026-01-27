"""
Analyze GDPVal tasks to categorize the skills being tested.

Usage:
    python analyze_skills.py --limit 10  # analyze first 10 tasks
    python analyze_skills.py              # analyze all tasks
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from gemini_client import GeminiClient

# Skill categories - designed for LLM research (SFT/RL training signals)
SKILL_CATEGORIES = """
## INPUT PROCESSING (multimodal SFT)
- table_parsing: Extract structured data from Excel/CSV tables
- document_ocr: Parse text/layout from PDFs, scanned docs
- image_understanding: Interpret charts, diagrams, screenshots
- multi_file_grounding: Cross-reference info across multiple files
- long_context_retrieval: Find specific info in long documents

## REASONING (CoT, process reward models)
- arithmetic: Basic math, percentages, conversions
- multi_step_planning: Break task into ordered subtasks
- constraint_reasoning: Satisfy multiple competing constraints
- temporal_reasoning: Handle dates, deadlines, schedules
- logical_deduction: If-then reasoning, policy application

## KNOWLEDGE (domain SFT, RAG)
- domain_procedural: Know HOW to do domain tasks (tax, audits)
- domain_factual: Know domain FACTS (rates, regulations)
- format_conventions: Know standard formats (1040, P&L, org charts)

## OUTPUT GENERATION (format SFT, RLHF)
- structured_output: Generate valid JSON, XML, code
- tabular_generation: Create well-formed tables, spreadsheets
- document_formatting: Apply headers, sections, professional layout
- specification_adherence: Match exact output requirements
- file_creation: Generate actual files (xlsx, docx, pdf)

## COMPOSITION (RL, agentic training)
- tool_orchestration: Chain multiple tools/operations correctly
- iterative_refinement: Self-correct based on intermediate results
- web_research: Search and synthesize from web sources
"""

# Model type categories
MODEL_TYPES = [
    "text_only",  # Pure text in, text out
    "multimodal_input",  # Needs to read images/PDFs/Excel visually
    "multimodal_output",  # Needs to generate files/images
    "multimodal_full",  # Both input and output are multimodal
    "tool_augmented",  # Requires external tool use (code exec, web)
]

ANALYSIS_PROMPT = """You are an LLM researcher analyzing evaluation tasks.

TASK: {prompt}

REFERENCE FILES: {reference_files}

SKILL TAXONOMY:
{categories}

MODEL TYPES:
- text_only: Pure text in/out, no file parsing needed
- multimodal_input: Must read PDFs/Excel/images (visual understanding)
- multimodal_output: Must generate files (xlsx, docx, pdf)
- multimodal_full: Both input and output are multimodal
- tool_augmented: Requires code execution, web search, or APIs

Analyze this task. Pick ALL relevant skills from each category (can be multiple).

Return JSON only (no markdown, no explanation):
{{
  "model_type": "multimodal_full",
  "input_skills": ["table_parsing", "multi_file_grounding"],
  "reasoning_skills": ["arithmetic", "constraint_reasoning"],
  "knowledge_skills": ["domain_procedural", "format_conventions"],
  "output_skills": ["tabular_generation", "file_creation"],
  "composition_skills": [],
  "primary_bottleneck": "hardest skill for current LLMs",
  "training_signal": "multimodal_sft|domain_sft|rl|tool_use",
  "difficulty": "easy|medium|hard"
}}"""


def extract_json(text: str) -> str:
    """Extract JSON from response, handling markdown and extra text."""
    text = text.strip()

    # Remove markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find JSON object boundaries
    start = text.find("{")
    if start == -1:
        return text

    # Find matching closing brace
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]  # Return partial if no closing brace


def analyze_task(client: GeminiClient, task: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single task and return skill classification."""
    prompt = ANALYSIS_PROMPT.format(
        prompt=task.get("prompt", "")[:2000],  # Truncate if too long
        reference_files=task.get("reference_files", []),
        categories=SKILL_CATEGORIES,
    )

    response = ""
    try:
        response = client.generate(
            prompt,
            system_prompt="Return only valid JSON. No markdown. No explanation.",
            max_tokens=3000,
            temperature=0.2,
        )
        print("response", response)

        # Extract and parse JSON
        json_str = extract_json(response)
        analysis = json.loads(json_str)

        # Flatten all skills into primary_skills for backward compat
        all_skills = []
        for key in [
            "input_skills",
            "reasoning_skills",
            "knowledge_skills",
            "output_skills",
            "composition_skills",
        ]:
            all_skills.extend(analysis.get(key, []))
        analysis["primary_skills"] = all_skills

        analysis["task_id"] = task.get("task_id", "unknown")
        analysis["sector"] = task.get("sector", "unknown")
        analysis["occupation"] = task.get("occupation", "unknown")
        return analysis

    except json.JSONDecodeError as e:
        return {
            "task_id": task.get("task_id", "unknown"),
            "error": f"JSON parse error: {e}",
            "raw_response": response[:300] if response else "",
        }
    except Exception as e:
        return {
            "task_id": task.get("task_id", "unknown"),
            "error": str(e),
        }


def load_existing_results(output_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing results and return dict keyed by task_id."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path) as f:
            data = json.load(f)
        existing = {r["task_id"]: r for r in data.get("tasks", [])}
        print(f"📂 Loaded {len(existing)} existing results from {output_path}")
        return existing
    except Exception as e:
        print(f"⚠️ Could not load existing results: {e}")
        return {}


def save_checkpoint(results: List[Dict[str, Any]], output_path: Path):
    """Save intermediate results."""
    output = {
        "checkpoint": True,
        "tasks": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def analyze_all_tasks(
    tasks: List[Dict[str, Any]],
    client: GeminiClient,
    output_path: Path,
    concurrency: int = 5,
    checkpoint_every: int = 5,
) -> List[Dict[str, Any]]:
    """Analyze tasks with checkpointing and resume support."""
    # Load existing results
    existing = load_existing_results(output_path)

    # Filter out already-analyzed tasks
    pending_tasks = [t for t in tasks if t.get("task_id") not in existing]
    results = list(existing.values())

    if not pending_tasks:
        print("✅ All tasks already analyzed!")
        return results

    total = len(tasks)
    already_done = len(existing)
    to_analyze = len(pending_tasks)

    print(
        f"\n📊 Total: {total} | Already done: {already_done} | To analyze: {to_analyze}"
    )
    print(f"🔄 Checkpointing every {checkpoint_every} tasks...")

    new_results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(analyze_task, client, task): task.get("task_id")
            for task in pending_tasks
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                new_results.append(result)
                results.append(result)

                # Progress indicator
                done = already_done + len(new_results)
                if "error" in result:
                    print(
                        f"  [{done}/{total}] ❌ {result['task_id'][:12]}: {result['error'][:40]}"
                    )
                else:
                    mtype = result.get("model_type", "?")
                    skills = ", ".join(result.get("primary_skills", [])[:2])
                    print(
                        f"  [{done}/{total}] ✓ {result['task_id'][:8]}... [{mtype}] {skills}"
                    )

                # Checkpoint every N results
                if len(new_results) % checkpoint_every == 0:
                    save_checkpoint(results, output_path)
                    print(f"  💾 Checkpoint saved ({len(results)} total)")

            except Exception as e:
                error_result = {"task_id": task_id, "error": str(e)}
                new_results.append(error_result)
                results.append(error_result)
                print(
                    f"  [{len(results)}/{total}] ❌ {str(task_id)[:12]}: {e}"
                )

    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate research-oriented summary statistics."""
    # Skill counts by category
    input_skills: Dict[str, int] = {}
    reasoning_skills: Dict[str, int] = {}
    knowledge_skills: Dict[str, int] = {}
    output_skills: Dict[str, int] = {}
    composition_skills: Dict[str, int] = {}

    # Meta counts
    model_type_counts: Dict[str, int] = {}
    bottleneck_counts: Dict[str, int] = {}
    training_signal_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    sector_counts: Dict[str, int] = {}
    errors = 0

    for r in results:
        if "error" in r:
            errors += 1
            continue

        # Count model types
        mtype = r.get("model_type", "unknown")
        model_type_counts[mtype] = model_type_counts.get(mtype, 0) + 1

        # Count by skill category
        for skill in r.get("input_skills", []):
            input_skills[skill] = input_skills.get(skill, 0) + 1
        for skill in r.get("reasoning_skills", []):
            reasoning_skills[skill] = reasoning_skills.get(skill, 0) + 1
        for skill in r.get("knowledge_skills", []):
            knowledge_skills[skill] = knowledge_skills.get(skill, 0) + 1
        for skill in r.get("output_skills", []):
            output_skills[skill] = output_skills.get(skill, 0) + 1
        for skill in r.get("composition_skills", []):
            composition_skills[skill] = composition_skills.get(skill, 0) + 1

        # Count bottlenecks and training signals
        bottleneck = r.get("primary_bottleneck", "unknown")
        bottleneck_counts[bottleneck] = (
            bottleneck_counts.get(bottleneck, 0) + 1
        )

        signal = r.get("training_signal", "unknown")
        training_signal_counts[signal] = (
            training_signal_counts.get(signal, 0) + 1
        )

        # Count difficulty and sector
        diff = r.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        sector = r.get("sector", "unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    def sort_dict(d):
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    return {
        "total_tasks": len(results),
        "successful": len(results) - errors,
        "errors": errors,
        # Model requirements
        "model_type_distribution": sort_dict(model_type_counts),
        # Research-relevant breakdowns
        "skills_by_category": {
            "input_processing": sort_dict(input_skills),
            "reasoning": sort_dict(reasoning_skills),
            "knowledge": sort_dict(knowledge_skills),
            "output_generation": sort_dict(output_skills),
            "composition": sort_dict(composition_skills),
        },
        # Training insights
        "primary_bottlenecks": sort_dict(bottleneck_counts),
        "recommended_training_signals": sort_dict(training_signal_counts),
        # Distribution stats
        "difficulty_distribution": difficulty_counts,
        "sector_distribution": sort_dict(sector_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GDPVal skills")
    parser.add_argument(
        "--input", default="gdp_eval_train.json", help="Input JSON file"
    )
    parser.add_argument(
        "--output", default="gdp_skills_analysis.json", help="Output file"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of tasks"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Parallel requests"
    )
    args = parser.parse_args()

    # Load tasks
    input_path = Path(args.input)
    print(f"Loading tasks from {input_path}...")

    with open(input_path) as f:
        tasks = json.load(f)

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"Limited to {args.limit} tasks")

    print(f"Loaded {len(tasks)} tasks")

    # Initialize client
    client = GeminiClient()
    output_path = Path(args.output)

    # Analyze tasks (with resume support)
    start_time = time.time()
    results = analyze_all_tasks(
        tasks,
        client,
        output_path=output_path,
        concurrency=args.concurrency,
        checkpoint_every=5,
    )
    elapsed = time.time() - start_time

    # Generate summary
    summary = summarize_results(results)
    summary["elapsed_seconds"] = round(elapsed, 2)

    # Save results
    # Save final results
    output = {
        "summary": summary,
        "tasks": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESEARCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {summary['successful']}/{summary['total_tasks']} analyzed")
    print(f"Time: {summary['elapsed_seconds']}s")

    print("\n🤖 MODEL TYPE REQUIRED:")
    for t, c in summary["model_type_distribution"].items():
        pct = round(100 * c / max(summary["successful"], 1))
        print(f"  {t}: {c} ({pct}%)")

    print("\n📥 INPUT SKILLS:")
    for s, c in list(
        summary["skills_by_category"]["input_processing"].items()
    )[:6]:
        print(f"  {s}: {c}")

    print("\n🧠 REASONING SKILLS:")
    for s, c in list(summary["skills_by_category"]["reasoning"].items())[:6]:
        print(f"  {s}: {c}")

    print("\n📚 KNOWLEDGE SKILLS:")
    for s, c in list(summary["skills_by_category"]["knowledge"].items())[:5]:
        print(f"  {s}: {c}")

    print("\n📤 OUTPUT SKILLS:")
    for s, c in list(
        summary["skills_by_category"]["output_generation"].items()
    )[:6]:
        print(f"  {s}: {c}")

    if summary["skills_by_category"]["composition"]:
        print("\n🔗 COMPOSITION SKILLS:")
        for s, c in list(summary["skills_by_category"]["composition"].items())[
            :5
        ]:
            print(f"  {s}: {c}")

    print("\n🎯 PRIMARY BOTTLENECKS:")
    for b, c in list(summary["primary_bottlenecks"].items())[:5]:
        print(f"  {b}: {c}")

    print("\n🔧 TRAINING SIGNALS:")
    for t, c in summary["recommended_training_signals"].items():
        print(f"  {t}: {c}")

    print(f"\n📊 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
