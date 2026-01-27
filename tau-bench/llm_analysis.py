#!/usr/bin/env python3
"""
Simple LLM Analysis Script for comparing tau-bench results.
Uses Azure OpenAI or Azure Anthropic to analyze why one model failed when another passed.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import AzureOpenAI

# Try to import Anthropic
try:
    from anthropic import AnthropicFoundry

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AnthropicFoundry = None
    ANTHROPIC_AVAILABLE = False

# Load environment variables
load_dotenv()

# Map alternate Azure env var names
if os.getenv("AZURE_SUBSCRIPTION_KEY") and not os.getenv("AZURE_API_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_SUBSCRIPTION_KEY")
if os.getenv("AZURE_ENDPOINT") and not os.getenv("AZURE_API_BASE"):
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE_ENDPOINT")


def get_azure_openai_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
    )


def get_azure_anthropic_client():
    """Initialize Azure Anthropic client."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "Anthropic not available. Install: pip install anthropic"
        )

    endpoint = os.getenv(
        "ANTHROPIC_AZURE_ENDPOINT",
        "https://ashish-alignment-resource.services.ai.azure.com/anthropic/",
    )
    api_key = os.getenv("AZURE_SUBSCRIPTION_KEY")

    if not api_key:
        raise ValueError(
            "AZURE_SUBSCRIPTION_KEY environment variable is required."
        )

    return AnthropicFoundry(
        api_key=api_key,
        base_url=endpoint,
    )


def test_llm_connection_openai(
    client: AzureOpenAI, model: str = "gpt-5-chat"
) -> bool:
    """Test Azure OpenAI connection."""
    print(f"Testing connection to Azure OpenAI ({model})...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'Connection successful!' in exactly those words.",
                },
            ],
            max_tokens=50,
        )
        result = response.choices[0].message.content
        print(f"✅ LLM Response: {result}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_llm_connection_anthropic(
    client, model: str = "claude-opus-4-5"
) -> bool:
    """Test Azure Anthropic connection."""
    print(f"Testing connection to Azure Anthropic ({model})...")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Connection successful!' in exactly those words.",
                },
            ],
            system="You are a helpful assistant.",
            thinking={
                "type": "enabled",
                "budget_tokens": 1024,
            },
        )
        # Extract text from response
        result = ""
        for block in response.content:
            if hasattr(block, "text"):
                result = block.text
                break
        print(f"✅ LLM Response: {result}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def load_results(file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load results file and index by task_id."""
    with open(file_path, "r") as f:
        results = json.load(f)
    return {r["task_id"]: r for r in results}


FAILURE_CATEGORIES = [
    "tool_response_error",
    "user_instruction_error",
    "tool_usage_error",
    "user_simulator_issue",
    "other",
]


def extract_failure_category(analysis: str) -> str:
    """Extract the failure category from the analysis text."""
    analysis_lower = analysis.lower()
    for cat in FAILURE_CATEGORIES:
        if cat in analysis_lower:
            return cat
    return "other"


def format_trajectory(traj: List[Dict[str, Any]], max_turns: int = 50) -> str:
    """Format trajectory for LLM consumption with FULL tool responses."""
    formatted = []
    for i, msg in enumerate(traj[:max_turns]):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "SYSTEM":
            continue

        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                formatted.append(
                    f"[Turn {i}] [{role}] TOOL_CALL: {func.get('name')}({func.get('arguments', '{}')})"
                )
        elif role == "TOOL":
            # Include FULL tool response - this is critical for analysis
            formatted.append(f"[Turn {i}] [TOOL_RESPONSE]:\n{content}")
        elif content:
            # Truncate assistant/user messages if very long
            if len(content) > 800:
                content = content[:800] + "..."
            formatted.append(f"[Turn {i}] [{role}] {content}")

    if len(traj) > max_turns:
        formatted.append(f"... ({len(traj) - max_turns} more turns truncated)")

    return "\n".join(formatted)


def analyze_failure(
    client: Any,
    task_id: int,
    task_info: Dict[str, Any],
    main_traj: List[Dict[str, Any]],
    sub_traj: List[Dict[str, Any]],
    main_reward_info: Dict[str, Any],
    sub_reward_info: Dict[str, Any],
    model: str = "gpt-5-chat",
    provider: str = "openai",
) -> str:
    """Analyze why the sub model failed when the main model passed."""

    main_formatted = format_trajectory(main_traj)
    sub_formatted = format_trajectory(sub_traj)

    # Extract task details
    instruction = task_info.get("instruction", "N/A")
    ground_truth_actions = task_info.get("actions", [])
    expected_outputs = task_info.get("outputs", [])

    gt_actions_summary = "\n".join(
        [
            f"{i + 1}. {a['name']}({json.dumps(a.get('kwargs', {}))})"
            for i, a in enumerate(ground_truth_actions)
        ]
    )

    # Extract actual actions taken
    main_actions = main_reward_info.get("actions", [])
    sub_actions = sub_reward_info.get("actions", [])

    main_actions_summary = "\n".join(
        [
            f"{i + 1}. {a['name']}({json.dumps(a.get('kwargs', {}))})"
            for i, a in enumerate(main_actions)
        ]
    )

    sub_actions_summary = "\n".join(
        [
            f"{i + 1}. {a['name']}({json.dumps(a.get('kwargs', {}))})"
            for i, a in enumerate(sub_actions)
        ]
    )

    prompt = f"""You are an expert at analyzing AI agent behavior in a retail customer service environment.

## Task ID: {task_id}

Compare two agent trajectories for the same task. The MAIN model succeeded (reward=1.0), but the SUB model failed (reward=0.0).
Your job is to find the EXACT reason why SUB failed.

---

## COMPLETE TASK INFORMATION

### User Instruction (what the simulated user wants):
{instruction}

### Expected Outputs (if any):
{json.dumps(expected_outputs) if expected_outputs else "None"}

### Ground Truth Actions (correct sequence):
{gt_actions_summary}

---

## MAIN MODEL (PASSED - reward=1.0)

### Actions Actually Taken by MAIN:
{main_actions_summary}

### Full MAIN Trajectory:
{main_formatted}

---

## SUB MODEL (FAILED - reward=0.0)

### Actions Actually Taken by SUB:
{sub_actions_summary}

### Full SUB Trajectory:
{sub_formatted}

---

## ANALYSIS REQUIREMENTS

CRITICAL: Read the TOOL_RESPONSE messages carefully - they contain the actual data.
- DO NOT assume an item_id is "fabricated" - verify it exists or not in the tool response
- Compare ATTRIBUTES of items, not just IDs
- Look at what the tool returned vs what each model selected

Your analysis must answer:

1. **What did the user actually want?** 
   - Parse the instruction carefully
   - Note any specific attribute requirements (size, color, brightness, etc.)

2. **What options were available?** 
   - From the TOOL_RESPONSE, list the relevant options with their exact attributes

3. **What did SUB choose vs MAIN?**
   - Show a comparison table if applicable

4. **Why is SUB's choice wrong?**
   - Be SPECIFIC: which attribute value is wrong?
   - Quote exact values from tool responses

5. **ROOT CAUSE CATEGORY** (REQUIRED - pick exactly ONE):

   CRITICAL: Identify the ROOT CAUSE, not the symptom/outcome.
   
   **Decision tree (follow in order):**
   
   Q1: Did the agent make an error related to TOOL RESPONSE data in understanding or reasoning the data?
       - Agent said "item/option doesn't exist" but it DOES exist in tool response → **tool_response_error**
       - Agent picked item with WRONG attributes from available options → **tool_response_error**
       - Agent miscounted/misread values from tool response → **tool_response_error**
       - Agent selected wrong item despite correct one being available → **tool_response_error**
       - If none of above, continue to Q2
   
   Q2: Did the agent misunderstand WHAT the user wanted?
       - Agent didn't understand "all" meant multiple items → **user_instruction_error**
       - Agent misunderstood attribute requirements → **user_instruction_error**
       - Agent missed conditional logic in user's request → **user_instruction_error**
       - If none of above, continue to Q3
   
   Q3: Did the agent call wrong tool or pass wrong arguments?
       - Wrong order_id, payment_method_id, etc. → **tool_usage_error**
       - Called wrong tool entirely → **tool_usage_error**
       - If none of above, continue to Q4
   
   Q4: Did the user simulator behave unexpectedly?
       - User gave confusing/contradictory responses → **user_simulator_issue**
       - If none of above → **other**

   **Category definitions:**

   - **tool_response_error**: Agent failed to correctly process/use tool response data
     - Said "option X doesn't exist" when it DOES exist
     - Picked item with wrong attributes (e.g., medium brightness instead of low)
     - Miscounted items, misread prices
     - Selected wrong item from available options
     - Failed to find/match the correct option even though it was there
     - Failed to reason based on the tool response
   
   - **user_instruction_error**: Agent misunderstood what the user wanted
     - Didn't understand "all" meant multiple items
     - Misinterpreted attribute requirements
     - Missed conditional logic in request
   
   - **tool_usage_error**: Agent called wrong tool or passed wrong arguments
     - Wrong IDs (order_id, payment_method_id, item_id)
     - Called wrong tool
   
   - **user_simulator_issue**: User simulator behaved unexpectedly
   
   - **other**: None of the above

6. **One-line Summary**: "[CATEGORY]: [specific error with quoted evidence]"
   
   Examples:
   - "tool_response_error: Agent said 'no purple S v-neck polyester exists' but item_id 9647292434 with those exact attributes was in tool response"
   - "user_instruction_error: Agent thought 'modify all t-shirts' meant only matching sizes, not changing size to S as requested"

Be forensic. Quote exact values."""

    system_msg = "You are a forensic AI behavior analyst. You find exact specific errors by comparing attribute values. Always quote actual values from tool responses."

    try:
        if provider == "anthropic":
            # Azure Anthropic API with thinking enabled
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                system=system_msg,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            )
            # Extract text from Anthropic response (skip thinking blocks)
            result = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    if hasattr(block, "text"):
                        result += block.text
            return result
        else:
            # Azure OpenAI API (default)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=2500,
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing: {e}"


def find_sub_only_failures(
    main_results: Dict[int, Dict[str, Any]],
    sub_results: Dict[int, Dict[str, Any]],
) -> List[int]:
    """Find tasks where main passed but sub failed."""
    failures = []
    common_tasks = set(main_results.keys()) & set(sub_results.keys())

    for task_id in sorted(common_tasks):
        main_reward = main_results[task_id].get("reward", 0)
        sub_reward = sub_results[task_id].get("reward", 0)

        if main_reward >= 1.0 and sub_reward < 1.0:
            failures.append(task_id)

    return failures


def analyze_task_wrapper(args: Tuple) -> Tuple[int, str]:
    """Wrapper for parallel execution."""
    client, task_id, main_task, sub_task, model, provider = args

    task_info = sub_task["info"]["task"]
    main_reward_info = main_task["info"].get("reward_info", {})
    sub_reward_info = sub_task["info"].get("reward_info", {})

    analysis = analyze_failure(
        client=client,
        task_id=task_id,
        task_info=task_info,
        main_traj=main_task.get("traj", []),
        sub_traj=sub_task.get("traj", []),
        main_reward_info=main_reward_info,
        sub_reward_info=sub_reward_info,
        model=model,
        provider=provider,
    )

    return task_id, analysis


def analyze_failures_parallel(
    client: Any,
    main_results: Dict[int, Dict[str, Any]],
    sub_results: Dict[int, Dict[str, Any]],
    task_ids: List[int],
    model: str = "gpt-5-chat",
    concurrency: int = 5,
    provider: str = "openai",
) -> Dict[int, str]:
    """Analyze multiple failures in parallel."""
    results = {}

    # Prepare arguments for each task
    tasks_args = []
    for task_id in task_ids:
        main_task = main_results[task_id]
        sub_task = sub_results[task_id]
        tasks_args.append(
            (client, task_id, main_task, sub_task, model, provider)
        )

    print(
        f"\nAnalyzing {len(task_ids)} tasks with concurrency={concurrency}..."
    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(analyze_task_wrapper, args): args[1]
            for args in tasks_args
        }

        for i, future in enumerate(as_completed(futures)):
            task_id = futures[future]
            try:
                tid, analysis = future.result()
                results[tid] = analysis
                print(f"  ✅ Task {tid} completed ({i + 1}/{len(task_ids)})")
            except Exception as e:
                results[task_id] = f"Error: {e}"
                print(f"  ❌ Task {task_id} failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based failure analysis for tau-bench results"
    )
    parser.add_argument(
        "--main", type=str, help="Path to main results file (the better model)"
    )
    parser.add_argument(
        "--sub",
        type=str,
        help="Path to sub results file (the model to analyze failures)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-chat",
        help="Model to use (default: gpt-5-chat for openai, claude-sonnet-4-20250514 for anthropic)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider: 'openai' (Azure OpenAI) or 'anthropic' (Azure Anthropic)",
    )
    parser.add_argument(
        "--task-id", type=int, default=None, help="Analyze a specific task ID"
    )
    parser.add_argument(
        "--test", action="store_true", help="Just test LLM connection"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Max number of tasks to analyze"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of parallel LLM calls",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Set default model based on provider if not specified
    if args.model == "gpt-5-chat" and args.provider == "anthropic":
        args.model = "claude-opus-4-5"

    # Initialize client based on provider
    if args.provider == "anthropic":
        print(f"Using Azure Anthropic provider with model: {args.model}")
        client = get_azure_anthropic_client()
    else:
        print(f"Using Azure OpenAI provider with model: {args.model}")
        client = get_azure_openai_client()

    # Test mode
    if args.test:
        if args.provider == "anthropic":
            success = test_llm_connection_anthropic(client, args.model)
        else:
            success = test_llm_connection_openai(client, args.model)
        return 0 if success else 1

    # Need both files for analysis
    if not args.main or not args.sub:
        print("Error: --main and --sub are required for analysis")
        print("Use --test to just test LLM connection")
        return 1

    # Load results
    print(f"Loading main results: {args.main}")
    main_results = load_results(args.main)
    print(f"  Found {len(main_results)} tasks")

    print(f"Loading sub results: {args.sub}")
    sub_results = load_results(args.sub)
    print(f"  Found {len(sub_results)} tasks")

    # Find failures
    failures = find_sub_only_failures(main_results, sub_results)
    print(f"\nFound {len(failures)} tasks where main passed but sub failed")
    print(f"Task IDs: {failures}")

    # Single task analysis (sequential)
    if args.task_id is not None:
        if args.task_id not in failures:
            print(f"\nTask {args.task_id} is not in the failure list.")
            if args.task_id in main_results and args.task_id in sub_results:
                main_r = main_results[args.task_id].get("reward", 0)
                sub_r = sub_results[args.task_id].get("reward", 0)
                print(f"  Main reward: {main_r}, Sub reward: {sub_r}")
            return 1

        # Single task - run sequentially
        main_task = main_results[args.task_id]
        sub_task = sub_results[args.task_id]
        task_info = sub_task["info"]["task"]
        main_reward_info = main_task["info"].get("reward_info", {})
        sub_reward_info = sub_task["info"].get("reward_info", {})

        print(f"\n{'=' * 80}")
        print(f"Task {args.task_id}")
        print(f"{'=' * 80}")
        print(f"Instruction: {task_info.get('instruction', 'N/A')}")
        print(f"\nAnalyzing with {args.model}...")

        analysis = analyze_failure(
            client=client,
            task_id=args.task_id,
            task_info=task_info,
            main_traj=main_task.get("traj", []),
            sub_traj=sub_task.get("traj", []),
            main_reward_info=main_reward_info,
            sub_reward_info=sub_reward_info,
            model=args.model,
            provider=args.provider,
        )

        print(f"\n{analysis}")
        return 0

    # Multiple tasks - run in parallel
    tasks_to_analyze = failures[: args.limit]

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS (PARALLEL)")
    print("=" * 80)

    results = analyze_failures_parallel(
        client=client,
        main_results=main_results,
        sub_results=sub_results,
        task_ids=tasks_to_analyze,
        model=args.model,
        concurrency=args.concurrency,
        provider=args.provider,
    )

    # Print results
    for task_id in tasks_to_analyze:
        sub_task = sub_results[task_id]
        instruction = sub_task["info"]["task"].get("instruction", "N/A")

        print(f"\n{'=' * 80}")
        print(f"Task {task_id}")
        print(f"{'=' * 80}")
        print(f"Instruction: {instruction[:200]}...")
        print(f"\n{results.get(task_id, 'No analysis available')}")

    # Save to file if requested
    if args.output:
        # Extract categories and build analyses
        analyses = []
        category_counts = {cat: 0 for cat in FAILURE_CATEGORIES}

        for tid in tasks_to_analyze:
            analysis_text = results.get(tid, "")
            category = extract_failure_category(analysis_text)
            category_counts[category] += 1

            analyses.append(
                {
                    "task_id": tid,
                    "instruction": sub_results[tid]["info"]["task"].get(
                        "instruction", ""
                    ),
                    "main_reward": main_results[tid].get("reward", 0),
                    "sub_reward": sub_results[tid].get("reward", 0),
                    "failure_category": category,
                    "analysis": analysis_text,
                }
            )

        output_data = {
            "main_file": args.main,
            "sub_file": args.sub,
            "model": args.model,
            "summary": {
                "total_failures_analyzed": len(tasks_to_analyze),
                "category_distribution": {
                    k: v for k, v in category_counts.items() if v > 0
                },
            },
            "category_definitions": {
                "tool_response_error": "Agent failed to correctly process/use tool response data (e.g., said option doesn't exist when it does, picked wrong item, miscounted)",
                "user_instruction_error": "Agent misunderstood what the user wanted (e.g., missed 'all', misunderstood attribute requirements)",
                "tool_usage_error": "Agent called wrong tool or passed wrong arguments (wrong IDs)",
                "user_simulator_issue": "User simulator behaved unexpectedly",
                "other": "None of the above categories apply",
            },
            "analyses": analyses,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

        # Print category summary
        print("\n📊 FAILURE CATEGORY SUMMARY:")
        print("-" * 40)
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / len(tasks_to_analyze) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")

    if len(failures) > args.limit:
        print(
            f"\n... {len(failures) - args.limit} more failures not shown (use --limit to increase)"
        )

    return 0


if __name__ == "__main__":
    exit(main())
