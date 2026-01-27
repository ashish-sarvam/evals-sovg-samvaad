# Copyright Sierra
"""
LLM-based failure analysis for tau-bench results.

Uses GPT-5-chat (Azure) to analyze failures with concrete labels by comparing
the model's trajectory against the ground truth actions.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Azure OpenAI
try:
    from openai import AzureOpenAI, AsyncAzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


# Failure labels with descriptions
FAILURE_LABELS = {
    "wrong_tool": "Agent called a tool that shouldn't have been called for this task",
    "missing_tool_call": "Agent did not call a required tool",
    "wrong_input": "Agent called the correct tool but with wrong argument values",
    "missing_input": "Agent called the correct tool but with missing required arguments",
    "extra_action": "Agent performed more actions than required (e.g., exchanged 2 items instead of 1)",
    "incomplete_action": "Agent started the task but didn't complete all required steps",
    "user_simulator_error": "The simulated user provided incorrect information causing the failure",
    "tool_error_not_recovered": "Agent got a tool error but failed to recover properly",
    "wrong_sequence": "Agent performed actions in the wrong order",
    "logic_error": "Agent made a logical reasoning error (e.g., misinterpreted conditions)",
    "other": "Other failure reason not covered by above categories",
}


@dataclass
class FailureAnalysisResult:
    """Result of LLM-based failure analysis."""
    label: str
    description: str
    failing_turn_index: int
    failing_turn_summary: str
    detailed_explanation: str
    ground_truth_summary: str
    actual_actions_summary: str


def extract_tool_calls_summary(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool calls from trajectory with their results."""
    tool_calls = []
    current_call = None
    
    for i, msg in enumerate(traj):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function"):
                    current_call = {
                        "turn": i,
                        "name": tc["function"].get("name"),
                        "arguments": tc["function"].get("arguments"),
                        "result": None,
                    }
        elif msg.get("role") == "tool" and current_call:
            current_call["result"] = msg.get("content", "")[:500]  # Truncate long results
            tool_calls.append(current_call)
            current_call = None
    
    return tool_calls


def format_trajectory_for_llm(traj: List[Dict[str, Any]], max_turns: int = 50) -> str:
    """Format trajectory for LLM analysis."""
    if not traj:
        return "(empty trajectory)"
    
    lines = []
    turn_count = 0
    
    for i, msg in enumerate(traj):
        if msg.get("role") == "system":
            continue
        
        turn_count += 1
        if turn_count > max_turns:
            lines.append(f"... (truncated, {len(traj) - i} more messages)")
            break
        
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                lines.append(f"Turn {i} [{role}]: TOOL_CALL {func.get('name')}({args})")
        elif content:
            # Truncate long content
            content_preview = content[:300].replace("\n", " ")
            if len(content) > 300:
                content_preview += "..."
            lines.append(f"Turn {i} [{role}]: {content_preview}")
    
    return "\n".join(lines)


def format_ground_truth(actions: List[Dict[str, Any]]) -> str:
    """Format ground truth actions for LLM."""
    if not actions:
        return "(no ground truth actions)"
    
    lines = []
    for i, action in enumerate(actions, 1):
        name = action.get("name", "unknown")
        kwargs = json.dumps(action.get("kwargs", {}))
        lines.append(f"{i}. {name}({kwargs})")
    
    return "\n".join(lines)


ANALYSIS_PROMPT = """You are an expert at analyzing AI agent failures in customer service tasks.

## Task Information
**Instruction given to the user (who the agent is helping):**
{instruction}

**Ground Truth Actions (what the agent SHOULD have done):**
{ground_truth}

## Agent's Actual Trajectory
{trajectory}

## Your Task
Analyze why the agent failed this task. Compare what the agent actually did vs what it should have done (ground truth).

## Failure Labels (choose ONE)
- `wrong_tool`: Agent called a tool that shouldn't have been called for this task
- `missing_tool_call`: Agent did not call a required tool  
- `wrong_input`: Agent called the correct tool but with wrong argument values
- `missing_input`: Agent called the correct tool but with missing required arguments
- `extra_action`: Agent performed more actions than required (e.g., exchanged 2 items when only 1 should be exchanged)
- `incomplete_action`: Agent started the task but didn't complete all required steps
- `user_simulator_error`: The simulated user provided incorrect information causing the failure
- `tool_error_not_recovered`: Agent got a tool error but failed to recover properly
- `wrong_sequence`: Agent performed actions in the wrong order
- `logic_error`: Agent made a logical reasoning error (e.g., misinterpreted task conditions)
- `other`: Other failure reason not covered by above categories

## Response Format (JSON)
{{
    "label": "<one of the labels above>",
    "description": "<1-2 sentence description of the specific failure>",
    "failing_turn_index": <turn number where the critical failure occurred, or -1 if unclear>,
    "failing_turn_summary": "<what happened at the failing turn>",
    "detailed_explanation": "<detailed 2-3 sentence explanation comparing expected vs actual behavior>",
    "ground_truth_summary": "<brief summary of what should have happened>",
    "actual_actions_summary": "<brief summary of what actually happened>"
}}

Respond with ONLY valid JSON, no markdown code blocks or other text."""


def _get_azure_client() -> "AzureOpenAI":
    """Create Azure OpenAI client from environment variables."""
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI SDK not available. Install with: pip install openai")
    
    endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_SUBSCRIPTION_KEY") or os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_ENDPOINT or AZURE_API_BASE environment variable required")
    if not api_key:
        raise ValueError("AZURE_SUBSCRIPTION_KEY or AZURE_API_KEY environment variable required")
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _get_async_azure_client() -> "AsyncAzureOpenAI":
    """Create async Azure OpenAI client from environment variables."""
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI SDK not available. Install with: pip install openai")
    
    endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_SUBSCRIPTION_KEY") or os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_ENDPOINT or AZURE_API_BASE environment variable required")
    if not api_key:
        raise ValueError("AZURE_SUBSCRIPTION_KEY or AZURE_API_KEY environment variable required")
    
    return AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


async def analyze_failure_async(
    traj: List[Dict[str, Any]],
    task_info: Dict[str, Any],
    model: str = "gpt-5-chat",
    api_base: Optional[str] = None,
) -> FailureAnalysisResult:
    """
    Analyze a failure using Azure OpenAI (async version).
    
    Args:
        traj: The conversation trajectory
        task_info: Task information including ground truth actions
        model: Azure deployment name (default: gpt-5-chat)
        api_base: API base URL (optional, uses env var if not provided)
    
    Returns:
        FailureAnalysisResult with detailed failure analysis
    """
    instruction = task_info.get("task", {}).get("instruction", "No instruction provided")
    gt_actions = task_info.get("task", {}).get("actions", [])
    
    trajectory_str = format_trajectory_for_llm(traj)
    ground_truth_str = format_ground_truth(gt_actions)
    
    prompt = ANALYSIS_PROMPT.format(
        instruction=instruction,
        ground_truth=ground_truth_str,
        trajectory=trajectory_str,
    )
    
    # Clean up model name if it has azure/ prefix
    if model.startswith("azure/"):
        model = model[6:]
    
    try:
        client = _get_async_azure_client()
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        
        return FailureAnalysisResult(
            label=result.get("label", "other"),
            description=result.get("description", "Analysis failed"),
            failing_turn_index=result.get("failing_turn_index", -1),
            failing_turn_summary=result.get("failing_turn_summary", ""),
            detailed_explanation=result.get("detailed_explanation", ""),
            ground_truth_summary=result.get("ground_truth_summary", ""),
            actual_actions_summary=result.get("actual_actions_summary", ""),
        )
    
    except json.JSONDecodeError as e:
        return FailureAnalysisResult(
            label="other",
            description=f"Failed to parse LLM response: {str(e)}",
            failing_turn_index=-1,
            failing_turn_summary="",
            detailed_explanation=f"Raw response: {content[:500] if 'content' in dir() else 'N/A'}",
            ground_truth_summary=ground_truth_str,
            actual_actions_summary=trajectory_str[:500],
        )
    except Exception as e:
        return FailureAnalysisResult(
            label="other",
            description=f"LLM analysis failed: {str(e)}",
            failing_turn_index=-1,
            failing_turn_summary="",
            detailed_explanation=str(e),
            ground_truth_summary=ground_truth_str,
            actual_actions_summary=trajectory_str[:500],
        )


def analyze_failure(
    traj: List[Dict[str, Any]],
    task_info: Dict[str, Any],
    model: str = "gpt-5-chat",
    api_base: Optional[str] = None,
) -> FailureAnalysisResult:
    """
    Analyze a failure using Azure OpenAI (sync version).
    
    Args:
        traj: The conversation trajectory
        task_info: Task information including ground truth actions
        model: Azure deployment name (default: gpt-5-chat)
        api_base: API base URL (optional)
    
    Returns:
        FailureAnalysisResult with detailed failure analysis
    """
    instruction = task_info.get("task", {}).get("instruction", "No instruction provided")
    gt_actions = task_info.get("task", {}).get("actions", [])
    
    trajectory_str = format_trajectory_for_llm(traj)
    ground_truth_str = format_ground_truth(gt_actions)
    
    prompt = ANALYSIS_PROMPT.format(
        instruction=instruction,
        ground_truth=ground_truth_str,
        trajectory=trajectory_str,
    )
    
    # Clean up model name if it has azure/ prefix
    if model.startswith("azure/"):
        model = model[6:]
    
    try:
        client = _get_azure_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        
        return FailureAnalysisResult(
            label=result.get("label", "other"),
            description=result.get("description", "Analysis failed"),
            failing_turn_index=result.get("failing_turn_index", -1),
            failing_turn_summary=result.get("failing_turn_summary", ""),
            detailed_explanation=result.get("detailed_explanation", ""),
            ground_truth_summary=result.get("ground_truth_summary", ""),
            actual_actions_summary=result.get("actual_actions_summary", ""),
        )
    
    except json.JSONDecodeError as e:
        return FailureAnalysisResult(
            label="other",
            description=f"Failed to parse LLM response: {str(e)}",
            failing_turn_index=-1,
            failing_turn_summary="",
            detailed_explanation=f"Raw response: {content[:500] if 'content' in dir() else 'N/A'}",
            ground_truth_summary=ground_truth_str,
            actual_actions_summary=trajectory_str[:500],
        )
    except Exception as e:
        return FailureAnalysisResult(
            label="other",
            description=f"LLM analysis failed: {str(e)}",
            failing_turn_index=-1,
            failing_turn_summary="",
            detailed_explanation=str(e),
            ground_truth_summary=ground_truth_str,
            actual_actions_summary=trajectory_str[:500],
        )


def analyze_failures_batch(
    failures: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    model: str = "gpt-5-chat",
    api_base: Optional[str] = None,
    max_concurrency: int = 5,
) -> List[FailureAnalysisResult]:
    """
    Analyze multiple failures in batch.
    
    Args:
        failures: List of (trajectory, task_info) tuples
        model: LLM model to use
        api_base: API base URL
        max_concurrency: Maximum concurrent requests
    
    Returns:
        List of FailureAnalysisResult
    """
    import asyncio
    
    async def _batch_analyze():
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _analyze_with_semaphore(traj, task_info):
            async with semaphore:
                return await analyze_failure_async(traj, task_info, model, api_base)
        
        tasks = [_analyze_with_semaphore(traj, info) for traj, info in failures]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(_batch_analyze())


# Quick fallback analysis (no LLM) for when LLM is not available
def analyze_failure_heuristic(
    traj: List[Dict[str, Any]],
    task_info: Dict[str, Any],
) -> FailureAnalysisResult:
    """
    Simple heuristic-based failure analysis (fallback when LLM not available).
    """
    if not traj:
        return FailureAnalysisResult(
            label="other",
            description="Empty trajectory",
            failing_turn_index=-1,
            failing_turn_summary="No trajectory recorded",
            detailed_explanation="The agent produced no conversation.",
            ground_truth_summary="",
            actual_actions_summary="",
        )
    
    # Extract actual tool calls
    actual_calls = extract_tool_calls_summary(traj)
    actual_tool_names = [c["name"] for c in actual_calls]
    
    # Get ground truth
    gt_actions = task_info.get("task", {}).get("actions", [])
    gt_tool_names = [a["name"] for a in gt_actions]
    
    # Check for missing tools
    missing = set(gt_tool_names) - set(actual_tool_names)
    if missing:
        return FailureAnalysisResult(
            label="missing_tool_call",
            description=f"Missing required tool calls: {missing}",
            failing_turn_index=-1,
            failing_turn_summary="Agent did not call required tools",
            detailed_explanation=f"Expected tools {gt_tool_names}, but agent only called {actual_tool_names}",
            ground_truth_summary=format_ground_truth(gt_actions),
            actual_actions_summary=str(actual_tool_names),
        )
    
    # Check for extra tools in final action
    if gt_actions:
        final_gt = gt_actions[-1]
        # Find matching actual call
        for call in reversed(actual_calls):
            if call["name"] == final_gt["name"]:
                try:
                    actual_args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
                    gt_args = final_gt.get("kwargs", {})
                    
                    # Check for extra items (common failure)
                    if "item_ids" in gt_args and "item_ids" in actual_args:
                        if len(actual_args["item_ids"]) > len(gt_args["item_ids"]):
                            return FailureAnalysisResult(
                                label="extra_action",
                                description=f"Agent exchanged/modified {len(actual_args['item_ids'])} items instead of {len(gt_args['item_ids'])}",
                                failing_turn_index=call["turn"],
                                failing_turn_summary=f"Called {call['name']} with extra items",
                                detailed_explanation=f"Expected item_ids={gt_args['item_ids']}, got {actual_args['item_ids']}",
                                ground_truth_summary=format_ground_truth(gt_actions),
                                actual_actions_summary=str(actual_args),
                            )
                    
                    # Check for wrong arguments
                    if actual_args != gt_args:
                        return FailureAnalysisResult(
                            label="wrong_input",
                            description=f"Wrong arguments for {call['name']}",
                            failing_turn_index=call["turn"],
                            failing_turn_summary=f"Called {call['name']} with wrong arguments",
                            detailed_explanation=f"Expected {gt_args}, got {actual_args}",
                            ground_truth_summary=format_ground_truth(gt_actions),
                            actual_actions_summary=str(actual_args),
                        )
                except Exception:
                    pass
    
    return FailureAnalysisResult(
        label="other",
        description="Could not determine specific failure reason",
        failing_turn_index=-1,
        failing_turn_summary="",
        detailed_explanation="Heuristic analysis could not identify the failure pattern",
        ground_truth_summary=format_ground_truth(gt_actions),
        actual_actions_summary=str(actual_tool_names),
    )

