# Copyright Sierra
"""
Failure type analysis for tau-bench results.

Based on the τ-bench paper (https://arxiv.org/pdf/2406.12045), failures are categorized by:

Fault Author:
- USER: User provided incorrect/hallucinated information not in the instruction
- AGENT: Agent took wrong action or used wrong arguments
- ENVIRONMENT: Other issues (environment bugs, etc.)

Fault Types (for agent failures):
- CALLED_WRONG_TOOL: Agent called an incorrect tool
- USED_WRONG_TOOL_ARGUMENT: Agent used correct tool but with wrong arguments
- GOAL_PARTIALLY_COMPLETED: Agent completed the task partially
- OTHER: Other types of failures
"""

import json
from typing import List, Dict, Any, Tuple


# Legend explaining each failure type
FAILURE_TYPE_LEGEND = {
    "authentication_failure": "Failed to authenticate user - wrong credentials used by simulated user",
    "no_trajectory": "Empty trajectory - no conversation recorded",
    "no_tool_calls": "Agent made no tool calls",
    "missing_tool_calls": "Agent did not call required tools",
    "wrong_arguments": "Agent used correct tool but with wrong arguments",
    "goal_partially_completed": "Task was only partially completed",
}


def extract_tool_calls(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all tool calls from a trajectory."""
    tool_calls = []
    for msg in traj:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function"):
                    tool_calls.append(
                        {
                            "name": tc["function"].get("name"),
                            "arguments": tc["function"].get("arguments"),
                        }
                    )
    return tool_calls


def find_failing_turn(
    traj: List[Dict[str, Any]],
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Find the turn where the failure occurred.

    Returns:
        (turn_index, failure_reason, failing_message)
    """
    if not traj:
        return -1, "Empty trajectory", {}

    for i, msg in enumerate(traj):
        # Check for tool errors
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            if "Error:" in content or "error" in content.lower():
                return i, f"Tool returned error: {content[:100]}", msg

        # Check for repeated messages (stuck in loop)
        if i > 2 and msg.get("role") == "assistant":
            prev_contents = [
                traj[j].get("content", "")
                for j in range(max(0, i - 3), i)
                if traj[j].get("role") == "assistant"
            ]
            if msg.get("content") in prev_contents:
                return i, "Agent stuck in repetitive loop", msg

    # If no specific failure found, return last message
    return (
        len(traj) - 1,
        "Task incomplete at end of conversation",
        traj[-1] if traj else {},
    )


def analyze_failure_type(
    traj: List[Dict[str, Any]], task_info: Dict[str, Any]
) -> Tuple[str, str, int, Dict[str, Any]]:
    """
    Analyze the type of failure based on trajectory and ground truth.

    Args:
        traj: The conversation trajectory
        task_info: Task information including ground truth actions

    Returns:
        (failure_type, description, failing_turn_index, failing_turn_details)
    """
    if not traj:
        return "no_trajectory", "Empty trajectory", -1, {}

    # Extract tool calls from trajectory
    actual_calls = extract_tool_calls(traj)

    # Get ground truth actions
    gt_actions = task_info.get("task", {}).get("actions", [])

    if not actual_calls:
        return (
            "no_tool_calls",
            "Agent made no tool calls",
            0,
            traj[0] if traj else {},
        )

    # Check for authentication failure (common pattern)
    for i, msg in enumerate(traj):
        if msg.get("role") == "tool" and "Error: user not found" in str(
            msg.get("content", "")
        ):
            return (
                "authentication_failure",
                "Failed to authenticate user - wrong credentials used",
                i,
                msg,
            )

    # Check if wrong tools were called
    gt_tool_names = {a["name"] for a in gt_actions}
    actual_tool_names = {c["name"] for c in actual_calls}

    missing_tools = gt_tool_names - actual_tool_names

    if missing_tools:
        failing_turn, reason, msg = find_failing_turn(traj)
        return (
            "missing_tool_calls",
            f"Missing tool calls: {missing_tools}",
            failing_turn,
            msg,
        )

    # Check for argument mismatches
    for gt_action in gt_actions:
        matching_calls = [
            c for c in actual_calls if c["name"] == gt_action["name"]
        ]
        if matching_calls:
            gt_kwargs = gt_action.get("kwargs", {})
            for call in matching_calls:
                try:
                    actual_kwargs = (
                        json.loads(call["arguments"])
                        if isinstance(call["arguments"], str)
                        else call["arguments"]
                    )
                    if gt_kwargs != actual_kwargs:
                        # Find the turn with this tool call
                        for i, msg in enumerate(traj):
                            if msg.get("tool_calls"):
                                for tc in msg["tool_calls"]:
                                    if (
                                        tc.get("function", {}).get("name")
                                        == gt_action["name"]
                                    ):
                                        return (
                                            "wrong_arguments",
                                            f"Tool '{gt_action['name']}' called with wrong arguments. Expected: {gt_kwargs}, Got: {actual_kwargs}",
                                            i,
                                            msg,
                                        )
                except Exception:
                    pass

    failing_turn, reason, msg = find_failing_turn(traj)
    return (
        "goal_partially_completed",
        "Task was not fully completed",
        failing_turn,
        msg,
    )

