import copy
import json
import re
from typing import Dict, List, Tuple


def clean_model_response(
    model_output: str,
) -> str:
    # Create a deep copy of the output to avoid modifying while iterating
    output = json.loads(model_output)
    cleaned_output = copy.deepcopy(output)

    if "tool_calls_nl" in cleaned_output:
        if len(cleaned_output["tool_calls_nl"]) > 0:
            if "audio" in cleaned_output:
                cleaned_output["audio"] = ""

    if (
        "retrieval_query" in cleaned_output and cleaned_output["retrieval_query"] != ""
    ) or ("rag_query" in cleaned_output and cleaned_output["rag_query"] != ""):
        if "audio" in cleaned_output:
            cleaned_output["audio"] = ""

        if "transition_state" in cleaned_output:
            cleaned_output.pop("transition_state")

        if "end_conversation" in cleaned_output:
            cleaned_output.pop("end_conversation")

        if "end_interaction" in cleaned_output:
            cleaned_output.pop("end_interaction")

        if "tool_calls_nl" in cleaned_output:
            cleaned_output.pop("tool_calls_nl")

    if (
        "transition_state" in cleaned_output
        and cleaned_output["transition_state"] != ""
    ):
        if "audio" in cleaned_output:
            cleaned_output["audio"] = ""

        if "tool_calls_nl" in cleaned_output:
            cleaned_output.pop("tool_calls_nl")

        if "end_conversation" in cleaned_output:
            cleaned_output.pop("end_conversation")

        if "end_interaction" in cleaned_output:
            cleaned_output.pop("end_interaction")

    if (
        "end_conversation" in cleaned_output and cleaned_output["end_conversation"]
    ) or ("end_interaction" in cleaned_output and cleaned_output["end_interaction"]):
        if "tool_calls_nl" in cleaned_output:
            cleaned_output.pop("tool_calls_nl")

    if (
        "end_interaction" in cleaned_output
        and cleaned_output["end_interaction"] == False
    ):
        cleaned_output.pop("end_interaction")

    if "audio" not in cleaned_output:
        cleaned_output["audio"] = ""

    return cleaned_output


def get_mutable_variables_from_system_prompt(
    system_prompt: str,
) -> List[Tuple[str, str]]:
    mutable_pattern = r"## Editable variables with their current value\s*([\s\S]*?)(?=\n(?:Current State:|Available state transitions:|## |$))"  # noqa
    mutable_vars_match = re.search(
        mutable_pattern,
        system_prompt,
        re.DOTALL,
    )
    if mutable_vars_match:
        content = mutable_vars_match.group(1).strip()
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        mutable_variables = []
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                mutable_variables.append((key.strip(), value.strip()))
        return mutable_variables
    else:
        return []


def filter_unchanged_variables(
    update_variables: Dict[str, str], system_prompt: str
) -> Dict[str, str]:
    """
    Filter out variables that are being updated to their current values.

    Args:
        update_variables: Dictionary with variable names as keys and new values as values
        system_prompt: The system prompt containing the current variable values

    Returns:
        Dictionary with only the variables that are actually changing
    """
    # Get the current mutable variables from the system prompt
    mutable_variables = get_mutable_variables_from_system_prompt(system_prompt)

    # Convert the list of tuples to a dictionary for easier lookup
    current_variables = {name: value for name, value in mutable_variables}

    # Filter out variables that aren't changing
    filtered_variables = {}
    for var_name, new_value in update_variables.items():
        # Only include the variable if it's not in the current variables
        # or if its value is actually changing
        if (
            var_name not in current_variables
            or str(new_value).strip() != str(current_variables[var_name]).strip()
        ):
            filtered_variables[var_name] = new_value

    return filtered_variables
