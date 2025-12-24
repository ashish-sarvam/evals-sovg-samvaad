import copy
import json

from .prompts.loader import load_template


def get_adapter(model_name):
    """
    Returns the appropriate adapter function based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing system prompt name and assistant response adapter functions.
    """
    adapter_map = {
        "Tota v7": ("tota_v7_system_prompt", tota_v7_chat_thread_adapter),
        "Tota v8": ("tota_v8_system_prompt", tota_v8_chat_thread_adapter),
    }

    # Default to no adapters if no match is found
    system_prompt_name = "tota_v6_system_prompt"

    def default_conversation_adapter(x, idx=None):
        return x

    conversation_adapter = default_conversation_adapter

    # Check if any key in the adapter_map is a substring of the model_name
    for key, adapters in adapter_map.items():
        if key in model_name:
            system_prompt_name, conversation_adapter = adapters
            break

    return system_prompt_name, conversation_adapter


def get_ouput_format_from_sys_prompt(system_prompt: str) -> list[str]:
    valid_conditions = []
    if '"rag_query"' in system_prompt:
        valid_conditions.append("rag_query")
    if '"retrieval_query"' in system_prompt:
        valid_conditions.append("retrieval_query")
    if '"update_variables"' in system_prompt:
        valid_conditions.append("update_variables")
    if '"tool_calls_nl"' in system_prompt:
        valid_conditions.append("tool_calls_nl")
    if '"transition_state"' in system_prompt:
        valid_conditions.append("transition_state")
    if '"end_interaction"' in system_prompt:
        valid_conditions.append("end_interaction")
    if '"audio"' in system_prompt:
        valid_conditions.append("audio")
    if "change_output_language" in system_prompt:
        valid_conditions.append("language_switch")
    if "change_language" in system_prompt:
        valid_conditions.append("change_language")

    return valid_conditions


def edit_system_prompt(system_prompt: str) -> str:
    valid_conditions = get_ouput_format_from_sys_prompt(system_prompt)
    template = load_template("tota_v8_system_prompt_guidelines")
    ADDITIONAL_INSTRUCTIONS_COMMON = template.render(valid_conditions=valid_conditions)

    if "# KB details for retrieval:" in system_prompt:
        first_half = system_prompt.split("# KB details for retrieval:")[0]
        second_half = system_prompt.split("# KB details for retrieval:")[1]
        return (
            first_half
            + ADDITIONAL_INSTRUCTIONS_COMMON
            + "# KB details for retrieval:\n"
            + second_half
        )

    elif "# Variables" in system_prompt:
        first_half = system_prompt.split("# Variables")[0]
        second_half = system_prompt.split("# Variables")[1]
        return (
            first_half + ADDITIONAL_INSTRUCTIONS_COMMON + "# Variables\n" + second_half
        )

    elif "# variables" in system_prompt:
        first_half = system_prompt.split("# variables")[0]
        second_half = system_prompt.split("# variables")[1]
        return (
            first_half + ADDITIONAL_INSTRUCTIONS_COMMON + "# variables\n" + second_half
        )

    else:
        return system_prompt + ADDITIONAL_INSTRUCTIONS_COMMON


def tota_v8_chat_thread_adapter(messages, idx):
    messages_copy = copy.deepcopy(messages)
    for message in messages_copy:
        if message["role"] == "system":
            message["content"] = edit_system_prompt(message["content"])

    return messages_copy


def tota_v9_chat_thread_adapter(messages, idx):
    system_prompt = messages[0]["content"]

    mutable_variables_with_values = False
    kb_details = False
    needs_tool_calls_nl = False
    state_transition = False

    if '"rag_query"' in system_prompt:
        kb_details = True
    if '"retrieval_query"' in system_prompt:
        kb_details = True
    if '"update_variables"' in system_prompt:
        mutable_variables_with_values = True
    if '"tool_calls_nl"' in system_prompt:
        needs_tool_calls_nl = True
    if '"transition_state"' in system_prompt:
        state_transition = True

    template = load_template("conversational_template_trimmed")
    conversational_guidelines = template.render(
        kb_details=kb_details,
        mutable_variables_with_values=mutable_variables_with_values,
        needs_tool_calls_nl=needs_tool_calls_nl,
        state_transition=state_transition,
    )

    first_half = messages[0]["content"].split("# General bot instructions")[0]
    second_half = messages[0]["content"].split("# General bot instructions")[1]
    messages[0]["content"] = (
        first_half
        + conversational_guidelines
        + "\n\n# General bot instructions\n\n"
        + second_half
    )

    return messages


def tota_v7_chat_thread_adapter(messages, idx):
    for message in messages:
        if message["role"] == "assistant":
            try:
                # Check if the message content is a JSON
                if not message["content"].strip().startswith("{"):
                    continue

                json_output = json.loads(message["content"])
                if "lines" in json_output:
                    for i, line in enumerate(json_output["lines"]):
                        if (
                            line.startswith("GI.1")
                            or line.startswith("GI.2")
                            or line.startswith("GI.3")
                            or line.startswith("GI.4")
                            or line.startswith("GI.5")
                        ):
                            json_output["lines"].pop(i)

                        if (
                            line.startswith("GI.6")
                            or line.startswith("GI.7")
                            or line.startswith("GI.8")
                        ):
                            json_output["lines"][i] = {
                                "GI.6": "GI.1",
                                "GI.7": "GI.2",
                                "GI.8": "GI.3",
                                "GI.8.1": "GI.3.1",
                                "GI.8.2": "GI.3.2",
                            }[line]

                    if len(json_output["lines"]) == 0:
                        json_output.pop("lines")

                message["content"] = json.dumps(json_output)
            except Exception:
                pass

    return messages


def render_variables(variables_list):
    render_variable_string = ""
    for variable_data in variables_list:
        render_variable_string += f"{variable_data[0]}: {variable_data[1]}\n"

    return render_variable_string


def system_prompt_renderer(
    prompt_name: str,
    current_language: str,
    languages_available: str,
    enable_agentic_lid: bool,
    global_prompt: str,
    instructions: str,
    kb_details: str,
    immutable_variables: list[tuple[str, str]],
    mutable_variables_with_values: list[tuple[str, str]],
    current_state_name: str,
    next_states: str,
    tools: str,
    needs_tool_calls_nl: bool,
    state_transition: bool,
    variables: bool,
) -> str:
    template = load_template(prompt_name)
    return template.render(
        current_language=current_language,
        languages_available=languages_available,
        enable_agentic_lid=enable_agentic_lid,
        global_prompt=global_prompt,
        instructions=instructions,
        kb_details=kb_details,
        immutable_variables=render_variables(immutable_variables),
        mutable_variables_with_values=render_variables(mutable_variables_with_values),
        current_state_name=current_state_name,
        next_states=next_states,
        tools=tools,
        needs_tool_calls_nl=needs_tool_calls_nl,
        state_transition=state_transition,
        llm_model_mode="inference",
        variables=variables,
    )


def convert_chat_thread_for_model(chat_thread, base_model_type):
    """
    Converts the chat thread to be compatible with the specified base model type.

    Args:
        chat_thread (list): The original chat thread with messages.
        base_model_type (str): The type of base model (e.g., "Mistral", "GPT", etc.).

    Returns:
        list: The converted chat thread compatible with the specified model.
    """
    converted_thread = copy.deepcopy(chat_thread)

    if base_model_type.upper() == "MISTRAL":
        # For Mistral models, convert tool messages to user messages
        for i, message in enumerate(converted_thread):
            if message.get("role") == "tool":
                converted_thread[i]["role"] = "user"

    if base_model_type.upper() == "GEMMA":
        for i, message in enumerate(converted_thread):
            if message.get("role") == "tool":
                converted_thread[i]["role"] = "user"
                converted_thread[i]["content"] = f"Tool response: {message['content']}"

    # Normalize tool role for common chat-only base models
    if base_model_type.upper() in {"GPT", "CLAUDE", "LLAMA", "GEMINI"}:
        for i, message in enumerate(converted_thread):
            if message.get("role") == "tool":
                converted_thread[i]["role"] = "user"

    # Anthropic (Claude) requires role to be one of user/assistant
    if base_model_type.upper() == "CLAUDE":
        for i, message in enumerate(converted_thread):
            role = message.get("role")
            if role not in ("user", "assistant"):
                converted_thread[i]["role"] = "user"

    # Gemini expects roles user/model; we map assistant->model earlier in the provider wrapper,
    # so here we just normalize tools as users
    if base_model_type.upper() == "GEMINI":
        for i, message in enumerate(converted_thread):
            role = message.get("role")
            if role == "tool":
                converted_thread[i]["role"] = "user"
            elif role == "assistant":
                # Gemini expects roles: user/model
                converted_thread[i]["role"] = "model"

    # if base_model_type.upper() == "QWEN3":
    #     chat_thread[-1]["content"] = chat_thread[-1]["content"] + " /no_think"

    # Add more model-specific conversions as needed
    # elif base_model_type.upper() == "LLAMA":
    #     # Specific conversions for Llama models
    #     pass

    return converted_thread
