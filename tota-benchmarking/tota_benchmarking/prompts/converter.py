from jinja2 import Template


def convert_system_prompt_to_structured(system_prompt: str) -> str:
    # add lines in output
    # add instruction numbers in output (GI, BI, SI)
    assert system_prompt.startswith("You are a bot in an audio call with a user.")
    assert "General bot instructions" in system_prompt
    assert "State specific instructions" in system_prompt

    kb_details = True if '"rag_query"' in system_prompt else False
    mutable_variables = True if '"update_variables"' in system_prompt else False
    state_transition = True if '"transition_state"' in system_prompt else False
    end_interaction = True if '"end_interaction"' in system_prompt else False
    needs_tool_calls_nl = True if '"tool_calls_nl"' in system_prompt else False

    jinja_current_template1 = """\
{ {% if kb_details %}"rag_query":"some query",{% endif %}{% if mutable_variables %}"update_variables":{"var1":"value","var2":"value"},{% endif %}{% if state_transition %}"transition_state":"next_state_name",{% endif %}"end_interaction":true|false,{%- if llm_model_mode == "datagen" %}{% if needs_tool_calls_nl %}"tool_calls":[{"name":"...","args": {...}},{"name":"...","args":{...}}, ...]{% endif %}{% endif %}{% if needs_tool_calls_nl %}"tool_calls_nl":["tool1: NL text with tool arguments","tool2: NL text with tool arguments"],{% endif %}"audio":"Text to be sent to the user"}
"""
    jinja_current_template2 = """\
{ {% if kb_details %}"rag_query":"some query",{% endif %}{% if mutable_variables %}"update_variables":{"var1":"value","var2":"value"},{% endif %}{% if state_transition %}"transition_state":"next_state_name",{% endif %}"end_interaction":true|false,{%- if llm_model_mode == "datagen" %}{% if needs_tool_calls_nl %}"tool_calls":[{"name":"...","args": {...}},{"name":"...","args":{...}}, ...]{% endif %}{% endif %}{% if needs_tool_calls_nl %}"tool_calls_nl":["tool1: NL text with tool arguments","tool2: NL text with tool arguments"],{% endif %}"audio":"Text to be sent to the user"}
- Descriptions of keys: {% if kb_details %}'rag_query' is a string to query a knowledge base and empty in case information retrieval is not required as per lines, {% endif %}{% if mutable_variables %}'update_variables' is a dict of variables to update and empty dict in case no variable needs updating as per lines, {% endif %}{% if state_transition %}'transition_state' has the name of the state to transition to or empty for no transition as per lines, {% endif %}end_interaction is boolean to end call as per lines, {% if needs_tool_calls_nl %}tool_calls_nl is list of natural language text describing arguments to any tools to call as per lines{% endif %}, audio is just the text for user as per lines.
"""
    template_current1 = Template(jinja_current_template1)
    output_current1 = template_current1.render(
        kb_details=kb_details,
        mutable_variables=mutable_variables,
        state_transition=state_transition,
        end_interaction=end_interaction,
        needs_tool_calls_nl=needs_tool_calls_nl,
    )
    template_current2 = Template(jinja_current_template2)
    output_current2 = template_current2.render(
        kb_details=kb_details,
        mutable_variables=mutable_variables,
        state_transition=state_transition,
        end_interaction=end_interaction,
        needs_tool_calls_nl=needs_tool_calls_nl,
    )
    jinja_structured_template = """\
{"lines":["GI.x","SIx.x.x"],{% if kb_details %}"rag_query":"some query",{% endif %}{% if mutable_variables %}"update_variables":{"var1":"value","var2":"value"},{% endif %}{% if state_transition %}"transition_state":"next_state_name",{% endif %}"end_interaction":true|false,{%- if llm_model_mode == "datagen" %}{% if needs_tool_calls_nl %}"tool_calls":[{"name":"...","args": {...}},{"name":"...","args":{...}}, ...]{% endif %}{% endif %}{% if needs_tool_calls_nl %}"tool_calls_nl":["tool1: NL text with tool arguments","tool2: NL text with tool arguments"],{% endif %}"audio":"Text to be sent to the user"}
- Descriptions of keys: 'lines' is a list of relevant line number/s of instruction/s to invoke in this turn, {% if kb_details %}'rag_query' is a string to query a knowledge base and empty in case information retrieval is not required as per lines, {% endif %}{% if mutable_variables %}'update_variables' is a dict of variables to update and empty dict in case no variable needs updating as per lines, {% endif %}{% if state_transition %}'transition_state' has the name of the state to transition to or empty for no transition as per lines, {% endif %}end_interaction is boolean to end call as per lines, {% if needs_tool_calls_nl %}tool_calls_nl is list of natural language text describing arguments to any tools to call as per lines{% endif %}, audio is just the text for user as per lines."""
    template_structured = Template(jinja_structured_template)
    output_structured = template_structured.render(
        kb_details=kb_details,
        mutable_variables=mutable_variables,
        state_transition=state_transition,
        end_interaction=end_interaction,
        needs_tool_calls_nl=needs_tool_calls_nl,
    )

    if '"lines"' not in system_prompt:
        assert output_current1 in system_prompt
        if output_current1 in system_prompt:
            if output_current2 in system_prompt:
                system_prompt = system_prompt.replace(
                    output_current2, output_structured
                )
            else:
                system_prompt = system_prompt.replace(
                    output_current1, output_structured
                )

        system_prompt = system_prompt.replace(
            "# Background information",
            "# Background information\n- You will see three sets of instructions: 'global', 'bot-specific' and 'state-specific' each with 'line' numbers. While responding, decide relevant lines given the conversation context.",
        )

    if "# Global instructions" not in system_prompt:
        # GI
        global_instructions = (
            system_prompt.split("# Background information")[1]
            .split("# General bot instructions")[0]
            .strip()
            .split("\n\n")
        )
        if len(global_instructions) > 1:
            global_instructions = global_instructions[1].strip().split("\n")
        else:
            global_instructions = []

        gi_count = 1
        for line in global_instructions:
            system_prompt = system_prompt.replace(line, f"GI.{gi_count}: {line[2:]}")
            gi_count += 1

    bot_instructions = (
        system_prompt.split("# General bot instructions")[1]
        .split("# State specific instructions")[0]
        .strip()
        .split("\n")
    )
    bi_count = 1
    for line in bot_instructions:
        system_prompt = system_prompt.replace(line, f"BI.{bi_count}: {line}")
        bi_count += 1

    if "# KB details" in system_prompt:
        state_instructions = (
            system_prompt.split("State specific instructions")[1]
            .split("# KB details")[0]
            .strip()
            .split("\n")
        )
    elif "# Variables" in system_prompt:
        state_instructions = (
            system_prompt.split("State specific instructions")[1]
            .split("# Variables")[0]
            .strip()
            .split("\n")
        )
    else:
        raise ValueError("Partitions not found in system prompt")

    si_count = 1
    for line in state_instructions:
        line = line.strip()
        if not line:
            continue
        system_prompt = system_prompt.replace(line, f"SI1.1.{si_count}: {line}")
        si_count += 1

    return system_prompt
