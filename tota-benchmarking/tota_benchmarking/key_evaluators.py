import json
import os
import re

from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from cachetools import cached
from dotenv import load_dotenv
from openai import AzureOpenAI

from .prompts.loader import load_template
from .schema import (
    GeneralEvaluationResult,
    LineEvaluationResult,
    SemanticSimilarityEvaluationResult,
    ToolCallsDetailedEvaluationResult,
    VariableUpdateEvaluationResult,
)
from .utils import filter_unchanged_variables

load_dotenv()


@cached({})
def get_azure_openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_URL"),
    )


@cached({})
def get_anthropic_client():
    # NOTE - run `gcloud auth application-default login` so that the vertex credentials are available
    return AsyncAnthropicVertex(
        project_id="gpu-reservation-sarvam",
        region="us-east5",
    )


ALL_LANGUAGES = [
    "hindi",
    "english",
    "bengali",
    "punjabi",
    "odia",
    "kannada",
    "malayalam",
    "marathi",
    "gujarati",
    "tamil",
    "telugu",
]


def extract_language(query: str):
    query = query.lower()
    for language in ALL_LANGUAGES:
        if language in query:
            return language
    return None


def extract_email(query: str):
    query = query.lower()
    # regex to extract email
    email_regex = r"[\w\.-]+@[\w\.-]+"
    email = re.search(email_regex, query)
    if email:
        return email.group(0)
    return None


def extract_phone_number(query: str):
    query = query.lower()
    # regex to extract the largest number
    phone_number_regex = r"\d+"
    phone_number = re.findall(phone_number_regex, query)
    if phone_number:
        return phone_number[0]
    return None


def extract_pincode(query: str):
    query = query.lower()
    pincode_regex = r"\d+"
    pincode = re.findall(pincode_regex, query)
    if pincode:
        return pincode[0]
    return None


def extract_date(query: str):
    query = query.lower()
    date_regex = r"\d{2}/\d{2}/\d{4}"
    date = re.findall(date_regex, query)
    if date:
        return date[0]
    return None


# async def evaluate_semantic_similarity(response, golden_response):
#     client = get_azure_openai_client()
#     prompt = load_template("semantic_similarity_prompt")
#     prompt = prompt.render(model_response=response, golden_response=golden_response)

#     RETRIES = 3
#     for _ in range(RETRIES):
#         response = client.chat.completions.create(
#             model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
#             messages=[
#                 {"role": "system", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#         )

#         try:
#             response_json = json.loads(response.choices[0].message.content)

#             if response_json["evaluation"] == "MATCH":
#                 return SemanticSimilarityEvaluationResult.MATCH
#             elif response_json["evaluation"] == "PARTIAL_MATCH":
#                 return SemanticSimilarityEvaluationResult.PARTIAL_MATCH
#             elif response_json["evaluation"] == "WRONG":
#                 return SemanticSimilarityEvaluationResult.WRONG
#             else:
#                 raise Exception(
#                     f"Invalid evaluation result: {response_json['evaluation']}"
#                 )
#         except Exception:
#             continue

#     # should this be wrong?
#     return SemanticSimilarityEvaluationResult.WRONG


async def evaluate_semantic_similarity(
    response,
    golden_response,
    prompt_name="semantic_similarity_prompt",
    last_user_message="",
):
    client = get_anthropic_client()
    prompt_template = load_template(prompt_name)
    prompt = prompt_template.render({})

    RETRIES = 3
    for _ in range(RETRIES):
        judge_response = await client.messages.create(
            model="claude-sonnet-4@20250514",
            max_tokens=5000,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Golden response: {golden_response}\nModel response: {response}""",  # noqa
                },
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        try:
            response_json = json.loads(
                judge_response.content[1].text.split("```json")[1].split("```")[0]
            )
            if response_json["evaluation"] == "MATCH":
                return SemanticSimilarityEvaluationResult.MATCH
            elif response_json["evaluation"] == "PARTIAL_MATCH":
                return SemanticSimilarityEvaluationResult.PARTIAL_MATCH
            elif response_json["evaluation"] == "WRONG":
                return SemanticSimilarityEvaluationResult.WRONG
            else:
                raise Exception(
                    f"Invalid evaluation result: {response_json['evaluation']}"
                )
        except Exception as e:
            print(f"Error evaluating semantic similarity: {str(e)}")
            continue

    return SemanticSimilarityEvaluationResult.WRONG


async def evaluate_lines(response, golden_response):
    if "lines" not in response and "lines" not in golden_response:
        return GeneralEvaluationResult.BOTH_KEYS_MISSING

    elif "lines" in response and "lines" in golden_response:
        try:
            response_lines = response["lines"]
            golden_lines = golden_response["lines"]

            response_set = set(response_lines)
            golden_set = set(golden_lines)

            matches = response_set.intersection(golden_set)
            matched_count = len(matches)

            # Return detailed statistics about matches and mismatches
            return LineEvaluationResult(
                matched_count=matched_count,
                total_response_count=len(response_lines),
                total_golden_count=len(golden_lines),
            )
        except Exception:
            return GeneralEvaluationResult.INCORRECT

    elif "lines" in response and "lines" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "lines" not in response and "lines" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_rag_query(response, golden_response, last_user_message):
    if not response.get("rag_query") and not golden_response.get("rag_query"):
        return GeneralEvaluationResult.BOTH_KEYS_MISSING

    elif "rag_query" in response and "rag_query" in golden_response:
        response_query = response["rag_query"].lower().strip()
        golden_query = golden_response["rag_query"].lower().strip()

        if response_query == golden_query:
            return GeneralEvaluationResult.CORRECT

        result = await evaluate_semantic_similarity(
            response_query,
            golden_query,
            prompt_name="rag_semantic_similarity_prompt",
            last_user_message=last_user_message,
        )
        return result

    elif "rag_query" in response and "rag_query" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "rag_query" not in response and "rag_query" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_transition_state(response, golden_response):
    if not response.get("transition_state") and not golden_response.get(
        "transition_state"
    ):
        return GeneralEvaluationResult.BOTH_KEYS_MISSING
    elif "transition_state" in response and "transition_state" in golden_response:
        response_transition_state = response["transition_state"]
        golden_transition_state = golden_response["transition_state"]

        if response_transition_state == golden_transition_state:
            return GeneralEvaluationResult.CORRECT
        else:
            return GeneralEvaluationResult.INCORRECT
    elif "transition_state" in response and "transition_state" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA
    else:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_update_variables(response, golden_response, system_prompt):
    if not response.get("update_variables") and not golden_response.get(
        "update_variables"
    ):
        return GeneralEvaluationResult.BOTH_KEYS_MISSING
    elif "update_variables" in response and "update_variables" in golden_response:
        response_update_variables = response["update_variables"]
        golden_update_variables = golden_response["update_variables"]

        filtered_response_update_variables = filter_unchanged_variables(
            response_update_variables, system_prompt
        )

        filtered_golden_update_variables = filter_unchanged_variables(
            golden_update_variables, system_prompt
        )

        response_update_variables = filtered_response_update_variables
        golden_update_variables = filtered_golden_update_variables

        # Count correct keys, correct values, incorrect keys, and incorrect values
        correct_keys = 0
        correct_values = 0
        incorrect_keys = 0
        incorrect_values = 0

        # Find common keys (variables that are present in both dictionaries)
        common_keys = set(response_update_variables.keys()).intersection(
            set(golden_update_variables.keys())
        )

        # Extra keys in response (not in golden)
        extra_keys = set(response_update_variables.keys()) - set(
            golden_update_variables.keys()
        )

        # Count correct keys and values
        correct_keys = len(common_keys)

        # Count incorrect keys
        incorrect_keys = len(extra_keys)

        # Check values for common keys
        for key in common_keys:
            if (
                str(response_update_variables[key]).lower().strip()
                == str(golden_update_variables[key]).lower().strip()
            ):
                correct_values += 1
            else:
                response_value = str(response_update_variables[key]).lower().strip()
                golden_value = str(golden_update_variables[key]).lower().strip()

                if (
                    len(response_value.split(" ")) > 1
                    or len(golden_value.split(" ")) > 1
                ):
                    result = await evaluate_semantic_similarity(
                        response_value, golden_value
                    )
                else:
                    result = response_value == golden_value

                if result:
                    correct_values += 1
                else:
                    incorrect_values += 1

        total_golden_count = len(golden_update_variables)

        # Return detailed statistics
        return VariableUpdateEvaluationResult(
            correct_keys=correct_keys,
            correct_values=correct_values,
            incorrect_keys=incorrect_keys,
            incorrect_values=incorrect_values,
            total_golden_count=total_golden_count,
        )
    elif "update_variables" in response and "update_variables" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "update_variables" not in response and "update_variables" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_tool_calls_nl(response, golden_response):
    if not response.get("tool_calls_nl") and not golden_response.get("tool_calls_nl"):
        return GeneralEvaluationResult.BOTH_KEYS_MISSING

    elif "tool_calls_nl" in response and "tool_calls_nl" in golden_response:
        response_tool_calls_nl = response["tool_calls_nl"]
        golden_tool_calls_nl = golden_response["tool_calls_nl"]

        # Initialize counters
        correct_tool_names = 0
        correct_queries = 0
        incorrect_tool_names = 0
        incorrect_queries = 0

        # Parse response and golden tool calls
        response_tool_data = {}
        golden_tool_data = {}

        # Dictionary to store detailed results for each tool
        tool_results = {}

        # Parse golden tool calls
        for tool_call in golden_tool_calls_nl:
            tool_call = tool_call.strip()
            if tool_call.startswith("tool:"):
                tool_call = tool_call[5:]
            try:
                parts = tool_call.split(":", 1)
                if len(parts) == 2:
                    tool_name = parts[0].strip()
                    query = parts[1].strip()
                    golden_tool_data[tool_name] = query

                    # Initialize tool result entry for each golden tool
                    tool_results[tool_name] = {
                        "name_status": "missing",
                        "query_status": None,
                    }
            except Exception:
                continue

        # Parse response tool calls
        for tool_call in response_tool_calls_nl:
            tool_call = tool_call.strip()
            if tool_call.startswith("tool:"):
                tool_call = tool_call[5:]
            try:
                if tool_call.startswith("tool:"):
                    tool_call = tool_call[5:]

                parts = tool_call.split(":", 1)
                if len(parts) == 2:
                    tool_name = parts[0].strip()
                    query = parts[1].strip()
                    response_tool_data[tool_name] = query

                    # If this tool isn't in the golden data, mark it as incorrect
                    if tool_name not in golden_tool_data:
                        tool_results[tool_name] = {
                            "name_status": "incorrect",  # Tool not in golden data
                            "query_status": None,
                        }
            except Exception:
                continue

        # Find common tool names (exact match required)
        common_tools = set(response_tool_data.keys()).intersection(
            set(golden_tool_data.keys())
        )

        # Count correct tool names
        correct_tool_names = len(common_tools)

        # Count incorrect tool names (extra tools in response)
        incorrect_tool_names = len(
            set(response_tool_data.keys()) - set(golden_tool_data.keys())
        )

        # Count missing tool names (tools in golden but not in response)
        missing_tool_names = len(
            set(golden_tool_data.keys()) - set(response_tool_data.keys())
        )

        # For tools with matching names, evaluate query semantic similarity
        for tool_name in common_tools:
            response_query = response_tool_data[tool_name].strip().lower()
            golden_query = golden_tool_data[tool_name].strip().lower()

            # Mark tool name as correct
            tool_results[tool_name]["name_status"] = "correct"

            if response_query == golden_query:
                tool_results[tool_name]["query_status"] = "correct"
                correct_queries += 1
                continue

            # special handling of some tools
            if tool_name == "change_output_language" or tool_name == "change_language":
                # extract language from query
                golden_language = extract_language(golden_query)
                response_language = extract_language(response_query)
                if golden_language == response_language:
                    tool_results[tool_name]["query_status"] = "correct"
                    correct_queries += 1
                    continue
                else:
                    tool_results[tool_name]["query_status"] = "incorrect"
                    incorrect_queries += 1
                    continue
            elif "email" in tool_name:
                golden_email = extract_email(golden_query)
                response_email = extract_email(response_query)
                if golden_email == response_email:
                    tool_results[tool_name]["query_status"] = "correct"
                    correct_queries += 1
                    continue
            elif "phone_number" in tool_name:
                golden_phone_number = extract_phone_number(golden_query)
                response_phone_number = extract_phone_number(response_query)
                if golden_phone_number == response_phone_number:
                    tool_results[tool_name]["query_status"] = "correct"
                    correct_queries += 1
                    continue
            elif "pincode" in tool_name or "pin_code" in tool_name:
                golden_pincode = extract_pincode(golden_query)
                response_pincode = extract_pincode(response_query)
                if golden_pincode == response_pincode:
                    tool_results[tool_name]["query_status"] = "correct"
                    correct_queries += 1
                    continue
            elif "date" in tool_name:
                golden_date = extract_date(golden_query)
                response_date = extract_date(response_query)
                if golden_date == response_date:
                    tool_results[tool_name]["query_status"] = "correct"
                    correct_queries += 1
                    continue

            similarity_result = await evaluate_semantic_similarity(
                response_query, golden_query
            )

            # Store the query evaluation result
            tool_results[tool_name]["query_status"] = similarity_result.value

            if similarity_result in [
                SemanticSimilarityEvaluationResult.MATCH,
                SemanticSimilarityEvaluationResult.PARTIAL_MATCH,
            ]:
                correct_queries += 1
            else:
                incorrect_queries += 1

        total_golden_count = len(golden_tool_calls_nl)

        # Return detailed statistics
        return ToolCallsDetailedEvaluationResult(
            correct_tool_names=correct_tool_names,
            correct_queries=correct_queries,
            incorrect_tool_names=incorrect_tool_names,
            incorrect_queries=incorrect_queries,
            missing_tool_names=missing_tool_names,
            total_golden_count=total_golden_count,
            tool_results=tool_results,
        )

    elif "tool_calls_nl" in response and "tool_calls_nl" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "tool_calls_nl" not in response and "tool_calls_nl" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_audio(response, golden_response, eval_type="JUDGE_LLM"):
    if not response.get("audio") and not golden_response.get("audio"):
        return GeneralEvaluationResult.BOTH_KEYS_MISSING

    elif "audio" in response and "audio" in golden_response:
        response_audio = response["audio"].strip().lower()
        golden_audio = golden_response["audio"].strip().lower()

        # Special case: both audio fields are empty strings - consider them matching
        if response_audio == golden_audio:
            return SemanticSimilarityEvaluationResult.MATCH

        elif response_audio == "" and golden_audio != "":
            return SemanticSimilarityEvaluationResult.WRONG

        elif response_audio != "" and golden_audio == "":
            return SemanticSimilarityEvaluationResult.WRONG

        result = await evaluate_semantic_similarity(response_audio, golden_audio)
        return result

    elif "audio" in response and "audio" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "audio" not in response and "audio" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_text(response, golden_response):
    if "text" not in response and "text" not in golden_response:
        return GeneralEvaluationResult.BOTH_KEYS_MISSING
    elif "text" in response and "text" in golden_response:
        response_text = response["text"]
        golden_text = golden_response["text"]

        result = await evaluate_semantic_similarity(response_text, golden_text)

        return result
    elif "text" in response and "text" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA
    else:
        return GeneralEvaluationResult.KEY_MISSING


async def evaluate_end_interaction(response, golden_response):
    if "end_interaction" not in response and "end_interaction" not in golden_response:
        return GeneralEvaluationResult.BOTH_KEYS_MISSING

    if "end_interaction" in response and "end_interaction" not in golden_response:
        return GeneralEvaluationResult.KEY_EXTRA

    elif "end_interaction" not in response and "end_interaction" in golden_response:
        return GeneralEvaluationResult.KEY_MISSING

    else:
        response_end_interaction = response["end_interaction"]
        golden_end_interaction = golden_response["end_interaction"]

        if response_end_interaction == golden_end_interaction:
            return GeneralEvaluationResult.CORRECT
        else:
            return GeneralEvaluationResult.INCORRECT
