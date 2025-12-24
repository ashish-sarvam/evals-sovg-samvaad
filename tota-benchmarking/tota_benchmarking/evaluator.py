"""
Module for evaluating model responses against golden responses.
"""

import json
import logging
from typing import Any, Dict, Union

from .key_evaluators import (
    evaluate_audio,
    evaluate_end_interaction,
    evaluate_lines,
    evaluate_rag_query,
    evaluate_text,
    evaluate_tool_calls_nl,
    evaluate_transition_state,
    evaluate_update_variables,
)
from .schema import (
    GeneralEvaluationResult,
    LineEvaluationResult,
    LineMatchStatus,
    SemanticSimilarityEvaluationResult,
    ToolCallsDetailedEvaluationResult,
    ToolCallsDetailedStatus,
    VariableUpdateEvaluationResult,
    VariableUpdateStatus,
)
from .utils import clean_model_response


async def evaluate_response(
    response: str,
    golden_response: str,
    system_prompt: str,
    last_user_message: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a model response against a golden response.

    Args:
        response: The model's response string
        golden_response: The golden (expected) response string

    Returns:
        Dict containing evaluation results for different aspects
    """
    try:
        cleaned_model_response = clean_model_response(response)
        cleaned_golden_response = clean_model_response(golden_response)

        evaluation_results = {}

        # Evaluate different aspects of the response
        # evaluation_results["lines"] = await evaluate_lines(response_dict, golden_dict)
        evaluation_results["rag_query"] = await evaluate_rag_query(
            cleaned_model_response, cleaned_golden_response, last_user_message
        )
        evaluation_results["transition_state"] = await evaluate_transition_state(
            cleaned_model_response, cleaned_golden_response
        )
        evaluation_results["update_variables"] = await evaluate_update_variables(
            cleaned_model_response, cleaned_golden_response, system_prompt
        )
        evaluation_results["tool_calls_nl"] = await evaluate_tool_calls_nl(
            cleaned_model_response, cleaned_golden_response
        )
        evaluation_results["audio"] = await evaluate_audio(
            cleaned_model_response, cleaned_golden_response
        )
        # evaluation_results["text"] = await evaluate_text(response_dict, golden_dict)
        evaluation_results["end_interaction"] = await evaluate_end_interaction(
            cleaned_model_response, cleaned_golden_response
        )

        # Store the cleaned responses in the results
        evaluation_results["cleaned_model_response"] = (
            json.dumps(cleaned_model_response).replace('": ', '":').replace(', "', ',"')
        )
        evaluation_results["cleaned_golden_response"] = (
            json.dumps(cleaned_golden_response)
            .replace('": ', '":')
            .replace(', "', ',"')
        )

    except json.JSONDecodeError as e:
        logging.exception(e, exc_info=True)
        evaluation_results = {
            # "lines": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "rag_query": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "transition_state": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "update_variables": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "tool_calls_nl": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "audio": GeneralEvaluationResult.JSON_DECODE_ERROR,
            # "text": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "end_interaction": GeneralEvaluationResult.JSON_DECODE_ERROR,
            "cleaned_model_response": response,
            "cleaned_golden_response": golden_response,
        }

    return evaluation_results


def process_evaluation_results(
    evaluation_results: Dict[str, Any],
) -> Dict[str, Union[bool, Dict[str, Any]]]:
    """
    Process raw evaluation results to determine if they indicate failures.

    Args:
        evaluation_results: Raw evaluation results from evaluate_response

    Returns:
        Dict with processed evaluation results and failure status
    """
    if evaluation_results == GeneralEvaluationResult.KEY_MISSING:
        return {"overall": "KEY_MISSING", "has_failed": True}

    processed_results = {}
    has_failed = False

    for key, result in evaluation_results.items():
        # Skip the cleaned responses in evaluation processing
        if key in ["cleaned_model_response", "cleaned_golden_response"]:
            continue

        # Handle each type of result and determine if it's a failure
        is_failure = False
        is_ignored = False  # Flag for BOTH_KEYS_MISSING case

        if isinstance(result, GeneralEvaluationResult):
            # For BOTH_KEYS_MISSING, mark as ignored
            if result == GeneralEvaluationResult.BOTH_KEYS_MISSING:
                is_ignored = True
                processed_results[key] = result.value
            else:
                # For other GeneralEvaluationResult, only CORRECT is considered a success
                is_failure = result != GeneralEvaluationResult.CORRECT
                processed_results[key] = result.value

        elif isinstance(result, SemanticSimilarityEvaluationResult):
            # For semantic evaluations, MATCH and PARTIAL_MATCH are considered correct
            is_failure = result == SemanticSimilarityEvaluationResult.WRONG
            processed_results[key] = result.value

        elif isinstance(result, ToolCallsDetailedEvaluationResult):
            # For tool calls, success if all tool names are correct and all queries match semantically
            tool_failure = (
                result.incorrect_tool_names > 0
                or result.missing_tool_names > 0
                or result.incorrect_queries > 0
            )
            is_failure = tool_failure

            # Create detailed tool evaluation result
            processed_results[key] = {
                "status": (
                    ToolCallsDetailedStatus.CORRECT.value
                    if not tool_failure
                    else ToolCallsDetailedStatus.ERROR.value
                ),
                "correct_tool_names": result.correct_tool_names,
                "correct_queries": result.correct_queries,
                "incorrect_tool_names": result.incorrect_tool_names,
                "incorrect_queries": result.incorrect_queries,
                "missing_tool_names": result.missing_tool_names,
                "total_golden_count": result.total_golden_count,
                "tool_details": result.tool_results,
            }

        elif isinstance(result, VariableUpdateEvaluationResult):
            # For variable updates, all keys and values must be correct
            variable_failure = (
                result.incorrect_keys > 0
                or result.incorrect_values > 0
                or (result.correct_keys < result.total_golden_count)
            )
            is_failure = variable_failure

            # Create detailed variable update evaluation result
            processed_results[key] = {
                "status": (
                    VariableUpdateStatus.CORRECT.value
                    if not variable_failure
                    else VariableUpdateStatus.ERROR.value
                ),
                "correct_keys": result.correct_keys,
                "correct_values": result.correct_values,
                "incorrect_keys": result.incorrect_keys,
                "incorrect_values": result.incorrect_values,
                "total_golden_count": result.total_golden_count,
            }

        elif isinstance(result, LineEvaluationResult):
            is_failure = result.match_status != LineMatchStatus.FULL_MATCH
            processed_results[key] = {
                "status": result.match_status.value,
                "matched_count": result.matched_count,
                "total_response_count": result.total_response_count,
                "total_golden_count": result.total_golden_count,
            }
        else:
            # For any other type of result
            processed_results[key] = str(result)

        # Update has_failed if any evaluation failed (and not ignored)
        if is_failure and not is_ignored:
            has_failed = True

    return {"evaluation_results": processed_results, "has_failed": has_failed}
