"""
Tests for the evaluator module.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock
from tota_benchmarking.evaluator import evaluate_response, process_evaluation_results
from tota_benchmarking.schema import (
    GeneralEvaluationResult,
    SemanticSimilarityEvaluationResult,
    LineEvaluationResult,
    LineMatchStatus,
    ToolCallsDetailedEvaluationResult,
    ToolCallsDetailedStatus,
    VariableUpdateEvaluationResult,
    VariableUpdateStatus,
)


@pytest.mark.asyncio
async def test_evaluate_response_json_decode_error():
    """Test evaluate_response handles JSON decode errors."""
    model_response = "invalid json"
    golden_response = "also invalid"

    result = await evaluate_response(model_response, golden_response)

    # Check that JSON decode errors are properly handled
    assert result["rag_query"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["transition_state"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["update_variables"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["tool_calls_nl"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["audio"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["end_interaction"] == GeneralEvaluationResult.JSON_DECODE_ERROR
    assert result["cleaned_model_response"] == model_response
    assert result["cleaned_golden_response"] == golden_response


@pytest.mark.asyncio
async def test_evaluate_response_valid_json():
    """Test evaluate_response with valid JSON responses."""
    model_response = json.dumps({"rag_query": "test query"})
    golden_response = json.dumps({"rag_query": "test query"})

    # Mock all the evaluator functions to return CORRECT
    with patch(
        "tota_benchmarking.evaluator.evaluate_rag_query",
        new=AsyncMock(return_value=GeneralEvaluationResult.CORRECT),
    ), patch(
        "tota_benchmarking.evaluator.evaluate_transition_state",
        new=AsyncMock(return_value=GeneralEvaluationResult.CORRECT),
    ), patch(
        "tota_benchmarking.evaluator.evaluate_update_variables",
        new=AsyncMock(
            return_value=VariableUpdateEvaluationResult(
                correct_keys=1,
                correct_values=1,
                incorrect_keys=0,
                incorrect_values=0,
                total_golden_count=1,
            )
        ),
    ), patch(
        "tota_benchmarking.evaluator.evaluate_tool_calls_nl",
        new=AsyncMock(
            return_value=ToolCallsDetailedEvaluationResult(
                correct_tool_names=1,
                correct_queries=1,
                incorrect_tool_names=0,
                incorrect_queries=0,
                missing_tool_names=0,
                total_golden_count=1,
                tool_results={},
            )
        ),
    ), patch(
        "tota_benchmarking.evaluator.evaluate_audio",
        new=AsyncMock(return_value=GeneralEvaluationResult.CORRECT),
    ), patch(
        "tota_benchmarking.evaluator.evaluate_end_interaction",
        new=AsyncMock(return_value=GeneralEvaluationResult.CORRECT),
    ):
        result = await evaluate_response(model_response, golden_response)

    # Check that results are properly stored
    assert result["rag_query"] == GeneralEvaluationResult.CORRECT
    assert result["transition_state"] == GeneralEvaluationResult.CORRECT
    assert isinstance(result["update_variables"], VariableUpdateEvaluationResult)
    assert isinstance(result["tool_calls_nl"], ToolCallsDetailedEvaluationResult)
    assert result["audio"] == GeneralEvaluationResult.CORRECT
    assert result["end_interaction"] == GeneralEvaluationResult.CORRECT


def test_process_evaluation_results_key_missing():
    """Test process_evaluation_results with KEY_MISSING result."""
    result = process_evaluation_results(GeneralEvaluationResult.KEY_MISSING)
    assert result == {"overall": "KEY_MISSING", "has_failed": True}


def test_process_evaluation_results_general_evaluation():
    """Test process_evaluation_results with GeneralEvaluationResult."""
    eval_results = {
        "test_key": GeneralEvaluationResult.CORRECT,
        "cleaned_model_response": "model response",
        "cleaned_golden_response": "golden response",
    }

    result = process_evaluation_results(eval_results)

    assert not result["has_failed"]
    assert (
        result["evaluation_results"]["test_key"]
        == GeneralEvaluationResult.CORRECT.value
    )


def test_process_evaluation_results_semantic_similarity():
    """Test process_evaluation_results with SemanticSimilarityEvaluationResult."""
    eval_results = {
        "test_key": SemanticSimilarityEvaluationResult.MATCH,
        "cleaned_model_response": "model response",
        "cleaned_golden_response": "golden response",
    }

    result = process_evaluation_results(eval_results)

    assert not result["has_failed"]
    assert (
        result["evaluation_results"]["test_key"]
        == SemanticSimilarityEvaluationResult.MATCH.value
    )


def test_process_evaluation_results_tool_calls():
    """Test process_evaluation_results with ToolCallsDetailedEvaluationResult."""
    tool_result = ToolCallsDetailedEvaluationResult(
        correct_tool_names=2,
        correct_queries=2,
        incorrect_tool_names=0,
        incorrect_queries=0,
        missing_tool_names=0,
        total_golden_count=2,
        tool_results={},
    )

    eval_results = {
        "tool_calls_nl": tool_result,
        "cleaned_model_response": "model response",
        "cleaned_golden_response": "golden response",
    }

    result = process_evaluation_results(eval_results)

    assert not result["has_failed"]
    assert (
        result["evaluation_results"]["tool_calls_nl"]["status"]
        == ToolCallsDetailedStatus.CORRECT.value
    )
    assert result["evaluation_results"]["tool_calls_nl"]["correct_tool_names"] == 2


def test_process_evaluation_results_variable_update():
    """Test process_evaluation_results with VariableUpdateEvaluationResult."""
    variable_result = VariableUpdateEvaluationResult(
        correct_keys=1,
        correct_values=1,
        incorrect_keys=1,
        incorrect_values=0,
        total_golden_count=2,
    )

    eval_results = {
        "update_variables": variable_result,
        "cleaned_model_response": "model response",
        "cleaned_golden_response": "golden response",
    }

    result = process_evaluation_results(eval_results)

    assert result["has_failed"]  # Should fail because incorrect_keys > 0
    assert (
        result["evaluation_results"]["update_variables"]["status"]
        == VariableUpdateStatus.ERROR.value
    )
    assert result["evaluation_results"]["update_variables"]["correct_keys"] == 1
    assert result["evaluation_results"]["update_variables"]["incorrect_keys"] == 1
