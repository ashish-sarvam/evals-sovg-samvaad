from tota_benchmarking.key_evaluators import (
    evaluate_tool_calls_nl,
    extract_email,
)
import pytest


@pytest.mark.asyncio
async def test_evaluate_language_change():
    golden_response = {
        "tool_calls_nl": [
            "change_output_language: Change language to Hindi",
        ]
    }
    model1_response = {
        "tool_calls_nl": [
            "change_output_language: Hindi",
        ]
    }
    model2_response = {
        "tool_calls_nl": [
            "change_output_language: Change language to English",
        ]
    }
    result1 = await evaluate_tool_calls_nl(model1_response, golden_response)
    assert result1.correct_tool_names == 1
    assert result1.correct_queries == 1
    assert result1.incorrect_tool_names == 0
    assert result1.incorrect_queries == 0

    result2 = await evaluate_tool_calls_nl(model2_response, golden_response)
    assert result2.correct_tool_names == 1
    assert result2.correct_queries == 0
    assert result2.incorrect_tool_names == 0
    assert result2.incorrect_queries == 1


@pytest.mark.asyncio
async def test_evaluate_email():
    golden_response = {
        "tool_calls_nl": [
            "verify_email: Please verify user email test@test.com",
        ]
    }
    model1_response = {
        "tool_calls_nl": [
            "verify_email: test@test.com",
        ]
    }
    model2_response = {
        "tool_calls_nl": [
            "verify_email: Please verify user email something@email.org",
        ]
    }

    result1 = await evaluate_tool_calls_nl(model1_response, golden_response)
    assert result1.correct_tool_names == 1
    assert result1.correct_queries == 1
    assert result1.incorrect_tool_names == 0
    assert result1.incorrect_queries == 0

    result2 = await evaluate_tool_calls_nl(model2_response, golden_response)
    print(result2.__dict__)
    assert result2.correct_tool_names == 1
    assert result2.correct_queries == 0
    assert result2.incorrect_tool_names == 0
    assert result2.incorrect_queries == 1


@pytest.mark.asyncio
async def test_evaluate_phone_number():
    golden_response = {
        "tool_calls_nl": [
            "verify_phone_number: Please verify user phone number 9876543210",
        ]
    }
    model1_response = {
        "tool_calls_nl": [
            "verify_phone_number: 9876543210",
        ]
    }
    model2_response = {
        "tool_calls_nl": [
            "verify_phone_number: 0123456789",
        ]
    }

    result1 = await evaluate_tool_calls_nl(model1_response, golden_response)
    assert result1.correct_tool_names == 1
    assert result1.correct_queries == 1
    assert result1.incorrect_tool_names == 0
    assert result1.incorrect_queries == 0

    result2 = await evaluate_tool_calls_nl(model2_response, golden_response)
    assert result2.correct_tool_names == 1
    assert result2.correct_queries == 0
    assert result2.incorrect_tool_names == 0
    assert result2.incorrect_queries == 1


@pytest.mark.asyncio
async def test_evaluate_pincode():
    golden_response = {
        "tool_calls_nl": [
            "pin_code_verification_tool: Verify the pin code 543210",
        ]
    }
    model1_response = {
        "tool_calls_nl": [
            "pin_code_verification_tool: 543210",
        ]
    }
    model2_response = {
        "tool_calls_nl": [
            "pin_code_verification_tool: 1234567890",
        ]
    }

    result1 = await evaluate_tool_calls_nl(model1_response, golden_response)
    assert result1.correct_tool_names == 1
    assert result1.correct_queries == 1
    assert result1.incorrect_tool_names == 0
    assert result1.incorrect_queries == 0

    result2 = await evaluate_tool_calls_nl(model2_response, golden_response)
    assert result2.correct_tool_names == 1
    assert result2.correct_queries == 0
    assert result2.incorrect_tool_names == 0
    assert result2.incorrect_queries == 1


@pytest.mark.asyncio
async def test_evaluate_payment_recieved_date():
    golden_response = {
        "tool_calls_nl": [
            "payment_recieved_date: Please verify the payment recieved date 10/01/2024",
        ]
    }
    model1_response = {
        "tool_calls_nl": [
            "payment_recieved_date: 10/01/2024",
        ]
    }
    model2_response = {
        "tool_calls_nl": [
            "payment_recieved_date: 10/01/2025",
        ]
    }

    result1 = await evaluate_tool_calls_nl(model1_response, golden_response)
    assert result1.correct_tool_names == 1
    assert result1.correct_queries == 1
    assert result1.incorrect_tool_names == 0
    assert result1.incorrect_queries == 0

    result2 = await evaluate_tool_calls_nl(model2_response, golden_response)
    assert result2.correct_tool_names == 1
    assert result2.correct_queries == 0
    assert result2.incorrect_tool_names == 0
    assert result2.incorrect_queries == 1
