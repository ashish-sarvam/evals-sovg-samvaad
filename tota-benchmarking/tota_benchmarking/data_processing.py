"""
Module for loading and processing benchmark datasets.
"""

import asyncio
import copy
import json
import logging
from typing import Any, Callable, Dict, Optional

from datasets import Dataset, load_dataset

from .evaluator import evaluate_response, process_evaluation_results
from .models import (
    call_anthropic_vertex_model,
    call_azure_model,
    call_gemini_model,
    call_groq_model,
    call_openai_model,
    call_sarvam_model,
)
from .prompt_converters import (
    convert_chat_thread_for_model,
    system_prompt_renderer,
    tota_v8_chat_thread_adapter,
    tota_v9_chat_thread_adapter,
)
from .prompts.converter import convert_system_prompt_to_structured

# Configure logging
logger = logging.getLogger(__name__)

# Suppress verbose logs from HTTP and API-related modules
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def load_benchmark_dataset(
    dataset_name: str, revision: str, token: Optional[str] = None
) -> Any:
    """
    Load the benchmark dataset from Hugging Face.

    Args:
        dataset_name: The name of the dataset on Hugging Face.
        revision: The revision of the dataset to use.
        token: Optional Hugging Face token for private datasets.

    Returns:
        The loaded dataset.

    Raises:
        Exception: If dataset loading fails.
    """
    try:
        if dataset_name.endswith(".jsonl"):
            with open(dataset_name, "r") as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list(data)
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                download_mode="force_redownload",
                revision=revision,
                token=token,
            )
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        logger.exception(e)
        raise


async def process_sample(
    sample: Dict[str, Any],
    idx: int,
    model_name: str,
    model_path: str,
    model_url: str,
    model_provider: str,
    base_model_type: str,
    llm_config: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_adapter: Callable,
    system_prompt_template_name: str,
    model_structured: bool,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    Process a single sample with the specified model.

    Args:
        sample: The sample from the dataset.
        idx: The index of the sample.
        model_name: The name of the model.
        model_path: The path to the model.
        model_url: The URL to the model API.
        model_provider: The provider of the model (SARVAM or AZURE).
        base_model_type: The base model type.
        llm_config: Configuration for the LLM.
        model_config: Model-specific configuration.
        conversation_adapter: Function to adapt the conversation thread.
        system_prompt_template_name: The name of the system prompt template.
        model_structured: Whether to use structured prompts.
        semaphore: Semaphore for concurrency control.

    Returns:
        Dict with the evaluation results.
    """
    async with semaphore:
        try:

            # Get the golden response using the conversation adapter
            sample_outputs = sample["output"]
            if isinstance(sample_outputs, str):
                sample_outputs = [sample_outputs]

            chat_thread = sample["chat_thread"]
            if "system_prompt" in sample.keys():
                system_message = sample["system_prompt"]
                if "Tota v8" in model_name:
                    system_message = tota_v8_chat_thread_adapter(
                        [{"role": "system", "content": system_message}], idx
                    )[0]["content"]
                if "Tota v9" in model_name:
                    system_message = tota_v9_chat_thread_adapter(
                        [{"role": "system", "content": system_message}], idx
                    )[0]["content"]
                if model_structured:
                    system_message = copy.deepcopy(
                        convert_system_prompt_to_structured(system_message)
                    )

                updated_chat_thread = []
                for turn in chat_thread:
                    turn_copy = copy.deepcopy(turn)
                    if turn_copy["role"] == "assistant":
                        if model_structured:
                            try:
                                turn_content_dict = json.loads(turn_copy["content"])
                                fixed_dict = {"lines": [], **turn_content_dict}
                                turn_copy["content"] = json.dumps(fixed_dict)
                            except Exception as e:
                                pass
                    updated_chat_thread.append(turn_copy)
                golden_responses = sample_outputs
            else:
                system_message = system_prompt_renderer(
                    system_prompt_template_name,
                    sample["current_language"],
                    sample["languages_available"],
                    sample["enable_agentic_lid"],
                    sample["global_prompt"],
                    sample["instructions"],
                    sample["kb_details"],
                    sample["immutable_variables"],
                    sample["mutable_variables_with_values"],
                    sample["current_state_name"],
                    sample["next_states"],
                    sample["tools"],
                    sample["needs_tool_calls_nl"],
                    sample["state_transition"],
                    sample["variables"],
                )
                golden_responses = [
                    conversation_adapter(
                        [{"role": "assistant", "content": sample_output}], idx
                    )[0]["content"]
                    for sample_output in sample_outputs
                ]
                updated_chat_thread = conversation_adapter(chat_thread, idx)

            updated_chat_thread = convert_chat_thread_for_model(
                updated_chat_thread, base_model_type
            )
            updated_conversation_thread = [
                {"role": "system", "content": system_message}
            ] + updated_chat_thread

            # print(json.dumps(updated_conversation_thread, indent=4))

            inference_time = 0
            # Call the appropriate model API
            if model_provider == "SARVAM":
                api_key = model_config.get("api_key")
                model_response, inference_time = await call_sarvam_model(
                    updated_conversation_thread,
                    model_path,
                    model_url,
                    llm_config,
                    api_key,
                )
            elif model_provider == "AZURE":
                model_response, inference_time = await call_azure_model(
                    updated_conversation_thread,
                    model_config.get("deployment", ""),
                    model_config.get("endpoint", ""),
                    model_config.get("api_key", ""),
                    llm_config,
                )
            elif model_provider == "OPENAI":
                model_response, inference_time = await call_openai_model(
                    updated_conversation_thread,
                    model_path,
                    model_config.get("api_key", ""),
                    llm_config,
                )
            elif model_provider == "ANTHROPIC_VERTEX":
                model_response, inference_time = await call_anthropic_vertex_model(
                    updated_conversation_thread,
                    model_path,
                    model_config.get("project_id", ""),
                    model_config.get("region", ""),
                    llm_config,
                )
            elif model_provider == "GEMINI":
                model_response, inference_time = await call_gemini_model(
                    updated_conversation_thread,
                    model_path,
                    model_config.get("api_key", ""),
                    llm_config,
                )
            elif model_provider == "GROQ":
                model_response, inference_time = await call_groq_model(
                    updated_conversation_thread,
                    model_path,
                    model_config.get("api_key", ""),
                    llm_config,
                )
            else:
                logger.error(f"Unknown provider {model_provider}")
                return {
                    "idx": idx,
                    "has_failed": True,
                    "error": f"Unknown provider {model_provider}",
                    "model_response": "",
                    "golden_response": "",
                    "evaluation_results": {
                        "error": f"Unknown provider {model_provider}"
                    },
                    "updated_conversation_thread": [],
                    "inference_time": inference_time,
                    "is_adequate": False,
                }
            if not model_response:
                logger.error(
                    f"No response received for sample {idx} with model {model_name}"
                )
                return {
                    "idx": idx,
                    "has_failed": True,
                    "error": "No model response received",
                    "model_response": "",
                    "golden_response": "",
                    "evaluation_results": {"error": "No model response received"},
                    "updated_conversation_thread": [],
                    "inference_time": inference_time,
                    "is_adequate": False,
                }

            # Evaluate the model response
            evaluator_results = await evaluate_response(
                model_response,
                golden_responses[0],
                system_message,
                chat_thread[-1]["content"],
            )
            # Process evaluation results
            processed_results = process_evaluation_results(evaluator_results)
            is_adequate = False
            if not processed_results["has_failed"]:
                is_adequate = True
            else:
                for candidate in golden_responses[1:]:
                    candidate_evaluator_results = await evaluate_response(
                        model_response,
                        candidate,
                        system_message,
                        chat_thread[-1]["content"],
                    )
                    candidate_processed_results = process_evaluation_results(
                        candidate_evaluator_results
                    )
                    if not candidate_processed_results["has_failed"]:
                        is_adequate = True
                        break

            # Get cleaned response from evaluation results
            cleaned_model_response = evaluator_results.get("cleaned_model_response", "")
            cleaned_golden_response = evaluator_results.get(
                "cleaned_golden_response", ""
            )

            return {
                "idx": idx,
                "has_failed": processed_results.get("has_failed", True),
                "model_response": cleaned_model_response,
                "golden_response": cleaned_golden_response,
                "evaluation_results": processed_results.get("evaluation_results", {}),
                "updated_conversation_thread": updated_conversation_thread,
                "inference_time": inference_time,
                "is_adequate": is_adequate,
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx} with model {model_name}: {e}")
            logger.exception(e)
            return {
                "idx": idx,
                "has_failed": True,
                "error": str(e),
                "model_response": "",
                "golden_response": "",
                "evaluation_results": {"error": str(e)},
                "updated_conversation_thread": [],
                "inference_time": 0,
                "is_adequate": False,
            }
