import sys

sys.path.append("/Users/rvk7895/Projects/sarvam_org_projects/tota-benchmarking")

import asyncio
import json
import logging
import random
from argparse import ArgumentParser
from typing import Any

import httpx
from cachetools import cached
from google.genai import Client, types
from google.genai.types import Content, GenerateContentConfig, Part
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

log_file = open("llm_logs.jsonl", "w")
error_log_file = open("llm_errors.jsonl", "w")


semaphore = asyncio.Semaphore(20)


class ValidationResult(BaseModel):
    score: float
    issues: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)

    def __add__(self, other: "ValidationResult") -> "ValidationResult":
        total_issues = self.issues + other.issues
        total_score = 0 if len(total_issues) > 0 else 1
        total_feedback = self.feedback + other.feedback
        return ValidationResult(
            score=total_score,
            issues=total_issues,
            feedback=total_feedback,
        )


def _get_thread_as_str(thread) -> str:
    """Get the voice thread as a single string"""
    thread_str = ""
    for msg in thread:
        if msg["role"] == "user":
            thread_str += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            try:
                content_dict = json.loads(msg["content"])
                audio = content_dict.get("audio")
                if audio:
                    thread_str += f"Assistant: {audio}\n"
            except Exception as e:
                thread_str += f"Assistant: {msg['content']}\n"
        elif msg["role"] == "tool":
            thread_str += f"Tool: {msg['content']}\n"

    return thread_str


@cached({})
def get_gemini_client():
    api_key = "AIzaSyC8i6Lt09w7edKP75FU2FlRz6YKVrL8q-A"
    return Client(
        api_key=api_key,
    )


@retry(
    wait=wait_random_exponential(min=4, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def call_gemini(prompt: str, json_mode: bool = False) -> tuple[str, str]:
    client = get_gemini_client()

    await asyncio.sleep(random.randint(1, 10) / 10)
    contents = [Content(role="user", parts=[Part(text=prompt)])]
    if json_mode:
        config = GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
    else:
        config = GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents=contents, config=config
    )

    return response.text, response.candidates[0].content.parts[0].text


# async def call_qwen_3(prompt: str) -> str:
#     headers = {"Content-Type": "application/json"}

#     payload = {
#         "stream": False,
#         "messages": [{"role": "user", "content": prompt}],
#         "model": "Qwen/Qwen3-235B-A22B",
#         "chat_template_kwargs": {"enable_thinking": True},
#     }
#     model_url = "http://10.67.27.60:9001/v1/chat/completions"

#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             url=model_url,
#             headers=headers,
#             json=payload,
#             timeout=600,
#         )
#         response.raise_for_status()
#     try:
#         # print("MODEL OUTPUT:")
#         # print(response.json())
#         model_output = response.json()["choices"][0]["message"]["content"]
#         log_file.write(json.dumps({"prompt": prompt, "response": model_output}) + "\n")
#     except Exception as e:
#         print(f"Error calling Sarvam model: {str(e)}")
#         model_output = None

#     return model_output


LOOSE_VERIFIER_PROMPT = """
You are an critical evaluator tasked with assessing whether a bot's response follows the given instructions correctly.

<bot_instructions>
{system_prompt}
</bot_instructions>

<conversation_thread>
{thread_str}
</conversation_thread>

<bot_output>
{output_str}
</bot_output>

## How to think
Clearly understand the conversation context, and especially the last user message.
**Think deeply** over every single application instruction given in bot_instructions, and check if the output is following or violating the instruction.

Extra Evaluation Guidelines:
- Evaluate only the current turn, i.e. the bot_output, based on the instructions provided and the past thread.
- Look into past thread for context only. Do not analyse the past thread issues, as the past cannot be changed.
"""

TIGHT_VERIFIER_PROMPT = """
You are a precise evaluator assessing model responses against explicit instructions with binary scoring (0 or 1).

### SCORING RUBRIC (Binary: 0 = Fail, 1 = Pass)

A response receives **Score 1** ONLY if it passes ALL applicable criteria below. Any single failure results in **Score 0**.

## MANDATORY VALIDATION CHECKLIST

### 1. STRUCTURAL COMPLIANCE
- [ ] All required keys for the current context are present
- [ ] No prohibited key combinations exist:
  - If `tool_calls_nl`, `rag_query`,  or `transition_state` present → `audio` must be empty ("") or absent
  - If `end_interaction: true` → must include appropriate farewell audio

### 2. INSTRUCTION ADHERENCE (Hierarchical Priority)
**Priority 1 - State-Specific Instructions:**
- [ ] Current conversation state instructions are followed completely
- [ ] State transitions occur ONLY when explicitly instructed

**Priority 2 - Message-Triggered Instructions:**
- [ ] Instructions triggered by user message content are executed
- [ ] User intent is addressed appropriately (no assumptions on ambiguous messages)

**Priority 3 - General Instructions:**
- [ ] General bot guidelines are followed when not overridden by higher priority instructions

### 3. VARIABLE MANAGEMENT
- [ ] Variables updated ONLY when explicitly mentioned in applicable instructions
- [ ] All updated variables exist in system prompt's "mutable variables" section
- [ ] Variable values match restricted options (if specified)
- [ ] Values derived correctly from conversation context or user message

### 4. TOOL & RAG USAGE
- [ ] Tools called ONLY from "Available tools" section
- [ ] Tool call format matches specified pattern exactly
- [ ] RAG queries extracted correctly from `last_user_message`
- [ ] Query format follows specified pattern

### 5. CONTENT APPROPRIATENESS
- [ ] Audio content aligns with current state instructions
- [ ] User questions/concerns from `last_user_message` are addressed
- [ ] Information accuracy matches defined variables
- [ ] No internal system information revealed
- [ ] Tone and engagement level appropriate

### 6. CONTEXT AWARENESS
- [ ] Response fits current conversation state
- [ ] Conversation history properly considered
- [ ] User's demonstrated interest/disinterest acknowledged

### 7. AUDIO CONTENT
- [ ] Audio content aligns with current state instructions
- [ ] User questions/concerns from the last user message are addressed
- [ ] Information accuracy matches defined variables
- [ ] No internal system information revealed
- [ ] Tone and engagement level appropriate
- [ ] Upon ambiguous user message on the last turn, the bot asks for clarification rather than guessing or assumingthe user's intent

<bot_instructions>
{system_prompt}
</bot_instructions>

<conversation_thread>
{thread_str}
</conversation_thread>

<bot_output>
{output_str}
</bot_output>
"""


async def evaluate_loose_verifier(
    system_prompt: str, thread_str: str, responses: str
) -> dict[int, ValidationResult]:
    results: dict[int, ValidationResult] = {}

    prompt = LOOSE_VERIFIER_PROMPT.format(
        system_prompt=system_prompt,
        thread_str=thread_str,
        output_str=responses,
    )
    prompt = prompt.replace("<bot_output>", "<candidate_outputs>")
    prompt = prompt.replace("</bot_output>", "</candidate_outputs>")
    prompt += """
Evaluate each of the candidate responses carefully.
Your final output should be a single score for all input candidates in the following format:
```json
{
    "Candidate-i" : {
        "reasons": <reason>
        "score": 0/1
    },
    "Candidate-j" : {
        "reasons": <reason>
        "score": 0/1
    },
    ...
}
```
"""

    try:
        response, _ = await call_gemini(prompt, json_mode=True)
        # response = await call_qwen_3(prompt)
        if response.startswith("```json"):
            response_dict = json.loads(response.split("```json")[1].split("```")[0])
        else:
            response_dict = json.loads(response)
    except Exception as e:
        error_log_file.write(
            json.dumps({"prompt": prompt, "response": response, "error": str(e)}) + "\n"
        )
        logging.exception(f"Error calling Qwen3: {e}")
        return results

    for candidate_id, output in response_dict.items():
        try:
            id = int(candidate_id.split("-")[1])
            structure_result = results.get(
                id, ValidationResult(score=0, issues=[], feedback=[])
            )
            if output["score"] == 0:
                results[id] = ValidationResult(
                    score=0,
                    issues=structure_result.issues + [output["reasons"]],
                    feedback=structure_result.feedback,
                )
            else:
                results[id] = ValidationResult(
                    score=1,
                    issues=structure_result.issues,
                    feedback=structure_result.feedback + [output["reasons"]],
                )
        except Exception as e:
            logging.info(f"Judge response: {response}")
            logging.exception(
                f"Error extracting result for candidate {candidate_id}: {e}"
            )

    return results


async def evaluate_tight_verifier(
    system_prompt: str, thread_str: str, responses: str
) -> dict[int, ValidationResult]:
    results: dict[int, ValidationResult] = {}

    prompt = TIGHT_VERIFIER_PROMPT.format(
        system_prompt=system_prompt,
        thread_str=thread_str,
        output_str=responses,
    )

    prompt = prompt.replace("<bot_output>", "<candidate_outputs>")
    prompt = prompt.replace("</bot_output>", "</candidate_outputs>")
    prompt += """
Evaluate each of the candidate responses carefully.
Your final output should be a single score for all input candidates in the following format:
```json
{
    "Candidate-i" : {
        "reasons": <reason>
        "score": 0/1
    },
    "Candidate-j" : {
        "reasons": <reason>
        "score": 0/1
    },
    ...
}
```
"""

    try:
        # response, _ = await call_gemini(prompt, json_mode=True)
        response = await call_qwen_3(prompt)
        response_dict = json.loads(response)
    except Exception as e:
        logging.exception(f"Error calling Gemini: {e}")
        return results

    for candidate_id, output in response_dict.items():
        try:
            id = int(candidate_id.split("-")[1])
            structure_result = results.get(
                id, ValidationResult(score=0, issues=[], feedback=[])
            )
            if output["score"] == 0:
                results[id] = ValidationResult(
                    score=0,
                    issues=structure_result.issues + [output["reasons"]],
                    feedback=structure_result.feedback,
                )
            else:
                results[id] = ValidationResult(
                    score=1,
                    issues=structure_result.issues,
                    feedback=structure_result.feedback + [output["reasons"]],
                )
        except Exception as e:
            logging.info(f"Judge response: {response}")
            logging.exception(
                f"Error extracting result for candidate {candidate_id}: {e}"
            )

    return results


async def llm_judge_evaluation(
    system_prompt: str,
    thread_str: str,
    responses: str,
    verifier_type: str,
    responses_list: list[str],
) -> tuple[dict[int, ValidationResult], str, str, list[str]]:
    async with semaphore:
        if verifier_type == "loose":
            return (
                await evaluate_loose_verifier(system_prompt, thread_str, responses),
                system_prompt,
                thread_str,
                responses_list,
            )
        elif verifier_type == "tight":
            return (
                await evaluate_tight_verifier(system_prompt, thread_str, responses),
                system_prompt,
                thread_str,
                responses_list,
            )
        else:
            raise ValueError(f"Invalid verifier type: {verifier_type}")


def analyze_results(
    results: list[tuple[dict[int, ValidationResult], str, str, list[str]]],
    model_candidate_number_map: dict[int, str],
) -> dict[str, dict[str, float]]:
    model_results = {
        v: {"score": 0.0, "total": 0, "pass_rate": 0.0}
        for k, v in model_candidate_number_map.items()
    }

    for result in results:
        for candidate_id, validation_result in result[0].items():
            model_name = model_candidate_number_map[candidate_id]
            model_results[model_name]["score"] += validation_result.score
            model_results[model_name]["total"] += 1

    for model_name, model_result in model_results.items():
        total = model_result["total"]
        if total == 0:
            model_result["pass_rate"] = 0.0
        else:
            model_result["pass_rate"] = model_result["score"] / total * 100

    return model_results


def dump_results(
    model_results: dict[str, dict[str, float]],
    sample_results: list[tuple[dict[int, ValidationResult], str, str, list[str]]],
    model_candidate_number_map: dict[int, str],
    verifier_type: str,
):
    final_results: dict[str, Any] = {}
    final_results["model_results"] = model_results
    final_results["sample_results"] = []

    for sample_result, system_prompt, thread_str, responses_list in sample_results:
        sample_result_dict: dict[str, Any] = {
            "system_prompt": system_prompt,
            "thread_str": thread_str,
            "model_results": {},
        }
        for candidate_id, validation_result in sample_result.items():
            model_name = model_candidate_number_map[candidate_id]
            sample_result_dict["model_results"][model_name] = {
                "model_response": responses_list[candidate_id - 1],
                "score": validation_result.score,
                "issues": validation_result.issues,
                "feedback": validation_result.feedback,
            }
        final_results["sample_results"].append(sample_result_dict)

    json.dump(final_results, open(f"results_{verifier_type}_gemini_2_5_flash.json", "w"))


async def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--verifier_type", type=str, required=True)
    parser.add_argument("--benchmark_file", type=str, required=False)
    args = parser.parse_args()

    # config = load_config(args.config)
    # models_config = config["models_config"]

    samples = {}
    if args.benchmark_file:
        benchmark_results = json.load(open(args.benchmark_file))
        samples = benchmark_results["samples"]

    tasks = []
    model_candidate_number_map: dict[int, str] = {}

    responses = ""
    for sample_id, sample in samples.items():
        responses_list = []
        system_prompt = sample["input_messages"][0]["content"]
        thread_str = _get_thread_as_str(sample["input_messages"])
        # Ensure the mapping for the golden response is always consistent (Candidate-1)
        if 1 not in model_candidate_number_map:
            model_candidate_number_map[1] = "golden_response"

        responses = f"Candidate-1:\n{sample['golden_response']}\n\n"
        responses_list.append(sample["golden_response"])
        candidate_counter = 2
        for model_name, model_result in sample["model_results"].items():
            responses += (
                f"Candidate-{candidate_counter}:\n{model_result['model_response']}\n\n"
            )
            if candidate_counter not in model_candidate_number_map:
                model_candidate_number_map[candidate_counter] = model_name
            responses_list.append(model_result["model_response"])
            candidate_counter += 1

        tasks.append(
            llm_judge_evaluation(
                system_prompt, thread_str, responses, args.verifier_type, responses_list
            )
        )

    results = []
    p_bar = tqdm(total=len(tasks), desc="Evaluating samples")

    for result in asyncio.as_completed(tasks):
        result = await result
        if result:
            results.append(result)
        p_bar.update(1)

    p_bar.close()

    model_results = analyze_results(results, model_candidate_number_map)
    dump_results(
        model_results,
        results,
        model_candidate_number_map,
        args.verifier_type,
    )


if __name__ == "__main__":
    asyncio.run(main())
