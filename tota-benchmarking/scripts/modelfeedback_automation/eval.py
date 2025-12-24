import json
import argparse
import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


load_dotenv()  # This will read from .env file

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_URL = os.getenv("AZURE_OPENAI_URL")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")


_client = None


def get_cached_azureopenai_client():
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_URL,
            default_headers={"Ocp-Apim-Subscription-Key": AZURE_OPENAI_API_KEY},
        )
    return _client


async def get_openai_response(prompt: str, model: str) -> str:
    try:
        client = get_cached_azureopenai_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Azure OpenAI Error] {str(e)}"


def filter_user_bot_messages(messages: List[Dict]) -> List[Dict]:
    return [msg for msg in messages if msg["role"] in ("user", "assistant", "tool")]


async def process_batch(batch: List[Dict], model_deployment: str) -> List[Dict]:
    """Process a single batch of failed samples."""
    # Build batch prompt
    prompt = (
        "Below are several failed bot responses. For each, briefly explain "
        "what went wrong in 1–2 sentences. Return a JSON list with "
        "'sample_id' and 'what_went_wrong'.\n\n"
    )
    prompt += "### Input:\n[\n"
    for item in batch:
        prompt += (
            json.dumps(
                {
                    "sample_id": item["sample_id"],
                    "conversation": item["conversation"],
                    "golden_response": item["golden_response"],
                    "model_response": item["model_response"],
                },
                indent=2,
            )
            + ",\n"
        )
    prompt += "]\n\n### Output:\n"

    # Get response
    response_str = await get_openai_response(prompt, model=model_deployment)

    try:
        parsed = json.loads(response_str)
        if isinstance(parsed, list):
            return parsed
        else:
            print(f"[Batch Parse Error] Expected list, got: {type(parsed)}")
            return []
    except Exception as e:
        print(f"[Batch Parse Error] Could not parse:\n{response_str}\n\n" f"Error: {e}")
        return []


async def analyze_failures(
    samples: Dict, model_name: str, model_deployment: str, batch_size: int = 5
) -> List[Dict]:
    analyses = []
    failed_samples = []

    # Collect all failed samples
    for sample_id, sample in samples.items():
        model_result = sample["model_results"].get(model_name)
        if model_result and model_result.get("has_failed"):
            conversation = filter_user_bot_messages(sample["input_messages"])
            golden_response = sample["golden_response"]
            model_response = model_result["model_response"]
            failed_samples.append(
                {
                    "sample_id": sample_id,
                    "conversation": conversation,
                    "golden_response": golden_response,
                    "model_response": model_response,
                }
            )

    print(f"Number of failed samples: {len(failed_samples)}")

    # Create batches
    batches = []
    for i in range(0, len(failed_samples), batch_size):
        batch = failed_samples[i : i + batch_size]
        batches.append(batch)

    # Process all batches in parallel
    print(
        f"Processing {len(batches)} batches with batch size {batch_size} in parallel..."
    )
    batch_results = await asyncio.gather(
        *[process_batch(batch, model_deployment) for batch in batches]
    )

    # Combine results from all batches
    for batch_result in batch_results:
        analyses.extend(batch_result)

    return analyses


async def generate_human_notes(
    failed_analyses: List[Dict], model_deployment: str
) -> str:
    """Generate human-like summary notes."""
    prompt = """You are an expert conversation evaluator reviewing a model's performance across a set of conversation samples with the user(the expected response vs the model response). You've been given a structured JSON breakdown of sample IDs and how the model went wrong in each.

Your task is to rewrite this evaluation in the style of a human reviewer's notes — concise, readable, and grouped by similar issues.

Here's what to aim for:
- Group sample IDs together when the issue is the same or highly similar.
- Use brief, informal explanations that still make the issue clear. The issue should be clear and understandable.
- Use bulleted or numbered lines, with each line starting with the sample ID(s), followed by a short summary of what went wrong.
- Avoid repeating the same explanation across many lines — consolidate wherever you can.
- Do not include model names, JSON formatting, or full sentences unless necessary.

Here's an example of the target format:
95, 148 – Language change tool call failed.
32, 60, 65, 73, 83 – Assumes user has seen the WhatsApp group; should have clarified
120 – Assumes user is salaried based on ambiguous input.
141, 145, 167 – Doesn’t clarify ambiguous response and ends the conversation.

Important instructions:
- Only include a sample ID in the summary if it clearly fits the issue category. Do not force every sample into a group.
- If a failure is unique or doesn't match others closely, leave it out of the summary.
- Your goal is not to account for every sample, but to identify the most common, clearly repeated error patterns.
- Focus on major failure modes that recur and can be actioned — ignore one-off issues.
- Each bullet should describe a coherent failure category, not a loose theme.

Now, here's the evaluation input for you to rewrite:"""

    prompt += f"\n{json.dumps(failed_analyses, indent=2)}\n"

    human_notes = await get_openai_response(prompt, model_deployment)
    return human_notes


# Example usage:
#
# python eval.py path/to/benchmark.json "Tota v8 (Qwen3-32B-opg1-sft)" --deployment_name my-azure-deployment --output_md_path results.md
#
# Required positional arguments:
#   json_path: Path to the benchmark JSON file (e.g., optimizations/benchmark.json)
#   model_name: Name of the model to analyze (e.g., 'Tota v8 (Qwen3-32B-opg1-sft)')
#
# Optional arguments:
#   --deployment_name: Azure OpenAI deployment name (default: value of AZURE_OPENAI_MODEL)
#   --output_md_path: Path to output Markdown file (default: markdown_output.md)


async def async_main():
    parser = argparse.ArgumentParser(
        description="LLM evaluation of bot behavior and failure analysis."
    )
    parser.add_argument("json_path", help="Path to the JSON file")
    parser.add_argument(
        "model_name",
        help="Model name to analyze (e.g., 'Tota v8 (Qwen3-32B-opg1-sft)')",
    )
    parser.add_argument(
        "--deployment_name",
        default=AZURE_OPENAI_MODEL,
        help="Azure OpenAI deployment name",
    )
    parser.add_argument(
        "--output_md_path",
        default="markdown_output.md",
        help="Path to output Markdown file",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    model_name = args.model_name
    model_deployment = args.deployment_name
    output_md_path = args.output_md_path

    print(f"\n🔍 Analyzing failed cases for: {model_name}\n")
    failed_analyses = await analyze_failures(samples, model_name, model_deployment)

    print(f"\n🔍 Generating summarized failure points for: {model_name}\n")
    human_notes = await generate_human_notes(failed_analyses, model_deployment)

    # Write results to Markdown file
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write("# FAILURE ANALYSIS\n\n")
        md_file.write(str(human_notes).strip() + "\n\n")
        md_file.write("# DETAILED FAILURE NOTES\n\n")
        md_file.write(json.dumps(failed_analyses, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(async_main())
