"""
LLM integration for verifier - supports both Gemini and DeepSeek.

Pass provider parameter to switch:
  - "gemini": Uses Gemini via main LLM module
  - "deepseek": Uses DeepSeek via direct HTTP API
"""

import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path for direct execution
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from conv_data_gen.config import config  # noqa: E402
from conv_data_gen.llm.llm import LLMClient, ProviderType  # noqa: E402

# ============================================================================
# DeepSeek Configuration
# ============================================================================
DEEPSEEK_API_URL = (
    "https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions"
)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-v3-2-251201"

# Retry config
MAX_RETRIES = 10
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds

# ============================================================================
# Gemini Client (singleton)
# ============================================================================
_gemini_client: Optional[LLMClient] = None


def _get_gemini_client() -> LLMClient:
    """Get or create the Gemini LLM client (singleton pattern)."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = LLMClient(provider=ProviderType.GEMINI)
    return _gemini_client


# ============================================================================
# DeepSeek Implementation
# ============================================================================
def _get_deepseek_response(
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = MAX_RETRIES,
) -> str:
    """Get response from DeepSeek API with retries."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"[DeepSeek] Requesting with model: {model}")
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "medium",
                },
                timeout=120,
            )

            # Check for rate limit or server errors
            if response.status_code == 429:
                raise Exception("Rate limited (429)")
            if response.status_code >= 500:
                raise Exception(f"Server error ({response.status_code})")

            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if "choices" not in data or not data["choices"]:
                raise Exception("Invalid response: no choices")

            print(f"[DeepSeek][Attempt {attempt + 1}] Success")
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                jitter = random.uniform(0, delay * 0.1)
                sleep_time = delay + jitter
                print(
                    f"[DeepSeek][Attempt {attempt + 1}] Failed: {e}. "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            else:
                print(
                    f"[DeepSeek][Attempt {attempt + 1}] Failed: {e}. "
                    "No more retries."
                )

    raise Exception(
        f"All {max_retries} attempts failed. Last error: {last_exception}"
    )


# ============================================================================
# Gemini Implementation
# ============================================================================
def _get_gemini_response(
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """Get response from Gemini via main LLM module."""
    client = _get_gemini_client()

    # Extract system prompt if present, otherwise use default
    system_prompt = (
        "You are an expert verifier. "
        "Verify and give the output in the given schema."
    )
    user_messages = messages

    # Check if first message is system
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", system_prompt)
        user_messages = messages[1:]

    result = client.get_llm_response_json(
        messages=user_messages,
        model=config.models.GEMINI_FLASH_2_5,
        system_prompt=system_prompt,
        max_tokens=4000,
    )

    return result.get("text", "")


# ============================================================================
# Public API
# ============================================================================
def get_llm_response(
    model: str,
    messages: List[Dict[str, str]],
    provider: str = "gemini",
) -> str:
    """
    Get LLM response using specified provider.

    Args:
        model: Model name (used for DeepSeek, ignored for Gemini)
        messages: List of message dicts with role and content
        provider: "gemini" or "deepseek"

    Returns:
        The text response from the LLM
    """
    if provider.lower() == "deepseek":
        return _get_deepseek_response(model, messages)
    else:
        return _get_gemini_response(model, messages)


def get_llm_responses_parallel(
    model: str,
    messages_list: List[List[Dict[str, str]]],
    max_workers: int = 5,
    provider: str = "gemini",
) -> List[Optional[str]]:
    """
    Process multiple message lists in parallel.

    Args:
        model: Model name (used for DeepSeek, ignored for Gemini)
        messages_list: List of message lists, each will be sent as a
            separate request
        max_workers: Maximum number of parallel workers
        provider: "gemini" or "deepseek"

    Returns:
        List of responses in the same order as the input messages_list
    """
    results: List[Optional[str]] = [None] * len(messages_list)

    # Concurrency fix: don't use more workers than items
    if messages_list:
        actual_workers = min(max_workers, len(messages_list))
    else:
        actual_workers = 1

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_index = {
            executor.submit(get_llm_response, model, messages, provider): i
            for i, messages in enumerate(messages_list)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = f"Error: {e}"

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test verifier LLM")
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "deepseek"],
        help="LLM provider to use",
    )
    args = parser.parse_args()

    print(f"Using provider: {args.provider}\n")

    # Quick test
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": 'What is 2+2? Reply in JSON: {"answer": <number>}',
        },
    ]

    print("Testing single request...")
    response = get_llm_response(DEEPSEEK_MODEL, test_messages, args.provider)
    print(f"Response: {response}")

    print("\nTesting parallel requests...")
    messages_list = [
        [{"role": "user", "content": "What is 1+1? Reply briefly."}],
        [{"role": "user", "content": "What is 2+2? Reply briefly."}],
        [{"role": "user", "content": "What is 3+3? Reply briefly."}],
    ]
    responses = get_llm_responses_parallel(
        DEEPSEEK_MODEL, messages_list, max_workers=3, provider=args.provider
    )
    for i, resp in enumerate(responses):
        print(f"Response {i + 1}: {resp}")
