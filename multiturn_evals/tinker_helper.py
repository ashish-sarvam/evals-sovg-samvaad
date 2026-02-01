"""
Tinker Model wrapper for use in generate_transcript.py

Provides a simple interface to get responses from Tinker fine-tuned models.
"""

import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Optional

# Add tinker-cookbook to path
_TINKER_COOKBOOK_PATH = Path(__file__).parent / "tinker-cookbook"
if str(_TINKER_COOKBOOK_PATH) not in sys.path:
    sys.path.insert(0, str(_TINKER_COOKBOOK_PATH))

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import get_text_content


def extract_harmony_content(raw_text: str) -> str:
    """
    Extract the first 'final' channel message from Harmony format output.

    The model outputs:
    <|channel|>final<|message|>CONTENT<|end|><|start|>assistant...

    We extract just the first CONTENT.
    """
    # Look for the first final channel message
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # Fallback: try to get any message content
    match = re.search(r"<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return cleaned text
    return raw_text.strip()


# Default configurations
DEFAULT_TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_MODEL_PATH = (
    "tinker://ec0153b2-e647-5497-9232-2e3258f76c95:train:0/sampler_weights/final"
)
DEFAULT_RENDERER = "gpt_oss_no_sysprompt"


class TinkerModel:
    """A wrapper around Tinker's sampling client."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: str = DEFAULT_MODEL_PATH,
        renderer_name: str = DEFAULT_RENDERER,
        api_key: str = DEFAULT_TINKER_API_KEY,
        max_tokens: int = 500,
        temperature: float = 0.5,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize Tinker clients
        self.service_client = tinker.ServiceClient(api_key=api_key)
        self.sampling_client = self.service_client.create_sampling_client(
            model_path=model_path
        )

        # Get tokenizer and renderer
        self.tokenizer = get_tokenizer(model_name)
        self.renderer = get_renderer(renderer_name, self.tokenizer)

    def get_response(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Get a response from the Tinker model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            The assistant's response text
        """
        prompt = self.renderer.build_generation_prompt(messages)

        # Get renderer stop sequences and add <|end|> token
        # The model may output <|end|> instead of <|return|> for final responses
        stop_sequences = self.renderer.get_stop_sequences()
        end_token = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        if len(end_token) == 1 and end_token[0] not in stop_sequences:
            stop_sequences = stop_sequences + end_token

        params = types.SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=stop_sequences,
        )

        # Retry logic for transient errors
        max_retries = 10
        delay = 1.0
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                future = self.sampling_client.sample(
                    prompt=prompt, sampling_params=params, num_samples=1
                )
                result = future.result()

                if result.sequences:
                    tokens = result.sequences[0].tokens
                    raw_text = self.tokenizer.decode(tokens)
                    print(f"Result: {raw_text}")
                    # Extract just the first message content from harmony format
                    parsed = extract_harmony_content(raw_text)
                    return parsed

                return ""
            
            except Exception as exc:
                last_exception = exc
                error_msg = str(exc).lower()
                
                # Check if retryable
                retryable_indicators = [
                    "rate limit", "timeout", "connection", "unavailable",
                    "502", "503", "504", "500", "overloaded", "capacity"
                ]
                is_retryable = any(ind in error_msg for ind in retryable_indicators)
                
                if attempt >= max_retries or not is_retryable:
                    print(f"[Retry] {'All retries exhausted' if attempt >= max_retries else 'Non-retryable error'}: {exc}")
                    raise
                
                jitter = random.uniform(0.5, 1.5)
                actual_delay = min(delay * jitter, 60.0)
                print(f"[Retry] Attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {actual_delay:.2f}s...")
                time.sleep(actual_delay)
                delay = min(delay * 2.0, 60.0)
        
        if last_exception:
            raise last_exception
        return ""


def main():
    """Quick test function for TinkerModel."""
    print("Initializing TinkerModel...")
    model = TinkerModel(
        max_tokens=1500,
        temperature=0.8,
    )

    messages = [
        {
            "role": "system",
            "content": "You are debt collection agent for IDFC bank, talk to user in hindi with mix of english. Imagine like a rural user, so talk in formal PSU style language. Your goal is to remind them to pay their due by tomorrow. You have to total nudge them 3 times, one by one. To nudge you can induce fear that there credit score will go down or you can say their will be penalties or you can say finally that they might face criminal charges. Overall be empathatetic.",
        },
        {"role": "user", "content": "Why are you calling me?"},
    ]

    print("\n" + "=" * 60)
    print("MESSAGES:")
    print("=" * 60)
    for msg in messages:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")

    print("\n" + "=" * 60)
    print("GENERATING RESPONSE...")
    print("=" * 60)

    response = model.get_response(messages)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
