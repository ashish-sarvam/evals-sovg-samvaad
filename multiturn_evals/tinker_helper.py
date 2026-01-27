"""
Tinker Model wrapper for use in generate_transcript.py

Provides a simple interface to get responses from Tinker fine-tuned models.
"""

import os
import re
from pathlib import Path
import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import get_text_content
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def extract_harmony_content(raw_text: str) -> str:
    """
    Extract the first 'final' channel message from Harmony format output.

    The model outputs:
    <|channel|>final<|message|>CONTENT<|end|><|start|>assistant...

    We extract just the first CONTENT.
    """
    # Look for the first final channel message
    match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)<\|end\|>',
        raw_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # Fallback: try to get any message content
    match = re.search(r'<\|message\|>(.*?)<\|end\|>', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return cleaned text
    return raw_text.strip()


# Default configurations
DEFAULT_TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_MODEL_PATH = (
    "tinker://2328525d-d99b-5b75-aaa1-779c1aa7db38:train:0/sampler_weights/final"
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

        params = types.SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=self.renderer.get_stop_sequences(),
        )

        future = self.sampling_client.sample(
            prompt=prompt, sampling_params=params, num_samples=1
        )
        result = future.result()

        if result.sequences:
            tokens = result.sequences[0].tokens
            raw_text = self.tokenizer.decode(tokens)
            # Extract just the first message content from harmony format
            parsed = extract_harmony_content(raw_text)
            return parsed

        return ""
