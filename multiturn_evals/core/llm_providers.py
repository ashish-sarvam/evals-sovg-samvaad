"""
Generic LLM integration module for multilingual evals.

Supports multiple providers: OpenAI, Azure OpenAI, Gemini (Vertex AI),
Anthropic, Tinker, Sarvam, Lepton.
"""

import json
import os
import re
import time
import random
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from openai import OpenAI, AzureOpenAI


# Retry configuration
MAX_RETRIES = 10
INITIAL_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds
BACKOFF_MULTIPLIER = 2.0
JITTER_RANGE = 0.5  # +/- 50% jitter

# Exception types to retry on
RETRYABLE_EXCEPTIONS = (
    Exception,  # Catch all for now, can be more specific
)

# Strings in error messages that indicate rate limiting
RATE_LIMIT_INDICATORS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "429",
    "quota exceeded",
    "quota_exceeded",
    "resource exhausted",
    "resource_exhausted",
    "overloaded",
    "capacity",
    "throttl",
    "retry after",
    "retry-after",
]


T = TypeVar("T")


def is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit or transient error)."""
    error_msg = str(exc).lower()
    
    # Check for rate limit indicators
    for indicator in RATE_LIMIT_INDICATORS:
        if indicator in error_msg:
            return True
    
    # Check for common transient errors
    transient_indicators = [
        "timeout",
        "connection",
        "temporary",
        "unavailable",
        "502",
        "503",
        "504",
        "internal server error",
        "500",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ]
    for indicator in transient_indicators:
        if indicator in error_msg:
            return True
    
    return False


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_DELAY,
    max_delay: float = MAX_DELAY,
    backoff_multiplier: float = BACKOFF_MULTIPLIER,
    jitter_range: float = JITTER_RANGE,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 10)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        backoff_multiplier: Multiplier for each retry (default: 2.0)
        jitter_range: Random jitter range as fraction (default: 0.5 = +/-50%)
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    
                    # Check if this is the last attempt
                    if attempt >= max_retries:
                        print(f"[Retry] All {max_retries} retries exhausted for {func.__name__}")
                        raise
                    
                    # Check if error is retryable
                    if not is_retryable_error(exc):
                        print(f"[Retry] Non-retryable error in {func.__name__}: {exc}")
                        raise
                    
                    # Calculate delay with jitter
                    jitter = random.uniform(1 - jitter_range, 1 + jitter_range)
                    actual_delay = min(delay * jitter, max_delay)
                    
                    print(
                        f"[Retry] Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {exc}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )
                    
                    time.sleep(actual_delay)
                    delay = min(delay * backoff_multiplier, max_delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")
        
        return wrapper
    return decorator


def llm_call_with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    **kwargs
) -> T:
    """
    Execute an LLM call with retry logic.
    
    This is a helper function for cases where you can't use the decorator.
    
    Args:
        func: The function to call
        *args: Positional arguments for the function
        max_retries: Maximum retry attempts
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
    """
    last_exception = None
    delay = INITIAL_DELAY
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            
            if attempt >= max_retries:
                print(f"[Retry] All {max_retries} retries exhausted")
                raise
            
            if not is_retryable_error(exc):
                print(f"[Retry] Non-retryable error: {exc}")
                raise
            
            jitter = random.uniform(1 - JITTER_RANGE, 1 + JITTER_RANGE)
            actual_delay = min(delay * jitter, MAX_DELAY)
            
            print(
                f"[Retry] Attempt {attempt + 1}/{max_retries} failed: {exc}. "
                f"Retrying in {actual_delay:.2f}s..."
            )
            
            time.sleep(actual_delay)
            delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")

from core.config import (
    AZURE_CONFIG,
    OPENAI_CONFIG,
    LEPTON_CONFIG,
    SARVAM_PRAVAH_CONFIG,
    GEMINI_CONFIG,
    MODEL_SETTINGS,
)


class ProviderType(str, Enum):
    """Supported LLM providers."""

    TINKER = "tinker"
    AZURE = "azure"
    OPENAI = "openai"
    LEPTON = "lepton"
    SARVAM = "sarvam"
    GEMINI = "gemini"
    GEMINI_VERTEX = "gemini_vertex"


@dataclass
class LLMMetadata:
    """Standardized metadata returned from all LLM providers."""

    provider: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseLLMProvider(ABC):
    """Common interface for different LLM providers."""

    def __init__(
        self,
        temperature: float,
        max_tokens: int,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Get response text from the provider."""


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: Optional[str] = None,
    ):
        super().__init__(temperature, max_tokens)
        self.model = model

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return "OpenAI"

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "model": self.model,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            print(f"Warning: Empty response from {self.model} (finish_reason: {finish_reason})")
            return "[Model returned empty response]"
        return content


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI API provider."""

    def __init__(
        self,
        deployment: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        endpoint: str,
        api_version: str,
    ):
        super().__init__(temperature, max_tokens)
        self.deployment = deployment

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    @property
    def name(self) -> str:
        return "AzureOpenAI"

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {
            "messages": messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
            "model": self.deployment,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            print(f"Warning: Empty response from {self.deployment} (finish_reason: {finish_reason})")
            return "[Model returned empty response]"
        return content


class GeminiProvider(BaseLLMProvider):
    """Gemini provider via Google AI Studio (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: str,
    ):
        super().__init__(temperature, max_tokens)
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return "Gemini"

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "model": self.model,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            print(f"Warning: Empty response from {self.model} (finish_reason: {finish_reason})")
            return "[Model returned empty response]"
        return content


class GeminiVertexProvider(BaseLLMProvider):
    """Gemini provider via Vertex AI (native SDK)."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        project_id: str,
        location: str,
        thinking_budget: Optional[int] = None,
    ):
        super().__init__(temperature, max_tokens)
        self.model_name = model
        self.thinking_budget = thinking_budget

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=project_id, location=location)
            self._GenerativeModel = GenerativeModel
            print(f"Gemini Vertex provider initialized: project={project_id}, location={location}")
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI SDK not available. Install google-cloud-aiplatform."
            ) from exc

    @property
    def name(self) -> str:
        return "GeminiVertex"

    def _convert_messages_to_gemini(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert OpenAI-style messages to Gemini Content format."""
        from vertexai.generative_models import Content, Part

        gemini_history = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                gemini_role = "model"
            elif role == "system":
                # System messages are handled separately
                continue
            else:
                gemini_role = "user"

            if content:
                gemini_history.append(
                    Content(role=gemini_role, parts=[Part.from_text(content)])
                )

        return gemini_history

    def _extract_system_prompt(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Extract system prompt from messages."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        system_prompt = self._extract_system_prompt(messages)

        model_kwargs = {"model_name": self.model_name}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model_instance = self._GenerativeModel(**model_kwargs)

        generation_config = {
            "max_output_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Add thinking config if specified
        if self.thinking_budget is not None:
            generation_config["thinking_config"] = {
                "thinking_budget": self.thinking_budget
            }

        gemini_history = self._convert_messages_to_gemini(messages)

        if len(gemini_history) > 1:
            chat = model_instance.start_chat(
                history=gemini_history[:-1],
                response_validation=False,
            )
            last_msg = gemini_history[-1]
            response = chat.send_message(
                last_msg.parts,
                generation_config=generation_config,
            )
        else:
            if gemini_history:
                content_parts = gemini_history[0].parts
            else:
                content_parts = messages[0].get("content", "") if messages else ""
            response = model_instance.generate_content(
                content_parts,
                generation_config=generation_config,
            )

        try:
            return response.text
        except Exception as exc:
            print(f"Failed to extract text from Gemini response: {exc}")
            return ""


class SarvamProvider(BaseLLMProvider):
    """Sarvam Pravah API provider with Harmony format extraction."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: str,
    ):
        super().__init__(temperature, max_tokens)
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return "Sarvam"

    def _extract_harmony_content(self, raw_text: str) -> str:
        """Extract the first 'final' channel message from Harmony format."""
        match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL
        )
        if match:
            return match.group(1).strip()

        if "<|channel|>analysis" in raw_text:
            cleaned = re.sub(
                r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>",
                "",
                raw_text,
                flags=re.DOTALL,
            )
            match = re.search(
                r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>", cleaned, re.DOTALL
            )
            if match:
                return match.group(1).strip()

        match = re.search(r"<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return raw_text.strip()

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "model": self.model,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            print(f"Warning: Empty response from {self.model} (finish_reason: {finish_reason})")
            return "[Model returned empty response]"

        return self._extract_harmony_content(content)


class TinkerProvider(BaseLLMProvider):
    """Tinker model provider."""

    def __init__(
        self,
        temperature: float,
        max_tokens: int,
    ):
        super().__init__(temperature, max_tokens)

        from tinker_helper import TinkerModel

        self._model = TinkerModel(temperature=temperature, max_tokens=max_tokens)

    @property
    def name(self) -> str:
        return "Tinker"

    @retry_with_backoff()
    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        return self._model.get_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def create_provider(
    provider_type: str,
    temperature: float,
    max_tokens: int,
) -> BaseLLMProvider:
    """Factory function to create LLM provider instances."""

    if provider_type == "tinker":
        return TinkerProvider(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider_type == "azure":
        deployment = AZURE_CONFIG["deployments"]["4_1_mini"]
        if isinstance(deployment, dict):
            deployment = deployment["deployment"]
        return AzureOpenAIProvider(
            deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=AZURE_CONFIG["api_key"],
            endpoint=AZURE_CONFIG["endpoint"],
            api_version=AZURE_CONFIG["api_version"],
        )

    elif provider_type == "openai":
        return OpenAIProvider(
            model=OPENAI_CONFIG.get("model", "gpt-4o"),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=OPENAI_CONFIG.get("api_key", ""),
            base_url=OPENAI_CONFIG.get("base_url"),
        )

    elif provider_type == "lepton":
        return OpenAIProvider(
            model=LEPTON_CONFIG.get("model", "benchmark-model"),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=LEPTON_CONFIG.get("api_key", ""),
            base_url=LEPTON_CONFIG.get("base_url"),
        )

    elif provider_type == "sarvam":
        return SarvamProvider(
            model=SARVAM_PRAVAH_CONFIG.get("model", "sarvam-gpt-oss-20b-finetune"),
            temperature=temperature,
            max_tokens=min(1000, max_tokens),  # Sarvam has 8192 context limit
            api_key=SARVAM_PRAVAH_CONFIG.get("api_key", ""),
            base_url=SARVAM_PRAVAH_CONFIG.get("base_url", ""),
        )

    elif provider_type == "gemini":
        return GeminiProvider(
            model=GEMINI_CONFIG.get("model", "gemini-2.5-flash-preview-05-20"),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=GEMINI_CONFIG.get("api_key", ""),
            base_url=GEMINI_CONFIG.get("base_url", ""),
        )

    elif provider_type == "gemini_vertex":
        return GeminiVertexProvider(
            model=os.getenv("GEMINI_VERTEX_MODEL", "gemini-2.5-flash-preview-05-20"),
            temperature=temperature,
            max_tokens=max_tokens,
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "0")) or None,
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def main():
    """Quick test for providers."""
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("LLM Provider Test")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Say hello in Hindi."},
    ]

    # Test Gemini via Google AI Studio
    print("\n[Testing Gemini via Google AI Studio]")
    try:
        provider = create_provider("gemini", temperature=0.5, max_tokens=100)
        response = provider.get_response(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
