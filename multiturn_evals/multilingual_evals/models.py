"""
Model wrappers for different LLM backends.

Provides a unified interface for Tinker, Azure OpenAI, OpenAI, and Lepton models.
"""

from typing import Optional, Protocol
from openai import AzureOpenAI, OpenAI

from multilingual_evals.config import AZURE_CONFIG, OPENAI_CONFIG, LEPTON_CONFIG, MODEL_SETTINGS


class LLMModel(Protocol):
    """Protocol defining the interface for LLM models."""

    def get_response(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str: ...


def get_deployment_name(config_value) -> str:
    """Extract deployment name from config value (string or dict with 'deployment' key)."""
    if isinstance(config_value, dict):
        return config_value["deployment"]
    return config_value


class AzureModel:
    """Azure OpenAI model wrapper."""

    def __init__(
        self,
        deployment = AZURE_CONFIG["deployments"]["4_1_mini"],
        temperature: float = MODEL_SETTINGS["user"]["temperature"],
        max_tokens: int = MODEL_SETTINGS["user"]["max_tokens"],
    ):
        self.deployment = get_deployment_name(deployment)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AzureOpenAI(
            api_version=AZURE_CONFIG["api_version"],
            azure_endpoint=AZURE_CONFIG["endpoint"],
            api_key=AZURE_CONFIG["api_key"],
        )

    def get_response(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # Build kwargs - only include temperature if set
        kwargs = {
            "messages": messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
            "model": self.deployment,
        }
        
        # Use provided temperature, fall back to instance temperature, skip if None
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            # Check if content was filtered
            finish_reason = response.choices[0].finish_reason
            print(f"Warning: Empty response from {self.deployment} (finish_reason: {finish_reason})")
            # Return a fallback response to continue conversation
            return "[Model returned empty response - possibly content filtered]"
        return content


class OpenAIModel:
    """Standard OpenAI API model wrapper."""

    def __init__(
        self,
        model: str = None,
        temperature: float = MODEL_SETTINGS["agent"]["temperature"],
        max_tokens: int = MODEL_SETTINGS["agent"]["max_tokens"],
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model or OPENAI_CONFIG.get("model", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Support custom base_url for OpenAI-compatible APIs
        client_kwargs = {"api_key": api_key or OPENAI_CONFIG.get("api_key")}
        if base_url or OPENAI_CONFIG.get("base_url"):
            client_kwargs["base_url"] = base_url or OPENAI_CONFIG.get("base_url")
        
        self.client = OpenAI(**client_kwargs)

    def get_response(
        self,
        messages: list[dict],
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
            return "[Model returned empty response - possibly content filtered]"
        return content


class TinkerModelAdapter:
    """
    Adapter for TinkerModel from tinker_helper.

    Wraps the existing TinkerModel to match the LLMModel interface.
    """

    def __init__(
        self,
        temperature: float = MODEL_SETTINGS["agent"]["temperature"],
        max_tokens: int = MODEL_SETTINGS["agent"]["max_tokens"],
    ):
        # Lazy import to avoid loading tinker unless needed
        from tinker_helper import TinkerModel

        self._model = TinkerModel(temperature=temperature, max_tokens=max_tokens)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        return self._model.get_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def create_agent_model(provider: str = None, temperature: float = None) -> LLMModel:
    """Create the agent model based on configured provider.
    
    Args:
        provider: Model provider - 'openai', 'azure', 'tinker', or 'lepton'. 
                  Defaults to OPENAI_CONFIG['provider'] or 'openai'.
        temperature: Optional temperature override for the model.
    
    Returns:
        LLMModel instance for the agent.
    """
    provider = provider or OPENAI_CONFIG.get("provider", "openai")
    temp = temperature if temperature is not None else MODEL_SETTINGS["agent"]["temperature"]
    
    if provider == "tinker":
        return TinkerModelAdapter(
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
        )
    elif provider == "azure":
        return AzureModel(
            deployment=AZURE_CONFIG["deployments"]["4_1_mini"],  # GPT-4.1-mini
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
        )
    elif provider == "lepton":
        # Lepton uses OpenAI-compatible API
        return OpenAIModel(
            model=LEPTON_CONFIG.get("model", "benchmark-model"),
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
            api_key=LEPTON_CONFIG.get("api_key", ""),
            base_url=LEPTON_CONFIG.get("base_url"),
        )
    else:  # Default to OpenAI
        return OpenAIModel(
            model=OPENAI_CONFIG.get("model"),
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
        )


def create_user_model() -> AzureModel:
    """Create the default user proxy model (Azure GPT-4.1-mini)."""
    return AzureModel(
        deployment=AZURE_CONFIG["deployments"]["4_1_mini"],  # GPT-4.1-mini
        temperature=MODEL_SETTINGS["user"]["temperature"],
        max_tokens=MODEL_SETTINGS["user"]["max_tokens"],
    )


def create_verifier_model(model_type: str = "azure") -> LLMModel:
    """Create a verifier model based on type.
    
    Default is GPT-5.2-chat for best judge quality.
    """
    if model_type == "tinker":
        return TinkerModelAdapter(
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
        )
    else:
        # Default to GPT-5.2-chat for verifier/judge
        return AzureModel(
            deployment=AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
        )
