"""
Model wrappers for different LLM backends.

Provides a unified interface for Tinker, Azure OpenAI, OpenAI, and Lepton models.
"""

from typing import Optional, Protocol
from openai import AzureOpenAI, OpenAI

from core.config import (
    AZURE_CONFIG,
    OPENAI_CONFIG,
    LEPTON_CONFIG,
    SARVAM_PRAVAH_CONFIG,
    GEMINI_CONFIG,
    MODEL_SETTINGS,
)
from core.llm_providers import retry_with_backoff


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
        deployment=AZURE_CONFIG["deployments"]["4_1_mini"],
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

    @retry_with_backoff()
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
            print(
                f"Warning: Empty response from {self.deployment} (finish_reason: {finish_reason})"
            )
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

    @retry_with_backoff()
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
            print(
                f"Warning: Empty response from {self.model} (finish_reason: {finish_reason})"
            )
            return "[Model returned empty response - possibly content filtered]"
        return content


def extract_harmony_content(raw_text: str) -> str:
    """
    Extract the first 'final' channel message from Harmony format output.

    The model outputs:
    <|channel|>final<|message|>CONTENT<|end|><|start|>assistant...

    We extract just the first CONTENT.
    """
    import re

    # Look for the first final channel message
    match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # Try analysis channel if no final channel (some models output analysis first)
    if "<|channel|>analysis" in raw_text:
        # Remove analysis block and look for final
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

    # Fallback: try to get any message content
    match = re.search(r"<\|message\|>(.*?)<\|end\|>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return original text (will be cleaned by runner)
    return raw_text.strip()


class SarvamModel:
    """Sarvam Pravah API model wrapper with Harmony format extraction."""

    def __init__(
        self,
        model: str = None,
        temperature: float = MODEL_SETTINGS["agent"]["temperature"],
        max_tokens: int = MODEL_SETTINGS["agent"]["max_tokens"],
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model or SARVAM_PRAVAH_CONFIG.get(
            "model", "sarvam-gpt-oss-20b-finetune"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs = {"api_key": api_key or SARVAM_PRAVAH_CONFIG.get("api_key", "")}
        if base_url or SARVAM_PRAVAH_CONFIG.get("base_url"):
            client_kwargs["base_url"] = base_url or SARVAM_PRAVAH_CONFIG.get("base_url")

        self.client = OpenAI(**client_kwargs)

    @retry_with_backoff()
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
            print(
                f"Warning: Empty response from {self.model} (finish_reason: {finish_reason})"
            )
            return "[Model returned empty response - possibly content filtered]"

        # Extract harmony content (similar to Tinker)
        parsed = extract_harmony_content(content)
        print(f"Parsed content: {parsed}")
        return parsed


class GeminiVertexModel:
    """Gemini model via Vertex AI (uses gcloud auth / ADC)."""

    def __init__(
        self,
        model: str,
        temperature: float = MODEL_SETTINGS["agent"]["temperature"],
        max_tokens: int = MODEL_SETTINGS["agent"]["max_tokens"],
        project_id: str = None,
        location: str = "us-central1",
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=project_id, location=location)
            self._GenerativeModel = GenerativeModel
            print(f"Gemini Vertex initialized: project={project_id}, location={location}, model={model}")
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI SDK not available. Run: pip install google-cloud-aiplatform"
            ) from exc

    def _convert_messages_to_gemini(self, messages: list[dict]) -> tuple:
        """Convert OpenAI-style messages to Gemini format."""
        from vertexai.generative_models import Content, Part

        system_prompt = None
        gemini_history = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if content:
                gemini_history.append(
                    Content(role=gemini_role, parts=[Part.from_text(content)])
                )

        return system_prompt, gemini_history

    @retry_with_backoff()
    def get_response(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        system_prompt, gemini_history = self._convert_messages_to_gemini(messages)

        model_kwargs = {"model_name": self.model_name}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model_instance = self._GenerativeModel(**model_kwargs)

        temp = temperature if temperature is not None else self.temperature
        generation_config = {
            "max_output_tokens": max_tokens or self.max_tokens,
            "temperature": temp,
        }

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
            return "[Gemini returned empty response]"


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

    @retry_with_backoff()
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
    temp = (
        temperature
        if temperature is not None
        else MODEL_SETTINGS["agent"]["temperature"]
    )

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
    elif provider == "sarvam":
        # Sarvam Pravah with Harmony format extraction
        # Note: Sarvam has 8192 token limit, so use smaller max_tokens
        return SarvamModel(
            model=SARVAM_PRAVAH_CONFIG.get("model", "sarvam-gpt-oss-20b-finetune"),
            temperature=temp,
            max_tokens=1000,  # Sarvam has 8192 context limit
            api_key=SARVAM_PRAVAH_CONFIG.get("api_key", ""),
            base_url=SARVAM_PRAVAH_CONFIG.get("base_url"),
        )
    elif provider == "gemini":
        # Google Gemini via Vertex AI (uses gcloud auth)
        return GeminiVertexModel(
            model=GEMINI_CONFIG.get("model", "gemini-2.5-flash-preview-05-20"),
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
            project_id=GEMINI_CONFIG.get("project_id", ""),
            location=GEMINI_CONFIG.get("location", "us-central1"),
        )
    else:  # Default to OpenAI
        return OpenAIModel(
            model=OPENAI_CONFIG.get("model"),
            temperature=temp,
            max_tokens=MODEL_SETTINGS["agent"]["max_tokens"],
        )


def create_user_model() -> LLMModel:
    """Create the user proxy model based on config."""
    from core.config import EVAL_DEFAULTS
    
    provider = EVAL_DEFAULTS.get("user_provider", "azure")
    model = EVAL_DEFAULTS.get("user_model", "gpt-4.1-mini")
    
    if provider == "azure":
        return AzureModel(
            deployment=model,
            temperature=MODEL_SETTINGS["user"]["temperature"],
            max_tokens=MODEL_SETTINGS["user"]["max_tokens"],
        )
    elif provider == "gemini":
        return GeminiVertexModel(
            model=model,
            temperature=MODEL_SETTINGS["user"]["temperature"],
            max_tokens=MODEL_SETTINGS["user"]["max_tokens"],
            project_id=GEMINI_CONFIG.get("project_id", ""),
            location=GEMINI_CONFIG.get("location", "us-central1"),
        )
    elif provider == "openai":
        return OpenAIModel(
            model=model,
            temperature=MODEL_SETTINGS["user"]["temperature"],
            max_tokens=MODEL_SETTINGS["user"]["max_tokens"],
            api_key=OPENAI_CONFIG.get("api_key", ""),
            base_url=OPENAI_CONFIG.get("base_url"),
        )
    else:
        # Default to Azure
        return AzureModel(
            deployment=model,
            temperature=MODEL_SETTINGS["user"]["temperature"],
            max_tokens=MODEL_SETTINGS["user"]["max_tokens"],
        )


def create_verifier_model(model_type: str = None) -> LLMModel:
    """Create a verifier model based on config or override.

    Default is from EVAL_DEFAULTS (Gemini Flash 3 Preview).
    """
    from core.config import EVAL_DEFAULTS
    
    # Use config defaults if not specified
    provider = model_type or EVAL_DEFAULTS.get("verifier_provider", "gemini")
    model = EVAL_DEFAULTS.get("verifier_model", "gemini-3-flash-preview")
    
    if provider == "tinker":
        return TinkerModelAdapter(
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
        )
    elif provider == "gemini":
        return GeminiVertexModel(
            model=model,
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
            project_id=GEMINI_CONFIG.get("project_id", ""),
            location=GEMINI_CONFIG.get("location", "us-central1"),
        )
    elif provider == "azure":
        return AzureModel(
            deployment=model if model else AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
        )
    elif provider == "openai":
        return OpenAIModel(
            model=model,
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
            api_key=OPENAI_CONFIG.get("api_key", ""),
            base_url=OPENAI_CONFIG.get("base_url"),
        )
    else:
        # Default to Gemini
        return GeminiVertexModel(
            model=model,
            temperature=MODEL_SETTINGS["verifier"]["temperature"],
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
            project_id=GEMINI_CONFIG.get("project_id", ""),
            location=GEMINI_CONFIG.get("location", "us-central1"),
        )
