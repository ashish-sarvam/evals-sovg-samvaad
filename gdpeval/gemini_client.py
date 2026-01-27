"""
Simple Gemini 3 Flash Preview client via Vertex AI.

Usage:
    from gemini_client import GeminiClient

    client = GeminiClient()
    response = client.generate("What is 2+2?")
    print(response)
"""

from typing import Any, Dict, List, Optional


class GeminiClient:
    """Simple client for Gemini 3 Flash Preview via Vertex AI."""

    def __init__(
        self,
        project_id: str = "text-475009",
        location: str = "global",
        model: str = "gemini-3-flash-preview",
    ):
        """
        Initialize Gemini client.

        Args:
            project_id: GCP project ID
            location: GCP region
            model: Model name
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:
            raise RuntimeError(
                "Install Vertex AI SDK: pip install google-cloud-aiplatform"
            ) from e

        self.project_id = project_id
        self.location = location
        self.model_name = model

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self._model = GenerativeModel(model_name=self.model_name)

        print(f"✓ Gemini client initialized: {self.model_name}")

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        thinking: bool = False,
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            max_tokens: Max output tokens
            temperature: Sampling temperature (0-2)
            thinking: Enable thinking mode (for supported models)

        Returns:
            Generated text response
        """
        from vertexai.generative_models import GenerativeModel

        # Recreate model with system instruction if provided
        if system_prompt:
            model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        # Build generation config
        gen_config: Dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add thinking config for models that support it
        if thinking:
            gen_config["thinking_config"] = {"thinking_budget": 1024}

        response = model.generate_content(prompt, generation_config=gen_config)

        return response.text

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        thinking: bool = False,
    ) -> str:
        """
        Multi-turn chat with Gemini.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system instruction
            max_tokens: Max output tokens
            temperature: Sampling temperature
            thinking: Enable thinking mode

        Returns:
            Generated text response
        """
        from vertexai.generative_models import Content, GenerativeModel, Part

        # Recreate model with system instruction if provided
        if system_prompt:
            model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:  # All but last
            role = "model" if msg["role"] == "assistant" else "user"
            history.append(
                Content(role=role, parts=[Part.from_text(msg["content"])])
            )

        # Build generation config
        gen_config: Dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if thinking:
            gen_config["thinking_config"] = {"thinking_budget": 1024}

        # Start chat and send last message
        chat = model.start_chat(history=history)
        last_msg = messages[-1]["content"]
        response = chat.send_message(last_msg, generation_config=gen_config)

        return response.text


# --- Quick test ---
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = GeminiClient()

    # Simple generation
    print("\n--- Simple Generation ---")
    resp = client.generate("What is the capital of France?")
    print(resp)

    # With thinking enabled
    print("\n--- With Thinking ---")
    resp = client.generate(
        "Explain why the sky is blue in simple terms.",
        thinking=True,
    )
    print(resp)

    # Multi-turn chat
    print("\n--- Multi-turn Chat ---")
    messages = [
        {"role": "user", "content": "My name is Ashish."},
        {"role": "assistant", "content": "Nice to meet you, Ashish!"},
        {"role": "user", "content": "What's my name?"},
    ]
    resp = client.chat(messages)
    print(resp)
