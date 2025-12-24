"""
Configuration handling for the benchmarking system.
Loads and validates configuration from config.json file.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load and validate the configuration from the config.json file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dict containing the validated configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    # Validate required configuration fields
    required_fields = ["dataset_name", "revision", "models_config"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required configuration field '{field}' missing")

    # Validate models_config
    validate_models_config(config["models_config"])

    # Set default values if not present
    if "llm_config" not in config:
        config["llm_config"] = {"temperature": 0.1, "max_tokens": 512}

    if "max_concurrent_tasks" not in config:
        config["max_concurrent_tasks"] = 5

    return config


def validate_models_config(models_config: List[Dict[str, Any]]) -> None:
    """
    Validate the models_config section of the config file.

    Args:
        models_config: List of model configurations.

    Raises:
        ValueError: If any model configuration is invalid.
    """
    if not models_config:
        raise ValueError("models_config cannot be empty")

    for idx, model_config in enumerate(models_config):
        # Check required fields
        required_fields = ["name", "provider", "base_model_type", "path"]
        for field in required_fields:
            if field not in model_config:
                raise ValueError(
                    f"Model config at index {idx} missing required field: {field}"
                )

        # Validate provider values
        valid_providers = [
            "SARVAM",
            "AZURE",
            "ANTHROPIC_VERTEX",
            "GEMINI",
            "GROQ",
            "OPENAI",
        ]
        if model_config["provider"] not in valid_providers:
            raise ValueError(
                f"Invalid provider '{model_config['provider']}' at index {idx}. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        # Check provider-specific required fields
        if model_config["provider"] == "AZURE":
            azure_fields = ["deployment", "endpoint", "api_key"]
            for field in azure_fields:
                if field not in model_config:
                    raise ValueError(
                        f"Azure model config at index {idx} missing required field: {field}"
                    )
        if model_config["provider"] == "ANTHROPIC_VERTEX":
            av_fields = ["project_id", "region"]
            for field in av_fields:
                if field not in model_config:
                    raise ValueError(
                        f"Anthropic Vertex model config at index {idx} missing required field: {field}"
                    )
        if model_config["provider"] == "GEMINI":
            if "api_key" not in model_config:
                raise ValueError(
                    f"Gemini model config at index {idx} missing required field: api_key"
                )
        if model_config["provider"] == "GROQ":
            if "api_key" not in model_config:
                raise ValueError(
                    f"Groq model config at index {idx} missing required field: api_key"
                )
        if model_config["provider"] == "OPENAI":
            if "api_key" not in model_config:
                raise ValueError(
                    f"OpenAI model config at index {idx} missing required field: api_key"
                )


def get_results_directory() -> Path:
    """
    Get or create the results directory.

    Returns:
        Path object to the results directory.
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir
