"""
Example configuration - Copy to config.py and fill in your values.

cp config.example.py config.py
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "eval_results"

# =============================================================================
# AZURE OPENAI - Fill in your values
# =============================================================================

AZURE_CONFIG = {
    "endpoint": "https://YOUR-RESOURCE.openai.azure.com/",
    "api_version": "2024-12-01-preview",
    "api_key": "YOUR_AZURE_API_KEY",
    "deployments": {
        "5_mini": "gpt-5-mini",          # For user proxy
        "5_2_chat": "gpt-5.2-chat",      # For verifier/judge
    },
}

# =============================================================================
# OPENAI (Standard API) - Recommended for most users
# =============================================================================

OPENAI_CONFIG = {
    "api_key": "YOUR_OPENAI_API_KEY",  # From https://platform.openai.com/api-keys
    "model": "gpt-4o",                  # Agent model (gpt-4o, gpt-4o-mini, etc.)
    "base_url": None,                   # Optional: for OpenAI-compatible APIs (e.g., vLLM, Ollama)
    "provider": "openai",               # Agent provider: 'openai', 'azure', or 'tinker'
}

# =============================================================================
# LEPTON (OpenAI-compatible hosted models)
# =============================================================================

LEPTON_CONFIG = {
    "base_url": "https://h1v6kgoi-sft-sovg-model.xenon.lepton.run/v1/",
    "api_key": "",
    "model": "benchmark-model",
}

# =============================================================================
# TINKER (Internal/Advanced use) - Skip if using OpenAI
# =============================================================================

TINKER_CONFIG = {
    "api_key": "YOUR_TINKER_API_KEY",
    "model_name": "openai/gpt-oss-120b",
    "model_path": "tinker://YOUR_MODEL_PATH",
    "renderer": "gpt_oss_no_sysprompt",
}

# =============================================================================
# EVALUATION DEFAULTS
# =============================================================================

EVAL_DEFAULTS = {
    "max_turns": 7,
    "agent_name": "dcs",
    "verifier_model": "azure",  # Uses GPT-5.2-chat
}

# Model settings
MODEL_SETTINGS = {
    "agent": {
        "temperature": 0.5,
        "max_tokens": 500,
    },
    "user": {
        "temperature": 0.7,
        "max_tokens": 200,
    },
    "verifier": {
        "temperature": None,  # Use API default
        "max_tokens": 3500,
    },
}
