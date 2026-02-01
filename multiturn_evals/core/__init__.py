"""
Multilingual Evaluation Framework

A modular framework for evaluating conversational AI agents across languages.

Structure:
    core/        - Core modules (config, models, runner, languages, cli)
    tasks/       - Evaluation task definitions
    agents/      - Agent system prompts
    eval_configs/- YAML evaluation configs

Usage:
    python run_evals.py --config eval_configs/eval_config.yaml
    python -m core --task multilingual --agent dcs
"""

from core.languages import (
    Language,
    SUPPORTED_LANGUAGES,
    get_language,
    get_supported_language_codes,
    get_supported_language_names,
)

__all__ = [
    "Language",
    "SUPPORTED_LANGUAGES",
    "get_language",
    "get_supported_language_codes",
    "get_supported_language_names",
]
