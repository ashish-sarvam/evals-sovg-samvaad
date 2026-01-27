"""
Multilingual Evaluation Framework

A modular framework for evaluating conversational AI agents across languages.

Structure:
    config.py    - All configuration (API keys, defaults)
    models.py    - LLM model wrappers (Tinker, Azure)
    languages.py - Language definitions
    runner.py    - Conversation execution infrastructure
    tasks/       - Evaluation task definitions
    cli.py       - Command-line interface

Usage:
    poetry run python -m multilingual_evals.cli --task multilingual
    poetry run python -m multilingual_evals.cli --task english_user --languages hi-en -v
"""

from multilingual_evals.languages import (
    Language,
    SUPPORTED_LANGUAGES,
    get_language,
    get_supported_language_codes,
    get_supported_language_names,
)

from multilingual_evals.tasks import (
    BaseTask,
    TaskConfig,
    get_task,
    list_tasks,
)

__all__ = [
    "Language",
    "SUPPORTED_LANGUAGES",
    "get_language",
    "get_supported_language_codes",
    "get_supported_language_names",
    "BaseTask",
    "TaskConfig",
    "get_task",
    "list_tasks",
]
