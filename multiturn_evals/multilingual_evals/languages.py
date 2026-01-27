"""
Supported languages configuration for multilingual evaluations.

This module provides a dataclass for language configurations and helper functions
for working with supported languages across the evaluation system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Language:
    """Represents a supported language configuration."""

    code: str  # ISO-style code (e.g., "hi-en", "bn-en", "en")
    name: str  # Full name (e.g., "Hindi", "Bengali")
    native_name: str  # Native script name (e.g., "हिन्दी", "বাংলা")
    is_bilingual: bool  # True if code-mixed with English (e.g., "hi-en")

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"

    @property
    def display_name(self) -> str:
        """Get display name with native script."""
        return f"{self.name} ({self.native_name})"

    @property
    def prompt_language(self) -> str:
        """Get the language name to use in prompts/system messages."""
        if self.is_bilingual:
            return f"{self.name} with mix of English"
        return self.name


# All supported languages
SUPPORTED_LANGUAGES: list[Language] = [
    Language(code="hi-en", name="Hindi", native_name="हिन्दी", is_bilingual=True),
    Language(code="bn-en", name="Bengali", native_name="বাংলা", is_bilingual=True),
    Language(code="gu-en", name="Gujarati", native_name="ગુજરાતી", is_bilingual=True),
    Language(code="kn-en", name="Kannada", native_name="ಕನ್ನಡ", is_bilingual=True),
    Language(code="ml-en", name="Malayalam", native_name="മലയാളം", is_bilingual=True),
    Language(code="mr-en", name="Marathi", native_name="मराठी", is_bilingual=True),
    Language(code="or-en", name="Odia", native_name="ଓଡ଼ିଆ", is_bilingual=True),
    Language(code="pa-en", name="Punjabi", native_name="ਪੰਜਾਬੀ", is_bilingual=True),
    Language(code="ta-en", name="Tamil", native_name="தமிழ்", is_bilingual=True),
    Language(code="te-en", name="Telugu", native_name="తెలుగు", is_bilingual=True),
    Language(code="en", name="English", native_name="English", is_bilingual=False),
]

# Language lookup by code
_LANGUAGE_BY_CODE: dict[str, Language] = {
    lang.code: lang for lang in SUPPORTED_LANGUAGES
}

# Language lookup by name (case-insensitive)
_LANGUAGE_BY_NAME: dict[str, Language] = {
    lang.name.lower(): lang for lang in SUPPORTED_LANGUAGES
}


def get_supported_language_codes() -> list[str]:
    """Get list of all supported language codes."""
    return [lang.code for lang in SUPPORTED_LANGUAGES]


def get_supported_language_names() -> list[str]:
    """Get list of all supported language names."""
    return [lang.name for lang in SUPPORTED_LANGUAGES]


def get_bilingual_languages() -> list[Language]:
    """Get all bilingual (code-mixed) languages."""
    return [lang for lang in SUPPORTED_LANGUAGES if lang.is_bilingual]


def get_language_by_code(code: str) -> Optional[Language]:
    """Get Language object by its code (e.g., 'hi-en')."""
    return _LANGUAGE_BY_CODE.get(code)


def get_language_by_name(name: str) -> Optional[Language]:
    """Get Language object by its name (case-insensitive)."""
    return _LANGUAGE_BY_NAME.get(name.lower())


def get_language(identifier: str) -> Optional[Language]:
    """
    Get Language object by code or name.

    Args:
        identifier: Either language code ('hi-en') or name ('Hindi')

    Returns:
        Language object if found, None otherwise
    """
    # Try code first
    lang = get_language_by_code(identifier)
    if lang:
        return lang
    # Try name
    return get_language_by_name(identifier)


def is_supported_language(identifier: str) -> bool:
    """Check if a language (by code or name) is supported."""
    return get_language(identifier) is not None


def get_total_language_count() -> int:
    """Get total number of supported languages."""
    return len(SUPPORTED_LANGUAGES)


def get_indic_languages() -> list[Language]:
    """Get all Indic languages (excluding pure English)."""
    return [lang for lang in SUPPORTED_LANGUAGES if lang.code != "en"]


def format_language_list(separator: str = ", ") -> str:
    """Format all supported languages as a readable string."""
    return separator.join(lang.name for lang in SUPPORTED_LANGUAGES)


# Export commonly used items
__all__ = [
    "Language",
    "SUPPORTED_LANGUAGES",
    "get_supported_language_codes",
    "get_supported_language_names",
    "get_bilingual_languages",
    "get_language_by_code",
    "get_language_by_name",
    "get_language",
    "is_supported_language",
    "get_total_language_count",
    "get_indic_languages",
    "format_language_list",
]
