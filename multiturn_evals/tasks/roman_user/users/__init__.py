"""User prompts registry for roman_user task.

Uses common user behaviors from language_users.
User writes in ROMANIZED script, agent should respond in native script.
"""

from tasks.language_users import get_user_behaviors, list_agents, list_users

# Language rules template for roman_user task
ROMAN_USER_RULES = """You are simulating a user in a phone conversation.
ALWAYS respond in ROMANIZED/TRANSLITERATED {LANGUAGE} using LATIN/ENGLISH script.

## CRITICAL: Script Rules
- Write ALL {LANGUAGE} words in ROMAN/LATIN script (English letters)
- Example for Hindi: Instead of "हाँ सही है", write "haan sahi hai"
- Example for Bengali: Instead of "হ্যাঁ ঠিক আছে", write "haan thik ache"
- NEVER use native Indic script

## Examples:
- "haan ji, main bol raha hoon"
- "nahi, yeh galat hai"
- "theek hai, dhanyavaad"

"""


def _build_prompt(behavior: dict) -> str:
    """Build a complete prompt from behavior dict."""
    return ROMAN_USER_RULES + behavior["behavior"]


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent."""
    behaviors = get_user_behaviors(agent_name)
    return {name: _build_prompt(b) for name, b in behaviors.items()}
