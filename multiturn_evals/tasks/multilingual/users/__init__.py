"""User prompts registry for multilingual task.

Uses common user behaviors from language_users and wraps with native script rules.
"""

from tasks.language_users import get_user_behaviors, list_agents as _list_agents, list_users as _list_users

# Special imports for agents with custom prompts
from tasks.multilingual.users import assistant

# Language rules template for multilingual task
MULTILINGUAL_RULES = """You are simulating a user in a phone conversation. Respond in {LANGUAGE} using NATIVE SCRIPT.

## CRITICAL: Script Rules
- Write {LANGUAGE} words in NATIVE SCRIPT (e.g., Hindi in Devanagari: हाँ, नहीं, धन्यवाद)
- ONLY English words stay in Roman/English script (survey, ministry, appointment, loan, EMI, etc.)
- Example for Hindi: "हाँ जी, survey 432 में Magfali सही है।"
- DO NOT use Roman transliteration for {LANGUAGE} words (wrong: "Haan ji, sahi hai")

"""


def _build_prompt(behavior: dict) -> str:
    """Build a complete prompt from behavior dict."""
    return MULTILINGUAL_RULES + behavior["behavior"]


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent."""
    # Special case: assistant has its own custom prompts
    if agent_name == "assistant":
        return assistant.USER_PROMPTS
    
    # For all other agents, use common behaviors with multilingual rules
    behaviors = get_user_behaviors(agent_name)
    return {name: _build_prompt(b) for name, b in behaviors.items()}


def list_agents() -> list[str]:
    """List available agents."""
    # Include both common agents and special ones
    agents = set(_list_agents())
    agents.add("assistant")
    return sorted(list(agents))


def list_users(agent_name: str) -> list[str]:
    """List available users for an agent."""
    if agent_name == "assistant":
        return list(assistant.USER_PROMPTS.keys())
    return _list_users(agent_name)
