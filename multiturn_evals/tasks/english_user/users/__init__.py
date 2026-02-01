"""User prompts registry for english_user task.

Uses common user behaviors from language_users.
User speaks ENGLISH, agent should respond in native language.
"""

from tasks.language_users import get_user_behaviors, list_agents, list_users

# Language rules template for english_user task
ENGLISH_USER_RULES = """You are simulating a user in a phone conversation. Respond in ENGLISH only.

## Rules:
- Respond ONLY in English
- Keep responses brief and natural
- The agent will respond in their native language - that's expected

"""


def _build_prompt(behavior: dict) -> str:
    """Build a complete prompt from behavior dict."""
    return ENGLISH_USER_RULES + behavior["behavior"]


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent."""
    behaviors = get_user_behaviors(agent_name)
    return {name: _build_prompt(b) for name, b in behaviors.items()}
