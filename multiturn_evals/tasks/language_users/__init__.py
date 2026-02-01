"""
Common user prompts shared across language-related tasks:
- multilingual (native script)
- english_user (user speaks English)
- colloquial (rural vs urban tone)
- roman_user (user writes in romanized)

Each agent has its own user file with USER_BEHAVIORS dict.
Tasks import these and wrap with their specific language/script rules.
"""

from tasks.language_users import dcs
from tasks.language_users import idfc_main
from tasks.language_users import tata_cap_sales
from tasks.language_users import uc_scheduling

# Registry: agent_name -> module with USER_BEHAVIORS
AGENT_USERS = {
    "dcs": dcs,
    "idfc_main": idfc_main,
    "tata_cap_sales": tata_cap_sales,
    "uc_scheduling": uc_scheduling,
}


def get_user_behaviors(agent_name: str) -> dict[str, dict]:
    """Get user behaviors for a specific agent."""
    if agent_name not in AGENT_USERS:
        available = ", ".join(AGENT_USERS.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    return AGENT_USERS[agent_name].USER_BEHAVIORS


def list_agents() -> list[str]:
    """List available agents."""
    return list(AGENT_USERS.keys())


def list_users(agent_name: str) -> list[str]:
    """List available users for an agent."""
    if agent_name not in AGENT_USERS:
        return []
    return list(AGENT_USERS[agent_name].USER_BEHAVIORS.keys())
