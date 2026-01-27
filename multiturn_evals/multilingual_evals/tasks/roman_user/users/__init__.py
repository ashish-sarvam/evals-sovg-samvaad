"""User prompts registry for roman_user task."""

from multilingual_evals.tasks.roman_user.users import dcs

# Registry: agent_name -> module with USER_PROMPTS
AGENT_USERS = {
    "dcs": dcs,
}


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent."""
    if agent_name not in AGENT_USERS:
        available = ", ".join(AGENT_USERS.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    return AGENT_USERS[agent_name].USER_PROMPTS


def list_agents() -> list[str]:
    """List available agents."""
    return list(AGENT_USERS.keys())


def list_users(agent_name: str) -> list[str]:
    """List available users for an agent."""
    if agent_name not in AGENT_USERS:
        return []
    return list(AGENT_USERS[agent_name].USER_PROMPTS.keys())
