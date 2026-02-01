"""User prompts registry for general assistant task."""

from tasks.general_assistant.users import assistant

# Registry: agent_name -> module with USER_PROMPTS
AGENT_USERS = {
    "assistant": assistant,
}


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent."""
    if agent_name not in AGENT_USERS:
        available = ", ".join(AGENT_USERS.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Options: {available}")
    return AGENT_USERS[agent_name].USER_PROMPTS


def list_agents() -> list[str]:
    """List available agents."""
    return list(AGENT_USERS.keys())


def list_users(agent_name: str) -> list[str]:
    """List available users for an agent."""
    if agent_name not in AGENT_USERS:
        return []
    return list(AGENT_USERS[agent_name].USER_PROMPTS.keys())
