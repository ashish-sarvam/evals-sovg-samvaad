"""User prompts registry for memory task.

Each agent has its own user file with conversation steps/flows.
User files define WHAT the simulated user does during the conversation.
Agent profiles (in user_profiles.py) define WHAT the agent has stored about the user.
"""

from tasks.memory.users import (
    general_assistant,
    idfc_main,
    tata_cap_sales,
    uc_scheduling,
    dcs,
)

# Registry: agent_name -> module with USER_PROMPTS
AGENT_USERS = {
    "general_assistant": general_assistant,
    "idfc_main": idfc_main,
    "tata_cap_sales": tata_cap_sales,
    "uc_scheduling": uc_scheduling,
    "dcs": dcs,
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
