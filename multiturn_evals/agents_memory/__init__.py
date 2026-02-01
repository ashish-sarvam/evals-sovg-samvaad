"""Agent configurations for memory/personalization evals.

These agents are used specifically for memory task where user profile
information is injected into the agent's system prompt.
"""

from . import general_assistant, idfc_main, dcs, tata_cap_sales, uc_scheduling

# Registry of all available agents for memory task
AGENTS = {
    "general_assistant": general_assistant,
    "idfc_main": idfc_main,
    "dcs": dcs,
    "tata_cap_sales": tata_cap_sales,
    "uc_scheduling": uc_scheduling,
}


def get_agent(name: str):
    """Get agent configuration by name."""
    if name not in AGENTS:
        available = ", ".join(AGENTS.keys())
        raise ValueError(f"Unknown agent: {name}. Available: {available}")
    return AGENTS[name]


def list_agents() -> list[str]:
    """List all available agent names."""
    return list(AGENTS.keys())
