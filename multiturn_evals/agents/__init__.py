"""Agent configurations for soft evals."""

from . import idfc_main, idfc_min, dcs, tata_cap_sales, uc_scheduling

# Registry of all available agents
AGENTS = {
    "idfc_main": idfc_main,
    "idfc_min": idfc_min,
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
