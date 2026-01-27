"""
Users module for conversationality task.

Blueprints are generated dynamically using the BlueprintGenerator.
Generated blueprints are saved as {agent_name}_generated.json in this folder.
"""

import json
from pathlib import Path
from typing import Optional


def get_blueprints_file(agent_name: str) -> Path:
    """Get path to blueprints file for an agent."""
    return Path(__file__).parent / f"{agent_name}_generated.json"


def load_blueprints(agent_name: str) -> Optional[dict]:
    """Load blueprints for an agent if they exist."""
    path = get_blueprints_file(agent_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def list_available_agents() -> list[str]:
    """List agents that have generated blueprints."""
    agents = []
    for f in Path(__file__).parent.glob("*_generated.json"):
        agents.append(f.stem.replace("_generated", ""))
    return agents


def get_blueprint_by_name(agent_name: str, blueprint_name: str) -> Optional[dict]:
    """Get a specific blueprint by name."""
    blueprints = load_blueprints(agent_name)
    if blueprints:
        for bp in blueprints.get("blueprints", []):
            if bp["name"] == blueprint_name:
                return bp
    return None
