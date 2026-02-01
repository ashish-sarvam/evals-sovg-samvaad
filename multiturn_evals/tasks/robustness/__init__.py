"""
Robustness Task

Tests how well agents handle noise, interruptions, and repetition requests
without getting stuck in loops or losing conversation state.

Key evaluation criteria:
- Reset rate: Did agent re-run opener/identity gate incorrectly?
- Repair quality: Did agent repeat only missing info?
- State continuity: Did agent resume the correct step?
- Over-asking: Did agent re-ask already-answered questions?
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

from tasks import BaseTask, TaskConfig
from core.languages import Language, SUPPORTED_LANGUAGES
from tasks.robustness.generator import RobustnessGenerator
from tasks.robustness.evaluator import RobustnessEvaluator


@dataclass
class RobustnessBlueprint:
    """A user blueprint for testing robustness/repetition handling."""

    name: str
    bucket: str  # One of the 6 buckets
    tags: list[str]
    difficulty: str
    description: str
    perturbation_turn: int  # Which turn the perturbation happens
    perturbation_count: int  # How many times user asks to repeat (1-3)
    behavior_steps: list[str]
    expected_state_at_perturbation: str  # What step agent should be at
    expected_recovery: str  # How agent should ideally recover
    failure_indicators: list[str]  # Signs of bad handling

    @classmethod
    def from_dict(cls, data: dict) -> "RobustnessBlueprint":
        return cls(
            name=data["name"],
            bucket=data["bucket"],
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
            description=data["description"],
            perturbation_turn=data.get("perturbation_turn", 2),
            perturbation_count=data.get("perturbation_count", 1),
            behavior_steps=data.get("behavior_steps", []),
            expected_state_at_perturbation=data.get(
                "expected_state_at_perturbation", ""
            ),
            expected_recovery=data.get("expected_recovery", ""),
            failure_indicators=data.get("failure_indicators", []),
        )

    def get_user_prompt(self, language: Language) -> str:
        """Generate user prompt for this robustness scenario."""
        steps_desc = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(self.behavior_steps)]
        )

        bucket_context = {
            "noise_opening": "You have bad phone signal from the START of the call.",
            "noise_slot": "Your signal becomes bad when agent asks about specific details (survey, crop, etc.).",
            "interruption_return": "You get interrupted mid-call (someone at door, another call) and return.",
            "partial_confirmation": "You partially confirm things but ask about something else mid-answer.",
            "confusion": "You genuinely don't understand the purpose and keep asking for clarification.",
            "repeated_perturbation": "You have multiple signal issues throughout - agent should eventually offer callback.",
        }.get(self.bucket, "You are testing agent's robustness.")

        return f"""You are simulating a user to test an AI agent's ability to handle interruptions and repetition requests WITHOUT getting stuck in loops.

## Scenario: {self.bucket.upper().replace('_', ' ')}
{bucket_context}

## Your Persona: {self.name}
{self.description}

## Your Behavior Pattern
Follow this sequence naturally:

{steps_desc}

## Critical Rules for Testing

### Language (VERY IMPORTANT):
- Write {language.name} words in NATIVE SCRIPT (Devanagari for Hindi: हाँ, रुको, नहीं)
- English words stay in English (survey, number, crop, call)
- Example: "रुको रुको, समझ नहीं आया, survey number क्या बोला?"
- Sound like a REAL person on phone - natural, not scripted

### Perturbation Rules:
- When you ask agent to repeat, be SPECIFIC or VAGUE based on the scenario
- Specific: "survey number दुबारा बोलो?" (agent should repeat ONLY that)
- Vague: "क्या बोला? समझ नहीं आया" (agent should summarize, not restart)
- Track what you've ALREADY confirmed - don't re-confirm those

### What You're Testing:
- Does agent repeat ONLY what's needed, or restart from scratch?
- Does agent remember what you already confirmed?
- Does agent get stuck repeating the same thing?
- After {self.perturbation_count}+ perturbations, does agent offer to call back?

### Conversation Flow:
- Start naturally, follow behavior steps
- After perturbation, see how agent recovers
- If agent handles it well, continue task
- If agent gets stuck/loops, express frustration
- Say **STOP** when conversation naturally ends OR agent is clearly stuck

Respond ONLY as the user. Never break character."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "bucket": self.bucket,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "description": self.description,
            "perturbation_turn": self.perturbation_turn,
            "perturbation_count": self.perturbation_count,
            "behavior_steps": self.behavior_steps,
            "expected_state_at_perturbation": self.expected_state_at_perturbation,
            "expected_recovery": self.expected_recovery,
            "failure_indicators": self.failure_indicators,
        }


class RobustnessTask(BaseTask):
    """Task for testing robustness/repetition handling."""

    def __init__(
        self,
        agent: str,
        blueprint: Optional[RobustnessBlueprint] = None,
        blueprint_name: Optional[str] = None,
    ):
        self.config = TaskConfig(
            name="robustness",
            description="Test robustness: noise handling, state continuity, no looping",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
            max_turns=12,  # Allow more turns for perturbation scenarios
        )
        self.agent = agent
        self.blueprint = blueprint
        self.blueprint_name = blueprint_name
        self.user = blueprint_name or "generated"

    def get_user_prompt(self, language: Language) -> str:
        """Get user prompt from blueprint."""
        if self.blueprint:
            return self.blueprint.get_user_prompt(language)
        raise ValueError("No blueprint set for this task")

    def get_verifier_system_prompt(self) -> str:
        """No individual verification - use robustness evaluator instead."""
        return ""

    def get_verifier_user_prompt(
        self, conversation: list[dict], language: Language
    ) -> str:
        """No individual verification."""
        return ""

    @staticmethod
    def list_users(agent: str) -> list[str]:
        """List available blueprints for an agent."""
        blueprints_file = Path(__file__).parent / "users" / f"{agent}_generated.json"
        if blueprints_file.exists():
            with open(blueprints_file) as f:
                data = json.load(f)
                return [b["name"] for b in data.get("blueprints", [])]
        return []

    @staticmethod
    def list_buckets() -> list[str]:
        """List the 6 test buckets."""
        return [
            "noise_opening",
            "noise_slot",
            "interruption_return",
            "partial_confirmation",
            "confusion",
            "repeated_perturbation",
        ]

    @classmethod
    def load_blueprints(cls, agent: str) -> list[RobustnessBlueprint]:
        """Load generated blueprints for an agent."""
        blueprints_file = Path(__file__).parent / "users" / f"{agent}_generated.json"
        if not blueprints_file.exists():
            return []

        with open(blueprints_file) as f:
            data = json.load(f)

        return [RobustnessBlueprint.from_dict(b) for b in data.get("blueprints", [])]

    @classmethod
    def get_blueprint(
        cls, agent: str, blueprint_name: str
    ) -> Optional[RobustnessBlueprint]:
        """Get a specific blueprint by name."""
        blueprints = cls.load_blueprints(agent)
        for bp in blueprints:
            if bp.name == blueprint_name:
                return bp
        return None

    @classmethod
    def get_blueprints_by_bucket(
        cls, agent: str, bucket: str
    ) -> list[RobustnessBlueprint]:
        """Get all blueprints for a specific bucket."""
        blueprints = cls.load_blueprints(agent)
        return [bp for bp in blueprints if bp.bucket == bucket]
