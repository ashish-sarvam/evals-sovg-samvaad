"""
Conversationality Task

Tests conversational robustness by generating user blueprints that stress-test
various conversational challenges, then comparing model trajectories.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

from tasks import BaseTask, TaskConfig
from core.languages import Language, SUPPORTED_LANGUAGES
from tasks.conversationality.generator import BlueprintGenerator
from tasks.conversationality.evaluator import PairwiseEvaluator


@dataclass
class Blueprint:
    """A user blueprint for testing conversational challenges."""
    name: str
    tags: list[str]
    difficulty: str
    challenge_start: str  # "immediate", "after_greeting", "mid_conversation"
    description: str
    behavior_steps: list[str]
    breaking_point: str
    expected_agent_behavior: list[str]
    
    @classmethod
    def from_dict(cls, data: dict) -> "Blueprint":
        return cls(
            name=data["name"],
            tags=data["tags"],
            difficulty=data["difficulty"],
            challenge_start=data.get("challenge_start", "after_greeting"),
            description=data["description"],
            behavior_steps=data.get("behavior_steps", data.get("turns", [])),
            breaking_point=data.get("breaking_point", ""),
            expected_agent_behavior=data.get("expected_agent_behavior", []),
        )
    
    def get_user_prompt(self, language: Language) -> str:
        """Generate user prompt from behavior steps."""
        # Format behavior steps as numbered list
        steps_desc = "\n".join([
            f"{i+1}. {step}" for i, step in enumerate(self.behavior_steps)
        ])
        
        timing_instruction = {
            "immediate": "Start with the challenge from your FIRST response.",
            "after_greeting": "Greet normally first, then introduce your challenge behavior.",
            "mid_conversation": "Cooperate for 2-3 turns, then introduce your challenge.",
        }.get(self.challenge_start, "Introduce challenge after initial greeting.")
        
        return f"""You are simulating a challenging user for stress-testing a conversational AI agent.

## Your Persona: {self.name}
{self.description}

## Your Behavior Pattern
Follow this sequence of behaviors (adapt exact words naturally, but follow the pattern):

{steps_desc}

## Timing
{timing_instruction}

## Language Rules - VERY IMPORTANT
- Write {language.name} words in NATIVE SCRIPT (e.g., Hindi in Devanagari: हाँ, अच्छा, नहीं)
- Write English words in English script (e.g., survey, number, call)
- Example for Hindi: "हाँ जी, बोलिए" NOT "Haan ji, boliye"
- Example for Hindi: "मैं survey के बारे में पूछ रहा था" (mix of Devanagari + English)
- Sound like a real person on the phone, natural and conversational
- Use realistic expressions in native script: "हाँ", "अच्छा", "नहीं", "क्या?", "मतलब"

## Critical Rules
1. STAY IN CHARACTER - you are this challenging user throughout
2. PUSH THE AGENT - don't make it easy for them, persist with your behavior
3. Follow the behavior pattern but use natural language, not exact scripts
4. Conversation should complete within 8-10 of YOUR turns
5. Say **STOP** only when:
   - The conversation naturally concludes (agent completes task or gives up)
   - You've been through all your behavior steps
   - 8-10 turns have passed

Respond ONLY as the user. Never break character or explain yourself."""


class ConversationalityTask(BaseTask):
    """Task for testing conversational robustness."""
    
    def __init__(
        self,
        agent: str,
        blueprint: Optional[Blueprint] = None,
        blueprint_name: Optional[str] = None,
    ):
        self.config = TaskConfig(
            name="conversationality",
            description="Test conversational robustness with challenging user behaviors",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
            max_turns=10,  # Allow more turns for complex conversations
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
        """No verification for this task - we use pairwise evaluation instead."""
        return ""
    
    def get_verifier_user_prompt(self, conversation: list[dict], language: Language) -> str:
        """No verification for this task."""
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
    def list_agents() -> list[str]:
        """List all available agents (from agents module)."""
        # Import here to avoid circular imports
        from agents import list_agents as list_all_agents
        return list_all_agents()
    
    @classmethod
    def load_blueprints(cls, agent: str) -> list[Blueprint]:
        """Load generated blueprints for an agent."""
        blueprints_file = Path(__file__).parent / "users" / f"{agent}_generated.json"
        if not blueprints_file.exists():
            return []
        
        with open(blueprints_file) as f:
            data = json.load(f)
        
        return [Blueprint.from_dict(b) for b in data.get("blueprints", [])]
    
    @classmethod
    def get_blueprint(cls, agent: str, blueprint_name: str) -> Optional[Blueprint]:
        """Get a specific blueprint by name."""
        blueprints = cls.load_blueprints(agent)
        for bp in blueprints:
            if bp.name == blueprint_name:
                return bp
        return None
