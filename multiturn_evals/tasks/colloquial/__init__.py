"""
Colloquial Task (Rural vs Urban)

Tests: Does the agent correctly adapt its tone for rural vs urban audiences?
Evaluates the degree of colloquialness (0=urban/casual, 100=rural/formal).
"""

from tasks import BaseTask, TaskConfig
from core.languages import Language, SUPPORTED_LANGUAGES

from tasks.colloquial.users import (
    get_user_prompts,
    get_user_tone,
    list_users,
    list_agents,
)
from tasks.colloquial.users.dcs import TONE_PROMPTS
from tasks.colloquial.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
    TONE_DESCRIPTIONS,
)


class ColloquialTask(BaseTask):
    """Tests agent's ability to adapt tone for rural vs urban audiences."""

    def __init__(self, agent: str, user: str):
        self.config = TaskConfig(
            name="colloquial",
            description="Test rural vs urban tone adaptation",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
        )
        self.agent = agent
        self.user = user
        self.expected_tone = get_user_tone(agent, user)
        
        # Validate agent and user
        user_prompts = get_user_prompts(agent)
        if user not in user_prompts:
            available = ", ".join(user_prompts.keys())
            raise ValueError(f"Unknown user: {user}. Available: {available}")
        
        self._user_prompts = user_prompts
        self._tone_prompt = TONE_PROMPTS.get(self.expected_tone, "")

    def get_tone_prompt(self, language: Language) -> str:
        """Get the tone prompt to inject into agent system prompt."""
        return self._tone_prompt.format(LANGUAGE=language.prompt_language)

    def get_user_prompt(self, language: Language) -> str:
        return self._user_prompts[self.user].format(LANGUAGE=language.prompt_language)

    def get_verifier_system_prompt(self) -> str:
        return VERIFIER_SYSTEM_PROMPT

    def get_verifier_user_prompt(
        self, conversation: list[dict], language: Language
    ) -> str:
        conv_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in conversation
        )
        return VERIFIER_USER_TEMPLATE.format(
            language=language.name,
            expected_tone=self.expected_tone,
            tone_description=TONE_DESCRIPTIONS.get(self.expected_tone, ""),
            conversation=conv_text,
        )

    @staticmethod
    def list_users(agent: str) -> list[str]:
        return list_users(agent)

    @staticmethod
    def list_agents() -> list[str]:
        return list_agents()
