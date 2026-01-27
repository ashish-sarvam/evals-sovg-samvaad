"""
Multilingual Task

Tests: Does the agent correctly converse in different Indian languages?
"""

from multilingual_evals.tasks import BaseTask, TaskConfig
from multilingual_evals.languages import Language, SUPPORTED_LANGUAGES

from multilingual_evals.tasks.multilingual.users import get_user_prompts, list_users, list_agents
from multilingual_evals.tasks.multilingual.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
)


class MultilingualTask(BaseTask):
    """Tests agent's ability to converse in different languages."""

    def __init__(self, agent: str, user: str):
        self.config = TaskConfig(
            name="multilingual",
            description="Test multilingual conversation ability",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
        )
        self.agent = agent
        self.user = user
        
        # Validate agent and user
        user_prompts = get_user_prompts(agent)
        if user not in user_prompts:
            available = ", ".join(user_prompts.keys())
            raise ValueError(f"Unknown user: {user}. Available: {available}")
        
        self._user_prompts = user_prompts

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
            language=language.name, conversation=conv_text
        )

    @staticmethod
    def list_users(agent: str) -> list[str]:
        return list_users(agent)

    @staticmethod
    def list_agents() -> list[str]:
        return list_agents()
