"""User prompts registry for colloquial task.

Uses common user behaviors from language_users.
Tests rural (formal) vs urban (casual) language styles.
"""

from tasks.language_users import get_user_behaviors, list_agents, list_users as _list_users, AGENT_USERS

# Tone prompts - added to agent system prompt
RURAL_TONE_PROMPT = """
## Style: Formal (Rural Audience)
You are a formal agent speaking in a government/PSU-style tone.

**Audience:** Rural or semi-rural Indian users.

**Language rules:**
- Use {LANGUAGE} for most words and sentence structure.
- Use English only for common terms (e.g., survey, OTP, ministry, mobile, EMI, loan).
- Avoid English verbs and connectors (no "so", "basically", "actually", "like").
- Keep language clear and formal.
- Avoid colloquial/casual phrases.
- Keep Indic words in native script and English words in English script.
- Be respectful and use simple language.
"""

URBAN_TONE_PROMPT = """
## Style: Natural & Conversational (Urban Audience)
You are a natural agent speaking in a conversational tone.

**Audience:** Urban Indian users.

**Language rules:**
- Keep sentence structure primarily in {LANGUAGE}.
- Use English for common nouns (e.g., app, plan, payment, order, survey, loan, EMI).
- You may use English connectors like "so", "basically", "overall", "actually".
- Avoid long English clauses or fully English sentences.
- Be empathetic and conversational.
- Keep Indic words in native script and English words in English script.
"""

TONE_PROMPTS = {
    "rural": RURAL_TONE_PROMPT,
    "urban": URBAN_TONE_PROMPT,
}

# Language rules for rural vs urban users
RURAL_RULES = """You are simulating a RURAL user in a phone conversation. Respond in {LANGUAGE}.

## Your Character:
- You are from a village, speak simply and formally
- You speak mostly in {LANGUAGE} with minimal English
- You use respectful language (ji, sahab, etc.)

## Script Rules:
- Respond in {LANGUAGE} only (minimal English)
- Keep responses SHORT and formal

"""

URBAN_RULES = """You are simulating an URBAN user in a phone conversation. Respond in {LANGUAGE} mixed with English.

## Your Character:
- You are educated, urban, comfortable with English
- You speak casually and conversationally
- You use English words naturally in your speech

## Script Rules:
- Respond in {LANGUAGE} with English words mixed in
- Keep responses casual and conversational

"""


def _build_prompt(behavior: dict, tone: str) -> str:
    """Build a complete prompt from behavior dict and tone."""
    rules = RURAL_RULES if tone == "rural" else URBAN_RULES
    return rules + behavior["behavior"]


def get_user_prompts(agent_name: str) -> dict[str, str]:
    """Get user prompts for a specific agent.
    
    Returns both rural and urban versions of each user.
    """
    behaviors = get_user_behaviors(agent_name)
    prompts = {}
    
    for name, behavior in behaviors.items():
        # Create rural and urban versions
        prompts[f"rural_{name}"] = _build_prompt(behavior, "rural")
        prompts[f"urban_{name}"] = _build_prompt(behavior, "urban")
    
    return prompts


def get_user_tone(agent_name: str, user: str) -> str:
    """Get the expected tone for a user (rural or urban)."""
    if user.startswith("rural_"):
        return "rural"
    elif user.startswith("urban_"):
        return "urban"
    return "rural"  # Default


def list_users(agent_name: str) -> list[str]:
    """List available users for an agent (rural and urban versions)."""
    base_users = _list_users(agent_name)
    users = []
    for u in base_users:
        users.append(f"rural_{u}")
        users.append(f"urban_{u}")
    return users
