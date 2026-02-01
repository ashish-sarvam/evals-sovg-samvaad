"""
Blueprint Generator

Uses GPT 5.2 chat to automatically generate varied user blueprints
for testing conversational robustness of any agent.
"""

import json
import re
from pathlib import Path
from typing import Optional

from core.models import AzureModel
from core.config import AZURE_CONFIG


BLUEPRINT_GENERATOR_PROMPT = """You are an expert at designing user simulation blueprints for testing conversational AI agents.

## Your Task
Given an agent's system prompt, generate 8-10 VARIED user blueprints that PUSH THE LIMITS of the agent's conversational abilities. Each blueprint should stress-test a specific challenge type. The goal is to find breaking points and edge cases.

## Agent Information
{agent_system_prompt}

## Blueprint Requirements

### 1. CHALLENGE CATEGORIES - Generate blueprints across these (pick 8-10):

**Audio/Signal Issues:**
- `noise_user`: Bad signal from the start or mid-call, "hello? hello?", keeps asking for repetition
- `echo_user`: Hears echo, gets confused, repeats themselves, talks over agent

**Out of Scope Requests:**
- `off_topic_asker`: Persistently asks about unrelated services (loans, schemes, complaints)
- `scheme_asker`: Keeps asking about PM Kisan, subsidies, other government schemes

**Sensitive/Difficult Topics:**
- `political_user`: Brings up politics, government criticism, party affiliations, gets heated
- `angry_user`: Frustrated, takes anger out on agent, uses harsh language
- `suspicious_user`: Convinced it's a scam, demands proof, threatens to report
- `emotionally_distressed`: Crying, overwhelmed, in crisis, needs emotional support before task

**Loop/Repetition Testing:**
- `loop_pusher`: Asks agent to repeat, "start from beginning", "phir se bolo"
- `persistent_denier`: Keeps saying "No" / "Nahi" to agreeing - denies information, denies loan payment refuses to cooperate
- `clarification_seeker`: Endless "what do you mean?", "samjha nahi", never satisfied

**Context/Flow Challenges:**
- `interruption_user`: Gets interrupted, forgets what was discussed, "kya bol rahe the aap?"
- `multitasking_user`: Distracted, gives one-word answers, "haan... kya?"
- `topic_switcher`: Abruptly changes topics, goes on tangents, hard to bring back

**Comprehension Issues:**
- `confused_elder`: Elderly, hard of hearing, doesn't understand purpose, needs multiple explanations
- `language_struggler`: Struggles to express, long pauses, incomplete sentences
- `misunderstander`: Completely misunderstands what agent wants, gives wrong info confidently

**Cooperation Issues:**
- `impatient_user`: "Jaldi bolo", "time nahi hai", wants to cut call
- `denier`: Denies all information agent has ("galat hai", "ye mera nahi hai")
- `skeptic`: Questions every statement, doesn't trust any data, wants proof for everything

**Edge Cases (Agent-Specific):**
- Analyze the agent's prompt and create 2-3 EXTREME challenges specific to this agent
- What would make this agent completely fail?
- What edge cases could break the workflow?

### 2. BEHAVIOR FLOW - Use HIGH-LEVEL behavioral instructions (NOT exact scripts):

Each blueprint should have `behavior_steps` - a list of high-level instructions for how the user behaves:

Example for noise_user:
```
behavior_steps: [
  "Answer call but immediately have signal issues - 'hello? hello? awaaz nahi aa rahi'",
  "Keep asking agent to repeat - can only hear partial words",
  "Get frustrated with bad signal - 'yahan network bahut kharab hai'",
  "Signal improves slightly - can hear but still unclear",
  "Finally understand and cooperate OR give up and ask to call back"
]
```

Example for persistent_denier:
```
behavior_steps: [
  "Greet normally",
  "When asked to confirm identity - say 'Nahi, galat number hai'",
  "When agent insists - keep denying 'Main wo nahi hoon'",
  "If agent mentions any data - deny it 'Ye galat information hai'",
  "Continue denying everything until agent gives up or finds a way through"
]
```

### 3. TIMING FLEXIBILITY:
- Challenge can start from TURN 1 (e.g., noise_user picks up with bad signal)
- Or start normally and challenge emerges at turn 2-4
- Specify `challenge_start`: "immediate" OR "after_greeting" OR "mid_conversation"

### 4. PUSH TO LIMITS:
- Each blueprint should be EXTREME enough to potentially break the agent
- User should persist with challenging behavior for 5-8 turns
- Don't make recovery easy - agent must really work for it
- Total conversation should be 8-10 USER TURNS max

### 5. METADATA:
- `name`: snake_case identifier
- `tags`: What it tests (e.g., ["denial_handling", "patience", "persistence"])
- `difficulty`: "medium" or "hard" (no easy ones - we're stress testing)
- `challenge_start`: "immediate" | "after_greeting" | "mid_conversation"
- `description`: One-line persona description
- `behavior_steps`: List of 5-8 high-level behavior instructions
- `breaking_point`: What would make a bad agent fail here?
- `expected_agent_behavior`: List of 3-5 ideal behaviors

### 6. LANGUAGE:
- User responses in NATURAL Hinglish (Hindi-English mix)
- Include realistic expressions: "haan", "achha", "matlab", "nahi nahi", "kya?"
- Match formality to user type

## Output Format

Return ONLY valid JSON:

{{
  "agent_name": "<agent identifier>",
  "agent_task_summary": "<one line summary>",
  "generation_notes": "<what challenges are most relevant for this agent>",
  "blueprints": [
    {{
      "name": "noise_user",
      "tags": ["noise_handling", "patience", "repetition_tolerance"],
      "difficulty": "medium",
      "challenge_start": "immediate",
      "description": "User with terrible phone signal throughout call",
      "behavior_steps": [
        "Pick up with bad signal - 'Hello? Hello? Kaun bol raha hai? Awaaz nahi aa rahi'",
        "Can only hear fragments - ask to repeat everything",
        "Get frustrated - 'Yaar network bahut kharab hai idhar'",
        "Partial understanding - give incomplete responses",
        "Keep struggling with signal for most of call",
        "Either cooperate at end OR ask agent to call back later"
      ],
      "breaking_point": "Agent might skip verification steps or give up too easily",
      "expected_agent_behavior": [
        "Patiently repeat information multiple times",
        "Use shorter, clearer sentences",
        "Don't skip mandatory verification despite difficulties",
        "Offer to call back if signal doesn't improve",
        "Stay calm and professional throughout"
      ]
    }},
    {{
      "name": "persistent_denier",
      "tags": ["denial_handling", "persistence", "loan_denial"],
      "difficulty": "hard",
      "challenge_start": "after_greeting",
      "description": "User who confirms identity but denies all loan/payment information",
      "behavior_steps": [
        "Greet normally - 'Haan boliye'",
        "Confirm identity when asked - 'Haan main hi hoon'",
        "When agent mentions loan/payment - 'Nahi nahi, maine koi loan nahi liya'",
        "Keep denying - 'Ye galat hai, mere naam pe koi loan nahi hai'",
        "Get frustrated - 'Aap galat information de rahe ho, check karo phir se'",
        "Firmly deny any outstanding amount - 'Maine sab chuka diya, kuch baaki nahi'",
        "Threaten to complain if agent insists - 'Aise galat calls karoge toh complaint karunga'"
      ],
      "breaking_point": "Agent might accept denial without proper verification, or get stuck arguing with user",
      "expected_agent_behavior": [
        "Stay calm despite repeated denials",
        "Provide specific details (dates, amounts) to verify",
        "Don't argue but politely restate facts",
        "Know when to escalate or end call professionally",
        "Document the denial appropriately"
      ]
    }}
  ]
}}

## CRITICAL REQUIREMENTS

1. Generate EXACTLY 8-10 blueprints
2. Each blueprint tests a DIFFERENT challenge type
3. Blueprints should PUSH LIMITS - we want to find agent's breaking points
4. Use HIGH-LEVEL behavior_steps, NOT exact turn-by-turn scripts
5. Include at least 2 AGENT-SPECIFIC extreme challenges
6. All conversations must complete within 8-10 USER TURNS
7. Return ONLY valid JSON, no other text

Now analyze the agent prompt and generate stress-test blueprints."""


class BlueprintGenerator:
    """Generates user blueprints for testing conversational robustness."""

    def __init__(self):
        self.model = AzureModel(
            deployment=AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=None,  # Use model default
            max_tokens=15000,  # Long output for all blueprints
        )

    def generate(self, agent_module) -> dict:
        """Generate user blueprints for an agent.

        Args:
            agent_module: Module containing SYSTEM_PROMPT and AGENT_NAME

        Returns:
            Dictionary with generated blueprints
        """
        # Build the prompt with agent's system prompt
        prompt = BLUEPRINT_GENERATOR_PROMPT.format(
            agent_system_prompt=agent_module.SYSTEM_PROMPT
        )

        print("Generating blueprints using GPT 5.2 chat...")
        print("This may take 30-60 seconds...")

        response = self.model.get_response(
            [
                {
                    "role": "system",
                    "content": "You are an expert test case designer for conversational AI. You output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        # Parse JSON from response
        try:
            # Try to extract JSON if wrapped in code blocks
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            blueprints = json.loads(response)

            # Validate structure
            if "blueprints" not in blueprints:
                raise ValueError("Response missing 'blueprints' key")

            if len(blueprints["blueprints"]) < 5:
                raise ValueError(
                    f"Only {len(blueprints['blueprints'])} blueprints generated, expected 8-10"
                )

            print(f"Successfully generated {len(blueprints['blueprints'])} blueprints")
            return blueprints

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print("Raw response:")
            print(response[:1000])
            raise

    def save(self, blueprints: dict, agent_name: str) -> Path:
        """Save generated blueprints to file.

        Args:
            blueprints: Generated blueprints dictionary
            agent_name: Name of the agent

        Returns:
            Path to saved file
        """
        # Ensure users directory exists
        users_dir = Path(__file__).parent / "users"
        users_dir.mkdir(parents=True, exist_ok=True)

        output_path = users_dir / f"{agent_name}_generated.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blueprints, f, ensure_ascii=False, indent=2)

        print(f"Saved blueprints to: {output_path}")
        return output_path

    def generate_and_save(self, agent_module, agent_name: str) -> dict:
        """Generate and save blueprints in one step.

        Args:
            agent_module: Module containing SYSTEM_PROMPT
            agent_name: Name of the agent (e.g., "dcs")

        Returns:
            Generated blueprints dictionary
        """
        blueprints = self.generate(agent_module)
        self.save(blueprints, agent_name)
        return blueprints

    def print_summary(self, blueprints: dict):
        """Print a summary of generated blueprints."""
        print("\n" + "=" * 60)
        print("GENERATED BLUEPRINTS SUMMARY")
        print("=" * 60)

        print(f"\nAgent: {blueprints.get('agent_name', 'unknown')}")
        print(f"Task: {blueprints.get('agent_task_summary', 'unknown')}")

        if blueprints.get("generation_notes"):
            print(f"Notes: {blueprints['generation_notes']}")

        print(f"\nTotal blueprints: {len(blueprints['blueprints'])}")
        print("\nBlueprints:")

        for bp in blueprints["blueprints"]:
            difficulty_icon = {"medium": "🟡", "hard": "🔴"}.get(
                bp.get("difficulty", "medium"), "⚪"
            )
            print(f"\n  {difficulty_icon} {bp['name']}")
            print(f"     Tags: {', '.join(bp.get('tags', []))}")
            print(
                f"     Challenge start: {bp.get('challenge_start', 'after_greeting')}"
            )
            print(f"     Description: {bp.get('description', '')}")
            steps = bp.get("behavior_steps", bp.get("turns", []))
            print(f"     Behavior steps: {len(steps)}")
            if bp.get("breaking_point"):
                print(f"     Breaking point: {bp['breaking_point'][:60]}...")

        print("\n" + "=" * 60)
