"""
Robustness Blueprint Generator

Uses GPT 5.2 chat to generate ~200 user blueprints across 6 buckets
for testing repetition handling and state continuity.
"""

import json
import re
from pathlib import Path
from typing import Optional

from core.models import AzureModel
from core.config import AZURE_CONFIG


ROBUSTNESS_GENERATOR_PROMPT = """You are an expert at designing test scenarios for conversational AI agents, specifically to test ROBUSTNESS - how well agents handle noise, interruptions, and repetition requests WITHOUT getting stuck in loops or losing state.

## Your Task
Given an agent's system prompt, generate test blueprints across 6 specific buckets. Generate 3-4 blueprints per bucket (~20 total).

## Agent Information
{agent_system_prompt}

## THE 6 TEST BUCKETS (Generate ~30-35 per bucket)

### BUCKET 1: noise_opening (Bad signal from START)
User picks up but has terrible signal immediately.
- Variations: Can't hear name, can't hear purpose, echoes, static
- Key test: Does agent repeat opener correctly or skip verification?
- Example perturbations:
  - "Hello? Hello? Kaun bol raha hai? Awaaz nahi aa rahi"
  - "Kya? Ministry? Kaunsi ministry? Dhire boliye"
  - "Signal bahut kharab hai, naam dobara boliye"

### BUCKET 2: noise_slot (Bad signal during specific questions)
User cooperates initially but signal dies when agent asks about SPECIFIC data (survey number, crop, amount, etc.)
- Variations: Can't hear number, can't hear crop name, partial hearing
- Key test: Does agent repeat ONLY the missed info or restart from beginning?
- Example perturbations:
  - "Survey number kya bola? 4... kya? Dobara boliye"
  - "Crop ka naam samajh nahi aaya, Mag...kya?"
  - "Ruk ruk, village ka naam kya tha?"

### BUCKET 3: interruption_return (User gets interrupted, returns)
User is cooperating but gets interrupted (door, another call, kid crying, etc.) and returns.
- Variations: Short interruption, long interruption, forgets context
- Key test: Does agent resume from correct step or restart?
- Example perturbations:
  - "Ek second... (returns) Haan ji, kya bol rahe the?"
  - "Ruko ruko, darwaza khatkhat... (returns) Achha haan, aage bolo"
  - "Hold on... sorry kya discuss kar rahe the hum?"

### BUCKET 4: partial_confirmation (Confirms but asks unrelated mid-answer)
User starts confirming something but suddenly asks about something else.
- Variations: Confirms crop but asks about agent name, confirms survey but asks about purpose
- Key test: Does agent handle tangent AND return to task, or get confused?
- Example perturbations:
  - "Haan haan crop sahi hai... waise aapka naam kya tha?"
  - "Survey 432... achha Ministry se ho na? Kaunsa department?"
  - "Magfali sahi hai, par ye data kahan se aaya aapke paas?"

### BUCKET 5: confusion (Doesn't understand purpose)
User genuinely doesn't understand what agent wants and keeps asking for clarification.
- Variations: Doesn't understand purpose, doesn't understand verification, suspicious of call
- Key test: Does agent explain clearly or keep repeating same thing?
- Example perturbations:
  - "Samjha nahi, aap chahte kya ho mujhse?"
  - "Matlab verify matlab? Kya verify karna hai?"
  - "Purpose kya hai is call ka? Ye survey kiske liye hai?"

### BUCKET 6: repeated_perturbation (2-3 noise issues → callback offer)
User has MULTIPLE signal issues throughout the call (2-3 times).
- Key test: After 2-3 perturbations, does agent offer to call back?
- This bucket should ALWAYS have perturbation_count >= 2
- Example flow:
  - Turn 2: "Hello? Awaaz nahi aa rahi"
  - Turn 4: "Ruk, signal chala gaya tha, kya bola?"
  - Turn 6: "Phir se? Yaar network bahut kharab hai idhar"
  - Expected: Agent offers callback by now

## BLUEPRINT STRUCTURE

Each blueprint MUST have:

```json
{{
  "name": "noise_opening_cant_hear_name",  // unique snake_case identifier
  "bucket": "noise_opening",  // MUST be one of the 6 buckets
  "tags": ["noise", "opening", "identity_verification"],
  "difficulty": "medium",  // "medium" or "hard"
  "description": "User can't hear agent's name due to bad signal at call start",
  "perturbation_turn": 1,  // Which turn perturbation happens (1-indexed)
  "perturbation_count": 1,  // How many times user asks to repeat (1-3)
  "behavior_steps": [
    "Pick up call but immediately have signal issues - 'Hello? Hello?'",
    "Ask agent to repeat name - 'Kaun bol raha hai? Naam samajh nahi aaya'",
    "Once agent repeats clearly, confirm - 'Achha Divya ji, haan boliye'",
    "Cooperate with verification normally after that"
  ],
  "expected_state_at_perturbation": "Agent should be in greeting/identity verification",
  "expected_recovery": "Agent should repeat ONLY their name and identity, not restart entire script",
  "failure_indicators": [
    "Agent repeats entire opener word-for-word",
    "Agent skips identity verification",
    "Agent gets stuck repeating same sentence",
    "Agent asks about crops before confirming user identity"
  ]
}}

**IMPORTANT: behavior_steps should ONLY describe USER actions/dialogue, NOT agent actions.**
- WRONG: "Agent repeats name slowly" (this is agent action)
- RIGHT: "Ask agent to repeat - 'Naam samajh nahi aaya'" (this is user action)
```

## VARIATIONS TO COVER (generate 3-4 per bucket)

### For noise_opening (3-4 blueprints):
- Can't hear agent's name at all
- Can't hear which ministry/organization
- Can't hear purpose of call
- Background noise (traffic, crowd)

### For noise_slot (3-4 blueprints):
- Can't hear survey number
- Can't hear crop name
- Partial hearing ("4...kya?")
- Signal dies mid-sentence

### For interruption_return (3-4 blueprints):
- Short pause, asks "kya bol rahe the?"
- Long pause, forgets context completely
- Returns with different topic first
- Multiple mini-interruptions

### For partial_confirmation (3-4 blueprints):
- Confirms crop, asks agent name
- Confirms survey, asks "ye data kahan se?"
- Mid-confirmation topic switch
- Asks if this is a scam mid-confirmation

### For confusion (3-4 blueprints):
- Doesn't understand "verify"
- Doesn't understand purpose
- Suspicious, asks for proof
- Elderly confusion

### For repeated_perturbation (3-4 blueprints):
- 2 noise issues, tests patience
- Mixed perturbations (noise + interruption)
- Escalating frustration, user asks for callback
- 3 issues, agent should offer callback

## LANGUAGE GUIDELINES

All example phrases should be in NATURAL Hinglish:
- "Haan" not "Yes"
- "Nahi samjha" not "Didn't understand"  
- "Dobara boliye" not "Please repeat"
- "Ruk ruk" for interruptions
- "Kya bola?" for clarification
- "Samajh nahi aaya" for confusion

## OUTPUT FORMAT

Return ONLY valid JSON:

{{
  "agent_name": "<agent identifier>",
  "total_blueprints": 20,
  "buckets_summary": {{
    "noise_opening": 3,
    "noise_slot": 3,
    "interruption_return": 3,
    "partial_confirmation": 3,
    "confusion": 4,
    "repeated_perturbation": 4
  }},
  "blueprints": [
    // All ~20 blueprints here
  ]
}}

## CRITICAL REQUIREMENTS

1. Generate EXACTLY ~20 blueprints (3-4 per bucket)
2. Each blueprint MUST have the "bucket" field set correctly
3. perturbation_count should be 1 for most, 2-3 for repeated_perturbation bucket
4. **behavior_steps should ONLY describe USER actions/dialogue - NO agent actions**
   - Each step = what the USER says or does
   - Include example dialogue in Hinglish
   - 4-6 user steps per blueprint
5. failure_indicators should list SPECIFIC bad behaviors to watch for
6. Return ONLY valid JSON, no other text

Now analyze the agent and generate comprehensive robustness test blueprints."""


class RobustnessGenerator:
    """Generates robustness test blueprints across 6 buckets."""

    def __init__(self):
        self.model = AzureModel(
            deployment=AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=None,  # Use model default for diversity
            max_tokens=15000,  # Sufficient for ~20 blueprints
        )

    def generate(self, agent_module) -> dict:
        """Generate robustness blueprints for an agent.

        Args:
            agent_module: Module containing SYSTEM_PROMPT and AGENT_NAME

        Returns:
            Dictionary with generated blueprints
        """
        prompt = ROBUSTNESS_GENERATOR_PROMPT.format(
            agent_system_prompt=agent_module.SYSTEM_PROMPT
        )

        print("Generating ~20 robustness blueprints using GPT 5.2 chat...")
        print("This may take 30-60 seconds...")

        response = self.model.get_response(
            [
                {
                    "role": "system",
                    "content": "You are an expert test case designer. You output only valid JSON with exactly the structure requested.",
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

            # Count by bucket
            bucket_counts = {}
            for bp in blueprints["blueprints"]:
                bucket = bp.get("bucket", "unknown")
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

            print(f"\nGenerated {len(blueprints['blueprints'])} total blueprints")
            print("By bucket:")
            for bucket, count in sorted(bucket_counts.items()):
                print(f"  - {bucket}: {count}")

            return blueprints

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print("Raw response (first 2000 chars):")
            print(response[:2000])
            raise

    def generate_by_bucket(self, agent_module, bucket: str, count: int = 4) -> dict:
        """Generate blueprints for a specific bucket only.

        Useful for generating in batches if full generation fails.
        """
        bucket_prompt = self._get_bucket_specific_prompt(bucket, count)

        prompt = f"""Generate {count} robustness test blueprints for the following bucket ONLY:

## Bucket: {bucket}
{bucket_prompt}

## Agent Information
{agent_module.SYSTEM_PROMPT}

Return ONLY valid JSON with structure:
{{
  "bucket": "{bucket}",
  "blueprints": [
    // {count} blueprints for this bucket
  ]
}}
"""

        print(f"Generating {count} blueprints for bucket: {bucket}...")

        response = self.model.get_response(
            [
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
        )

        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse: {e}")
            raise

    def _get_bucket_specific_prompt(self, bucket: str, count: int) -> str:
        """Get detailed prompt for a specific bucket."""
        prompts = {
            "noise_opening": f"""
Generate {count} scenarios where user has bad signal FROM THE START of the call.
Variations: Can't hear name, can't hear ministry, echoes, static, complete cutoff.
Key test: Does agent repeat opener correctly or skip verification?
""",
            "noise_slot": f"""
Generate {count} scenarios where signal dies during SPECIFIC questions (survey number, crop, etc.)
User cooperates initially, but can't hear specific data.
Key test: Does agent repeat ONLY missed info or restart from beginning?
""",
            "interruption_return": f"""
Generate {count} scenarios where user gets interrupted (door, another call, family) and returns.
Variations: Short pause, long pause, forgets what was discussed.
Key test: Does agent resume from correct step or restart?
""",
            "partial_confirmation": f"""
Generate {count} scenarios where user starts confirming but asks unrelated question mid-answer.
"Haan crop sahi hai... waise aapka naam kya tha?"
Key test: Does agent handle tangent AND return to task?
""",
            "confusion": f"""
Generate {count} scenarios where user genuinely doesn't understand purpose.
Keeps asking "kya verify karna hai?", "purpose kya hai?", "samjha nahi".
Key test: Does agent explain clearly or keep repeating same thing?
""",
            "repeated_perturbation": f"""
Generate {count} scenarios with MULTIPLE (2-3) perturbations throughout call.
After 2-3 issues, agent should offer to call back.
Set perturbation_count to 2 or 3 for all in this bucket.
""",
        }
        return prompts.get(bucket, "")

    def save(self, blueprints: dict, agent_name: str) -> Path:
        """Save generated blueprints to file."""
        users_dir = Path(__file__).parent / "users"
        users_dir.mkdir(parents=True, exist_ok=True)

        output_path = users_dir / f"{agent_name}_generated.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blueprints, f, ensure_ascii=False, indent=2)

        print(f"Saved blueprints to: {output_path}")
        return output_path

    def generate_and_save(self, agent_module, agent_name: str) -> dict:
        """Generate and save blueprints in one step."""
        blueprints = self.generate(agent_module)
        self.save(blueprints, agent_name)
        return blueprints

    def print_summary(self, blueprints: dict):
        """Print a summary of generated blueprints."""
        print("\n" + "=" * 70)
        print("ROBUSTNESS BLUEPRINTS SUMMARY")
        print("=" * 70)

        print(f"\nAgent: {blueprints.get('agent_name', 'unknown')}")
        print(f"Total blueprints: {len(blueprints['blueprints'])}")

        # Group by bucket
        by_bucket = {}
        for bp in blueprints["blueprints"]:
            bucket = bp.get("bucket", "unknown")
            if bucket not in by_bucket:
                by_bucket[bucket] = []
            by_bucket[bucket].append(bp)

        print("\nBy Bucket:")
        for bucket in [
            "noise_opening",
            "noise_slot",
            "interruption_return",
            "partial_confirmation",
            "confusion",
            "repeated_perturbation",
        ]:
            count = len(by_bucket.get(bucket, []))
            print(f"  {bucket}: {count} blueprints")

            # Show first 3 examples
            for bp in by_bucket.get(bucket, [])[:3]:
                print(f"    - {bp['name']}: {bp.get('description', '')[:50]}...")

        print("\n" + "=" * 70)
