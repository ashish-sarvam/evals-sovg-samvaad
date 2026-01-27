"""
Pairwise Evaluator

Compares two conversation trajectories and judges which model performed better
across multiple criteria.
"""

import json
import re
from typing import Optional

from multilingual_evals.models import AzureModel
from multilingual_evals.config import AZURE_CONFIG


PAIRWISE_EVALUATOR_PROMPT = """You are an expert evaluator comparing two AI agent conversation trajectories for a VOICE-BASED phone call system.

## Context
Both trajectories are from the SAME scenario with the SAME user blueprint. The only difference is which AI model powered the agent. This is a VOICE conversation - the text you see would be spoken aloud to a real person on a phone call. The trajectories may have a different path depending on the user messages, don't judge for the path.

## User Blueprint Information
**Name:** {blueprint_name}
**Description:** {blueprint_description}
**Challenge Type:** {blueprint_tags}
**What This Tests:** {expected_behaviors}

## Trajectory A (Model: {model_a})
{trajectory_a}

## Trajectory B (Model: {model_b})
{trajectory_b}

## Evaluation Criteria

Rate EACH trajectory on these 7 criteria (0-100 scale):

### 1. Voice Naturalness (0-100)
**This is for VOICE/PHONE - judge how it would SOUND when spoken aloud**
- Does it sound natural when read aloud? (Not robotic or scripted)
- Are sentences SHORT and easy to speak? (Long sentences = unnatural for voice)
- Does it use natural speech patterns? ("हाँ जी", "अच्छा", "ठीक है")
- Does it avoid over-formal or overly polite phrasing?
- Does it sound like a real conversation, not a written document?
- NATURAL: "अच्छा, आपका नाम बताइए" 
- UNNATURAL: "Sir, मैं आपसे विनम्र निवेदन करता हूँ कि कृपया अपना शुभ नाम बताने की कृपा करें"
- Score 80+: Sounds like a real human on a phone call, brief and natural
- Score 50-79: Acceptable but sometimes sounds scripted or verbose
- Score <50: Robotic, overly formal, or unnaturally long sentences

### 2. Language Compliance (PASS/FAIL)
**Check if agent follows Indic script + English code-mixing pattern**
- Is the primary language in correct Indic script (Hindi in Devanagari, Bengali in Bengali script, etc.)?
- Are English words written in English script (not transliterated)?
- Example PASS: "आपका survey number क्या है?" (Hindi in Devanagari, English in Roman)
- Example FAIL: "Aapka survey number kya hai?" (Hindi in Roman script - WRONG)
- PASS: Indic language in native script, English words in English
- FAIL: Any use of Roman script for Indic words

### 3. Empathy & Emotional Intelligence (0-100)
**How well does agent handle user's emotional state?**
- Does it acknowledge user's frustration/confusion/distress?
- Does it show patience with difficult users?
- Does it de-escalate tense situations appropriately?
- Does it adapt tone based on user's emotional state?
- Does it avoid being dismissive or cold?
- Score 80+: Excellent emotional awareness and appropriate response
- Score 50-79: Some empathy shown but could be better
- Score <50: Cold, dismissive, or escalates tension

### 4. Recovery & Adaptability (0-100)
**How well does agent handle challenges and adapt?**
- Does it recover gracefully from difficult situations?
- Does it adapt approach when current method isn't working?
- Does it find alternative ways to proceed when stuck?
- Does it handle noise/interruptions/confusion smoothly?
- Score 80+: Excellent recovery, adapts seamlessly
- Score 50-79: Manages challenges but with some difficulty
- Score <50: Gets stuck, fails to adapt, or handles poorly

### 5. No Loops / Progression (0-100)
**Does conversation move forward without getting stuck?**
- Does agent avoid repeating the same thing verbatim?
- Does it progress the conversation forward?
- Does it recognize when to try a different approach?
- Does it avoid circular patterns?
- Score 80+: Smooth progression, no repetition issues
- Score 50-79: Minor repetition but conversation moves forward
- Score <50: Gets stuck in loops or repeats excessively

### 6. Brevity & Clarity (0-100)
**IMPORTANT: Shorter is often better for voice. Verbose does NOT mean more conversational.**
- Does it deliver correct information in SHORT, natural sentences?
- Does it avoid unnecessary elaboration or over-explanation?
- Is each response easy to understand when HEARD (not read)?
- Does it get to the point quickly without filler?
- PREFER: "आपका survey number बताइए" over "Sir, मैं आपसे request करना चाहूँगा कि आप please अपना survey number share करें ताकि मैं आपकी details verify कर सकूँ"
- Score 80+: Concise, gets to point, natural brevity
- Score 50-79: Sometimes too wordy but acceptable
- Score <50: Verbose, over-explains, sounds like reading a script

### 7. Professional Boundaries (0-100)
**Does agent maintain appropriate boundaries?**
- Does it stay on topic or redirect politely?
- Does it handle off-topic requests appropriately?
- Does it know when to end a difficult conversation?
- Does it avoid getting drawn into arguments?
- Score 80+: Excellent boundary management
- Score 50-79: Generally good but some lapses
- Score <50: Poor boundaries, gets derailed easily

### 8. Overall Preference
- If you were the USER receiving this phone call, which agent would you prefer?
- Consider: How comfortable, helpful, and natural did it feel?
- Choose: "A", "B", or "Tie"

## Output Format

Return ONLY valid JSON. For each criterion, include "evidence" with specific quotes from the conversations that justify your score.

{{
  "voice_naturalness": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote from A that shows naturalness/unnaturalness>"],
      "b_examples": ["<quote from B that shows naturalness/unnaturalness>"]
    }}
  }},
  "language_compliance": {{
    "pass_a": <true/false>,
    "pass_b": <true/false>,
    "reason": "<1-2 sentence explanation>",
    "evidence": {{
      "a_examples": ["<quote showing correct/incorrect script usage>"],
      "b_examples": ["<quote showing correct/incorrect script usage>"]
    }}
  }},
  "empathy": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote showing empathy or lack thereof>"],
      "b_examples": ["<quote showing empathy or lack thereof>"]
    }}
  }},
  "recovery": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote showing how A handled a challenge>"],
      "b_examples": ["<quote showing how B handled a challenge>"]
    }}
  }},
  "no_loops": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote showing repetition or smooth progression>"],
      "b_examples": ["<quote showing repetition or smooth progression>"]
    }}
  }},
  "brevity_clarity": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote showing concise/verbose response>"],
      "b_examples": ["<quote showing concise/verbose response>"]
    }}
  }},
  "professional_boundaries": {{
    "score_a": <0-100>,
    "score_b": <0-100>,
    "reason": "<1-2 sentence comparison>",
    "evidence": {{
      "a_examples": ["<quote showing boundary handling>"],
      "b_examples": ["<quote showing boundary handling>"]
    }}
  }},
  "overall_preference": {{
    "winner": "<A/B/Tie>",
    "reason": "<2-3 sentence explanation>",
    "key_differentiators": ["<main reason 1>", "<main reason 2>"]
  }},
  "summary": "<3-4 sentence overall comparison highlighting key differences>"
}}

Be objective and fair. Judge based on how these conversations would feel as actual VOICE phone calls. Always cite specific examples from the conversations to justify your scores."""


class PairwiseEvaluator:
    """Evaluates pairs of conversation trajectories."""

    def __init__(self):
        self.model = AzureModel(
            deployment=AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=None,  # Use default for consistent evaluation
            max_tokens=4000,  # Increased for detailed evidence
        )

    def format_trajectory(self, conversation: list[dict]) -> str:
        """Format a conversation for display."""
        lines = []
        for msg in conversation:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def evaluate(
        self,
        trajectory_a: list[dict],
        trajectory_b: list[dict],
        model_a: str,
        model_b: str,
        blueprint: dict,
    ) -> dict:
        """Evaluate two trajectories against each other.

        Args:
            trajectory_a: First conversation (list of messages)
            trajectory_b: Second conversation (list of messages)
            model_a: Name of model A
            model_b: Name of model B
            blueprint: Blueprint dictionary with metadata

        Returns:
            Evaluation results dictionary
        """
        prompt = PAIRWISE_EVALUATOR_PROMPT.format(
            blueprint_name=blueprint.get("name", "unknown"),
            blueprint_description=blueprint.get("description", ""),
            blueprint_tags=", ".join(blueprint.get("tags", [])),
            expected_behaviors="\n".join(
                f"- {b}" for b in blueprint.get("expected_agent_behavior", [])
            ),
            model_a=model_a,
            model_b=model_b,
            trajectory_a=self.format_trajectory(trajectory_a),
            trajectory_b=self.format_trajectory(trajectory_b),
        )

        response = self.model.get_response(
            [
                {
                    "role": "system",
                    "content": "You are an expert evaluator of conversational AI. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        # Parse JSON
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            result = json.loads(response)

            # Add metadata
            result["model_a"] = model_a
            result["model_b"] = model_b
            result["blueprint"] = blueprint.get("name", "unknown")

            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse evaluation JSON: {e}")
            return {
                "error": str(e),
                "raw_response": response[:500],
                "model_a": model_a,
                "model_b": model_b,
            }

    def calculate_aggregate(self, evaluations: list[dict]) -> dict:
        """Calculate aggregate scores across multiple evaluations.

        Args:
            evaluations: List of evaluation results

        Returns:
            Aggregate statistics
        """
        valid_evals = [e for e in evaluations if "error" not in e]

        if not valid_evals:
            return {"error": "No valid evaluations"}

        # Aggregate by criteria (scored)
        scored_criteria = [
            "voice_naturalness",
            "empathy",
            "recovery",
            "no_loops",
            "brevity_clarity",
            "professional_boundaries",
        ]

        # Boolean criteria
        boolean_criteria = ["language_compliance"]

        aggregate = {
            "total_comparisons": len(valid_evals),
            "criteria_scores": {},
            "boolean_criteria": {},
            "wins": {"a": 0, "b": 0, "tie": 0},
        }

        # Scored criteria aggregation
        for criterion in scored_criteria:
            scores_a = [e[criterion]["score_a"] for e in valid_evals if criterion in e]
            scores_b = [e[criterion]["score_b"] for e in valid_evals if criterion in e]

            if scores_a and scores_b:
                aggregate["criteria_scores"][criterion] = {
                    "avg_a": sum(scores_a) / len(scores_a),
                    "avg_b": sum(scores_b) / len(scores_b),
                    "a_wins": sum(1 for a, b in zip(scores_a, scores_b) if a > b),
                    "b_wins": sum(1 for a, b in zip(scores_a, scores_b) if b > a),
                    "ties": sum(1 for a, b in zip(scores_a, scores_b) if a == b),
                }

        # Boolean criteria aggregation
        for criterion in boolean_criteria:
            passes_a = [
                e[criterion].get("pass_a", False) for e in valid_evals if criterion in e
            ]
            passes_b = [
                e[criterion].get("pass_b", False) for e in valid_evals if criterion in e
            ]

            if passes_a and passes_b:
                aggregate["boolean_criteria"][criterion] = {
                    "pass_rate_a": sum(passes_a) / len(passes_a) * 100,
                    "pass_rate_b": sum(passes_b) / len(passes_b) * 100,
                    "a_passes": sum(passes_a),
                    "b_passes": sum(passes_b),
                }

        # Overall preference counts
        for e in valid_evals:
            if "overall_preference" in e:
                winner = e["overall_preference"].get("winner", "").upper()
                if winner == "A":
                    aggregate["wins"]["a"] += 1
                elif winner == "B":
                    aggregate["wins"]["b"] += 1
                else:
                    aggregate["wins"]["tie"] += 1

        return aggregate

    def print_comparison(self, evaluation: dict):
        """Print a formatted comparison result."""
        print("\n" + "-" * 50)
        print(f"Blueprint: {evaluation.get('blueprint', 'unknown')}")
        print(
            f"Comparing: {evaluation.get('model_a', '?')} vs {evaluation.get('model_b', '?')}"
        )
        print("-" * 50)

        if "error" in evaluation:
            print(f"ERROR: {evaluation['error']}")
            return

        scored_criteria = [
            "voice_naturalness",
            "empathy",
            "recovery",
            "no_loops",
            "brevity_clarity",
            "professional_boundaries",
        ]

        # Print scored criteria
        for criterion in scored_criteria:
            if criterion in evaluation:
                c = evaluation[criterion]
                score_a = c.get("score_a", "?")
                score_b = c.get("score_b", "?")
                winner = (
                    "A" if score_a > score_b else ("B" if score_b > score_a else "=")
                )
                print(f"  {criterion}: A={score_a} vs B={score_b} [{winner}]")
                if c.get("reason"):
                    print(f"    → {c['reason'][:80]}...")

        # Print boolean criteria (language_compliance)
        if "language_compliance" in evaluation:
            c = evaluation["language_compliance"]
            pass_a = "PASS" if c.get("pass_a", False) else "FAIL"
            pass_b = "PASS" if c.get("pass_b", False) else "FAIL"
            print(f"  language_compliance: A={pass_a} vs B={pass_b}")
            if c.get("reason"):
                print(f"    → {c['reason'][:80]}...")

        if "overall_preference" in evaluation:
            pref = evaluation["overall_preference"]
            print(f"\n  OVERALL: {pref.get('winner', '?')}")
            if pref.get("reason"):
                print(f"    → {pref['reason']}")

        if "summary" in evaluation:
            print(f"\n  Summary: {evaluation['summary']}")
