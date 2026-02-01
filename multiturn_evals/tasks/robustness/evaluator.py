"""
Robustness Evaluator

LLM-as-judge evaluator specifically for robustness/repetition handling.
Evaluates 4 key criteria:
1. Reset rate: Did agent incorrectly re-run opener/identity gate?
2. Repair quality: Did agent repeat only missing info?
3. State continuity: Did agent resume the correct step?
4. Over-asking: Did agent re-ask already-answered questions?
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.models import AzureModel
from core.config import AZURE_CONFIG, MODEL_SETTINGS


ROBUSTNESS_EVALUATOR_PROMPT = """You are an expert evaluator assessing how well a conversational AI agent handles NOISE, INTERRUPTIONS, and REPETITION REQUESTS.

## Context
The user in this conversation had signal issues, interruptions, or confusion and asked the agent to repeat/clarify information. Your job is to evaluate how the agent handled this.

## Blueprint Information
- **Bucket**: {bucket}
- **Scenario**: {description}
- **Perturbation Turn**: {perturbation_turn}
- **Expected State at Perturbation**: {expected_state}
- **Expected Recovery**: {expected_recovery}
- **Known Failure Indicators**: {failure_indicators}

## Conversation to Evaluate
```
{conversation}
```

## EVALUATION CRITERIA (Score each 1-5)

### 1. Reset Rate (Lower is better → Invert to: "No Unnecessary Reset")
**Question**: Did the agent incorrectly restart from the beginning when a simple repeat was sufficient?

- 5 (Excellent): Agent NEVER restarted unnecessarily. Only repeated the specific missed info.
- 4 (Good): Agent mostly avoided restarts, maybe one minor over-explanation.
- 3 (Average): Agent partially restarted (repeated opener/name unnecessarily once).
- 2 (Poor): Agent restarted/repeated opener multiple times when not needed.
- 1 (Bad): Agent fully restarted from scratch every time user asked to repeat.

**Red Flags (score 1-2)**:
- Re-introducing themselves after user already acknowledged who they are
- "Ministry of Agriculture से Divya बोल रही हूँ" repeated when user just asked about survey number
- Full greeting repeated when only one detail was missed

### 2. Repair Quality (Higher is better)
**Question**: When asked to repeat, did the agent repeat ONLY what was missed, or did they over-repeat?

- 5 (Excellent): Agent precisely repeated only the missed information.
- 4 (Good): Agent repeated missed info + small context, appropriate.
- 3 (Average): Agent repeated more than needed but not egregiously.
- 2 (Poor): Agent repeated large chunks unnecessarily.
- 1 (Bad): Agent repeated everything from the start or gave irrelevant info.

**Good Examples**:
- User: "Survey number samajh nahi aaya" → Agent: "Survey number 432, Lavana village"
- User: "Crop ka naam kya tha?" → Agent: "Magfali"

**Bad Examples**:
- User: "Survey number samajh nahi aaya" → Agent: "Ministry of Agriculture से... survey number 432..."
- Agent gives full script instead of targeted answer

### 3. State Continuity (Higher is better)  
**Question**: After the perturbation, did the agent resume from the correct step?

- 5 (Excellent): Agent perfectly resumed exactly where they left off.
- 4 (Good): Agent resumed correctly with minor redundancy.
- 3 (Average): Agent resumed but skipped a step or went back one step.
- 2 (Poor): Agent went back multiple steps unnecessarily.
- 1 (Bad): Agent completely lost track and restarted or got stuck in a loop.

**Check for**:
- If user confirmed identity, agent shouldn't re-verify identity
- If user confirmed survey 432, agent should move to next survey (not ask 432 again)
- Agent should remember what was already confirmed

### 4. Over-Asking (Lower over-asking is better → Score "Efficiency")
**Question**: Did the agent re-ask questions that were already answered?

- 5 (Excellent): Agent NEVER re-asked anything already confirmed.
- 4 (Good): Agent avoided re-asking, maybe minor clarification.
- 3 (Average): Agent re-asked one thing that was already confirmed.
- 2 (Poor): Agent re-asked multiple already-confirmed things.
- 1 (Bad): Agent repeatedly asked the same questions, ignoring previous answers.

**Red Flags**:
- "क्या आप Rameshbhai Patel जी हैं?" after user already confirmed
- "Survey 432 में crop Magfali है?" after user already said yes
- Same question appearing 3+ times

### 5. Looping Detection (Critical - Boolean)
**Question**: Did the agent get stuck in a repetitive loop?

LOOP INDICATORS:
- Same sentence repeated 3+ times with minimal variation
- Agent saying essentially the same thing turn after turn
- No progress in conversation despite user responding
- Robotic, non-adaptive repetition

Score: PASS (no loop) or FAIL (loop detected)

### 6. Callback Offer (For repeated_perturbation bucket only)
**Question**: After 2-3 perturbations, did the agent offer to call back later?

- If perturbation_count >= 2: Agent SHOULD offer callback
- If perturbation_count == 1: Not required, mark as N/A

Score: PASS (offered callback or N/A), PARTIAL (hinted but didn't explicitly offer), FAIL (didn't offer despite multiple issues)

## OUTPUT FORMAT

Return ONLY valid JSON:

{{
  "scores": {{
    "no_unnecessary_reset": {{
      "score": <1-5>,
      "reason": "<specific evidence from conversation>",
      "evidence": ["<quote1>", "<quote2>"]
    }},
    "repair_quality": {{
      "score": <1-5>,
      "reason": "<specific evidence>",
      "evidence": ["<quote showing good/bad repair>"]
    }},
    "state_continuity": {{
      "score": <1-5>,
      "reason": "<what step agent was at, what step they resumed at>",
      "evidence": ["<quote showing correct/incorrect state>"]
    }},
    "over_asking_efficiency": {{
      "score": <1-5>,
      "reason": "<did agent re-ask confirmed things?>",
      "evidence": ["<quotes of re-asking if any>"]
    }},
    "looping": {{
      "result": "PASS" | "FAIL",
      "reason": "<evidence of loop or lack thereof>",
      "evidence": ["<repeated phrases if loop detected>"]
    }},
    "callback_offer": {{
      "result": "PASS" | "PARTIAL" | "FAIL" | "N/A",
      "reason": "<did agent offer to call back?>"
    }}
  }},
  "overall_score": <1-5 average of numeric scores>,
  "summary": "<2-3 sentence overall assessment>",
  "critical_issues": ["<list any major failures>"],
  "positive_behaviors": ["<list good handling examples>"]
}}

Be STRICT in evaluation. A score of 5 should be rare - only for truly excellent handling.
Look for SPECIFIC evidence in the conversation for each criterion.
Quote actual agent messages as evidence."""


class RobustnessEvaluator:
    """Evaluates agent robustness using LLM-as-judge."""

    def __init__(self):
        self.model = AzureModel(
            deployment=AZURE_CONFIG["deployments"]["5_2_chat"],
            temperature=None,
            max_tokens=MODEL_SETTINGS["verifier"]["max_tokens"],
        )
        self._print_lock = threading.Lock()

    def _sync_print(self, *args, **kwargs):
        """Thread-safe print."""
        with self._print_lock:
            print(*args, **kwargs)

    def evaluate_trajectory(
        self,
        trajectory: dict,
        blueprint: dict,
    ) -> dict:
        """Evaluate a single trajectory.

        Args:
            trajectory: Dict with conversation, model, etc.
            blueprint: Blueprint dict with expected behaviors

        Returns:
            Evaluation result dict
        """
        # Format conversation for prompt
        conv_text = ""
        for msg in trajectory.get("conversation", []):
            role = msg["role"].upper()
            content = msg["content"]
            conv_text += f"{role}: {content}\n\n"

        # Build prompt
        prompt = ROBUSTNESS_EVALUATOR_PROMPT.format(
            bucket=blueprint.get("bucket", "unknown"),
            description=blueprint.get("description", ""),
            perturbation_turn=blueprint.get("perturbation_turn", "unknown"),
            expected_state=blueprint.get("expected_state_at_perturbation", ""),
            expected_recovery=blueprint.get("expected_recovery", ""),
            failure_indicators=json.dumps(blueprint.get("failure_indicators", []), ensure_ascii=False),
            conversation=conv_text,
        )

        response = self.model.get_response(
            [
                {
                    "role": "system",
                    "content": "You are a strict evaluator. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        # Parse response
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            evaluation = json.loads(response)
            
            # Add metadata
            evaluation["trajectory_metadata"] = {
                "model": trajectory.get("model"),
                "blueprint": trajectory.get("blueprint"),
                "language": trajectory.get("language"),
                "timestamp": trajectory.get("timestamp"),
            }
            
            return evaluation

        except json.JSONDecodeError as e:
            self._sync_print(f"Failed to parse evaluation: {e}")
            return {
                "error": str(e),
                "raw_response": response[:1000],
            }

    def evaluate_all_trajectories(
        self,
        trajectories_dir: Path,
        blueprints: list[dict],
        output_dir: Path,
        max_workers: int = 10,
    ) -> dict:
        """Evaluate all trajectories in a directory.

        Args:
            trajectories_dir: Directory containing trajectory JSON files
            blueprints: List of blueprint dicts (to match with trajectories)
            output_dir: Where to save evaluation results
            max_workers: Number of parallel workers

        Returns:
            Summary statistics
        """
        # Create blueprint lookup
        blueprint_lookup = {bp["name"]: bp for bp in blueprints}
        
        # Find all trajectory files
        trajectory_files = list(trajectories_dir.glob("**/*.json"))
        self._sync_print(f"Found {len(trajectory_files)} trajectory files")

        results = []
        results_lock = threading.Lock()

        def evaluate_file(traj_file: Path) -> Optional[dict]:
            """Evaluate a single trajectory file."""
            try:
                with open(traj_file) as f:
                    trajectory = json.load(f)
                
                blueprint_name = trajectory.get("blueprint")
                blueprint = blueprint_lookup.get(blueprint_name, {})
                
                if not blueprint:
                    self._sync_print(f"Warning: No blueprint found for {blueprint_name}")
                    blueprint = {"name": blueprint_name, "bucket": "unknown"}
                
                self._sync_print(f"Evaluating: {traj_file.name}")
                evaluation = self.evaluate_trajectory(trajectory, blueprint)
                
                # Save individual result
                result_file = output_dir / f"{traj_file.stem}_eval.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(evaluation, f, ensure_ascii=False, indent=2)
                
                with results_lock:
                    results.append(evaluation)
                
                return evaluation
                
            except Exception as e:
                self._sync_print(f"Error evaluating {traj_file}: {e}")
                return None

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_file, f): f 
                for f in trajectory_files
            }
            
            for future in as_completed(futures):
                traj_file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._sync_print(f"Error processing {traj_file}: {e}")

        # Generate summary statistics
        summary = self._generate_summary(results)
        
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self._sync_print(f"\nSaved {len(results)} evaluations to {output_dir}")
        self._sync_print(f"Summary saved to {summary_file}")
        
        return summary

    def _generate_summary(self, results: list[dict]) -> dict:
        """Generate summary statistics from evaluation results."""
        if not results:
            return {"error": "No results to summarize"}

        # Filter out errors
        valid_results = [r for r in results if "scores" in r]
        
        if not valid_results:
            return {"error": "No valid results", "total": len(results)}

        # Aggregate scores
        score_sums = {
            "no_unnecessary_reset": 0,
            "repair_quality": 0,
            "state_continuity": 0,
            "over_asking_efficiency": 0,
        }
        score_counts = {k: 0 for k in score_sums}
        
        loop_results = {"PASS": 0, "FAIL": 0}
        callback_results = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "N/A": 0}
        
        # Track by bucket
        by_bucket = {}
        
        for result in valid_results:
            scores = result.get("scores", {})
            
            # Aggregate numeric scores
            for key in score_sums:
                if key in scores and "score" in scores[key]:
                    score_sums[key] += scores[key]["score"]
                    score_counts[key] += 1
            
            # Track boolean results
            if "looping" in scores:
                loop_result = scores["looping"].get("result", "PASS")
                loop_results[loop_result] = loop_results.get(loop_result, 0) + 1
            
            if "callback_offer" in scores:
                callback_result = scores["callback_offer"].get("result", "N/A")
                callback_results[callback_result] = callback_results.get(callback_result, 0) + 1
            
            # Track by bucket
            bucket = result.get("trajectory_metadata", {}).get("blueprint", "unknown")
            # Extract bucket from blueprint name if possible
            for b in ["noise_opening", "noise_slot", "interruption_return", 
                      "partial_confirmation", "confusion", "repeated_perturbation"]:
                if b in bucket:
                    bucket = b
                    break
            
            if bucket not in by_bucket:
                by_bucket[bucket] = {"count": 0, "scores": {k: [] for k in score_sums}}
            by_bucket[bucket]["count"] += 1
            for key in score_sums:
                if key in scores and "score" in scores[key]:
                    by_bucket[bucket]["scores"][key].append(scores[key]["score"])

        # Calculate averages
        avg_scores = {}
        for key in score_sums:
            if score_counts[key] > 0:
                avg_scores[key] = round(score_sums[key] / score_counts[key], 2)
            else:
                avg_scores[key] = None

        # Calculate overall
        valid_avgs = [v for v in avg_scores.values() if v is not None]
        overall_avg = round(sum(valid_avgs) / len(valid_avgs), 2) if valid_avgs else None

        # Bucket averages
        bucket_summaries = {}
        for bucket, data in by_bucket.items():
            bucket_summaries[bucket] = {
                "count": data["count"],
                "avg_scores": {
                    k: round(sum(v) / len(v), 2) if v else None
                    for k, v in data["scores"].items()
                }
            }

        return {
            "total_evaluated": len(valid_results),
            "total_errors": len(results) - len(valid_results),
            "timestamp": datetime.now().isoformat(),
            "overall_average": overall_avg,
            "average_scores": avg_scores,
            "looping_results": loop_results,
            "callback_results": callback_results,
            "by_bucket": bucket_summaries,
            "interpretation": {
                "no_unnecessary_reset": "Higher = Agent didn't restart unnecessarily (5 = perfect)",
                "repair_quality": "Higher = Agent repeated only what was needed (5 = perfect)",
                "state_continuity": "Higher = Agent resumed from correct step (5 = perfect)",
                "over_asking_efficiency": "Higher = Agent didn't re-ask confirmed things (5 = perfect)",
                "looping": "PASS = No loop detected, FAIL = Agent got stuck",
                "callback_offer": "For repeated_perturbation: PASS = Offered callback appropriately",
            },
        }

    def print_summary(self, summary: dict):
        """Print a formatted summary."""
        print("\n" + "=" * 70)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal Evaluated: {summary.get('total_evaluated', 0)}")
        print(f"Errors: {summary.get('total_errors', 0)}")
        print(f"Overall Average Score: {summary.get('overall_average', 'N/A')}/5")
        
        print("\n--- Average Scores (out of 5) ---")
        for key, value in summary.get("average_scores", {}).items():
            display_name = key.replace("_", " ").title()
            print(f"  {display_name}: {value if value else 'N/A'}")
        
        print("\n--- Looping Results ---")
        loop = summary.get("looping_results", {})
        print(f"  PASS (no loop): {loop.get('PASS', 0)}")
        print(f"  FAIL (loop detected): {loop.get('FAIL', 0)}")
        
        print("\n--- Callback Offer (repeated_perturbation only) ---")
        callback = summary.get("callback_results", {})
        print(f"  PASS: {callback.get('PASS', 0)}")
        print(f"  PARTIAL: {callback.get('PARTIAL', 0)}")
        print(f"  FAIL: {callback.get('FAIL', 0)}")
        print(f"  N/A: {callback.get('N/A', 0)}")
        
        if summary.get("by_bucket"):
            print("\n--- By Bucket ---")
            for bucket, data in summary["by_bucket"].items():
                avg_overall = [v for v in data["avg_scores"].values() if v]
                avg = round(sum(avg_overall) / len(avg_overall), 2) if avg_overall else "N/A"
                print(f"  {bucket}: {data['count']} trajectories, avg score: {avg}")
        
        print("\n" + "=" * 70)
