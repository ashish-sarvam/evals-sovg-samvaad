"""
Analyze training data for repetitive patterns and conversation flow issues.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import argparse


def extract_assistant_endings(messages: list) -> list:
    """Extract how assistant messages end (last sentence/question)."""
    endings = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"].strip()
            # Get last sentence (after last period, or last question)
            sentences = re.split(r'[।.?]', content)
            last = sentences[-1].strip() if sentences[-1].strip() else (sentences[-2].strip() if len(sentences) > 1 else content)
            if last:
                endings.append(last)
    return endings


def find_repetitive_patterns(data: list[dict], min_repeat: int = 3) -> dict:
    """Find patterns that repeat across conversations."""
    all_endings = []
    
    for item in data:
        messages = item.get("messages", [])
        endings = extract_assistant_endings(messages)
        all_endings.extend(endings)
    
    # Normalize and count
    normalized = [e.lower().strip() for e in all_endings]
    counter = Counter(normalized)
    
    return {k: v for k, v in counter.most_common(50) if v >= min_repeat}


def check_escalation_regression(messages: list) -> dict:
    """Check if conversation regresses after escalation (e.g., legal action mentioned then goes back to soft warning)."""
    escalation_keywords = {
        "legal": ["legal action", "legal", "कानूनी"],
        "blacklist": ["blacklist", "ban", "mana", "रोक"],
        "credit_score": ["credit score", "cibil", "क्रेडिट स्कोर"],
    }
    
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    
    escalation_timeline = []
    for i, msg in enumerate(assistant_msgs):
        content = msg["content"].lower()
        level = 0
        if any(kw in content for kw in escalation_keywords["credit_score"]):
            level = 1
        if any(kw in content for kw in escalation_keywords["legal"]):
            level = 2
        if any(kw in content for kw in escalation_keywords["blacklist"]):
            level = 3
        escalation_timeline.append(level)
    
    # Check for regression (going from higher level to lower)
    regressions = []
    for i in range(1, len(escalation_timeline)):
        if escalation_timeline[i] < escalation_timeline[i-1] and escalation_timeline[i-1] >= 2:
            regressions.append({
                "turn": i,
                "from_level": escalation_timeline[i-1],
                "to_level": escalation_timeline[i]
            })
    
    return {
        "timeline": escalation_timeline,
        "regressions": regressions,
        "has_regression": len(regressions) > 0
    }


def check_same_ending_pattern(messages: list, threshold: float = 0.6) -> dict:
    """Check if assistant messages end with similar patterns too often."""
    endings = extract_assistant_endings(messages)
    
    if len(endings) < 3:
        return {"same_ending_ratio": 0, "is_repetitive": False}
    
    # Check for common patterns
    date_patterns = ["कब तक", "कब", "date", "when", "तारीख"]
    payment_patterns = ["pay कर", "payment", "पेमेंट", "जमा कर"]
    
    date_count = sum(1 for e in endings if any(p in e.lower() for p in date_patterns))
    payment_count = sum(1 for e in endings if any(p in e.lower() for p in payment_patterns))
    
    same_pattern_ratio = max(date_count, payment_count) / len(endings)
    
    return {
        "total_assistant_turns": len(endings),
        "date_ask_count": date_count,
        "payment_ask_count": payment_count,
        "same_ending_ratio": round(same_pattern_ratio, 2),
        "is_repetitive": same_pattern_ratio >= threshold,
        "sample_endings": endings[:5]
    }


def check_exit_strategy(messages: list) -> dict:
    """Check if conversation has proper exit after escalation."""
    exit_patterns = [
        "madad", "help", "मदद",
        "plan", "प्लान",
        "baad mein", "later", "बाद में",
        "callback", "call back",
        "note", "नोट"
    ]
    
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    
    has_legal_warning = any(
        "legal" in m["content"].lower() or "कानूनी" in m["content"].lower()
        for m in assistant_msgs
    )
    
    has_exit_after_warning = False
    legal_seen = False
    for msg in assistant_msgs:
        content = msg["content"].lower()
        if "legal" in content or "कानूनी" in content:
            legal_seen = True
        if legal_seen and any(p in content for p in exit_patterns):
            has_exit_after_warning = True
            break
    
    return {
        "has_legal_warning": has_legal_warning,
        "has_exit_strategy": has_exit_after_warning,
        "needs_fix": has_legal_warning and not has_exit_after_warning
    }


def analyze_file(filepath: str) -> list[dict]:
    """Analyze a single JSONL file."""
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                messages = item.get("messages", [])
                
                result = {
                    "line": line_num,
                    "num_messages": len(messages),
                    "ending_analysis": check_same_ending_pattern(messages),
                    "escalation_analysis": check_escalation_regression(messages),
                    "exit_analysis": check_exit_strategy(messages),
                }
                
                # Flag problematic samples
                result["issues"] = []
                if result["ending_analysis"]["is_repetitive"]:
                    result["issues"].append("repetitive_endings")
                if result["escalation_analysis"]["has_regression"]:
                    result["issues"].append("escalation_regression")
                if result["exit_analysis"]["needs_fix"]:
                    result["issues"].append("no_exit_strategy")
                
                results.append(result)
                
            except json.JSONDecodeError as e:
                print(f"JSON error at line {line_num}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze training data for conversation issues")
    parser.add_argument("filepath", help="Path to JSONL training file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    print(f"Analyzing: {args.filepath}\n")
    
    results = analyze_file(args.filepath)
    
    # Summary
    total = len(results)
    repetitive = sum(1 for r in results if "repetitive_endings" in r["issues"])
    regression = sum(1 for r in results if "escalation_regression" in r["issues"])
    no_exit = sum(1 for r in results if "no_exit_strategy" in r["issues"])
    any_issue = sum(1 for r in results if r["issues"])
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:           {total}")
    print(f"Repetitive endings:      {repetitive} ({100*repetitive/total:.1f}%)")
    print(f"Escalation regression:   {regression} ({100*regression/total:.1f}%)")
    print(f"No exit strategy:        {no_exit} ({100*no_exit/total:.1f}%)")
    print(f"Samples with any issue:  {any_issue} ({100*any_issue/total:.1f}%)")
    print("=" * 60)
    
    if args.verbose:
        print("\nPROBLEMATIC SAMPLES:")
        for r in results:
            if r["issues"]:
                print(f"\nLine {r['line']}: {r['issues']}")
                print(f"  Endings: {r['ending_analysis']['sample_endings'][:3]}")
    
    # Show problematic line numbers for filtering
    problem_lines = [r["line"] for r in results if r["issues"]]
    if problem_lines:
        print(f"\nLines to review/filter: {problem_lines[:20]}{'...' if len(problem_lines) > 20 else ''}")


if __name__ == "__main__":
    main()
