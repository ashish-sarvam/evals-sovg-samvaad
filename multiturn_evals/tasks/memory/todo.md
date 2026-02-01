# Memory/Personalization Training Data Generation

## Goal
Create bulk training data to improve model's ability to use stored user context appropriately.

---

## Current State ✓
- [x] Eval framework with 6 verification criteria
- [x] User profiles with domain-specific info (financial, work, farming)
- [x] Agent-specific business context
- [x] User blueprints (conversation flows)
- [x] Verifier passes agent's stored info for accurate evaluation

---

## Step 1: Generate Diverse Agents (1000+)

### 1.1 Use LLM to generate realistic agent configurations
- Input: Existing agent examples (IDFC, Tata Cap, UC, DCS, General)
- Output: New agent configs with:
  - Agent role/domain (banking, insurance, e-commerce, healthcare, govt, etc.)
  - System prompt
  - First user message
  - Realistic business context templates

### 1.2 Filter for realism
- Use small/fast model to score realism (1-5)
- Keep agents scoring >= 4
- Ensure domain diversity (not all banking)

### 1.3 Agent domains to cover
- [ ] Banking (collection, sales, support)
- [ ] Insurance (claims, renewal, sales)
- [ ] E-commerce (order status, returns, support)
- [ ] Healthcare (appointment, follow-up, prescription)
- [ ] Government (scheme info, document status)
- [ ] Telecom (recharge, plan upgrade, complaint)
- [ ] Travel (booking, cancellation, support)
- [ ] Education (admission, fees, course info)

---

## Step 2: Generate User Profiles per Agent

### 2.1 For each agent, generate matching user profile
- Personal info (name, location, age, profession)
- Domain-relevant info:
  - Banking → account details, salary, loan history
  - Healthcare → medical history, appointments
  - E-commerce → order history, preferences
  - Govt → scheme enrollment, documents

### 2.2 Generate business context
- What agent knows from CRM/database
- Previous interactions history
- Current status (e.g., pending payment, open ticket)

### 2.3 Ensure info density
- Each profile should have 5-10 usable facts
- Mix of: must-use, could-use, should-not-use info

---

## Step 3: Generate User Blueprints (Conversation Flows)

### 3.1 For each agent, create 2-3 user blueprints
- Different user goals/scenarios
- Different user moods (cooperative, confused, busy, frustrated)

### 3.2 Blueprint should define
- User's goal (what they want to achieve)
- Conversation flow (turn-by-turn guidance)
- What info user should NOT volunteer
- Expected agent behavior (use X info, don't ask Y)

---

## Step 4: Generate Training Conversations

### 4.1 Positive examples (agent does it right)
- Uses stored info when relevant
- Doesn't ask redundant questions
- Personalizes naturally
- Uses business context appropriately

### 4.2 Negative examples (agent fails) - for contrastive learning
- Asks for info it already has
- Ignores relevant stored context
- Uses info at wrong time (creepy)
- Over-personalizes (uses name every sentence)

### 4.3 Edge cases to cover
- [ ] User corrects stored info ("Actually I moved to Delhi")
- [ ] Stored info is partially relevant
- [ ] Multiple stored facts, only some relevant
- [ ] User asks about something agent has no stored info for
- [ ] Agent should ask clarifying question (not redundant)

---

## Step 5: Quality Control & Filtering

### 5.1 Automated verification
- Run verifier on all generated conversations
- Filter: keep only 6/6 pass for positive, specific failures for negative

### 5.2 Diversity checks
- Ensure language diversity (hi-en, ta-en, etc.)
- Ensure domain diversity
- Ensure scenario diversity

### 5.3 Manual spot-check
- Sample 5% for human review
- Check for unrealistic patterns

---

## Step 6: Training Data Format

### 6.1 SFT format
```json
{
  "system_prompt": "...(includes user context)...",
  "conversation": [...],
  "label": "positive" | "negative",
  "failure_type": null | "redundant_question" | "ignored_context" | ...
}
```

### 6.2 Preference/DPO format
```json
{
  "system_prompt": "...",
  "conversation_context": [...previous turns...],
  "chosen": "agent response using memory well",
  "rejected": "agent response ignoring memory"
}
```

---

## Step 7: Metrics to Track

- Overall pass rate (6 criteria)
- Per-criterion pass rate
- Domain-wise performance
- Language-wise performance
- Failure type distribution

---

## Open Questions

1. How many conversations per agent? (10? 50? 100?)
2. What ratio of positive:negative examples?
3. Should negative examples be synthetically created or natural failures?
4. How to handle agents where memory is less relevant?
5. Multi-turn memory (info shared in turn 2, used in turn 5)?

---

## Quick Start Commands

```bash
# Run eval for all agents
python -m multilingual_evals --task memory --agent general_assistant,idfc_main,tata_cap_sales,uc_scheduling,dcs --languages hi-en -p 5

# Run single agent
python -m multilingual_evals --task memory --agent idfc_main --user cooperative --languages hi-en
```
