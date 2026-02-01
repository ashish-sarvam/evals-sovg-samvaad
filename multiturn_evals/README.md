# Multiturn Evals

Evaluation framework for testing multilingual conversational agents.

## Setup

```bash
cd multiturn_evals
poetry install
```

Copy config and add your API keys:
```bash
cp multilingual_evals/config.example.py multilingual_evals/config.py
# Edit config.py with your API keys
```

## Available Tasks

| Task | Description |
|------|-------------|
| `multilingual` | Test language compliance (Indic script usage) |
| `english_user` | Test language maintenance when user speaks English |
| `colloquial` | Test rural vs urban tone adaptation |
| `conversationality` | Test conversational robustness with challenging users |
| `robustness` | Test repetition handling, state continuity, no looping |

## Quick Commands

### 1. Multilingual Task

Test if agent maintains correct Indic script:

```bash
# Single language (Hindi)
poetry run python -m multilingual_evals --task multilingual --agent dcs --user cooperative --model lepton --languages hi-en

# All languages, parallel
poetry run python -m multilingual_evals --task multilingual --agent dcs --user cooperative --model lepton -p 11

# All users
poetry run python -m multilingual_evals --task multilingual --agent dcs --model lepton -p 5
```

### 2. English User Task

Test if agent maintains Indic language when user speaks English:

```bash
poetry run python -m multilingual_evals --task english_user --agent dcs --user cooperative --model lepton -p 5
```

### 3. Colloquial Task

Test rural vs urban tone:

```bash
poetry run python -m multilingual_evals --task colloquial --agent dcs --user rural_cooperative --model lepton -p 5
```

### 4. Conversationality Task

Test robustness with challenging user behaviors:

```bash
# Step 1: Generate blueprints (one-time)
poetry run python -m multilingual_evals --task conversationality --agent dcs --mode generate-blueprints

# Step 2: Generate trajectories for each model
poetry run python -m multilingual_evals --task conversationality --agent dcs --mode generate --model tinker --user noise_user
poetry run python -m multilingual_evals --task conversationality --agent dcs --mode generate --model azure --user noise_user

# Step 3: Run pairwise evaluation (Tinker vs Azure)
poetry run python -m multilingual_evals --task conversationality --agent dcs --mode evaluate -p 10

# Step 4: View results
poetry run python -m multilingual_evals --task conversationality --agent dcs --mode results
```

### 5. Robustness Task

Test how well agent handles noise, interruptions, and repetition requests without looping:

```bash
# Step 1: Generate ~20 blueprints across 6 buckets (one-time)
poetry run python -m multilingual_evals --task robustness --agent dcs --mode generate-blueprints

# Step 2: Generate trajectories
# All blueprints
poetry run python -m multilingual_evals --task robustness --agent dcs --mode generate --model tinker

# Single bucket only
poetry run python -m multilingual_evals --task robustness --agent dcs --mode generate --model tinker --bucket noise_opening

# Specific blueprints
poetry run python -m multilingual_evals --task robustness --agent dcs --mode generate --model tinker --user noise_opening_cant_hear_name

# Step 3: Run LLM evaluation
poetry run python -m multilingual_evals --task robustness --agent dcs --mode evaluate -p 10

# Step 4: View results
poetry run python -m multilingual_evals --task robustness --agent dcs --mode results
```

**Robustness Buckets:**
| Bucket | Description |
|--------|-------------|
| `noise_opening` | Bad signal from START of call |
| `noise_slot` | Bad signal during specific questions (survey/crop) |
| `interruption_return` | User gets interrupted, returns |
| `partial_confirmation` | Confirms but asks unrelated mid-answer |
| `confusion` | Doesn't understand purpose |
| `repeated_perturbation` | Multiple issues → should offer callback |

**Evaluation Criteria:**
- **No Unnecessary Reset**: Did agent avoid restarting from scratch?
- **Repair Quality**: Did agent repeat only what was missed?
- **State Continuity**: Did agent resume from correct step?
- **Over-asking Efficiency**: Did agent avoid re-asking confirmed info?
- **Looping Detection**: Did agent get stuck in loops?
- **Callback Offer**: Did agent offer callback after multiple issues?

## Model Providers

Use `--model` to specify the agent model:

| Model | Description |
|-------|-------------|
| `tinker` | Internal Tinker model (default) |
| `azure` | Azure OpenAI (GPT-4.1-mini) |
| `lepton` | Lepton hosted model |
| `openai` | Standard OpenAI API |

## Common Options

```bash
--task TASK          # Task to run (required)
--agent AGENT        # Agent to test (required)
--user USER          # User persona (comma-separated for multiple)
--model MODEL        # Agent model provider
--languages LANGS    # Comma-separated language codes (e.g., hi-en,bn-en)
-p N, --parallel N   # Run N languages in parallel
--skip-verification  # Skip verification step
--verbose            # Show detailed output
--temperature TEMP   # Override agent temperature
--bucket BUCKET      # For robustness task: filter by test bucket
--mode MODE          # For conversationality/robustness: generate-blueprints, generate, evaluate, results
```

## Output Structure

Results are saved to:
```
artifacts/eval_results/
  {task}/
    {agent}/
      {user}/
        {model}/
          {task}_{language}_{timestamp}.json
          summary_{timestamp}.json
```

## Available Languages

```
hi-en (Hindi), bn-en (Bengali), gu-en (Gujarati), kn-en (Kannada),
ml-en (Malayalam), mr-en (Marathi), or-en (Odia), pa-en (Punjabi),
ta-en (Tamil), te-en (Telugu), en (English)
```

## List Available Options

```bash
# Show available tasks, agents, and users
poetry run python -m multilingual_evals
```
