export OPENAI_API_BASE="https://h1v6kgoi-qwen3-30b-a3b-tools.xenon.lepton.run/v1"
export OPENAI_API_KEY="EMPTY"

python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --model-provider openai \
  --user-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 2 \
  --start-index 0 \
  --end-index 100
