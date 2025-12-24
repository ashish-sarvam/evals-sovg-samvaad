## TAU BENCH:

cd tau-bench
export LEPTON_API_TOKEN=""

python run.py \
    --model "/ashish_sarvam_ai/nemo-rlvr/results/qwen3-30b-a3b-sft-2nodes/step_3364-hf" \
    --user-model "/ashish_sarvam_ai/nemo-rlvr/results/qwen3-30b-a3b-sft-2nodes/step_3364-hf" \
    --model-provider openai \
    --user-model-provider openai \
    --api-base "https://h1v6kgoi-qwen30ba3-sft-fc-2-9-max16-2.xenon.lepton.run/v1/" \
    --user-api-base "https://h1v6kgoi-qwen30ba3-sft-fc-2-9-max16-2.xenon.lepton.run/v1/" \
    --env retail \
    --start-index 0 \
    --end-index 100
