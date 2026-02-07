#!/bin/bash
export PYTHONPATH=$(pwd)/src/code
DATASET=("SWE-Bench-verified")
# FEEDBACK_TYPES=("test_feedback" "compiler_feedback" "llm_skilled_feedback" "llm_expert_feedback" "minimal_feedback" "mixed_feedback")
FEEDBACK_TYPES=("test_feedback" "compiler_feedback")
declare -A MODELS=(
     ["GPT"]="gpt-4o-2024-11-20"
    # ["Claude"]="claude-3-5-sonnet-20241022"
    #  ["GLM"]="glm-4-plus"
    #  ["Qwen"]="qwen2.5-72b-instruct"
    #  ["Deepseek"]="deepseek-r1-250528"
)

for DATASET in "${DATASET[@]}"; do
  for MODEL in "${!MODELS[@]}"; do
      VERSION="${MODELS[$MODEL]}"

      for FEEDBACK in "${FEEDBACK_TYPES[@]}"; do

          echo "Single-round fixing for model $MODEL ($VERSION), feedback $FEEDBACK, dataset $DATASET"

          python src/code/evaluate.py \
              --dataset "$DATASET" \
              --model "$MODEL" \
              --version "$VERSION" \
              --feedback "$FEEDBACK" \
              --function "single_fix"
      done
  done
done
