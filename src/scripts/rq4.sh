#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd)/src/code
MODEL_NAME="Claude"
MODEL_VERSION="claude-3-5-sonnet-20241022"
FEEDBACK="test_feedback"
DATASET="CoderEval"

BASE_CMD="python src/code/evaluate.py --dataset ${DATASET} --model ${MODEL_NAME} --version ${MODEL_VERSION} --feedback ${FEEDBACK} --function single_fix"

experiments=(
    "Baseline (no additional flags)           | "
    # "Single-round fix without persona         | --no_persona"
    # "Single-round fix with cot        | --is_cot"
    # "Single-round fix without docstring       | --no_docstring"
    # "Single-round fix without context         | --no_context"
    # "Single-round fix with few-shot        | --is_few_shot"
    # "Single-round fix without instructions | --no_instructions"
    # "Single-round fix with ES-Shot        | --is_es_shot"
    # "Single-round fix with SA            | --is_sa"
    # "Single-round fix with SG_ICL    | --is_sg_icl"
    # "Single-round fix with SBP            | --is_sbp"
    # "Single-round fix with RR            | --is_rr"
)

for exp in "${experiments[@]}"; do
    IFS='|' read -r name flags <<< "${exp}"
    echo "======================================================================"
    echo "Running experiment: ${name}"
    echo "Flags: ${flags}"
    eval "${BASE_CMD} ${flags}"
    echo -e "Experiment completed: ${name}\n"
done

echo "All experiments completed successfully."