from __future__ import annotations

import logging
import traceback

from feedback.evaluation_feedback import (
    build_feedback_mapping,
    build_feedback_mapping_swebench,
    build_repair_prompt_for_round,
    get_mixed_feedback,
)
from evaluation.evaluation_types import task_id
from feedback.feedback import run_test
from core.patch_generator import generate_patch_from_function_fix
from swebench_ext.swebench_helpers import apply_and_test_patch
from core.utils import extract_repaired_code, get_model_response, truncate_feedback


class EvaluatorBase:
    def __init__(self, dataset: str, options, logger: logging.Logger):
        self.dataset = dataset
        self.options = options
        self.logger = logger

    def run_candidate_round(
        self,
        candidate,
        ques: dict,
        feedback_type: str,
        model_version: str,
        current_round: int,
        **kwargs,
    ) -> bool:
        raise NotImplementedError


class LocalEvaluator(EvaluatorBase):
    def run_candidate_round(
        self,
        candidate,
        ques: dict,
        feedback_type: str,
        model_version: str,
        current_round: int,
        **kwargs,
    ) -> bool:
        current_code = candidate.current_code
        feedback_mapping = build_feedback_mapping(self.dataset, ques, current_code)
        if current_round == 1:
            current_feedback = candidate.repair_history[0]["feedback"]
        else:
            current_feedback = feedback_mapping[feedback_type]()

        if feedback_type == "test_feedback":
            current_feedback = truncate_feedback(current_feedback)

        prompt = build_repair_prompt_for_round(
            ques,
            current_code,
            current_feedback,
            self.dataset,
            self.options,
        )
        self.logger.info(
            f"prompt:\n{prompt}\n",
            extra={"task_id": task_id(ques), "round": current_round, "stage": "prompt"},
        )
        response = get_model_response(model_version, prompt)
        fixed_code = extract_repaired_code(response)
        self.logger.info(
            f"response:\n{response}\n",
            extra={
                "task_id": task_id(ques),
                "round": current_round,
                "stage": "response",
            },
        )

        if not fixed_code:
            candidate.record_round(current_round, "", current_feedback, False)
            return False

        new_exit_code, _ = run_test(
            self.dataset,
            fixed_code,
            ques.get("_id", None),
            ques.get("test", None),
        )
        is_true = new_exit_code in (0, 5)
        candidate.record_round(current_round, fixed_code, current_feedback, is_true)
        return not is_true


class SweBenchEvaluator(EvaluatorBase):
    def run_candidate_round(
        self,
        candidate,
        ques: dict,
        feedback_type: str,
        model_version: str,
        current_round: int,
        **kwargs,
    ) -> bool:
        swe_ctx = kwargs.get("swe_ctx")
        timeout = kwargs.get("timeout")

        current_code = candidate.current_code
        last_round = candidate.repair_history[-1]
        last_test_feedback = last_round.get("test_feedback") or last_round["feedback"]

        if feedback_type == "test_feedback":
            current_feedback = last_test_feedback
        elif feedback_type == "mixed_feedback":
            current_feedback = get_mixed_feedback(
                "SWE-Bench-verified",
                current_code,
                ques,
                {"test_feedback": last_test_feedback},
            )
        else:
            feedback_mapping = build_feedback_mapping_swebench(ques, current_code)
            current_feedback = feedback_mapping.get(feedback_type, lambda: "")()

        prompt = build_repair_prompt_for_round(
            ques,
            current_code,
            current_feedback,
            "SWE-Bench-verified",
            self.options,
        )
        self.logger.info(
            f"prompt:\n{prompt}\n",
            extra={"task_id": task_id(ques), "round": current_round, "stage": "prompt"},
        )
        response = get_model_response(model_version, prompt)
        fixed_code = extract_repaired_code(response)
        self.logger.info(
            f"response:\n{response}\n",
            extra={
                "task_id": task_id(ques),
                "round": current_round,
                "stage": "response",
            },
        )

        if not fixed_code:
            candidate.record_round(
                current_round,
                "",
                current_feedback,
                False,
                test_feedback=last_test_feedback,
            )
            return False

        if swe_ctx is None or timeout is None:
            candidate.record_round(
                current_round,
                fixed_code,
                current_feedback,
                False,
                test_feedback=last_test_feedback,
            )
            return False

        success, patch, _, error = generate_patch_from_function_fix(ques, fixed_code)
        if not success or not patch:
            feedback_msg = error or "Patch generation failed"
            candidate.record_round(
                current_round,
                fixed_code,
                feedback_msg,
                False,
                test_feedback=last_test_feedback,
            )
            return False

        try:
            resolved, test_fb = apply_and_test_patch(
                patch,
                swe_ctx,
                f"{model_version}_{feedback_type}",
                timeout,
                self.logger,
            )
        except Exception as exc:
            self.logger.error(
                f"Error applying/testing patch: {exc}\n{traceback.format_exc()}",
                extra={
                    "task_id": task_id(ques),
                    "round": current_round,
                    "stage": "error",
                },
            )
            resolved, test_fb = False, str(exc)

        candidate.record_round(
            current_round,
            fixed_code,
            current_feedback,
            resolved,
            test_feedback=test_fb,
        )
        return not resolved


def get_evaluator(dataset: str, options, logger: logging.Logger) -> EvaluatorBase:
    if dataset == "SWE-Bench-verified":
        return SweBenchEvaluator(dataset, options, logger)
    return LocalEvaluator(dataset, options, logger)
