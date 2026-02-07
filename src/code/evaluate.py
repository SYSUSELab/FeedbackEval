import logging
import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# SWE-bench harness imports
ROOT_DIR = Path(__file__).resolve().parents[2]
SWEBENCH_DIR = ROOT_DIR / "SWE-bench"
if SWEBENCH_DIR.exists():
    sys.path.insert(0, str(SWEBENCH_DIR))
    sys.path.insert(0, str(SWEBENCH_DIR / "swebench"))

from feedback.feedback import run_test
from core.utils import (
    FEEDBACK_TYPES,
    read_jsonl,
    write_jsonl,
    setup_logging,
    RunContext,
    ProgressiveCache,
)
from core.patch_generator import generate_patch_from_function_fix
from evaluation.evaluation_types import PromptOptions, RepairCandidate, task_id
from evaluation.evaluators import get_evaluator
from swebench_ext.swebench_helpers import (
    SWEBENCH_TIMEOUT,
    build_swebench_context,
    cleanup_swebench_context,
    ensure_eval_script,
    run_swebench_batch_evaluation,
    update_predictions_with_results,
)

from swebench.harness.constants import KEY_INSTANCE_ID, RUN_EVALUATION_LOG_DIR

logger = logging.getLogger("evaluate")


def _single_save_path(
    model_name: str,
    model_version: str,
    feedback: str,
    dataset: str,
    options: PromptOptions,
) -> str:
    if dataset == "SWE-Bench-verified":
        save_dir = os.path.join("results", model_name, dataset, "single")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{model_version}_{feedback}.predictions.jsonl")

    if all(
        [
            options.use_docstring,
            options.use_context,
            options.use_persona,
            options.use_instructions,
        ]
    ) and not any(
        [
            options.use_cot,
            options.use_few_shot,
            options.use_sa,
            options.use_sg_icl,
            options.use_sbp,
            options.use_rr,
            options.use_es_shot,
        ]
    ):
        save_dir = os.path.join("results", model_name, dataset, "single")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{model_version}_{feedback}.jsonl")

    save_dir = os.path.join("results/rq4-prompt")
    os.makedirs(save_dir, exist_ok=True)
    config_suffix = (
        f"doc_{int(options.use_docstring)}_ctx_{int(options.use_context)}_"
        f"per_{int(options.use_persona)}_cot_{int(options.use_cot)}_shot_"
        f"{int(options.use_few_shot)}_ins_{int(options.use_instructions)}_sa_{int(options.use_sa)}_sg_{int(options.use_sg_icl)}"
        f"_sbp_{int(options.use_sbp)}_rr_{int(options.use_rr)}_es_{int(options.use_es_shot)}"
    )
    logger.info(
        f"config_suffix={config_suffix}",
        extra={"stage": "config"},
    )
    return os.path.join(save_dir, f"{model_version}_{feedback}_{config_suffix}.jsonl")


def fix_code(
    file_path,
    models: list[tuple[str, str]],
    feedback_types: list[str],
    dataset,
    options: PromptOptions,
    max_rounds: int,
):
    if dataset == "HumanEval":
        id_field = "task_id"
    elif dataset == "CoderEval":
        id_field = "_id"
    elif dataset == "SWE-Bench-verified":
        id_field = "instance_id"
    else:
        id_field = "instance_id"

    # Create caches for each (model, version, feedback_type)
    caches = {}

    for model_name, model_version in models:
        for ft in feedback_types:
            if max_rounds == 1:
                if len(feedback_types) > 1:
                    pass

                save_path = _single_save_path(
                    model_name,
                    model_version,
                    ft,
                    dataset,
                    options,
                )
            else:
                save_dir = os.path.join("results", model_name, dataset, "multi")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, f"{model_version}_multi_round_{ft}.jsonl"
                )

            caches[(model_name, model_version, ft)] = ProgressiveCache(
                save_path, id_field=id_field
            )
            logger.info(
                f"Cache initialized for {model_name} {model_version} {ft} at {save_path}",
                extra={"stage": "init"},
            )

    _run_fix_code(
        file_path,
        models,
        feedback_types,
        dataset,
        options,
        max_rounds=max_rounds,
        caches=caches,
    )


def _run_fix_code(
    file_path,
    models: list[tuple[str, str]],
    feedback_types: list[str],
    dataset,
    options: PromptOptions,
    max_rounds: int,
    caches: Dict[tuple[str, str, str], ProgressiveCache] | None = None,
):
    fixed_list = []
    ques_list = read_jsonl(file_path)
    logger.info(
        f"Evaluating file: {file_path}",
        extra={"stage": "start"},
    )
    desc = "Fixing code" if max_rounds == 1 else "Multi-Round Fixing code"
    cache_lock = threading.Lock()

    # Prepare tasks with pending items
    pending_tasks = []
    for ques in ques_list:
        task_key = task_id(ques)
        pending_items = []

        for model_name, model_version in models:
            for ft in feedback_types:
                cache_key = (model_name, model_version, ft)
                if caches:
                    if cache_key in caches and not caches[cache_key].has(task_key):
                        pending_items.append((model_name, model_version, ft))
                else:
                    pending_items.append((model_name, model_version, ft))

        if pending_items:
            pending_tasks.append((ques, pending_items))

    def _process_ques_with_multiple_feedbacks(
        ques: dict, pending_items: list[tuple[str, str, str]]
    ) -> list[tuple[str, dict]]:
        task_key = task_id(ques)
        results = []

        swe_multi = dataset == "SWE-Bench-verified" and max_rounds > 1
        swe_ctx = None
        evaluator = None

        try:
            evaluator = get_evaluator(dataset, options, logger)
            if swe_multi:
                swe_ctx, ques = build_swebench_context(ques, logger)

            for model_name, model_version, feedback_type in pending_items:
                task_logger = None
                handler = None
                try:
                    if swe_multi:
                        run_id = f"swe-multi-{model_version.replace('/', '__')}-{feedback_type}"
                        log_dir = (
                            RUN_EVALUATION_LOG_DIR / run_id / ques[KEY_INSTANCE_ID]
                        )
                        log_dir.mkdir(parents=True, exist_ok=True)
                        swe_ctx["run_id"] = run_id
                        swe_ctx["log_dir"] = log_dir

                        # Setup instance-specific logger
                        task_logger = logging.Logger(
                            f"evaluate.{task_key}_{model_version}_{feedback_type}"
                        )
                        handler = logging.FileHandler(log_dir / "run_instance.log")
                        handler.setFormatter(
                            logging.Formatter(
                                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                            )
                        )
                        task_logger.addHandler(handler)
                        task_logger.setLevel(logging.INFO)

                        ensure_eval_script(swe_ctx, log_dir)

                    else:
                        task_logger = logger

                    list_results = ques.get("false_results", [])
                    candidate_processes = [
                        RepairCandidate.from_candidate(
                            i, candidate, feedback_type, dataset, ques
                        )
                        for i, candidate in enumerate(list_results)
                    ]
                    active_candidates = candidate_processes[:]
                    current_round = 1

                    while current_round <= max_rounds and active_candidates:
                        next_active_candidates = []
                        for candidate_proc in active_candidates:
                            try:
                                if swe_multi:
                                    evaluator.logger = task_logger
                                    needs_next_round = evaluator.run_candidate_round(
                                        candidate_proc,
                                        ques,
                                        feedback_type,
                                        model_version,
                                        current_round,
                                        swe_ctx=swe_ctx,
                                        timeout=SWEBENCH_TIMEOUT,
                                    )
                                else:
                                    evaluator.logger = task_logger
                                    needs_next_round = evaluator.run_candidate_round(
                                        candidate_proc,
                                        ques,
                                        feedback_type,
                                        model_version,
                                        current_round,
                                    )
                                if needs_next_round:
                                    next_active_candidates.append(candidate_proc)
                            except Exception as e:
                                err_msg = f"Error during round {current_round} code generation: {e}"
                                task_logger.error(
                                    err_msg + f"\n{traceback.format_exc()}",
                                    extra={
                                        "task_id": task_key,
                                        "round": current_round,
                                        "stage": "error",
                                    },
                                )
                                candidate_proc.record_round(
                                    current_round,
                                    candidate_proc.current_code,
                                    f"Exception: {str(e)}",
                                    False,
                                    test_feedback=traceback.format_exc(),
                                )

                        active_candidates = next_active_candidates
                        current_round += 1

                    if max_rounds == 1:
                        if dataset == "SWE-Bench-verified":
                            candidate = candidate_processes[0]
                            round_record = candidate.repair_history[-1]
                            fixed_code = round_record["generate_code"]

                            success, patch, buggy_code, error = (
                                generate_patch_from_function_fix(ques, fixed_code)
                            )
                            record = {
                                "instance_id": ques["instance_id"],
                                "model_name_or_path": f"{model_version}_{feedback_type}",
                                "model_patch": patch if success else "",
                                "buggy_code": buggy_code or "",
                                "fixed_code": fixed_code,
                                "isTrue": False,
                                "patch_generated": success,
                                "patch_error": error if not success else None,
                                "feedback_type": feedback_type,
                            }
                        else:
                            record = {
                                "_id": task_key,
                                "error": "Not supported for single_fix multi-feedback non-SWE-bench",
                            }

                        results.append((task_key, record))
                        cache_key = (model_name, model_version, feedback_type)
                        if caches and cache_key in caches:
                            with cache_lock:
                                caches[cache_key].append(record)

                    else:
                        candidate_results = [
                            candidate.to_result() for candidate in candidate_processes
                        ]
                        if dataset == "SWE-Bench-verified":
                            record = {
                                "instance_id": ques["instance_id"],
                                "repair_results": candidate_results,
                                "repo": ques["repo"],
                                "base_commit": ques["base_commit"],
                                "file_path": ques["file_path"],
                                "test_patch": ques["test_patch"],
                                "difficulty": ques["difficulty"],
                                "function_name": ques["function_name"],
                                "problem_statement": ques.get("problem_statement"),
                                "correct_code": ques["correct_code"],
                                "feedback_type": feedback_type,
                                "model_name_or_path": model_name,
                                "model_version": model_version,
                            }
                            results.append((task_key, record))
                            cache_key = (model_name, model_version, feedback_type)
                            if caches and cache_key in caches:
                                with cache_lock:
                                    caches[cache_key].append(record)
                finally:
                    if handler:
                        handler.close()
                        if task_logger:
                            task_logger.removeHandler(handler)

        finally:
            if swe_ctx:
                cleanup_swebench_context(swe_ctx, logger)

        return results

    max_workers = int(os.getenv("LLM_MAX_WORKERS", "5"))
    with tqdm(total=len(pending_tasks), desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _process_ques_with_multiple_feedbacks, ques, items
                ): task_id(ques)
                for ques, items in pending_tasks
            }
            for future in as_completed(future_to_task):
                try:
                    results_list = future.result()
                    if results_list is None:
                        failed_task_key = future_to_task[future]
                        logger.error(
                            f"Task {failed_task_key} returned None",
                            extra={"task_id": failed_task_key, "stage": "error"},
                        )
                        continue

                    for completed_task_key, record in results_list:
                        fixed_list.append(record)
                        # Cache already updated in worker
                except Exception as e:
                    failed_task_key = future_to_task[future]
                    logger.error(
                        f"Error processing task {failed_task_key}: {e}\n{traceback.format_exc()}",
                        extra={"task_id": failed_task_key, "stage": "error"},
                    )
                finally:
                    pbar.update(1)

    return fixed_list


def pass_rate_single_round(input_path, dataset):
    logger.info(
        f"Calculating score for {input_path}",
        extra={"stage": "score"},
    )
    eval_data = read_jsonl(input_path)

    if dataset == "SWE-Bench-verified":
        run_id = f"score_{int(time.time())}"
        logger.info(
            f"Running SWE-bench evaluation in pass_rate_single_round, run_id={run_id}",
            extra={"stage": "evaluate"},
        )
        run_swebench_batch_evaluation(
            predictions_path=input_path,
            run_id=run_id,
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            timeout=900,
            logger=logger,
        )
        update_predictions_with_results(input_path, run_id)
        eval_data = read_jsonl(input_path)
        num_tot = len(eval_data)
        num_accept = sum(1 for item in eval_data if item.get("isTrue", False))
        logger.info(
            f"Score: {num_accept / num_tot * 100:.2f}, {num_accept}/{num_tot}",
            extra={"stage": "score"},
        )
        return

    num_accept, num_tot = 0, 0
    for data in tqdm(eval_data, total=len(eval_data), desc="Calculating score"):
        for result in data["fixed_results"]:
            fixed_code = result["fixed_code"]
            if fixed_code:
                num_tot += 1
                if "isTrue" in result:
                    num_accept += result["isTrue"]
                else:
                    exit_code, test_feedback = run_test(
                        dataset,
                        fixed_code,
                        data.get("_id", None),
                        data.get("test", None),
                    )
                    result["isTrue"] = exit_code in (0, 5)
                    if exit_code not in (0, 5):
                        result["test_feedback"] = test_feedback
                    num_accept += result["isTrue"]

    write_jsonl(input_path, eval_data)
    logger.info(
        f"Score: {num_accept / num_tot * 100:.2f}, {num_accept}/{num_tot}",
        extra={"stage": "score"},
    )


def pass_rate_multi_round(input_path):
    pass_rate_per_round = defaultdict(int)
    total = 0
    logger.info(
        f"Evaluating file:{input_path}",
        extra={"stage": "score"},
    )
    eval_data = read_jsonl(input_path)

    for ques in eval_data:
        for result in ques["repair_results"]:
            if all(record["generate_code"] for record in result["repair_history"]):
                total += 1
            for record in result["repair_history"]:
                if record["round"] not in pass_rate_per_round:
                    pass_rate_per_round[record["round"]] = 0
                if record["isTrue"]:
                    pass_rate_per_round[record["round"]] += 1

    sorted_rounds = sorted(pass_rate_per_round.keys())
    cumulative_passed = 0

    for round_num in sorted_rounds:
        cumulative_passed += pass_rate_per_round[round_num]
        pass_rate = cumulative_passed / total if total > 0 else 0
        logger.info(
            f"Round {round_num}: Pass rate = {pass_rate:.2%}, {cumulative_passed}/{total}",
            extra={"stage": "score", "round": round_num},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="CoderEval or HumanEval")
    parser.add_argument(
        "--model", type=str, required=True, nargs="+", help="Model name(s)"
    )
    parser.add_argument(
        "--version", type=str, required=True, nargs="+", help="Model version(s)"
    )
    parser.add_argument(
        "--feedback",
        type=str,
        required=True,
        choices=FEEDBACK_TYPES,
        nargs="+",
        help="Type of feedback, can be multiple",
    )
    parser.add_argument(
        "--function",
        type=str,
        required=True,
        choices=["single_fix", "single_score", "multi_fix", "multi_score"],
        help="Function to run",
    )
    parser.add_argument(
        "--no_docstring", action="store_false", help="Whether to use docstring"
    )
    parser.add_argument(
        "--no_context", action="store_false", help="Whether to use context"
    )
    parser.add_argument(
        "--no_persona", action="store_false", help="Whether to use persona"
    )
    parser.add_argument(
        "--is_cot", action="store_true", help="Whether to use chain of thought"
    )
    parser.add_argument(
        "--is_few_shot", action="store_true", help="Whether to use few-shot"
    )
    parser.add_argument(
        "--no_instructions", action="store_false", help="Whether to use instructions"
    )
    parser.add_argument(
        "--is_es_shot", action="store_true", help="Whether to use ES-Shot"
    )
    parser.add_argument("--is_sa", action="store_true", help="Whether to use SA")
    parser.add_argument(
        "--is_sg_icl", action="store_true", help="Whether to use SG-ICL"
    )
    parser.add_argument("--is_sbp", action="store_true", help="Whether to use SBP")
    parser.add_argument("--is_rr", action="store_true", help="Whether to use RR")
    args = parser.parse_args()

    if len(args.model) != len(args.version):
        raise ValueError("Number of models must match number of versions")

    global logger
    logger = setup_logging(
        RunContext(
            dataset=args.dataset,
            model=args.model[0] if len(args.model) == 1 else "multi_models",
            version=args.version[0] if len(args.version) == 1 else "multi_versions",
            feedback=args.feedback,
            function=args.function,
        ),
        "evaluate",
    )

    options = PromptOptions(
        use_docstring=args.no_docstring,
        use_context=args.no_context,
        use_persona=args.no_persona,
        use_cot=args.is_cot,
        use_few_shot=args.is_few_shot,
        use_instructions=args.no_instructions,
        use_es_shot=args.is_es_shot,
        use_sa=args.is_sa,
        use_sg_icl=args.is_sg_icl,
        use_sbp=args.is_sbp,
        use_rr=args.is_rr,
    )

    if args.function == "single_fix":
        if len(args.feedback) > 1:
            raise ValueError("single_fix mode does not support multiple feedback types")
        input_path = os.path.join(
            "dataset", args.dataset, f"{args.dataset}_feedback_test.jsonl"
        )
        fix_code(
            input_path,
            list(zip(args.model, args.version)),
            args.feedback,
            args.dataset,
            options,
            max_rounds=1,
        )
    elif args.function == "single_score":
        if len(args.model) > 1:
            raise ValueError("single_score mode does not support multiple models")
        if len(args.feedback) > 1:
            raise ValueError(
                "single_score mode does not support multiple feedback types"
            )
        input_path = _single_save_path(
            args.model[0],
            args.version[0],
            args.feedback[0],
            args.dataset,
            options,
        )
        pass_rate_single_round(input_path, args.dataset)
    elif args.function == "multi_fix":
        input_path = os.path.join(
            "dataset", args.dataset, f"{args.dataset}_feedback_test.jsonl"
        )
        fix_code(
            input_path,
            list(zip(args.model, args.version)),
            args.feedback,
            args.dataset,
            options,
            max_rounds=3,
        )
    elif args.function == "multi_score":
        if len(args.model) > 1:
            raise ValueError("multi_score mode does not support multiple models")
        feedback_str = args.feedback[0] if len(args.feedback) == 1 else "multi_feedback"
        input_path = os.path.join(
            "results",
            args.model[0],
            args.dataset,
            "multi",
            f"{args.version[0]}_multi_round_{feedback_str}.jsonl",
        )
        pass_rate_multi_round(input_path)


if __name__ == "__main__":
    main()
