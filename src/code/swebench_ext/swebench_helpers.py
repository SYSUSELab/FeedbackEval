from __future__ import annotations

import json
import logging
import os
from pathlib import Path, PurePosixPath

import docker
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_build import build_container
from swebench.harness.docker_utils import cleanup_container, copy_to_container
from swebench_ext.batch_run_evaluation import run_prediction_in_container
from swebench.harness.test_spec.test_spec import make_test_spec

from core.patch_generator import extract_test_summary

SWEBENCH_NAMESPACE = os.getenv("SWEBENCH_NAMESPACE") or None
SWEBENCH_INSTANCE_TAG = os.getenv("SWEBENCH_INSTANCE_TAG", "latest")
SWEBENCH_ENV_TAG = os.getenv("SWEBENCH_ENV_TAG", "latest")
SWEBENCH_TIMEOUT = int(os.getenv("SWEBENCH_TIMEOUT", "600"))

_SWEBENCH_DATASET_CACHE = {}


def get_swebench_instance(
    instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Verified"
) -> dict:
    """Get full SWE-bench instance from dataset cache."""
    from datasets import load_dataset

    if dataset_name not in _SWEBENCH_DATASET_CACHE:
        _SWEBENCH_DATASET_CACHE[dataset_name] = load_dataset(dataset_name, split="test")

    dataset = _SWEBENCH_DATASET_CACHE[dataset_name]
    instance = next(
        (item for item in dataset if item["instance_id"] == instance_id), None
    )
    if not instance:
        raise ValueError(f"Instance {instance_id} not found in {dataset_name}")
    return dict(instance)


def build_swebench_context(
    ques: dict,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    client = docker.from_env()
    instance_id = ques.get("instance_id")
    if "version" not in ques:
        full_instance = get_swebench_instance(instance_id)
        ques = {**full_instance, **ques}

    test_spec = make_test_spec(
        ques,
        namespace=SWEBENCH_NAMESPACE,
        instance_image_tag=SWEBENCH_INSTANCE_TAG,
        env_image_tag=SWEBENCH_ENV_TAG,
    )

    container_run_id = f"swe-multi-{instance_id}-container"
    container = build_container(
        test_spec,
        client,
        container_run_id,
        logger,
        nocache=False,
        force_rebuild=False,
    )
    container.start()

    swe_ctx = {
        "client": client,
        "container": container,
        "test_spec": test_spec,
        "eval_script_copied": False,
    }
    return swe_ctx, ques


def apply_and_test_patch(
    patch_text: str,
    swe_ctx: dict,
    model_name: str,
    timeout: int,
    logger: logging.Logger,
):
    container = swe_ctx["container"]
    test_spec = swe_ctx["test_spec"]
    run_id = swe_ctx["run_id"]

    prediction = {
        KEY_MODEL: model_name,
        KEY_PREDICTION: patch_text,
        KEY_INSTANCE_ID: test_spec.instance_id,
    }

    try:
        run_prediction_in_container(
            test_spec=test_spec,
            pred=prediction,
            container=container,
            run_id=run_id,
            logger=logger,
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)

    model_dir = model_name.replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_dir / test_spec.instance_id
    report_path = log_dir / LOG_REPORT
    test_output_path = log_dir / LOG_TEST_OUTPUT

    if report_path.exists():
        report = json.loads(report_path.read_text())
        resolved = report.get(test_spec.instance_id, {}).get("resolved", False)
        if resolved:
            return True, "Tests passed"

    if test_output_path.exists():
        test_output = test_output_path.read_text()
        return False, extract_test_summary(test_output)

    return False, "Evaluation report not found"


def ensure_eval_script(swe_ctx: dict, log_dir: Path) -> None:
    if swe_ctx["eval_script_copied"]:
        return

    eval_file = log_dir / "eval.sh"
    eval_file.write_text(swe_ctx["test_spec"].eval_script)
    copy_to_container(swe_ctx["container"], eval_file, PurePosixPath("/eval.sh"))
    swe_ctx["eval_script_copied"] = True


def cleanup_swebench_context(swe_ctx: dict, logger: logging.Logger) -> None:
    cleanup_container(swe_ctx["client"], swe_ctx["container"], logger)


def run_swebench_batch_evaluation(
    predictions_path: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    logger: logging.Logger,
) -> None:
    command = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "SWE-bench/SWE-bench_Verified",
        "--split",
        "test",
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
    ]
    logger.info(f"Running SWE-bench evaluation with command: {' '.join(command)}")
    try:
        import subprocess

        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"SWE-bench evaluation failed: {exc}")
    except FileNotFoundError:
        logger.error(
            "'python' command not found. Please ensure python is in your PATH."
        )


def update_predictions_with_results(predictions_path: str, run_id: str) -> None:
    from core.utils import read_jsonl, write_jsonl

    predictions = read_jsonl(predictions_path)
    updated = []

    for pred in predictions:
        instance_id = pred.get("instance_id")
        model_name = pred.get("model_name_or_path", "model").replace("/", "__")
        base_dir = Path("logs") / "run_evaluation" / run_id / model_name / instance_id
        report_file = base_dir / "report.json"
        test_output_file = base_dir / "test_output.txt"

        if report_file.exists():
            report = json.loads(report_file.read_text())
            pred["isTrue"] = report.get(instance_id, {}).get("resolved", False)
            if not pred["isTrue"] and test_output_file.exists():
                pred["test_feedback"] = extract_test_summary(
                    test_output_file.read_text()
                )
        else:
            pred["isTrue"] = False
            if not pred.get("patch_generated", True):
                pred["test_feedback"] = pred.get(
                    "patch_error", "Patch generation failed"
                )
            else:
                pred["test_feedback"] = "Evaluation report not found"

        pred.pop("patch_generated", None)
        pred.pop("patch_error", None)
        updated.append(pred)

    write_jsonl(predictions_path, updated)
