from __future__ import annotations

import docker
import json
import platform
import threading
import traceback
import os
from collections import defaultdict

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath
from tqdm.auto import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def reset_container_state(
    container: docker.models.containers.Container, test_spec: TestSpec, logger
) -> bool:
    """Resets the git repository in the container to the base commit."""
    try:
        # Extract base_commit from repo_script_list
        base_commit = None
        for cmd in test_spec.repo_script_list:
            if "git reset --hard" in cmd:
                # Extract commit hash from command like "git reset --hard <commit>"
                parts = cmd.split()
                if len(parts) >= 4:
                    base_commit = parts[3]
                    break

        if not base_commit:
            logger.error(
                f"Could not find base_commit in repo_script_list for {test_spec.instance_id}"
            )
            return False

        # Check for unclean state first
        status_out = container.exec_run(
            "git status --porcelain", workdir=DOCKER_WORKDIR
        )
        if status_out.exit_code != 0 or status_out.output.decode(UTF8).strip():
            logger.warning(
                f"Container for {test_spec.instance_id} is in an unclean state. Resetting..."
            )

        # Reset to base commit
        reset_result = container.exec_run(
            f"git reset --hard {base_commit}", workdir=DOCKER_WORKDIR, user=DOCKER_USER
        )
        if reset_result.exit_code != 0:
            logger.error(
                f"Failed to git reset for {test_spec.instance_id}: {reset_result.output.decode(UTF8)}"
            )
            return False

        # Clean untracked files
        clean_result = container.exec_run(
            "git clean -fd", workdir=DOCKER_WORKDIR, user=DOCKER_USER
        )
        if clean_result.exit_code != 0:
            logger.warning(
                f"git clean had issues for {test_spec.instance_id}: {clean_result.output.decode(UTF8)}"
            )
            # Don't fail on clean errors, just warn

        logger.info(
            f"Container for {test_spec.instance_id} has been reset to commit {base_commit}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Exception while resetting container for {test_spec.instance_id}: {e}"
        )
        return False


def run_prediction_in_container(
    test_spec: TestSpec,
    pred: dict,
    container: docker.models.containers.Container,
    run_id: str,
    logger,
    timeout: int | None,
):
    """Applies a single prediction and runs the test, assuming container is ready."""
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }

    if not reset_container_state(container, test_spec, logger):
        raise EvaluationError(
            instance_id, "Failed to reset container state before evaluation.", logger
        )

    # Copy model prediction as patch file to container
    patch_file = Path(log_dir / "patch.diff")
    patch_file.write_text(pred[KEY_PREDICTION] or "")
    copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

    # Attempt to apply patch
    applied_patch = False
    for git_apply_cmd in GIT_APPLY_CMDS:
        val = container.exec_run(
            f"{git_apply_cmd} {DOCKER_PATCH}",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        if val.exit_code == 0:
            logger.info(
                f"Apply patch PASSED for {model_name_or_path}:\n{val.output.decode(UTF8)}"
            )
            applied_patch = True
            break
        else:
            logger.info(
                f"Failed to apply patch with '{git_apply_cmd}' for {model_name_or_path}"
            )

    if not applied_patch:
        logger.info(
            f"Apply patch FAILED for {model_name_or_path}:\n{val.output.decode(UTF8)}"
        )
        raise EvaluationError(
            instance_id,
            f"Failed to apply patch for {model_name_or_path}",
            logger,
        )

    # Run eval script
    test_output, timed_out, total_runtime = exec_run_with_timeout(
        container, "/bin/bash /eval.sh", timeout
    )
    test_output_path = log_dir / LOG_TEST_OUTPUT
    logger.info(f"Test runtime for {model_name_or_path}: {total_runtime:.2f} seconds")
    with open(test_output_path, "w") as f:
        f.write(test_output)
        if timed_out:
            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise EvaluationError(
                instance_id,
                f"Test for {model_name_or_path} timed out after {timeout} seconds.",
                logger,
            )

    # Get report
    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=test_output_path,
        include_tests_status=True,
    )
    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=4))

    return {
        "completed": True,
        "resolved": report.get(instance_id, {}).get("resolved", False),
    }


def run_instance_batch(
    test_spec: TestSpec,
    predictions: list,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    max_reuse_count: int = 10,
) -> dict:
    """
    Run a batch of predictions for a single instance, reusing a container.
    """
    instance_id = test_spec.instance_id
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / "batch_logs" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(instance_id, log_dir / LOG_INSTANCE)

    container = None
    results = {}
    reuse_counter = 0

    try:
        for i, pred in enumerate(predictions):
            model_name = pred.get(KEY_MODEL, "None").replace("/", "__")
            results[model_name] = {"completed": False, "resolved": False}

            try:
                # Build container if it doesn't exist or max reuse is hit
                if container is None or reuse_counter >= max_reuse_count:
                    if container:
                        cleanup_container(client, container, logger)
                    container = build_container(
                        test_spec,
                        client,
                        run_id,
                        logger,
                        nocache=False,
                        force_rebuild=force_rebuild or (reuse_counter > 0),
                    )
                    container.start()
                    logger.info(f"Container for {instance_id} started: {container.id}")
                    # Copy eval script once per container
                    eval_file = log_dir / "eval.sh"
                    eval_file.write_text(test_spec.eval_script)
                    copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
                    reuse_counter = 0

                # Run single prediction
                result = run_prediction_in_container(
                    test_spec, pred, container, run_id, logger, timeout
                )
                results[model_name] = result
                reuse_counter += 1

            except (EvaluationError, BuildImageError) as e:
                logger.error(
                    f"Error evaluating {model_name} for {instance_id}: {e}\n{traceback.format_exc()}"
                )
            except Exception:
                logger.error(
                    f"Unexpected error evaluating {model_name} for {instance_id}: \n{traceback.format_exc()}"
                )

    finally:
        if container:
            cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)

    return results


def run_instances_batch(
    grouped_predictions: dict,
    instances: list,
    max_workers: int,
    run_id: str,
    timeout: int,
    max_reuse_count: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
):
    client = docker.from_env()
    test_specs = {
        i[KEY_INSTANCE_ID]: make_test_spec(
            i,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )
        for i in instances
    }

    payloads = []
    for instance_id, preds in grouped_predictions.items():
        if instance_id not in test_specs:
            print(f"Warning: Skipping instance {instance_id} not found in dataset.")
            continue
        test_spec = test_specs[instance_id]
        payloads.append(
            (
                test_spec,
                preds,
                False,  # rm_image is handled at the end
                False,  # force_rebuild is handled at the beginning
                client,
                run_id,
                timeout,
                max_reuse_count,
            )
        )

    print(f"Running {len(payloads)} instances in batch mode...")
    pbar = tqdm(total=len(payloads), desc="Batch Evaluation")

    def run_batch_with_progress(*args):
        run_instance_batch(*args)
        pbar.update()

    run_threadpool(run_batch_with_progress, payloads, max_workers)
    print("All instance batches run.")


def main(
    predictions_path: str,
    dataset_name: str,
    split: str,
    run_id: str,
    max_workers: int,
    timeout: int,
    max_reuse_count: int,
    instance_ids: list | None = None,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    open_file_limit: int = 4096,
):
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    # Load all predictions and group them
    all_preds = get_predictions_from_file(predictions_path, dataset_name, split)
    grouped_predictions = defaultdict(list)
    for pred in all_preds:
        if instance_ids and pred[KEY_INSTANCE_ID] not in instance_ids:
            continue
        grouped_predictions[pred[KEY_INSTANCE_ID]].append(pred)

    if not grouped_predictions:
        print("No predictions found for the given instances.")
        return

    # Load dataset info - only for instances we have predictions for
    instance_ids_with_preds = list(grouped_predictions.keys())
    dataset = load_swebench_dataset(dataset_name, split, instance_ids_with_preds)

    client = docker.from_env()
    if namespace is None:
        build_env_images(
            client,
            dataset,
            force_rebuild=False,
            max_workers=max_workers,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )

    run_instances_batch(
        grouped_predictions,
        dataset,
        max_workers,
        run_id,
        timeout,
        max_reuse_count,
        namespace,
        instance_image_tag,
        env_image_tag,
    )

    print("\\nBatch evaluation run completed.")
    print(f"Logs and results are in: {RUN_EVALUATION_LOG_DIR}/{run_id}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run SWE-bench evaluation in batch mode with container reuse.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--predictions_path",
        type=str,
        required=True,
        help="Path to a single JSONL file containing all predictions to run.",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Lite",
        type=str,
        help="Name of dataset.",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="test", help="Split of the dataset."
    )
    parser.add_argument(
        "-id", "--run_id", type=str, required=True, help="Run ID for logging."
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max parallel instances."
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=1800, help="Timeout for each test run."
    )
    parser.add_argument(
        "--max_reuse_count",
        type=int,
        default=10,
        help="Max times to reuse a container before rebuilding.",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Optional: Specific instance IDs to run.",
    )
    parser.add_argument(
        "-n",
        "--namespace",
        type=optional_str,
        default="swebench",
        help="Namespace for Docker images.",
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag."
    )
    parser.add_argument(
        "--env_image_tag", type=str, default="latest", help="Environment image tag."
    )
    parser.add_argument(
        "--open_file_limit",
        type=int,
        default=4096,
        help="Open file limit (Linux only).",
    )

    args = parser.parse_args()
    main(**vars(args))
