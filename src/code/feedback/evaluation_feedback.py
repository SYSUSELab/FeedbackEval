from __future__ import annotations

import ast
import sys
import subprocess
import textwrap
from pathlib import Path
from typing import Callable, Dict

from feedback.feedback import run_test, run_pylint, analyze_pylint_message
from prompts.template import build_gpt_prompt, build_gpt_gt_prompt, build_repair_prompt
from core.utils import get_model_response, truncate_feedback

# SWE-bench harness import for file retrieval
ROOT_DIR = Path(__file__).resolve().parents[2]
SWEBENCH_DIR = ROOT_DIR / "SWE-bench"
if SWEBENCH_DIR.exists():
    sys.path.insert(0, str(SWEBENCH_DIR))
    sys.path.insert(0, str(SWEBENCH_DIR / "swebench"))

from swebench.harness.utils import get_repo_file


def build_repair_prompt_for_round(
    ques: dict,
    current_code: str,
    current_feedback: str,
    dataset: str,
    options,
) -> str:
    return build_repair_prompt(
        solution=current_code,
        feedback=current_feedback,
        docstring=ques.get("docstring", None) if options.use_docstring else None,
        context=ques.get("oracle_context", None) if options.use_context else None,
        problem_statement=ques.get("problem_statement", None),
        current_task=ques,
        dataset=dataset,
        is_persona=options.use_persona,
        is_cot=options.use_cot,
        is_few_shot=options.use_few_shot,
        is_instructions=options.use_instructions,
        is_es_shot=options.use_es_shot,
        is_sa=options.use_sa,
        is_sg_icl=options.use_sg_icl,
        is_sbp=options.use_sbp,
        is_rr=options.use_rr,
    )


def build_feedback_mapping(
    dataset: str, ques: dict, current_code: str
) -> Dict[str, Callable[[], str]]:
    return {
        "test_feedback": lambda: run_test(
            dataset,
            current_code,
            ques.get("_id", None),
            ques.get("test", None),
        )[1],
        "compiler_feedback": lambda: run_pylint(current_code),
        "llm_feedback": lambda: get_model_response(
            "gpt-4o-mini",
            build_gpt_prompt(
                dataset,
                current_code,
                ques.get("docstring", None),
                ques.get("oracle_context", None),
            ),
        ),
        "llm_gt_feedback": lambda: get_model_response(
            "gpt-4o-mini",
            build_gpt_gt_prompt(
                dataset,
                current_code,
                ques["correct_code"],
                ques.get("docstring", None),
                ques.get("oracle_context", None),
            ),
        ),
        "minimal_feedback": lambda: "The code is wrong. Please fix it.",
        "mixed_feedback": lambda: get_mixed_feedback(dataset, current_code, ques),
    }


def build_feedback_mapping_swebench(
    ques: dict, current_code: str
) -> Dict[str, Callable[[], str]]:
    """Feedback providers for SWE-bench multi-round to honor chosen feedback type."""
    return {
        "compiler_feedback": lambda: compiler_feedback_swebench(ques, current_code),
        "llm_skilled_feedback": lambda: get_model_response(
            "gpt-4o-mini",
            build_gpt_prompt(
                "SWE-Bench-verified",
                current_code,
                ques.get("docstring", None),
                ques.get("oracle_context", None),
            ),
        ),
        "llm_expert_feedback": lambda: get_model_response(
            "gpt-4o-mini",
            build_gpt_gt_prompt(
                "SWE-Bench-verified",
                current_code,
                ques.get("correct_code"),
                ques.get("docstring", None),
                ques.get("oracle_context", None),
            ),
        ),
        "simple_feedback": lambda: "The code is wrong. Please fix it.",
    }


def get_function_boundaries(source_code: str) -> list[tuple[str, int, int]]:
    """Return (function_name, start_line, end_line) tuples; empty if parse fails."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    boundaries = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_lineno = getattr(node, "end_lineno", node.lineno)
            if node.body:
                end_lineno = max(
                    child.end_lineno
                    for child in ast.walk(node)
                    if hasattr(child, "end_lineno")
                )
            boundaries.append((node.name, node.lineno, end_lineno))
    return boundaries


def replace_buggy_with_candidate(
    full_content: str, buggy_code: str | None, candidate_code: str
) -> str:
    if buggy_code and buggy_code in full_content:
        return full_content.replace(buggy_code, candidate_code, 1)
    return full_content


def pylint_messages_with_lines(code_content: str) -> list[dict]:
    """Run pylint and keep line numbers so we can filter by function span."""
    code_lines = code_content.splitlines()
    format_string = "{line}:{C}:{msg_id}:{obj}:{module}:{msg}:{symbol}"
    process = subprocess.Popen(
        [
            "pylint",
            "--disable=C,R",
            "--from-stdin",
            "lint.py",
            f"--msg-template='{format_string}'",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    normalized = textwrap.dedent(code_content).lstrip()
    stdout, _ = process.communicate(input=normalized)
    messages = []
    for line in stdout.splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            continue
        try:
            lineno = int(parts[0])
            resp = {
                "line": lineno,
                "line_content": (
                    code_lines[lineno - 1] if 0 <= lineno - 1 < len(code_lines) else ""
                ),
                "category": parts[1],
                "diagnostic_type": parts[2],
                "related_object": parts[3],
                "message": parts[5],
            }
            messages.append(
                {
                    "line": lineno,
                    "text": analyze_pylint_message(resp),
                }
            )
        except Exception:
            continue
    return messages


def compiler_feedback_swebench(ques: dict, candidate_code: str) -> list[str]:
    """Run pylint on the full file and keep only diagnostics inside target function."""
    file_path = ques.get("file_path")
    repo = ques.get("repo")
    base_commit = ques.get("base_commit")
    function_name = ques.get("function_name")
    buggy_code = ques.get("buggy_code")

    if not (file_path and repo and base_commit):
        return run_pylint(candidate_code)

    full_content = get_repo_file(repo, base_commit, file_path)
    if not full_content:
        return run_pylint(candidate_code)

    patched_content = replace_buggy_with_candidate(
        full_content, buggy_code, candidate_code
    )
    boundaries = get_function_boundaries(patched_content)
    target_span = None
    if function_name:
        for name, start, end in boundaries:
            if name == function_name:
                target_span = (start, end)
                break

    messages = pylint_messages_with_lines(patched_content)
    if target_span:
        start, end = target_span
        messages = [m for m in messages if start <= m["line"] <= end]

    return [m["text"] for m in messages]


def get_mixed_feedback(dataset, code, ques_data, existing_feedbacks=None):
    existing_feedbacks = existing_feedbacks or {}

    # Get or compute llm_expert_feedback
    llm_expert_feedback = existing_feedbacks.get("llm_expert_feedback")
    if llm_expert_feedback is None:
        llm_expert_feedback = get_model_response(
            "gpt-4o-mini",
            build_gpt_gt_prompt(
                dataset,
                code,
                ques_data.get("correct_code"),
                ques_data.get("docstring", None),
                ques_data.get("oracle_context", None),
            ),
        )

    # Get or compute test_feedback
    test_feedback = existing_feedbacks.get("test_feedback")
    if test_feedback is None:
        # For SWE-Bench, test_feedback must be provided via existing_feedbacks
        if dataset != "SWE-Bench-verified":
            test_feedback = run_test(
                dataset, code, ques_data.get("_id", None), ques_data.get("test", None)
            )[1]
    if test_feedback:
        test_feedback = truncate_feedback(test_feedback)
    else:
        test_feedback = ""

    # Get or compute compiler_feedback
    compiler_feedback = existing_feedbacks.get("compiler_feedback")
    if compiler_feedback is None:
        if dataset == "SWE-Bench-verified":
            compiler_feedback = compiler_feedback_swebench(ques_data, code)
        else:
            compiler_feedback = run_pylint(code)
    if isinstance(compiler_feedback, list):
        compiler_feedback = "\n".join(compiler_feedback)

    # Combine all feedbacks
    feedback_parts = [
        "The code is wrong. Please fix it.",
        str(llm_expert_feedback),
        "Here is some additional feedback information from the test cases and static analysis tools for your reference:",
        f"{test_feedback}\n{compiler_feedback}",
    ]
    return "\n".join(feedback_parts)
