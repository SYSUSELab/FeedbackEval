import difflib
import re
from typing import Tuple
from swebench.harness.constants import (
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)
from swebench.harness.utils import get_repo_file


def get_indentation(s: str) -> str:
    """Return the leading whitespace of the first line."""
    if not s:
        return ""
    first_line = s.splitlines()[0] if s else ""
    return first_line[: len(first_line) - len(first_line.lstrip())]


def detect_body_indentation(code: str) -> str:
    """Detect function body indentation, skipping def line and empty/comment lines."""
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        stripped = line.lstrip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        ):
            continue
        return line[: len(line) - len(stripped)]
    return ""


def generate_patch_from_function_fix(
    instance: dict, fixed_function_code: str
) -> Tuple[bool, str, str, str]:
    """
    Build a unified diff patch for a single-function fix using remote repo content.

    Returns (success, patch, buggy_code, error_message).
    On failure, patch is an empty string and error_message describes the cause.
    """

    repo = instance.get("repo")
    base_commit = instance.get("base_commit")
    file_path = instance.get("file_path")

    # Extract buggy_code from false_results[0]["generate_code"].
    false_results = instance.get("false_results", [])
    if not false_results or not isinstance(false_results, list):
        return (
            False,
            "",
            "",
            "Missing or invalid false_results in instance",
        )

    buggy_code = false_results[0].get("generate_code")

    if not all([repo, base_commit, file_path, buggy_code]):
        return (
            False,
            "",
            "",
            "Missing repo/base_commit/file_path or buggy_code from false_results",
        )

    original_content = get_repo_file(repo, base_commit, file_path)
    if not original_content:
        return False, "", buggy_code, "Unable to fetch original file"

    # Verify that the provided buggy_code is actually in the original file
    if buggy_code not in original_content:
        return (
            False,
            "",
            buggy_code,
            "The provided 'buggy_code' was not found in the original file content.",
        )

    if not fixed_function_code:
        return False, "", buggy_code, "Empty fixed_function_code provided"

    # --- Indentation calibration start ---
    buggy_def_indent = get_indentation(buggy_code)
    fixed_def_indent = get_indentation(fixed_function_code)

    # If def indentation matches, align function body indentation.
    if buggy_def_indent == fixed_def_indent:
        buggy_body_indent = detect_body_indentation(buggy_code)
        fixed_body_indent = detect_body_indentation(fixed_function_code)

        # Adjust body indentation if needed.
        if buggy_body_indent != fixed_body_indent:
            indent_diff = len(buggy_body_indent) - len(fixed_body_indent)

            fixed_lines = fixed_function_code.splitlines(keepends=True)
            calibrated_lines = []

            for i, line in enumerate(fixed_lines):
                if i == 0:
                    calibrated_lines.append(line)
                    continue

                if not line.strip():
                    calibrated_lines.append(line)
                    continue

                # Shift indentation by the difference.
                line_stripped = line.lstrip()
                current_indent = line[: len(line) - len(line_stripped)]

                if indent_diff > 0:
                    new_indent = " " * indent_diff + current_indent
                else:
                    spaces_to_remove = min(len(current_indent), abs(indent_diff))
                    new_indent = current_indent[spaces_to_remove:]

                calibrated_lines.append(new_indent + line_stripped)

            calibrated_fixed_function_code = "".join(calibrated_lines)
        else:
            calibrated_fixed_function_code = fixed_function_code
    else:
        # If def indentation differs, realign based on def indentation.
        fixed_lines = fixed_function_code.splitlines(keepends=True)
        calibrated_lines = []

        for line in fixed_lines:
            if not line.strip():
                calibrated_lines.append(line)
                continue

            line_stripped = line.lstrip()
            current_indent = line[: len(line) - len(line_stripped)]

            if current_indent.startswith(fixed_def_indent):
                relative_indent = current_indent[len(fixed_def_indent) :]
                new_line = buggy_def_indent + relative_indent + line_stripped
                calibrated_lines.append(new_line)
            else:
                calibrated_lines.append(line)

        calibrated_fixed_function_code = "".join(calibrated_lines)
    # --- Indentation calibration end ---

    if buggy_code == calibrated_fixed_function_code:
        return False, "", buggy_code, "No changes generated"

    patched_content = original_content.replace(
        buggy_code, calibrated_fixed_function_code, 1
    )
    if patched_content == original_content:
        # This could happen if buggy_code appears multiple times and we replace the wrong one,
        # but given the single-function context, it's more likely a subtle string difference.
        return False, "", buggy_code, "Replacement failed; content unchanged"

    original_lines = original_content.splitlines(keepends=True)
    patched_lines = patched_content.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    patch_str = "".join(diff_lines)

    if not patch_str.strip():
        return False, "", buggy_code, "Empty patch generated"

    return True, patch_str, buggy_code, ""


def extract_test_summary(full_log: str) -> str:
    """Extract a concise error summary from pytest or Django test output."""
    # 1. Extract the core test output region if available.
    if START_TEST_OUTPUT in full_log and END_TEST_OUTPUT in full_log:
        content = full_log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    else:
        content = full_log

    lines = content.strip().split("\n")

    # 2. Pattern helpers.
    ansi_pattern = r"(?:\x1b\[[0-9;]*m)*"
    pytest_failure_start = re.compile(
        ansi_pattern + r"={3,}\s*(FAILURES|ERRORS)\s*={3,}", re.I
    )
    pytest_summary_start = re.compile(
        ansi_pattern + r"={3,}\s*short test summary", re.I
    )
    django_error_start = re.compile(r"^(ERROR:|FAIL:)")
    test_result_line = re.compile(
        r"(FAILED|PASSED|ERROR|Ran).*\d+\s+(passed|failed|error|exception|tests?)", re.I
    )

    # 3. Collect error blocks.
    error_blocks = []
    current_block = []
    in_error_block = False

    for line in lines:
        # Stop at pytest short test summary.
        if pytest_summary_start.match(line):
            if current_block:
                error_blocks.append(current_block)
                current_block = []
            break

        # pytest FAILURES separator.
        if pytest_failure_start.match(line):
            if current_block:
                error_blocks.append(current_block)
            current_block = [line]
            in_error_block = True
            continue

        # Django ERROR:/FAIL:
        if django_error_start.match(line):
            if current_block:
                error_blocks.append(current_block)
            current_block = [line]
            in_error_block = True
            continue

        # Collect block content.
        if in_error_block:
            current_block.append(line)
            # End when reaching summary line.
            if test_result_line.search(line):
                error_blocks.append(current_block)
                current_block = []
                break

    # Keep the last block if present.
    if current_block and in_error_block:
        error_blocks.append(current_block)

    # 4. Fallback to tail if nothing matched.
    if not error_blocks:
        # Scan backwards for a summary line.
        for i in range(len(lines) - 1, max(0, len(lines) - 100), -1):
            if test_result_line.search(lines[i]):
                start = max(0, i - 15)
                end = min(len(lines), i + 5)
                return "\n".join(lines[start:end])
        return "\n".join(lines[-50:])

    # 5. Join all error blocks.
    extracted = []
    for block in error_blocks:
        extracted.extend(block)
        extracted.append("")

    result = "\n".join(extracted).strip()

    return result
