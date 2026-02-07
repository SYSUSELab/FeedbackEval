from tqdm import tqdm
import sys
import os
import textwrap

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from core.client import LlmClient, LlmConfig
from core.utils import read_jsonl, setup_logging, RunContext, ProgressiveCache
from prompts.template import build_gpt_prompt, build_gpt_gt_prompt
from feedback.diagnostic_handlers import E0602_handler, E1101_handler, E0102_handler
import subprocess
import multiprocessing
import sys
import os
import json
import ast
import tempfile


def run_pytest(code, timeout=10):
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode("utf-8"))
            temp_filename = temp_file.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--tb=short",
                "-q",
                temp_filename,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(temp_filename),
        )
        exit_code = result.returncode
        res = result.stdout.splitlines()[
            4:-4
        ]  # Keep only the relevant lines from pytest output
        test_feedback = "\n".join(res)
    except subprocess.TimeoutExpired:
        exit_code = -1
        test_feedback = f"Execution timed out after {timeout} seconds."
    except Exception as e:
        exit_code = -2
        test_feedback = str(e)
    finally:
        os.remove(temp_filename)

    return exit_code, test_feedback


class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            self._cconn.send(exception)

    def join(self, timeout):
        super().join(timeout)

        if self.is_alive():
            self.terminate()
        super().join()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


dict_std_nonestd = {
    "/home/travis/builds/repos/standalone/neo4j-_meta-deprecated.py": "/home/travis/builds/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
    "/home/travis/builds/repos/standalone/neo4j-work-query-unit_of_work.py": "/home/travis/builds/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
    "/home/travis/builds/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py": "/home/travis/builds/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py",
}


def run_coder_eval_test(_id, code):
    f = open("CoderEval4Python.json", "r", encoding="utf-8")
    content = f.read()
    f.close()

    content_json = json.loads(content)
    collection = {}
    for l in content_json["RECORDS"]:
        collection[l["_id"]] = l
    kk = 0
    project_path = "/home/travis/builds/repos/"
    dict_id_file = {}
    for keyy in collection:
        dictTemp = collection[keyy]
        save_data = (
            project_path
            + "standalone/"
            + dictTemp["file_path"].replace(".py", "").replace("/", "-")
            + "-"
            + dictTemp["name"]
            + ".py"
        )
        if save_data in dict_std_nonestd.keys():
            save_data = dict_std_nonestd[save_data]
            if os.path.exists(save_data):
                kk += 1
                dict_id_file[dictTemp["_id"]] = save_data
        elif os.path.exists(save_data):
            kk += 1
            dict_id_file[dictTemp["_id"]] = save_data
        else:
            file_path = dictTemp["file_path"]
            if (
                project_path + dictTemp["project"].replace("/", "---")
                == "/home/travis/builds/repos/neo4j---neo4j-python-driver"
            ):
                save_data = os.path.join(
                    project_path + dictTemp["project"].replace("/", "---") + "/src",
                    file_path,
                ).replace(".py", "_" + dictTemp["name"] + "_passk_validte.py")
            else:
                save_data = os.path.join(
                    project_path + dictTemp["project"].replace("/", "---"), file_path
                ).replace(".py", "_" + dictTemp["name"] + "_passk_validte.py")
            if save_data in dict_std_nonestd.keys():
                save_data = dict_std_nonestd[save_data]

            if os.path.exists(save_data):
                kk += 1
                dict_id_file[dictTemp["_id"]] = save_data

    content_doc = collection[_id]
    if content_doc is None:
        return
    f_save_data = open(dict_id_file[str(_id)], "r")
    file_content = f_save_data.read()
    f_save_data.close()
    file_content_list = file_content.split("\n")

    ast_file = ast.parse(file_content)
    start_indent = 0
    new_data = ""
    for node in ast.walk(ast_file):
        if isinstance(node, ast.FunctionDef):
            temp_method_name = node.name
            if (
                content_doc["name"] != temp_method_name
                and "_" + content_doc["name"] != temp_method_name
            ):
                continue
            start_line = node.lineno
            end_line = node.end_lineno
            indent_s = file_content_list[start_line - 1]
            tttt = indent_s.lstrip(" ")
            start_indent = len(indent_s) - len(tttt)
            new_data = ""
            for i in range(0, start_line - 1):
                new_data += file_content_list[i]
                new_data += "\n"
            new_data += "<insert generated code here>\n"
            for i in range(end_line, len(file_content_list)):
                new_data += file_content_list[i]
                new_data += "\n"
    assert new_data != ""
    code_num = 0
    code_list = code.split("\n")
    tttt = code_list[0].lstrip(" ")
    code_indent = len(code_list[0]) - len(tttt)
    new_code = ""
    if start_indent > code_indent:
        str_a = ""
        for iii in range(0, start_indent - code_indent):
            str_a += " "
        for ccc in code_list:
            ttz = str_a + ccc
            new_code += ttz
            new_code += "\n"
    else:
        new_code = code
    out_data = new_data.replace("<insert generated code here>", new_code)
    save_data_new = dict_id_file[str(_id)]
    f = open(save_data_new.replace(".py", str(code_num) + ".py"), "w")
    f.write(out_data)
    f.close()
    try:
        process = subprocess.Popen(
            [sys.executable, save_data_new.replace(".py", str(code_num) + ".py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, error = process.communicate(timeout=30)
        exit_code = process.returncode
        test_feedback = error.decode("utf-8") if exit_code != 0 and error else ""
        return exit_code, test_feedback
    except subprocess.TimeoutExpired:
        return -1, "Execution timed out after 30 seconds."


def run_test(dataset, code, _id=None, test=None):
    if dataset == "CoderEval":
        exit_code, test_feedback = run_coder_eval_test(_id, code)
    elif dataset == "HumanEval":
        check_code = f"{code}\n{test}"
        exit_code, test_feedback = run_pytest(check_code)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    return exit_code, test_feedback


def run_pylint(code_content):
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
    code_content = textwrap.dedent(code_content).lstrip()
    stdout, stderr = process.communicate(input=code_content)
    results = []
    for line in stdout.splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            continue
        try:
            resp = {
                "line": int(parts[0]),
                "line_content": code_lines[int(parts[0]) - 1],
                "category": parts[1],
                "diagnostic_type": parts[2],
                "related_object": parts[3],
                "message": parts[5],
            }
            feedback = analyze_pylint_message(resp)
            results.append(feedback)
        except:
            pass
    return results


def analyze_pylint_message(diagnostic_body) -> str:
    diagnostic_type = diagnostic_body["diagnostic_type"]
    default_diagnostic_message = (
        "In line: "
        + diagnostic_body["line_content"]
        + " . "
        + diagnostic_body["message"]
    )

    diagnostic_handlers = {
        "E0213": lambda _: "",  # Skip 'no-self-argument' error
        "E0001": lambda _: default_diagnostic_message,
        "E0602": E0602_handler,  # undefined-variable
        "E1101": E1101_handler,  # no-member
        "E0102": E0102_handler,  # function redefined
    }
    handler = diagnostic_handlers.get(diagnostic_type)
    if handler:
        return handler(diagnostic_body)
    return default_diagnostic_message


def _get_llm_feedback(
    llm_client,
    skilled_prompt: str,
    expert_prompt: str,
    task_id,
):
    prompts = [skilled_prompt, expert_prompt]
    types = ["skilled", "expert"]

    results = llm_client.batch_complete(prompts)

    skilled_feedback = results[0]
    expert_feedback = results[1]

    for p_type, prompt, feedback in zip(types, prompts, results):
        logger.info(
            f"LLM Interaction ({p_type}):\n"
            f"--- PROMPT ---\n{prompt}\n"
            f"--- RESPONSE ---\n{feedback}\n",
            extra={"task_id": task_id, "stage": f"llm_{p_type}"},
        )

    return skilled_feedback, expert_feedback


def eval_feedback(dataset, file_path):
    global logger
    logger = setup_logging(
        RunContext(dataset=dataset, function="feedback"),
        "feedback",
    )
    data_list = read_jsonl(file_path)
    llm_client = LlmClient(LlmConfig(model="gpt-4o-mini"))
    output_path = f"dataset/{dataset}/{dataset}_feedback.jsonl"
    if dataset == "SWE-Bench-verified":
        id_field = "instance_id"
    else:
        id_field = "_id" if any("_id" in item for item in data_list) else "task_id"
    cache = ProgressiveCache(output_path, id_field=id_field)

    for data in tqdm(
        data_list, total=len(data_list), desc="Processing feedback dataset:"
    ):

        task_id = data.get(id_field)
        if cache.has(task_id):
            logger.info(
                f"Skipping cached task {task_id}",
                extra={"task_id": task_id, "stage": "cache"},
            )
            continue

        filtered_results = []
        if dataset == "SWE-Bench-verified":
            buggy_code = data.get("buggy_code") or data.get("generate_code")
            if not buggy_code:
                logger.error(
                    f"Missing buggy_code for {task_id}",
                    extra={"task_id": task_id, "stage": "input"},
                )
                continue
            data["correct_code"] = data.get("fixed_code")
            problem_statement = data.get("problem_statement")
            list_results = [
                {
                    "source": "swe-bench",
                    "generate_code": buggy_code,
                    "test_feedback": data.get("test_feedback"),
                }
            ]
        else:
            problem_statement = None
            list_results = data["false_results"]
        for result in list_results:
            try:
                if dataset != "SWE-Bench-verified":
                    exit_code, test_feedback = run_test(
                        dataset,
                        result["generate_code"],
                        data.get("_id", None),
                        data.get("test", None),
                    )
                    if exit_code in (0, 5):
                        continue
                    result["test_feedback"] = test_feedback

                result["compiler_feedback"] = run_pylint(result["generate_code"])
                llm_skilled_prompt = build_gpt_prompt(
                    dataset,
                    result["generate_code"],
                    data.get("docstring", None),
                    data.get("oracle_context", None),
                    problem_statement=problem_statement,
                )
                llm_expert_prompt = build_gpt_gt_prompt(
                    dataset,
                    result["generate_code"],
                    data.get("correct_code", None),
                    data.get("docstring", None),
                    data.get("oracle_context", None),
                    problem_statement=problem_statement,
                )
                skilled_feedback, expert_feedback = _get_llm_feedback(
                    llm_client,
                    llm_skilled_prompt,
                    llm_expert_prompt,
                    task_id,
                )
                result["llm_skilled_feedback"] = skilled_feedback
                result["llm_expert_feedback"] = expert_feedback
                result["minimal_feedback"] = "The code is wrong. Please fix it."
                filtered_results.append(result)
            except Exception as e:
                logger.error(f"Error getting feedback: {e}")
                continue
        data["false_results"] = filtered_results
        cache.append(data)

    logger.info(
        f"Results saved to {output_path}",
        extra={"stage": "save"},
    )


if __name__ == "__main__":
    eval_feedback(
        "SWE-Bench-verified",
        "dataset/SWE-Bench-verified/SWE-Bench-verified_feedback.jsonl",
    )
