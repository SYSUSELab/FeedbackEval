import os
from jinja2 import Environment, FileSystemLoader
from retrieval.bm25 import find_best_example

TEMPLATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "templates")
)
_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)


def _render_template(template_name: str, **context) -> str:
    return _env.get_template(template_name).render(**context)


def build_mutant_prompt(ori_code):
    return _render_template("mutant_prompt.j2", ori_code=ori_code).strip()


def build_gpt_prompt(
    dataset, code, docstring=None, context=None, problem_statement=None
):
    if dataset == "CoderEval":
        prompt = _render_template(
            "gpt_prompt_coder_eval.j2",
            code=code,
            docstring=docstring,
            context=context,
        ).strip()
    elif dataset == "HumanEval":
        prompt = _render_template(
            "gpt_prompt_human_eval.j2",
            code=code,
        ).strip()
    elif dataset == "SWE-Bench-verified":
        prompt = _render_template(
            "gpt_prompt_swe_bench_verified.j2",
            code=code,
            problem_statement=problem_statement,
        ).strip()
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    return prompt


def build_gpt_gt_prompt(
    dataset, code, correct_code, docstring=None, context=None, problem_statement=None
):
    if dataset == "CoderEval":
        prompt = _render_template(
            "gpt_gt_prompt_coder_eval.j2",
            code=code,
            docstring=docstring,
            context=context,
            correct_code=correct_code,
        ).strip()
    elif dataset == "HumanEval":
        prompt = _render_template(
            "gpt_gt_prompt_human_eval.j2",
            code=code,
            correct_code=correct_code,
        ).strip()
    elif dataset == "SWE-Bench-verified":
        prompt = _render_template(
            "gpt_gt_prompt_swe_bench_verified.j2",
            code=code,
            correct_code=correct_code,
            problem_statement=problem_statement,
        ).strip()
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    return prompt


def build_repair_prompt(
    solution,
    feedback,
    docstring=None,
    context=None,
    problem_statement=None,
    current_task=None,
    dataset="CoderEval",
    is_persona=True,
    is_cot=False,
    is_few_shot=False,
    is_instructions=True,
    is_es_shot=False,
    is_sa=False,
    is_sg_icl=False,
    is_sbp=False,
    is_rr=False,
):
    persona = "You are a professional code repair assistant skilled at fixing code errors based on the @@Feedback."
    best_example = None
    if is_es_shot:
        best_example = (
            find_best_example(current_task or {}, dataset) if current_task else None
        )

    return _render_template(
        "repair_prompt.j2",
        persona=persona,
        solution=solution,
        feedback=feedback,
        docstring=docstring,
        context=context,
        problem_statement=problem_statement,
        is_persona=is_persona,
        is_cot=is_cot,
        is_few_shot=is_few_shot,
        is_instructions=is_instructions,
        is_es_shot=is_es_shot,
        is_sa=is_sa,
        is_sg_icl=is_sg_icl,
        is_sbp=is_sbp,
        is_rr=is_rr,
        best_example=best_example,
    )
