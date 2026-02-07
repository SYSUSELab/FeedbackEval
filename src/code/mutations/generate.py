import os
import sys
import re
import logging
from core.utils import DataLoader, ProgressiveCache, setup_logging, RunContext
from tqdm import tqdm
from prompts.template import build_mutant_prompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.client import LlmClient, LlmConfig

logger = logging.getLogger("generate")


def generate_false_results(llm, prompt, attempts=3):
    """Generate false results using the LLM."""
    false_results = []
    results = llm.batch_complete([prompt] * attempts)
    for result in results:
        if result is None:
            continue
        try:
            match = re.search(r"```python\n(.*?)```", result, re.DOTALL)
            extracted_code = match.group(1).strip()
            false_results.append(
                {"source": "llm-based", "generate_code": extracted_code}
            )
        except Exception as e:
            logger.warning(f"error:{e}", extra={"stage": "parse"})
            continue
    return false_results


class Generator:
    def __init__(self, file_path, sample_size=-1):
        self.file_path = file_path
        self.sample_size = sample_size
        self.original_data = DataLoader(self.file_path, self.sample_size).data

    def generate_mutants(self):
        output_path = ""
        mut_list = []

        dataset = "HumanEval" if "HumanEval" in self.file_path else "CoderEval"
        global logger
        logger = setup_logging(
            RunContext(dataset=dataset, function="generate"), "generate"
        )

        if "HumanEval" in self.file_path:
            output_path = f"../../output/human_eval/llm_mutants.jsonl"
            cache = ProgressiveCache(output_path, id_field="task_id")
            for data in tqdm(
                self.original_data,
                total=len(self.original_data),
                desc="Generating Mutants",
            ):
                if cache.has(data.get("task_id")):
                    continue
                ori_code = f"{data['prompt']}\n{data['canonical_solution']}"
                prompt = build_mutant_prompt(ori_code)
                llm = LlmClient(LlmConfig(model="gpt-4o-mini"))
                false_results = generate_false_results(llm, prompt)
                record = {
                    "task_id": data["task_id"],
                    "false_results": false_results,
                    "test": f"{data['test']}\ncheck({data['entry_point']})",
                }
                mut_list.append(record)
                cache.append(record)
        elif "CoderEval" in self.file_path:
            output_path = f"../../output/coder_eval/llm_mutants.jsonl"
            cache = ProgressiveCache(output_path, id_field="_id")
            for data in tqdm(
                self.original_data,
                total=len(self.original_data),
                desc="Generating Mutants",
            ):
                if cache.has(data.get("_id")):
                    continue
                ori_code = data["code"]
                prompt = build_mutant_prompt(ori_code)
                llm = LlmClient(LlmConfig(model="gpt-4o-mini"))
                false_results = generate_false_results(llm, prompt)
                record = {"_id": data["_id"], "false_results": false_results}
                mut_list.append(record)
                cache.append(record)
        else:
            raise ValueError("Invalid file path.")

        logger.info(f"Results saved to {output_path}", extra={"stage": "save"})


if __name__ == "__main__":
    # generator = Generator(f'../input/CoderEval4Python.json')
    generator = Generator(f"../../input/HumanEval.jsonl")
    generator.generate_mutants()
