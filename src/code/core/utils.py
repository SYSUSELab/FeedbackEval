import json
import random
import re
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.client import LlmClient, LlmConfig

logger = logging.getLogger("utils")

# 最大反馈长度，从环境变量读取，默认 5000
MAX_FEEDBACK_LENGTH = int(os.getenv("MAX_FEEDBACK_LENGTH", "5000"))

FEEDBACK_TYPES = [
    "test_feedback",
    "compiler_feedback",
    "llm_skilled_feedback",
    "llm_expert_feedback",
    "minimal_feedback",
    "mixed_feedback",
]


def truncate_feedback(feedback, max_length: int = None):
    """
    截断过长的反馈信息，从尾部截断

    Args:
        feedback: 原始反馈（字符串或其他可转换为字符串的类型）
        max_length: 最大长度阈值，不指定则使用 MAX_FEEDBACK_LENGTH

    Returns:
        截断后的反馈字符串（如果超过阈值）或原始值
    """
    if max_length is None:
        max_length = MAX_FEEDBACK_LENGTH

    if not feedback:
        return feedback

    # 确保转换为字符串
    feedback_str = str(feedback) if not isinstance(feedback, str) else feedback

    if len(feedback_str) <= max_length:
        return feedback_str

    # 从尾部截断，保留前面的内容
    truncated = feedback_str[:max_length] + f"\n\n... truncated ..."

    return truncated


@dataclass(frozen=True)
class RunContext:
    dataset: str
    model: str | None = None
    version: str | None = None
    feedback: str | None = None
    function: str | None = None


class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        runtime_extra = kwargs.pop("extra", {}) or {}
        merged = {**self.extra, **runtime_extra}
        context_parts = []
        for key in (
            "dataset",
            "module",
            "function",
            "model",
            "version",
            "feedback",
            "task_id",
            "round",
            "stage",
        ):
            value = merged.get(key)
            if value is not None:
                context_parts.append(f"{key}={value}")
        prefix = f"[{', '.join(context_parts)}]" if context_parts else ""
        return (f"{prefix} {msg}" if prefix else msg, kwargs)


def setup_logging(context: RunContext, module_name: str):
    """General logging setup function."""
    log_dir_parts = ["logs", context.dataset, module_name]
    if context.function:
        log_dir_parts.append(context.function)
    if context.model:
        log_dir_parts.append(context.model)
    if context.version:
        log_dir_parts.append(context.version)
    if context.feedback:
        if isinstance(context.feedback, list):
            if len(context.feedback) == 1:
                log_dir_parts.append(context.feedback[0])
            else:
                log_dir_parts.append("multi_feedback")
        else:
            log_dir_parts.append(context.feedback)

    log_dir = os.path.join(*log_dir_parts)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
        ],
        force=True,
    )

    logger = logging.getLogger(module_name)
    return ContextLoggerAdapter(
        logger,
        {
            "dataset": context.dataset,
            "module": module_name,
            "function": context.function,
            "model": context.model,
            "version": context.version,
            "feedback": context.feedback,
        },
    )


class DataLoader:
    def __init__(self, file_path, sample_size=-1):
        self.file_path = file_path
        self.sample_size = sample_size
        self.data = self._load_data()

    def _load_data(self):
        if "HumanEval" in self.file_path:
            return self._load_human_eval()
        elif "CoderEval" in self.file_path:
            return self._load_coder_eval()
        else:
            raise ValueError("Invalid file path")

    def _load_human_eval(self):
        data_list = []
        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_data = json.loads(line.strip())
                data_list.append(json_data)
        if self.sample_size == -1:
            return data_list
        return random.sample(data_list, self.sample_size)

    def _load_coder_eval(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.sample_size == -1:
            return data["RECORDS"]
        return random.sample(data["RECORDS"], self.sample_size)


def read_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line.strip())
            data_list.append(json_data)
    return data_list


def write_jsonl(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")


def get_model_response(model_version, prompt):
    try:
        config = LlmConfig(model=model_version)
        llm = LlmClient(config)
        generate_result = llm.complete(prompt)
        return generate_result
    except Exception as e:
        logger.error(f"Error during code generation: {e}")


def extract_repaired_code(generated_text):
    try:
        match = re.search(
            r"<repaired_code>(.*?)</repaired_code>", generated_text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        else:
            raise ValueError(
                "No code found between <repaired_code> tags in the generated result."
            )
    except Exception as e:
        logger.error(f"Error extracting repaired code: {e}")
        return None


class ProgressiveCache:
    """Incremental JSONL cache with resume support."""

    def __init__(self, file_path: str, id_field: str = "_id"):
        self.file_path = file_path
        self.id_field = id_field
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.completed_ids = self._load_existing_ids()

    def _load_existing_ids(self) -> set[str]:
        completed = set()
        if not os.path.exists(self.file_path):
            return completed
        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    record_id = record.get(self.id_field)
                    if record_id is not None:
                        completed.add(str(record_id))
                except json.JSONDecodeError:
                    continue
        return completed

    def has(self, record_id: str | int | None) -> bool:
        if record_id is None:
            return False
        return str(record_id) in self.completed_ids

    def append(self, record: dict) -> None:
        record_id = record.get(self.id_field)
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if record_id is not None:
            self.completed_ids.add(str(record_id))

    def append_many(self, records: list[dict]) -> None:
        if not records:
            return
        with open(self.file_path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_id = record.get(self.id_field)
                if record_id is not None:
                    self.completed_ids.add(str(record_id))
