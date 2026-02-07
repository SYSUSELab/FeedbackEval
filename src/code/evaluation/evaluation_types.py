from __future__ import annotations

from dataclasses import dataclass

from feedback.evaluation_feedback import get_mixed_feedback


@dataclass(frozen=True)
class PromptOptions:
    use_docstring: bool
    use_context: bool
    use_persona: bool
    use_cot: bool
    use_few_shot: bool
    use_instructions: bool
    use_es_shot: bool
    use_sa: bool
    use_sg_icl: bool
    use_sbp: bool
    use_rr: bool


@dataclass
class RepairCandidate:
    id: int
    source: str
    current_code: str
    repair_history: list

    @classmethod
    def from_candidate(
        cls,
        idx: int,
        candidate: dict,
        feedback: str,
        dataset: str,
        ques: dict,
    ) -> "RepairCandidate":
        initial_feedback = candidate.get(feedback, None)
        if feedback == "mixed_feedback":
            initial_feedback = get_mixed_feedback(
                dataset, candidate["generate_code"], ques, candidate
            )
        initial_test_feedback = candidate.get("test_feedback", "")
        initial_record = {
            "round": 0,
            "generate_code": candidate["generate_code"],
            "feedback": initial_feedback,
            "isTrue": False,
        }
        if initial_test_feedback:
            initial_record["test_feedback"] = initial_test_feedback
        return cls(
            id=idx,
            source=candidate["source"],
            current_code=candidate["generate_code"],
            repair_history=[initial_record],
        )

    def record_round(
        self,
        round_idx: int,
        code: str,
        feedback: str,
        is_true: bool,
        test_feedback: str = None,
    ):
        record = {
            "round": round_idx,
            "generate_code": code,
            "feedback": feedback,
            "isTrue": is_true,
        }
        if test_feedback is not None:
            record["test_feedback"] = test_feedback
        self.repair_history.append(record)
        self.current_code = code

    def to_result(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "repair_history": self.repair_history,
        }


def task_id(ques: dict) -> str:
    return str(ques.get("_id") or ques.get("task_id") or ques.get("instance_id"))
