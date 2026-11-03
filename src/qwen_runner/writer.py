from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlResultWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids = self._load_processed_ids()

    def _load_processed_ids(self) -> set[str]:
        if not self.output_path.exists():
            return set()

        processed_ids: set[str] = set()
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                processed_ids.add(str(payload["question_id"]))
        return processed_ids

    def write_result(
        self,
        *,
        question_id: str,
        question: str,
        correct_answer: str,
        raw_model_work: dict[str, Any],
    ) -> None:
        payload = {
            "question_id": question_id,
            "question": question,
            "correct_answer": correct_answer,
            "raw_model_work": raw_model_work,
        }

        line = json.dumps(payload, ensure_ascii=False)
        # Unicode line/paragraph separators are legal raw inside JSON strings but
        # break line-oriented jsonl readers — escape them (corpus gotcha).
        line = (
            line.replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
            .replace("\u0085", "\\u0085")
        )
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        self.processed_ids.add(question_id)
