from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRecord:
    question_id: str
    question: str
    correct_answer: str
    image: str | None = None


def load_dataset_records(dataset_path: Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            image = payload.get("image")
            if dataset_path.stem == "hle" and image is not None:
                continue

            records.append(
                BenchmarkRecord(
                    question_id=str(payload["question_id"]),
                    question=str(payload["question"]),
                    correct_answer=str(payload["correct_answer"]),
                    image=image,
                )
            )

    return records
