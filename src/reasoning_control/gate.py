"""Portable loading and inference for the frozen event-span gate."""
import argparse
import json
from pathlib import Path

from .gate_runtime.controller import SecondsObserver
from .gate_runtime.features import EVENTS, validate
from .gate_runtime.policy import observation_from_dict
from .io import read_jsonl


def load(config_path, root=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if root is None:
        candidates = [p for p in config_path.parents if (p / "models" / "gate").is_dir()]
        if not candidates:
            raise ValueError("Pass root containing models/gate")
        root = candidates[0]
    weights = Path(config["weights"])
    config["weights"] = str(weights if weights.is_absolute() else Path(root) / weights)
    return SecondsObserver(config)


def detect_and_decide(detector, gate, observation):
    """The caller supplies only completed segments and native token coordinates."""
    row = dict(observation)
    segments = row.get("segments", [])
    predictions = detector.predict_segments([s["text"] for s in segments])
    row["segments"] = [{"token_start": s["token_start"], "token_end": s["token_end"],
                         "available_at": s["available_at"],
                         "events": [p["events"][event] for event in EVENTS],
                         "span": p["span_index"]} for s, p in zip(segments, predictions)]
    validate(observation_from_dict(row))
    return {"observation": row, "decision": gate.decide(row)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector", help="If set, annotate supplied completed segment texts first")
    args = parser.parse_args()
    gate = load(args.config, args.root)
    detector = None
    if args.detector:
        from .detector import Detector
        detector = Detector(args.detector)
    with open(args.output, "x", encoding="utf-8") as stream:
        for row in read_jsonl(args.input):
            if not isinstance(row.get("prompt_tokens"), int) or row["prompt_tokens"] < 0:
                raise ValueError("prompt_tokens must be a nonnegative integer")
            out = detect_and_decide(detector, gate, row) if detector else gate.decide(row)
            stream.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
