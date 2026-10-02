"""The lexical detector used by the frozen gate, with a standalone trainer."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from .gate_runtime.features import EVENTS, SPANS
from .io import read_jsonl, sha256, write_json


def context(current, previous=""):
    return "CURRENT " + current[-1600:] + " PREVIOUS " + previous[-800:]


class Detector:
    """Load only trusted project checkpoints; pickle contains executable objects."""
    def __init__(self, checkpoint):
        with open(checkpoint, "rb") as stream:
            self.bundle = pickle.load(stream)
        if len(self.bundle["events"]) != len(EVENTS):
            raise ValueError("Event order/width does not match the release taxonomy")

    def predict_contexts(self, contexts):
        if not contexts:
            return []
        x = self.bundle["vectorizer"].transform(contexts)
        probabilities = np.stack([model.predict_proba(x)[:, 1]
                                  for model in self.bundle["events"]], axis=1)
        spans = self.bundle["spans"].predict(x)
        return [{"events": dict(zip(EVENTS, map(float, row))),
                 "span": SPANS[int(span)], "span_index": int(span)}
                for row, span in zip(probabilities, spans)]

    def predict_segments(self, segments):
        return self.predict_contexts([context(text, segments[i-1] if i else "")
                                      for i, text in enumerate(segments)])


def train(input_path, output_path, folds=(1, 2), seed=20260924):
    """Fit the historical detector on the explicitly recorded question folds."""
    contexts, targets, operators, known = [], [], [], []
    groups = set()
    for row in read_jsonl(input_path):
        if row["fold"] not in folds:
            continue
        groups.add(row["group"])
        segments = row["segments"]
        events = row["events"]
        spans = row["spans"]
        mask = row["event_known"]
        if not len(segments) == len(events) == len(spans) == len(mask):
            raise ValueError("Mismatched segment/annotation lengths")
        contexts.extend(context(text, segments[i-1] if i else "")
                        for i, text in enumerate(segments))
        targets.extend(events); operators.extend(spans); known.extend(mask)
    if not contexts:
        raise ValueError("No training segments in the requested folds")
    targets = np.asarray(targets, dtype=np.int8)
    operators = np.asarray(operators, dtype=np.int8)
    known = np.asarray(known, dtype=bool)
    if targets.shape != (len(contexts), len(EVENTS)) or not np.isin(targets, [0, 1]).all():
        raise ValueError("Invalid binary event labels")
    if not np.isin(operators, np.arange(-1, len(SPANS))).all():
        raise ValueError("Invalid span labels")
    vectorizer = HashingVectorizer(n_features=65536, ngram_range=(1, 2),
                                   alternate_sign=False, norm="l2", dtype=np.float32)
    x = vectorizer.transform(contexts)
    parameters = dict(loss="log_loss", alpha=1e-5, max_iter=12, tol=1e-3, random_state=seed)
    models = [SGDClassifier(**parameters).fit(x[known], targets[known, j])
              for j in range(len(EVENTS))]
    span_model = SGDClassifier(**parameters).fit(x[operators >= 0], operators[operators >= 0])
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(dict(vectorizer=vectorizer, events=models, spans=span_model), stream)
    write_json(output_path.with_suffix(".json"), {
        "input_sha256": sha256(input_path), "checkpoint_sha256": sha256(output_path),
        "folds": list(folds), "seed": seed, "questions": len(groups),
        "segments": len(contexts), "known_event_segments": int(known.sum()),
        "known_span_segments": int((operators >= 0).sum()),
        "events": list(EVENTS), "spans": list(SPANS), "parameters": parameters,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--input", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--folds", default="1,2")
    fit.add_argument("--seed", type=int, default=20260924)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True, help='JSONL: {"segments": ["text", ...]}')
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.input, args.output, tuple(map(int, args.folds.split(","))), args.seed)
    else:
        model = Detector(args.checkpoint)
        with open(args.output, "x", encoding="utf-8") as stream:
            for row in read_jsonl(args.input):
                out = {"id": row.get("id"), "segments": model.predict_segments(row["segments"])}
                stream.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
