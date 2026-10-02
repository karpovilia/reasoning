"""Create the unpublished corpus layout and validate trace/annotation joins."""
import argparse
import hashlib
import json
from pathlib import Path

from .gate_runtime.features import EVENTS, SPANS
from .io import read_jsonl, write_json


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize(catalog_path, destination):
    """Create directories and parameter records only; never download data."""
    catalog_path, destination = Path(catalog_path), Path(destination)
    catalog = json.loads(catalog_path.read_text())
    for entry in catalog["runs"]:
        relative = Path(entry["run_id"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Run directory must remain inside destination")
        run = json.loads((catalog_path.parent / entry["config"]).read_text())
        paths = [(destination / relative / "run.json", run)]
        for annotation in run["annotations"]:
            name = annotation["annotation_run_id"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Annotation run id must be a single path component")
            paths.append((destination / relative / "annotations" / name / "run.json", annotation))
        for path, contents in paths:
            if path.exists():
                if json.loads(path.read_text()) != contents:
                    raise FileExistsError(f"Existing configuration differs: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, contents)
    for name in ("detector", "gate", "annotator"):
        (destination / "training" / name).mkdir(parents=True, exist_ok=True)
    return len(catalog["runs"])


def validate_trace(row):
    required = ("schema_version", "trace_id", "benchmark", "question_id", "solver_run_id",
                "reasoning", "reasoning_sha256", "source", "segments")
    for field in required:
        if field not in row:
            raise ValueError(f"Trace lacks {field}")
    if row["schema_version"] != 1 or digest(row["reasoning"]) != row["reasoning_sha256"]:
        raise ValueError("Trace version or text hash differs")
    previous_end = 0
    for index, segment in enumerate(row["segments"]):
        start, end = segment["char_start"], segment["char_end"]
        if (segment["seg_id"] != index or type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= len(row["reasoning"])):
            raise ValueError("Invalid segment coordinates/order")
        if row["reasoning"][start:end] != segment["text"]:
            raise ValueError("Segment does not match the untouched source text")
        previous_end = end


def validate_annotation(row, trace):
    if row["schema_version"] != 1:
        raise ValueError("Unsupported annotation version")
    if row["trace_id"] != trace["trace_id"] or row["reasoning_sha256"] != trace["reasoning_sha256"]:
        raise ValueError("Annotation points to another trace/text revision")
    if not row.get("annotation_run_id") or not row.get("source"):
        raise ValueError("Annotation model/config and source must be identified")
    n = len(trace["segments"])
    lo, hi = row["lo"], row["hi"]
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= n:
        raise ValueError("Invalid target window")
    clo, chi = row["context_lo"], row["context_hi"]
    if type(clo) is not int or type(chi) is not int or not 0 <= clo <= lo < hi <= chi <= n:
        raise ValueError("Invalid annotation context")
    at = row["available_at_char"]
    if type(at) is not int or not trace["segments"][chi - 1]["char_end"] <= at <= len(trace["reasoning"]):
        raise ValueError("Annotation availability precedes text seen by its labeler")
    if row["status"] not in ("accepted", "partial", "quarantined"):
        raise ValueError("Unknown annotation status")
    if row["status"] == "quarantined":
        if row["events"] or row["spans"]:
            raise ValueError("Quarantined output cannot be used as accepted labels")
        return
    seen = set()
    for event in row["events"]:
        slot = (event["seg_id"], event["type"])
        if type(slot[0]) is not int or not lo <= slot[0] < hi or slot[1] not in EVENTS or slot in seen:
            raise ValueError("Invalid or duplicate canonical event")
        seen.add(slot)
    coverage = [0] * (hi - lo)
    for span in row["spans"]:
        a, b = span["seg_start"], span["seg_end"]
        if type(a) is not int or type(b) is not int or not lo <= a < b <= hi or span["type"] not in SPANS:
            raise ValueError("Invalid canonical span")
        for i in range(a, b):
            coverage[i - lo] += 1
    if any(v > 1 for v in coverage) or row["status"] == "accepted" and not all(v == 1 for v in coverage):
        raise ValueError("Overlapping spans or missing span coverage in an accepted window")


def validate_files(traces, annotations):
    index = {}
    for row in read_jsonl(traces):
        validate_trace(row)
        if row["trace_id"] in index:
            raise ValueError("Duplicate trace id")
        index[row["trace_id"]] = row
    seen = set()
    count = 0
    for row in read_jsonl(annotations):
        if row["annotation_id"] in seen or row["trace_id"] not in index:
            raise ValueError("Duplicate annotation id or orphan annotation")
        validate_annotation(row, index[row["trace_id"]])
        seen.add(row["annotation_id"])
        count += 1
    return {"traces": len(index), "annotation_windows": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--catalog", default="data/releases/2026-10-02/catalog.json")
    init.add_argument("--out", default="data/corpora/release-20261002")
    check = sub.add_parser("validate")
    check.add_argument("--traces", required=True)
    check.add_argument("--annotations", required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps({"runs_prepared": initialize(args.catalog, args.out), "data_downloaded": False}))
    else:
        print(json.dumps(validate_files(args.traces, args.annotations)))


if __name__ == "__main__":
    main()
