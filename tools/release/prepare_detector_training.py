#!/usr/bin/env python3
"""Export the trusted audited pickle in its original order, retaining unknowns."""
import argparse
import gzip
import io
import json
from pathlib import Path
import pickle

from reasoning_control.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path,
                        help="Trusted local prepared.pkl from the 2026-09-24 audit")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with args.prepared.open("rb") as stream:
        traces, _ = pickle.load(stream)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Empty gzip filename and fixed mtime make the output bytes reproducible.
    with args.out.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                for trace in traces:
                    record = {k: trace[k] for k in ("key", "group", "fold", "segments")}
                    record.update(events=trace["events"].tolist(), spans=trace["ops"].tolist(),
                                  event_known=trace["event_known"].tolist())
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
    write_json(args.out.with_name(args.out.name + ".manifest.json"), {
        "source_sha256": sha256(args.prepared), "output_sha256": sha256(args.out),
        "traces": count, "order": "unchanged from audited prepared.pkl",
        "unknown_events": "event_known=false", "unknown_spans": -1,
    })
    print(json.dumps({"traces": count, "output_sha256": sha256(args.out)}))


if __name__ == "__main__":
    main()
