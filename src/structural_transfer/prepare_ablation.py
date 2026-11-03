"""EVENT_SPAN_TRANSFORMER_ABLATION_20261013: structural inputs and remainder targets for the E / S / E+S cells.

Source: the frozen corpus of the article model v2 (span_transformer_2026_10_07/v2/data/sequences.npz and metadata.json),
unchanged.  Nothing of the text is read.  For every completed segment j of a trace the 41 input numbers are split into
three groups; a cell multiplies the groups it must not see by zero during training and evaluation alike.

  common (3): log position of the segment, log characters of the segment, log characters read so far
  S      (2): log age of the current span in segments, log characters of the current span so far
  E     (36): own events of the segment (6) and their known flags (6); two decaying memories of the events (12, the
              old columns 29:41, they never reset at a span boundary); cumulative count of every event over the
              prefix (6); segments since the last event of every type, or since the start if there was none (6)
The old columns 17:29 (events counted inside the current span and inside the previous span) reset at span boundaries
and therefore reveal S; they are not used by any cell.  The span type of the current and of the previous span are
embeddings of the model and belong to S.
Remainder targets after segment j (only for traces whose natural end is observed; a trace cut by a limit has no
remainder): segments, span starts and characters that follow segment j.  Characters are characters; no token unit
exists in this corpus."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

GROUPS = dict(common=(0, 3), S=(3, 5), E=(5, 41))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(x, span_start, offsets, end_observed):
    N = len(x)
    xin, rem, rem_mask = np.zeros((N, 41), np.float32), np.zeros((N, 3), np.float32), np.zeros(N, bool)
    xin[:, 0], xin[:, 1], xin[:, 2] = x[:, 1], x[:, 2], x[:, 4]
    xin[:, 3], xin[:, 4] = x[:, 0], x[:, 3]
    xin[:, 5:17] = x[:, 5:17]
    xin[:, 17:29] = x[:, 29:41]
    chars = np.rint(np.expm1(x[:, 2].astype(np.float64) * 10))
    for i in range(len(offsets) - 1):
        a, z = offsets[i], offsets[i + 1]
        ev = x[a:z, 5:11].astype(np.float64)
        cum = ev.cumsum(0)
        xin[a:z, 29:35] = np.log1p(cum) / 5
        pos = np.arange(1, z - a + 1, dtype=np.float64)[:, None]
        last = np.maximum.accumulate(np.where(ev > 0, pos, 0.0), 0)      # position of the last event of the type, 0 if none yet
        xin[a:z, 35:41] = np.log1p(pos - last) / 10
        if end_observed[i]:
            n = z - a
            st = span_start[a:z].astype(np.int64)
            c = chars[a:z]
            rem[a:z, 0] = n - 1 - np.arange(n)
            rem[a:z, 1] = st.sum() - st.cumsum()                         # span starts after segment j
            rem[a:z, 2] = c.sum() - c.cumsum()
            rem_mask[a:z] = True
    return xin, rem, rem_mask


def main(data_dir, out):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())
    with np.load(data_dir / "sequences.npz") as z:
        x, span_start, offsets, split = z["x"], z["span_start"], z["offsets"], z["split"]
    assert len(meta["traces"]) == len(offsets) - 1 and all(t["offset"] == offsets[i] for i, t in enumerate(meta["traces"]))
    end_observed = np.array([t["end_observed"] for t in meta["traces"]])
    xin, rem, rem_mask = build(x, span_start, offsets, end_observed)
    np.savez_compressed(out, xin=xin, rem=rem, rem_mask=rem_mask)
    rec = dict(task_id="EVENT_SPAN_TRANSFORMER_ABLATION_20261013", groups=GROUPS, segments=int(len(x)), traces=int(len(offsets) - 1),
               traces_with_observed_natural_end={s: int(sum(1 for t in meta["traces"] if t["end_observed"] and t["split"] == s)) for s in ("train", "validation", "test")},
               segments_with_remainder={s: int((rem_mask & (split == k)).sum()) for k, s in enumerate(("train", "validation", "test"))},
               sha256={"sequences.npz": sha(data_dir / "sequences.npz"), "metadata.json": sha(data_dir / "metadata.json"), "ablation_inputs.npz": sha(out), "prepare_ablation.py": sha(Path(__file__).resolve())})
    Path(str(out)[:-4] + "_BUILD.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
