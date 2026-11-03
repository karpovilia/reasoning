"""PROCESS_FORECAST_GATE_20261013: reference bridge and forward of the matched pair P_text (T) and P_ES_matched (TES).

For the consumer of C_text / C_matched.  The inputs of a prefix are the 41 numbers, the span type and the type of the
previous span built by the accepted reference pes_reference.build_inputs from three raw rows of the completed segments
(characters, 0/1 events, operator index), plus two text vectors: the frozen MiniLM vector of every completed segment and
the vector of the question.  T ignores the events, the spans and the span types by construction; they may be given or
set to zero, the forecast is the same.  The classes are imported from the accepted training code (matched.py, which
checks the sum of ablation.py before it runs it); nothing is copied.  No fit and no gradient here.

encode         the text vectors of a list of texts by the procedure of the corpus table: tokenizer of the frozen encoder,
               truncation to 256 word pieces, mean of the last hidden states over the attention mask, L2 norm, float16.
forward_last   the forecast after the last completed segment of a prefix, with the window layout of the training; the
               outputs and their units are those of pes_reference.named.
check          for every given weight: the forward on rebuilt inputs and on vectors encoded again from the raw texts
               reproduces the validation forecasts stored by the training code on the H200; a prefix alone gives the
               forecast of the same position in the window with later segments; seconds per prefix on the CPU.

  python matched_reference.py check --corpus DIR --texts texts.jsonl.gz --encoder DIR --fits DIR --src DIR --provider DIR --out JSON [--fit T_seed17 ...]
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CFG = dict(width=512, layers=12, heads=8, context=256)
TEXT = 384
_ENCODERS = {}


def load_code(src, provider):
    sys.path.insert(0, str(provider))
    sys.path.insert(0, str(src))
    import matched as X                                                 # checks the sum of the accepted ablation.py before executing it
    import pes_reference as R
    return X, R


def load_model(X, R, checkpoint, cell, expect_sha=None):
    if expect_sha is not None:
        assert X.sha(checkpoint) == expect_sha, "the checkpoint is not the one of the manifest"
    model = X.Predictor(cell, dict(operators=R.OPERATORS, solvers=R.SOLVERS, benchmarks=R.BENCHMARKS), CFG)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), int(state["step"])


def encode(texts, encoder_dir, batch=64):
    """Vectors of the texts, float16 as in the corpus table."""
    if str(encoder_dir) not in _ENCODERS:
        from transformers import AutoModel, AutoTokenizer
        _ENCODERS[str(encoder_dir)] = (AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True), AutoModel.from_pretrained(encoder_dir, local_files_only=True).eval())
    tok, enc = _ENCODERS[str(encoder_dir)]
    out = np.zeros((len(texts), TEXT), np.float16)
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            t = tok(list(texts[s:s + batch]), padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = enc(**t).last_hidden_state
            m = t["attention_mask"][..., None]
            out[s:s + batch] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp_min(1), p=2, dim=1).numpy().astype(np.float16)
    return out


@torch.no_grad()
def run(model, xin, mode, prev, solver, bench, text, prompt):
    """text (n, 384): vectors of the n segments; prompt (384,): the vector of the question, the same at every position."""
    n = len(xin)
    assert n <= CFG["context"] and text.shape == (n, TEXT) and prompt.shape == (TEXT,)
    b = dict(xin=torch.from_numpy(np.ascontiguousarray(xin, np.float32))[None], mode=torch.from_numpy(np.asarray(mode, np.int64))[None], prev_mode=torch.from_numpy(np.asarray(prev, np.int64))[None],
             solver=torch.full((1, n), solver, dtype=torch.long), bench=torch.full((1, n), bench, dtype=torch.long), pad=torch.zeros((1, n), dtype=torch.bool),
             text=torch.from_numpy(np.asarray(text, np.float16)).float()[None], prompt=torch.from_numpy(np.asarray(prompt, np.float16)).float()[None, None].expand(1, n, TEXT))
    return {k: v[0].float() for k, v in model(b).items()}


def forward_last(R, model, xin, mode, prev, solver, bench, text, prompt):
    """Forecast after the last completed segment of the prefix (rows 0..j of build_inputs and of the text vectors)."""
    lo = R.window(len(xin) - 1)
    return {k: v[-1] for k, v in R.named(run(model, xin[lo:], mode[lo:], prev[lo:], solver, bench, text[lo:], prompt)).items()}


def forward_all(R, model, xin, mode, prev, solver, bench, text, prompt):
    n, parts = len(xin), []
    for s in range(0, n, R.SCORED):
        lo, hi = max(0, s - R.WARM), min(n, s + R.SCORED)
        o = R.named(run(model, xin[lo:hi], mode[lo:hi], prev[lo:hi], solver, bench, text[lo:hi], prompt))
        parts.append({k: v[s - lo:] for k, v in o.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def check(a):
    torch.set_num_threads(a.threads)
    X, R = load_code(a.src, a.provider)
    corpus = Path(a.corpus)
    meta = json.loads((corpus / "metadata.json").read_text())
    assert meta["operators"] == R.OPERATORS and meta["solvers"] == R.SOLVERS and meta["benchmarks"] == R.BENCHMARKS
    with np.load(corpus / "sequences.npz") as z:
        d = {k: z[k] for k in ("x", "mode", "prev_mode", "solver", "bench", "split", "offsets", "gate", "text_id", "prompt_id")}
    with gzip.open(a.texts, "rt", encoding="utf8") as f:
        texts = [json.loads(line) for line in f]
    off, n_seg = d["offsets"], np.diff(d["offsets"])
    val = np.flatnonzero((d["split"][off[:-1]] == 1) & (n_seg <= a.max_segments))
    order = val[np.argsort(n_seg[val], kind="stable")]
    pick = sorted(set(order[np.linspace(0, len(order) - 1, a.traces).astype(int)].tolist()))
    chars_all = np.rint(np.expm1(d["x"][:, 2].astype(np.float64) * 10))
    # the inputs of the picked traces from the raw rows and the raw texts only
    built, t0 = {}, time.time()
    for i in pick:
        s, e = int(off[i]), int(off[i + 1])
        assert all(len(texts[int(t)]) == int(c) for t, c in zip(d["text_id"][s:e], chars_all[s:e]))
        xin, mode, prev = R.build_inputs(chars_all[s:e], d["x"][s:e, 5:11], d["mode"][s:e])
        vec = encode([texts[int(t)] for t in d["text_id"][s:e]] + [texts[int(d["prompt_id"][s])]], a.encoder)
        built[i] = (xin, mode, prev, vec[:-1], vec[-1])
    rep = dict(task="PROCESS_FORECAST_GATE_20261013: reference bridge and forward of the matched pair; no fit",
               sha256={"matched_reference.py": X.sha(Path(__file__).resolve()), "matched.py": X.sha(Path(X.__file__).resolve()), "ablation.py": X.ABLATION_SHA, "pes_reference.py": X.sha(Path(R.__file__).resolve())},
               torch=torch.__version__, threads=a.threads, traces=len(pick), segments=int(sum(len(v[0]) for v in built.values())), seconds_to_build_and_encode=time.time() - t0, weights={})
    rng = np.random.default_rng(20261013)
    keys = ("gate", "dest", "events", "next_span_marginal")
    for fit in a.fit:
        cell, seed = fit.split("_seed")
        folder = Path(a.fits) / fit
        done = json.loads((folder / "DONE.json").read_text())
        model, step = load_model(X, R, folder / "best.pt", cell, done["best_checkpoint_sha256"])
        with np.load(folder / "predictions_validation.npz") as z:
            st = {k: z[k] for k in z.files}
        where = {int(v): k for k, v in enumerate(st["index"])}
        mx, single, positions = {k: 0.0 for k in keys + ("rem_relative", "length_log")}, [], 0
        blind = 0.0
        for i in pick:
            s = int(off[i])
            xin, mode, prev, text, prompt = built[i]
            sv, bn = int(d["solver"][s]), int(d["bench"][s])
            o = forward_all(R, model, xin, mode, prev, sv, bn, text, prompt)
            lab = np.flatnonzero(d["gate"][s:s + len(xin)] >= 0)
            rows = np.array([where[s + int(j)] for j in lab])
            positions += len(lab)
            for k in keys:
                mx[k] = max(mx[k], float(np.abs(o[k][lab] - st[k][rows].astype(np.float32)).max()))
            mx["rem_relative"] = max(mx["rem_relative"], float((np.abs(o["rem"][lab] - st["rem"][rows]) / np.maximum(1.0, st["rem"][rows])).max()))
            mx["length_log"] = max(mx["length_log"], float(np.abs(np.log1p(o["next_segment_chars"][lab]) - np.minimum(st["length"][rows], 15)).max()))
            if cell == "T":                                             # T must not read the structure: zero structure and any span types give the same forecast
                z0 = xin.copy()
                z0[:, 3:41] = 0
                o0 = forward_all(R, model, z0, np.zeros_like(mode), np.full_like(prev, 3), sv, bn, text, prompt)
                blind = max(blind, max(float(np.abs(o0[k] - o[k]).max()) for k in keys + ("rem",)))
            for j in rng.choice(lab, size=min(a.prefixes, len(lab)), replace=False):
                j = int(j)
                t1 = time.time()
                f = forward_last(R, model, xin[:j + 1], mode[:j + 1], prev[:j + 1], sv, bn, text[:j + 1], prompt)
                single.append((j + 1 - R.window(j), time.time() - t1, float(np.abs(f["gate"] - o["gate"][j]).max()), float((np.abs(f["rem"] - o["rem"][j]) / np.maximum(1.0, o["rem"][j])).max())))
        single = np.array(single)
        short = single[single[:, 0] <= 64]
        rep["weights"][fit] = dict(cell=cell, seed=int(seed), selected_step=step, checkpoint_sha256=done["best_checkpoint_sha256"], labelled_positions_compared=positions,
                                   max_abs_difference_from_the_stored_h200_forecasts=mx, prefixes_checked_one_by_one=int(len(single)), prefix_vs_window_max_gate_difference=float(single[:, 2].max()),
                                   prefix_vs_window_max_relative_remainder_difference=float(single[:, 3].max()),
                                   seconds_per_prefix_window_up_to_64_segments=dict(n=int(len(short)), median=float(np.median(short[:, 1])) if len(short) else None),
                                   T_forecast_change_when_structure_is_zeroed=blind if cell == "T" else None)
        print(fit, json.dumps(rep["weights"][fit]), flush=True)
        assert max(mx[k] for k in keys) < 5e-3 and mx["rem_relative"] < 5e-3 and single[:, 2].max() < 1e-4 and blind == 0.0, (mx, blind)
    rep["passed"] = True
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps(rep, indent=1) + "\n")
    tmp.replace(a.out)
    print("MATCHED REFERENCE CHECK PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("check",))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--fits", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", nargs="+", default=["T_seed17", "TES_seed17", "T_seed29", "TES_seed29", "T_seed43", "TES_seed43"])
    ap.add_argument("--traces", type=int, default=24)
    ap.add_argument("--prefixes", type=int, default=4)
    ap.add_argument("--max-segments", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    check(ap.parse_args())
