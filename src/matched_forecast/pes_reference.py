"""PROCESS_FORECAST_GATE_20261013, process provider: reference bridge and forward of the accepted P_ES weights.

Accepted weights: ES_seed17 / ES_seed29 / ES_seed43 of endfix_run_v1 (EVENT_SPAN_TRANSFORMER_ABLATION_20261013): 12 layers,
width 512, 8 heads, FF 2048, context 256, 38 042 797 parameters.  The class Predictor is imported from the accepted
ablation.py (checked by its sha256), never copied.  No fit and no gradient here.

build_inputs   the 41 numbers, the span type and the type of the previous span of every completed segment, from three raw
               facts of a prefix: characters of each completed segment, 0/1 events of each segment (order EVENTS), operator
               index of each segment (order OPERATORS).  Row j depends on segments 0..j only.
forward_last   the forecast after the last completed segment of a prefix, with the window layout of the training.
check          (1) build_inputs reproduces the stored inputs of the whole corpus; (2) the forward on the rebuilt inputs
               reproduces the predictions stored by the training code on the H200 for the validation part, both with the
               windows of the training and prefix by prefix; (3) seconds per prefix on the CPU.

  python pes_reference.py check --corpus DIR --inputs ablation_inputs_endfix_v2.npz --weights DIR --stored DIR --src DIR --out JSON
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OPERATORS = ["SETUP", "DERIVING", "EXPLORING", "VERIFYING", "REVISING", "CONCLUDING", "GROUNDING"]   # GROUNDING has no position in the corpus
EVENTS = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
SOLVERS = ["UNKNOWN", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-122B-A10B-FP8",
           "Qwen/Qwen3.6-35B-A3B-FP8", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-reasoner",
           "deepseek-v4-flash-0731-q3", "gemma-4-26b-a4b-it-nitro", "gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen3.6-35b-a3b-nitro"]
BENCHMARKS = ["UNKNOWN", "bbh", "gpqa_diamond", "hle", "math500"]       # index 0 of both lists has no training position: an untrained row
CFG = dict(width=512, layers=12, heads=8, context=256)
NO_PREVIOUS_SPAN, SCORED, WARM = 7, 64, 192
ABLATION_SHA = "a92f8b859376121fb4b5685e5f6c8ff6351dafd182ad24788e528506301e0790"
WEIGHTS = {17: "9836ccf43c5d09c903ce2e313bbc74661e6f95e8d57802b822b0fb7497ce5148", 29: "d0f3b6a14e22190c7477616038580b1047fdc27f912fc1cd7c47ef7f6f3006c2",
           43: "f0ae5271d06979834f0021de02b75007cb3738262ac738971dfa16ad155c40e6"}
GATE = ["STAY", "SWITCH", "END"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_inputs(chars, events, modes):
    """chars (n,) characters of each completed segment; events (n, 6) zeros and ones; modes (n,) operator index 0..5.
    Returns xin (n, 41) float32, mode (n,) int64, prev_mode (n,) int64.  A span starts where the operator changes."""
    chars, events, modes = np.asarray(chars, np.float64), np.asarray(events, np.float64), np.asarray(modes, np.int64)
    n = len(chars)
    assert events.shape == (n, 6) and modes.shape == (n,) and n >= 1 and np.isin(events, (0, 1)).all() and (modes >= 0).all() and (modes < 7).all()
    pos = np.arange(1, n + 1, dtype=np.float64)
    start = np.ones(n, bool)
    start[1:] = modes[1:] != modes[:-1]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))     # index of the first segment of the current span
    spent = chars.cumsum()
    before = spent - chars                                              # characters before segment j
    xin = np.zeros((n, 41), np.float32)
    xin[:, 0] = np.log1p(pos) / 10                                      # common: position, characters of the segment, characters read
    xin[:, 1] = np.log1p(chars) / 10
    xin[:, 2] = np.log1p(spent) / 10
    xin[:, 3] = np.log1p(np.arange(n) - first + 1.0) / 10               # S: age of the current span in segments, its characters so far
    xin[:, 4] = np.log1p(spent - before[first]) / 10
    xin[:, 5:11] = events                                               # E: events of the segment
    xin[:, 11:17] = 1                                                   # E: 'known' flags, equal to one at every position of the corpus
    memory, m, decay = np.zeros((n, 12)), np.zeros(12), np.repeat([np.exp(-1 / 4), np.exp(-1 / 32)], 6)
    for j in range(n):
        m = m * decay if j else m
        m = m + np.tile(events[j], 2)
        memory[j] = m
    xin[:, 17:29] = np.log1p(memory) / 5                                # E: two decaying memories, never reset at a span boundary
    xin[:, 29:35] = np.log1p(events.cumsum(0)) / 5                      # E: count of every event over the prefix
    last = np.maximum.accumulate(np.where(events > 0, pos[:, None], 0.0), 0)
    xin[:, 35:41] = np.log1p(pos[:, None] - last) / 10                  # E: segments since the last event of the type, or since the start
    prev = np.full(n, NO_PREVIOUS_SPAN, np.int64)
    has = first > 0
    prev[has] = modes[first[has] - 1]
    return xin, modes.copy(), prev


def load_model(src, checkpoint, seed):
    sys.path.insert(0, str(src))
    import ablation as A
    assert sha(Path(A.__file__).resolve()) == ABLATION_SHA, "ablation.py is not the accepted training code"
    assert sha(checkpoint) == WEIGHTS[seed], "the checkpoint is not the accepted one of this seed"
    model = A.Predictor("ES", dict(operators=OPERATORS, solvers=SOLVERS, benchmarks=BENCHMARKS), CFG)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), int(state["step"])


def named(o):
    """Raw outputs -> forecasts with their units.  Every forecast at segment j speaks about what follows segment j."""
    pi = torch.softmax(o["wait_pi"], -1)
    return dict(gate=F.softmax(o["gate"], -1).numpy(),                                   # P(same span, another span, natural end) for the next step
                dest=F.softmax(o["dest"], -1).numpy(),                                   # operator of the next segment; trained and scored at a change of span
                events=torch.sigmoid(o["events"]).numpy(),                               # six events of the next segment
                next_segment_chars=np.expm1(np.minimum(o["length"].numpy(), 15)),       # characters of the next segment
                next_span_marginal=(pi[..., None] * torch.softmax(o["wait_mark"], -1)).sum(-2).numpy(),   # type of the next boundary: 7 operators and END
                wait_pi=pi.numpy(), wait_scale_chars=o["wait_logscale"].exp().numpy(), wait_shape=o["wait_shape"].numpy(),   # three Weibull parts, characters to the boundary
                rem=np.clip(np.expm1(np.minimum(o["rem"].numpy(), 20)), 0, None))        # point remainder: segments, span starts, characters


def window(j):
    """First segment of the window in which the training scored segment j."""
    return max(0, SCORED * (j // SCORED) - WARM)


@torch.no_grad()
def run(model, xin, mode, prev, solver, bench):
    n = len(xin)
    assert n <= CFG["context"]
    b = dict(xin=torch.from_numpy(np.ascontiguousarray(xin, np.float32))[None], mode=torch.from_numpy(np.asarray(mode, np.int64))[None], prev_mode=torch.from_numpy(np.asarray(prev, np.int64))[None],
             solver=torch.full((1, n), solver, dtype=torch.long), bench=torch.full((1, n), bench, dtype=torch.long), pad=torch.zeros((1, n), dtype=torch.bool))
    return {k: v[0].float() for k, v in model(b).items()}


def forward_last(model, xin, mode, prev, solver, bench):
    """Forecast after the last completed segment of the prefix (rows 0..j of build_inputs)."""
    j = len(xin) - 1
    lo = window(j)
    o = run(model, xin[lo:], mode[lo:], prev[lo:], solver, bench)
    return {k: v[-1] for k, v in named(o).items()}


def forward_all(model, xin, mode, prev, solver, bench):
    """Forecasts at every segment of a trace with the windows of the training."""
    n, parts = len(xin), []
    for s in range(0, n, SCORED):
        lo, hi = max(0, s - WARM), min(n, s + SCORED)
        o = named(run(model, xin[lo:hi], mode[lo:hi], prev[lo:hi], solver, bench))
        parts.append({k: v[s - lo:] for k, v in o.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def check(a):
    torch.set_num_threads(a.threads)
    corpus = Path(a.corpus)
    meta = json.loads((corpus / "metadata.json").read_text())
    assert meta["operators"] == OPERATORS and meta["events"] == EVENTS and meta["solvers"] == SOLVERS and meta["benchmarks"] == BENCHMARKS
    if (corpus / "config.json").exists():
        c = json.loads((corpus / "config.json").read_text())
        assert all(c[k] == v for k, v in CFG.items())
    with np.load(corpus / "sequences.npz") as z:
        d = {k: z[k] for k in ("x", "mode", "prev_mode", "solver", "bench", "split", "offsets", "trace", "gate")}
    with np.load(a.inputs) as z:
        xin_stored = z["xin"]
    rep = dict(task_id="PROCESS_FORECAST_GATE_20261013", what="reference bridge and forward of the accepted P_ES weights; no fit",
               sha256={"pes_reference.py": sha(Path(__file__).resolve()), "ablation.py": ABLATION_SHA, "sequences.npz": sha(corpus / "sequences.npz"), "metadata.json": sha(corpus / "metadata.json"),
                       "inputs": sha(a.inputs)}, torch=torch.__version__, threads=a.threads)
    # (1) the bridge against the stored inputs of every trace
    off = d["offsets"]
    chars_all = np.rint(np.expm1(d["x"][:, 2].astype(np.float64) * 10))
    worst, unequal, rebuilt = 0.0, 0, np.zeros_like(xin_stored)
    t0 = time.time()
    for i in range(len(off) - 1):
        s, e = int(off[i]), int(off[i + 1])
        x, m, p = build_inputs(chars_all[s:e], d["x"][s:e, 5:11], d["mode"][s:e])
        rebuilt[s:e] = x
        assert np.array_equal(m, d["mode"][s:e]) and np.array_equal(p, d["prev_mode"][s:e]), ("span types", i)
    cut_rng, cuts = np.random.default_rng(20261013), 0                  # a prefix alone gives the same rows as the whole trace
    for i in cut_rng.choice(len(off) - 1, size=600, replace=False):
        s, e = int(off[i]), int(off[i + 1])
        k = int(cut_rng.integers(1, e - s + 1))
        x, m, p = build_inputs(chars_all[s:s + k], d["x"][s:s + k, 5:11], d["mode"][s:s + k])
        assert np.array_equal(x, rebuilt[s:s + k]) and np.array_equal(p, d["prev_mode"][s:s + k]), ("prefix", i, k)
        cuts += 1
    diff = np.abs(rebuilt - xin_stored)
    worst, unequal = float(diff.max()), int((rebuilt != xin_stored).sum())
    rep["bridge"] = dict(traces=int(len(off) - 1), positions=int(len(xin_stored)), max_abs_difference=worst, unequal_numbers=unequal, numbers=int(xin_stored.size),
                         unequal_by_column={int(k): int(v) for k, v in enumerate((rebuilt != xin_stored).sum(0)) if v}, mode_and_prev_mode_equal=True, prefixes_rebuilt_alone_and_equal_to_the_rows_of_the_whole_trace=cuts, seconds=time.time() - t0)
    print(json.dumps(rep["bridge"]), flush=True)
    assert worst < 1e-6, worst
    # (2) the forward against the stored predictions of the validation part
    n_seg = np.diff(off)
    val = np.flatnonzero(d["split"][off[:-1]] == 1)
    order = val[np.argsort(n_seg[val], kind="stable")]
    pick = sorted(set(order[np.linspace(0, len(order) - 1, a.traces).astype(int)].tolist()) | set(order[-3:].tolist() if a.long else []))
    rng = np.random.default_rng(20261013)
    rep["forward"] = {}
    keys = dict(gate="gate", dest="dest", events="events", next_span_marginal="next_span_marginal")
    for seed in (17, 29, 43):
        model, step = load_model(a.src, Path(a.weights) / f"ES_seed{seed}" / "best.pt", seed)
        with np.load(Path(a.stored) / f"ES_seed{seed}" / "predictions_validation.npz") as z:
            st = {k: z[k] for k in z.files}
        where = {int(v): k for k, v in enumerate(st["index"])}
        mx = {k: 0.0 for k in list(keys) + ["rem_relative", "length_log"]}
        positions, t0, single = 0, time.time(), []
        for i in pick:
            s, e = int(off[i]), int(off[i + 1])
            if e - s > a.max_segments:
                e = s + a.max_segments
            o = forward_all(model, rebuilt[s:e], d["mode"][s:e], d["prev_mode"][s:e], int(d["solver"][s]), int(d["bench"][s]))
            lab = np.flatnonzero(d["gate"][s:e] >= 0)
            rows = np.array([where[s + int(j)] for j in lab])
            positions += len(lab)
            for k, name in keys.items():
                mx[k] = max(mx[k], float(np.abs(o[k][lab] - st[name][rows].astype(np.float32)).max()))
            mx["rem_relative"] = max(mx["rem_relative"], float((np.abs(o["rem"][lab] - st["rem"][rows]) / np.maximum(1.0, st["rem"][rows])).max()))
            mx["length_log"] = max(mx["length_log"], float(np.abs(np.log1p(o["next_segment_chars"][lab]) - np.minimum(st["length"][rows], 15)).max()))
            for j in rng.choice(lab, size=min(a.prefixes, len(lab)), replace=False):   # prefix by prefix: only segments 0..j are given
                t1 = time.time()
                f = forward_last(model, rebuilt[s:s + int(j) + 1], d["mode"][s:s + int(j) + 1], d["prev_mode"][s:s + int(j) + 1], int(d["solver"][s]), int(d["bench"][s]))
                single.append((int(j) + 1 - window(int(j)), time.time() - t1, float(np.abs(f["gate"] - o["gate"][j]).max()), float((np.abs(f["rem"] - o["rem"][j]) / np.maximum(1.0, o["rem"][j])).max())))
        single = np.array(single)
        short, full = single[single[:, 0] <= 64], single[single[:, 0] > 192]
        rep["forward"][f"ES_seed{seed}"] = dict(selected_step=step, traces=len(pick), labelled_positions_compared=positions, max_abs_difference_from_the_stored_h200_predictions=mx,
                                                seconds_all_windows=time.time() - t0, prefixes_checked_one_by_one=int(len(single)),
                                                prefix_vs_window_max_gate_difference=float(single[:, 2].max()), prefix_vs_window_max_relative_remainder_difference=float(single[:, 3].max()),
                                                seconds_per_prefix_window_up_to_64_segments=dict(n=int(len(short)), median=float(np.median(short[:, 1])) if len(short) else None, max=float(short[:, 1].max()) if len(short) else None),
                                                seconds_per_prefix_window_over_192_segments=dict(n=int(len(full)), median=float(np.median(full[:, 1])) if len(full) else None, max=float(full[:, 1].max()) if len(full) else None))
        print(seed, json.dumps(rep["forward"][f"ES_seed{seed}"]), flush=True)
        assert max(mx[k] for k in keys) < 2e-3 and mx["rem_relative"] < 2e-3 and single[:, 2].max() < 1e-4, mx
    rep["passed"] = True
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps(rep, indent=1) + "\n")
    tmp.replace(a.out)
    print("PES REFERENCE CHECK PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("check",))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", type=int, default=40)
    ap.add_argument("--prefixes", type=int, default=6)
    ap.add_argument("--max-segments", type=int, default=768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--long", action="store_true")
    check(ap.parse_args())
