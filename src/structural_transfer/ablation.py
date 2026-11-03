"""EVENT_SPAN_TRANSFORMER_ABLATION_20261013: the article predictor 12 / 512 / 8 with structural inputs only, cells E, S, E+S.

One architecture and one set of output heads for the three cells; a cell differs only in the input groups that are
multiplied by zero (and, for E, in the span-type embeddings replaced by the 'unknown' index) in training and in
evaluation alike.  No text and no question embedding enter any cell; the contrastive next-text target of v2 is removed
for all cells.  Backbone, heads, losses and schedule of span_transformer_2026_10_07/v2 are imported or repeated:
12 layers, width 512, 8 heads, FF 2048, dropout 0.1, context 256, batch 64, 8 epochs, AdamW 3e-4, weight decay 0.01,
warm-up 100 steps and cosine decay, gradient norm 1; checkpoint chosen by validation joint NLL + 0.2 wait NLL.
New, identical for the cells: a head with three numbers, the remainder in segments, span starts and characters after
the current segment, L1 on log1p, only on traces whose natural end is observed, weight 0.2.
A cell and a seed start from the same initial weights, the same batch order and the same dropout stream."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CELLS = dict(E=dict(S=False, E=True), S=dict(S=True, E=False), ES=dict(S=True, E=True))
SEEDS, REM_WEIGHT, CKPT_EVERY = (17, 29, 43), 0.2, 200
GROUPS = dict(common=(0, 3), S=(3, 5), E=(5, 41))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Predictor(nn.Module):
    def __init__(self, cell, meta, cfg):
        super().__init__()
        self.cell, self.K = cell, len(meta["operators"])
        w = cfg["width"]
        self.mode = nn.Embedding(self.K + 1, 24)
        self.prev = nn.Embedding(self.K + 1, 16)
        self.solver = nn.Embedding(len(meta["solvers"]), 16)
        self.bench = nn.Embedding(len(meta["benchmarks"]), 8)
        self.proj = nn.Linear(41 + 64, w)
        self.position = nn.Embedding(cfg["context"], w)
        layer = nn.TransformerEncoderLayer(w, cfg["heads"], w * 4, .1, batch_first=True, norm_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        for module in self.core.modules():                              # as in v2: cloned layers must not start identical
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        self.norm = nn.LayerNorm(w)
        self.gate, self.dest, self.events, self.length = nn.Linear(w, 3), nn.Linear(w, self.K), nn.Linear(w, 6), nn.Linear(w, 1)
        self.wait = nn.Linear(w, 3 * (3 + self.K + 1))
        self.rem = nn.Linear(w, 3)
        with torch.no_grad():
            bias = self.wait.bias.view(3, 3 + self.K + 1)
            bias[:, 1] = 7.
            bias[:, 2] = 0.
        keep = torch.zeros(41)
        keep[slice(*GROUPS["common"])] = 1
        if CELLS[cell]["S"]:
            keep[slice(*GROUPS["S"])] = 1
        if CELLS[cell]["E"]:
            keep[slice(*GROUPS["E"])] = 1
        self.register_buffer("keep", keep)

    def forward(self, b):
        x = b["xin"] * self.keep
        mode, prev = b["mode"], b["prev_mode"]
        if not CELLS[self.cell]["S"]:                                   # the span types are S: the 'unknown' index instead
            mode, prev = torch.full_like(mode, self.K), torch.full_like(prev, self.K)
        h = self.proj(torch.cat((x, self.mode(mode), self.prev(prev), self.solver(b["solver"]), self.bench(b["bench"])), dim=-1))
        n = h.shape[1]
        h = h + self.position(torch.arange(n, device=h.device))[None]
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=h.device), 1)
        h = self.norm(self.core(h, mask=mask, src_key_padding_mask=b["pad"]))
        p = self.wait(h).view(*h.shape[:-1], 3, 3 + self.K + 1)
        return dict(gate=self.gate(h), dest=self.dest(h), events=self.events(h), length=self.length(h).squeeze(-1), wait_pi=p[..., 0], wait_logscale=p[..., 1].clamp(-3, 16),
                    wait_shape=(F.softplus(p[..., 2]) + .2).clamp(max=5), wait_mark=p[..., 3:], rem=self.rem(h))


def losses(M, B, o, b):
    base = B.losses(o, b)                                               # v2: joint + .5 dest + .25 event + .05 length
    valid = b["score"] & (b["wait_y"] > 0)
    w = b["weight"] * valid
    loss = base + .2 * (M.wait_nll(o, b["wait_y"], b["wait_observed"], b["next_span"]) * w).sum() / w.sum().clamp_min(1e-8)
    wr = b["weight"] * (b["score"] & b["rem_mask"])
    rem = ((o["rem"] - torch.log1p(b["rem"])).abs().sum(-1) * wr).sum() / wr.sum().clamp_min(1e-8)
    return loss + REM_WEIGHT * rem


def make_batch(d, ids, device, T):
    chunks = d["chunks"][ids]
    size = len(chunks)
    index, pad, score = np.zeros((size, T), np.int64), np.ones((size, T), bool), np.zeros((size, T), bool)
    for i, (s, n, w, ti, sp) in enumerate(chunks):
        index[i, :n] = np.arange(s, s + n)
        pad[i, :n] = False
        score[i, w:n] = True
    fields = ["xin", "mode", "prev_mode", "gate", "dest", "event_y", "event_mask", "length_y", "solver", "bench", "weight", "wait_y", "wait_observed", "next_span", "rem", "rem_mask"]
    b = {k: torch.from_numpy(d[k][index]).to(device) for k in fields}
    for k in ("mode", "prev_mode", "gate", "dest", "solver", "bench", "next_span"):
        b[k] = b[k].long()
    b["pad"], b["score"], b["index"] = torch.from_numpy(pad).to(device), torch.from_numpy(score).to(device), index
    return b


@torch.no_grad()
def evaluate(M, B, model, d, split, device, cfg, save_to=None):
    chunks = np.flatnonzero(d["chunks"][:, 4] == split)
    keys = ("gate", "dest", "events", "length", "wait_pi", "wait_logscale", "wait_shape", "wait_mark", "rem")
    pieces, outs = [], {k: [] for k in keys}
    model.eval()
    for start in range(0, len(chunks), 128):
        b = make_batch(d, chunks[start:start + 128], device, cfg["context"])
        o = model(b)
        m = (b["score"] & (b["gate"] >= 0)).cpu().numpy()
        pieces.append(b["index"][m])
        for k in keys:
            v = o[k]
            v = F.softmax(v, -1) if k in ("gate", "dest") else torch.sigmoid(v) if k == "events" else v
            outs[k].append(v.float().cpu().numpy()[m])
    ids = np.concatenate(pieces)
    pred = {k: np.concatenate(v) for k, v in outs.items()}
    order = np.argsort(ids)
    ids, pred = ids[order], {k: v[order] for k, v in pred.items()}
    assert np.array_equal(ids, np.flatnonzero((d["split"] == split) & (d["gate"] >= 0)))
    result = B.summarize_predictions(d, ids, pred)
    cpu = {k: torch.from_numpy(v) for k, v in pred.items() if k.startswith("wait_")}
    nll = M.wait_nll(cpu, torch.from_numpy(d["wait_y"][ids]), torch.from_numpy(d["wait_observed"][ids]), torch.from_numpy(d["next_span"][ids])).numpy()
    mp = (torch.softmax(cpu["wait_pi"], -1)[..., None] * torch.softmax(cpu["wait_mark"], -1)).sum(-2).numpy()
    valid = d["wait_y"][ids] > 0
    observed = valid & d["wait_observed"][ids]
    question, y, ns = d["question"][ids], d["gate"][ids], d["next_span"][ids]
    rem_true, rem_ok = d["rem"][ids], d["rem_mask"][ids]
    rem_pred = np.clip(np.expm1(np.minimum(pred["rem"], 20)), 0, None)
    pg = np.clip(pred["gate"], 1e-8, 1)
    for row in result["question_metrics"]:
        q = question == row["question"]
        v, ob, en, rm = q & valid, q & observed, q & (y == 2), q & rem_ok
        if v.any():
            row["wait_joint_nll"] = float(nll[v].mean())
        if ob.any():
            yy, probs = ns[ob], mp[ob]
            row["next_span_nll"] = float(-np.log(np.clip(probs[np.arange(len(yy)), yy], 1e-9, 1)).mean())
            row["next_span_accuracy"] = float((probs.argmax(-1) == yy).mean())
        if en.any():
            row["end_nll_at_natural_end"] = float(-np.log(pg[en, 2]).mean())
        if rm.any():
            e = np.abs(rem_pred[rm] - rem_true[rm])
            row["rem_mae_segments"], row["rem_mae_spans"], row["rem_mae_chars"] = (float(e[:, k].mean()) for k in range(3))
    for key in ("wait_joint_nll", "next_span_nll", "next_span_accuracy", "end_nll_at_natural_end", "rem_mae_segments", "rem_mae_spans", "rem_mae_chars"):
        vals = [r[key] for r in result["question_metrics"] if key in r]
        result["scores"][key] = float(np.mean(vals)) if vals else None
        result["scores"][key + "_questions"] = len(vals)
    result["scores"]["selection_objective"] = result["scores"]["joint_nll"] + .2 * result["scores"]["wait_joint_nll"]
    if save_to is not None:
        tmp = Path(str(save_to)[:-4] + ".tmp.npz")
        np.savez_compressed(tmp, index=ids.astype(np.int64), gate=pred["gate"].astype(np.float32), dest=pred["dest"].astype(np.float32), events=pred["events"].astype(np.float16),
                            length=pred["length"].astype(np.float32), next_span_marginal=mp.astype(np.float32), wait_nll=nll.astype(np.float32), rem=rem_pred.astype(np.float32))
        os.replace(tmp, save_to)
        result["predictions_sha256"] = sha(save_to)
    return result


def load(v2_dir, inputs):
    from . import base as B, model as M
    B.ROOT = Path(v2_dir)
    M.ROOT = Path(v2_dir)
    M.CONFIG = json.loads((Path(v2_dir) / "config.json").read_text())
    d, meta = B.load_data()                                             # checks the frozen sha256 of the corpus
    with np.load(inputs) as z:
        d.update(xin=z["xin"], rem=z["rem"], rem_mask=z["rem_mask"])
        for k in ("gate", "next_span", "wait_observed", "wait_y"):      # corrected repeat: targets of the recovered END statuses replace the frozen ones
            if k in z.files:
                assert z[k].shape == d[k].shape
                d[k] = z[k].astype(d[k].dtype)
    return B, M, d, meta, dict(M.CONFIG)


def lr_at(cfg, step, total):
    return cfg["lr"] * min(1., (step + 1) / cfg["warmup_steps"]) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total)))


def fit(B, M, d, meta, cfg, out, cell, seed, device, deadline, eval_splits, max_steps=None, tag=None, stop_at=None):
    folder = out / "fits" / (tag or f"{cell}_seed{seed}")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "DONE.json").exists():
        return json.loads((folder / "DONE.json").read_text())
    train_ids = np.flatnonzero(d["chunks"][:, 4] == 0)
    per_epoch = math.ceil(len(train_ids) / cfg["batch"])
    total = per_epoch * cfg["epochs"] if max_steps is None else max_steps
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = Predictor(cell, meta, cfg).to(device)
    start_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for k, v in sorted(model.state_dict().items()) if k != "keep")).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.01, fused=True)
    step, best, order, pos, t_used = 0, float("inf"), rng.permutation(train_ids), 0, 0.0
    ck = folder / "resume.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)     # generator states must stay on the CPU
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        for group_state in opt.state.values():
            for k, v in group_state.items():
                if torch.is_tensor(v):
                    group_state[k] = v.to(device)
        step, best, order, pos, t_used = st["step"], st["best"], st["order"], st["position"], st["seconds"]
        rng.bit_generator.state = st["numpy_rng"]
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        with open(out / "ATTEMPTS.jsonl", "a") as f:
            f.write(json.dumps(dict(event="resume", fit=folder.name, step=step, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))) + "\n")
    t0 = time.time()

    def checkpoint():
        atomic_torch(ck, dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step, best=best, order=order, position=pos, numpy_rng=rng.bit_generator.state,
                              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), seconds=t_used + time.time() - t0))
    while step < total:
        if time.time() > deadline or step == stop_at:
            checkpoint()
            raise TimeoutError("the batch deadline is reached; the state is saved")
        if pos >= len(order):
            order, pos = rng.permutation(train_ids), 0
        ids = order[pos:pos + cfg["batch"]]
        pos += len(ids)
        b = make_batch(d, ids, device, cfg["context"])
        model.train()
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = lr_at(cfg, step, total)
        loss = losses(M, B, model(b), b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite training loss")
            atomic_json(out / "HEARTBEAT.json", dict(fit=folder.name, step=step, total=total, loss=value, seconds=t_used + time.time() - t0, time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        if step % CKPT_EVERY == 0 and step % per_epoch != 0 and step != total:
            checkpoint()
        if step % per_epoch == 0 or step == total:
            val = evaluate(M, B, model, d, 1, device, cfg)
            score = val["scores"]["selection_objective"]
            with (folder / "curve.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=step, epoch=step / per_epoch, training_loss=float(loss.detach()), validation=val["scores"], seconds=t_used + time.time() - t0)) + "\n")
            print(folder.name, "step", step, "validation objective %.5f joint %.5f wait %.5f rem segs %.3f" % (score, val["scores"]["joint_nll"], val["scores"]["wait_joint_nll"], val["scores"]["rem_mae_segments"]),
                  "seconds %.0f" % (t_used + time.time() - t0), flush=True)
            if score < best:
                best = score
                atomic_torch(folder / "best.pt", dict(model=model.state_dict(), step=step))
            checkpoint()
    state = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    results = {}
    for split in eval_splits:
        name = {1: "validation", 2: "test"}[split]
        results[name] = evaluate(M, B, model, d, split, device, cfg, folder / f"predictions_{name}.npz")
        atomic_json(folder / f"metrics_{name}.json", results[name])
    done = dict(fit=folder.name, cell=cell, seed=seed, steps=step, selected_step=state["step"], best_validation_objective=best, parameters=sum(p.numel() for p in model.parameters()),
                start_state_sha256=start_sha, best_checkpoint_sha256=sha(folder / "best.pt"), seconds=t_used + time.time() - t0, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                scores={k: v["scores"] for k, v in results.items()})
    atomic_json(folder / "DONE.json", done)
    ck.unlink()                                                         # the resume state is large and is not needed after the fit
    return done


def set_status(out, key, status):
    p = out / "MANIFEST.json"
    m = json.loads(p.read_text()) if p.exists() else dict(task_id="EVENT_SPAN_TRANSFORMER_ABLATION_20261013", elements={f"{c}_seed{s}": "pending" for s in SEEDS for c in CELLS})
    m["elements"][key] = status
    atomic_json(p, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True, help="directory with the frozen base.py, model.py, config.json and data/ of the article model v2")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("precheck", "bench", "train"), required=True)
    ap.add_argument("--eval-splits", default="1")
    ap.add_argument("--seconds", type=float, default=14400)
    a = ap.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA device is required; no CPU fallback"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)
    B, M, d, meta, cfg = load(Path(a.v2), a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "precheck":
        return precheck(B, M, d, meta, cfg, out)
    if a.mode == "bench":
        return bench(B, M, d, meta, cfg, out)
    clock = out / "CLOCK.json"
    if not clock.exists():
        atomic_json(clock, dict(first_start=time.time(), deadline=time.time() + a.seconds))
    deadline = json.loads(clock.read_text())["deadline"]                # a restart does not extend the limit
    with open(out / "ATTEMPTS.jsonl", "a") as f:
        f.write(json.dumps(dict(event="start", time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=sha(Path(__file__).resolve()), inputs=sha(a.inputs), device=torch.cuda.get_device_name(0))) + "\n")
    splits = [int(x) for x in a.eval_splits.split(",")]
    for seed in SEEDS:
        for cell in CELLS:
            key = f"{cell}_seed{seed}"
            set_status(out, key, "running")
            try:
                fit(B, M, d, meta, cfg, out, cell, seed, "cuda", deadline, splits)
            except BaseException as e:
                set_status(out, key, f"stopped: {type(e).__name__}: {e}")
                raise
            set_status(out, key, "done")
    atomic_json(out / "COMPLETE.json", dict(fits=9, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=sha(Path(__file__).resolve()), inputs=sha(a.inputs), config=cfg, eval_splits=splits))
    print("ABLATION COMPLETE", flush=True)


def bench(B, M, d, meta, cfg, out):
    train_ids = np.flatnonzero(d["chunks"][:, 4] == 0)
    per_epoch = math.ceil(len(train_ids) / cfg["batch"])
    torch.manual_seed(17)
    model = Predictor("ES", meta, cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.01, fused=True)
    rng = np.random.default_rng(17)
    order = rng.permutation(train_ids)
    times = []
    for k in range(60):
        t0 = time.time()
        b = make_batch(d, order[k * cfg["batch"]:(k + 1) * cfg["batch"]], "cuda", cfg["context"])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = losses(M, B, model(b), b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    step = float(np.median(times[10:]))
    val_chunks = np.flatnonzero(d["chunks"][:, 4] == 1)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, 128 * 8, 128):
            model(make_batch(d, val_chunks[start:start + 128], "cuda", cfg["context"]))
    torch.cuda.synchronize()
    val = (time.time() - t0) / 8 * math.ceil(len(val_chunks) / 128)
    rep = dict(parameters=sum(p.numel() for p in model.parameters()), train_chunks=int(len(train_ids)), steps_per_epoch=per_epoch, total_steps=per_epoch * cfg["epochs"], seconds_per_step=step,
               loss_after_60_steps=float(loss.detach()), validation_forward_seconds_estimate=val, validation_chunks=int(len(val_chunks)),
               fit_seconds_estimate=per_epoch * cfg["epochs"] * step + (cfg["epochs"] + 1) * val * 1.5, cuda_peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
               state_megabytes=sum(v.numel() * v.element_size() for v in model.state_dict().values()) / 2 ** 20, device=torch.cuda.get_device_name(0), torch=torch.__version__)
    rep["nine_fits_seconds_estimate"] = 9 * rep["fit_seconds_estimate"]
    atomic_json(out / "BENCH.json", rep)
    print(json.dumps(rep, indent=1))


def precheck(B, M, d, meta, cfg, out):
    rep = {}
    ids = np.flatnonzero(d["chunks"][:, 4] == 0)[:4]
    b = make_batch(d, ids, "cuda", cfg["context"])
    models = {}
    for cell in CELLS:
        torch.manual_seed(5)
        torch.cuda.manual_seed_all(5)
        models[cell] = Predictor(cell, meta, cfg).cuda().eval()
    sd = {c: {k: v for k, v in m.state_dict().items() if k != "keep"} for c, m in models.items()}
    assert all(torch.equal(sd["E"][k], sd[c][k]) for c in ("S", "ES") for k in sd["E"])
    rep["same_initial_weights_in_the_three_cells"] = True
    with torch.no_grad():
        base = {c: m(b) for c, m in models.items()}

        def change(fn):
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            fn(b2)
            return {c: max(float((m(b2)[k] - base[c][k]).abs().max()) for k in ("gate", "dest", "rem", "wait_pi")) for c, m in models.items()}

        def ev(b2):
            b2["xin"][..., 5:41] = torch.rand_like(b2["xin"][..., 5:41])

        def sp(b2):
            b2["xin"][..., 3:5] += 1.0
            b2["mode"] = (b2["mode"] + 1) % 7
            b2["prev_mode"] = (b2["prev_mode"] + 1) % 7

        def tg(b2):
            for k in ("gate", "dest", "next_span"):
                b2[k] = (b2[k] + 1).clamp(max=2)
            b2["rem"] += 5
            b2["rem_mask"] = ~b2["rem_mask"]
            b2["event_y"] = 1 - b2["event_y"]
            b2["wait_y"] += 3

        def fut(b2):
            b2["xin"][:, 100:] += 1.0
            b2["mode"][:, 100:] = (b2["mode"][:, 100:] + 1) % 7
        e, s, t = change(ev), change(sp), change(tg)
        rep["change_of_event_inputs"], rep["change_of_span_inputs"], rep["change_of_targets_and_masks"] = e, s, t
        assert e["S"] == 0 and e["E"] > 1e-4 and e["ES"] > 1e-4, e          # S ignores every event input
        assert s["E"] == 0 and s["S"] > 1e-4 and s["ES"] > 1e-4, s          # E ignores every span input
        assert max(t.values()) == 0, t                                      # no target and no mask is an input
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        fut(b2)
        rep["future_change_max_effect_on_positions_before_100"] = {c: max(float((m(b2)[k][:, :100] - base[c][k][:, :100]).abs().max()) for k in ("gate", "dest", "rem")) for c, m in models.items()}
        assert max(rep["future_change_max_effect_on_positions_before_100"].values()) < 1e-4
    # the event channels of a prefix do not depend on later segments and never reset at a span boundary
    from . import prepare_ablation as P
    x = d["x"]
    a, z = int(d["offsets"][3]), int(d["offsets"][4])
    xin0, _, _ = P.build(x[a:z], d["span_start"][a:z], np.array([0, z - a]), np.array([True]))
    x2 = x[a:z].copy()
    cut = (z - a) // 2
    x2[cut:, 5:11] = 1 - x2[cut:, 5:11]
    xin1, _, _ = P.build(x2, d["span_start"][a:z], np.array([0, z - a]), np.array([True]))
    assert np.array_equal(xin0[:cut], xin1[:cut])
    st2 = d["span_start"][a:z].copy()
    st2[1:] = ~st2[1:]
    xin2, rem2, _ = P.build(x[a:z], st2, np.array([0, z - a]), np.array([True]))
    assert np.array_equal(xin0[:, 5:41], xin2[:, 5:41]) and np.array_equal(xin0, d["xin"][a:z])
    n = z - a
    _, rem0, m0 = P.build(x[a:z], d["span_start"][a:z], np.array([0, n]), np.array([True]))
    assert rem0[-1].tolist() == [0, 0, 0] and rem0[0, 0] == n - 1 and rem0[0, 1] == d["span_start"][a + 1:z].sum() and m0.all()
    _, _, m1 = P.build(x[a:z], d["span_start"][a:z], np.array([0, n]), np.array([False]))
    assert not m1.any()
    rep["inputs"] = "event channels of a prefix ignore later segments and span boundaries; remainder targets are 0 at the last segment and absent on traces without an observed natural end"
    # a short real pipeline on the fit part only: two tiny fits with a restart; nothing of it is a result
    small = dict(cfg, epochs=1)
    a_ = fit(B, M, d, meta, small, out / "smoke_straight", "ES", 17, "cuda", time.time() + 900, [], max_steps=6, tag="x")
    try:
        fit(B, M, d, meta, small, out / "smoke_resumed", "ES", 17, "cuda", time.time() + 900, [], max_steps=6, tag="x", stop_at=3)
    except TimeoutError:
        pass
    assert (out / "smoke_resumed" / "fits" / "x" / "resume.pt").exists() and not (out / "smoke_resumed" / "fits" / "x" / "DONE.json").exists()
    b_ = fit(B, M, d, meta, small, out / "smoke_resumed", "ES", 17, "cuda", time.time() + 900, [], max_steps=6, tag="x")
    w1 = torch.load(out / "smoke_straight" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    w2 = torch.load(out / "smoke_resumed" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    rep["restart"] = dict(stopped_at_step=3, weights_max_abs_difference_after_6_steps=max(float((w1[k].float() - w2[k].float()).abs().max()) for k in w1),
                          objective_straight=a_["best_validation_objective"], objective_resumed=b_["best_validation_objective"], same_start=a_["start_state_sha256"] == b_["start_state_sha256"])
    assert a_["start_state_sha256"] == b_["start_state_sha256"] and abs(a_["best_validation_objective"] - b_["best_validation_objective"]) < 1e-2
    for sub in ("smoke_straight", "smoke_resumed"):
        (out / sub / "fits" / "x" / "best.pt").unlink()
    rep["cuda_peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    atomic_json(out / "PRECHECK.json", rep)
    print(json.dumps(rep, indent=1))
    print("PRECHECK PASSED")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
