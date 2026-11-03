"""PROCESS_FORECAST_GATE_20261013: the matched pair P_text / P_ES_matched (input sets T and TES) on the corrected END targets.

One class, one code, one schedule for both input sets.  The training code is the accepted code of the corrected ablation
repeat, ablation.py next to this file: it is imported unchanged after its sha256 is checked, and its fit(), evaluate() and
losses() run as they are.  This file adds only what the pair needs:
  * the first layer reads 873 numbers: the 105 of the accepted predictor plus the frozen MiniLM vector of the current
    completed segment (384) and of the question (384), taken from the frozen table embeddings.npy by text_id / prompt_id;
  * two input sets through the accepted mask: T opens the common group, the solver, the benchmark and the text; the span
    and event columns 3..40 are multiplied by zero and both span-type embeddings take the 'unknown' index 7.  TES opens
    the span and event columns and the span types as well.  Nothing else differs;
  * a seed gives both input sets the same initial weights, the same batch order and the same dropout stream;
  * no contrastive next-text loss: the accepted ES recipe has none.
Targets and masks: ablation_inputs_endfix_v2.npz only (recovered END statuses), read by the accepted load().
Modes: cpucheck (no card: structure, masks, causality, gradients on the real data), probe (one card: the same checks, a
short real pipeline with a restart, a timing of the full recipe), train (six fits under one clock).
  python matched.py --v2 V2 --inputs INPUTS --embeddings embeddings.npy --out OUT --mode cpucheck|probe|train [--seconds N]"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ABLATION_SHA = "a92f8b859376121fb4b5685e5f6c8ff6351dafd182ad24788e528506301e0790"
INPUTS_SHA = "1038f35c99135ce545424db5fa50fff72a0f1282193cdd063e02eb69975a3da9"
EMBEDDINGS_SHA = "f749007bbfa9ef4c8730f8c2bb0c297e6b33f30c59b72458f3649a535d800cf3"
V2_FILES = {"base.py": "a1e30f6f255c518bb8cd729e4ffa8ae5470a7524301ea48d21329436b1eafde4", "model.py": "86140369d00f3162de0a9c297f0224b706f5637fbd17c89ef103d61f8d5d491a",
            "config.json": "1dda53c5e9abcf74f14efd3252399c09e415c6444bf54b7e111b22b52c5d3bc9"}
TASK = "PROCESS_FORECAST_GATE_20261013 matched pair P_text / P_ES_matched"
CELLS = dict(T=dict(S=False, E=False), TES=dict(S=True, E=True))
SEEDS, TEXT, PARAMETERS, MEMORY_LIMIT_GIB = (17, 29, 43), 384, 38436013, 20
# columns of the first layer
COLS = dict(common=(0, 3), S=(3, 5), E=(5, 41), mode=(41, 65), prev=(65, 81), solver=(81, 97), bench=(97, 105), text=(105, 489), prompt=(489, 873))
SHARED = ("common", "solver", "bench", "text", "prompt")
STRUCTURE = ("S", "E", "mode", "prev")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def accepted(name, digest):
    """An accepted file next to this one is executed only after its sha256 is checked."""
    p = HERE / f"{name}.py"
    assert sha(p) == digest, f"{name}.py is not the accepted file"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = accepted("ablation", ABLATION_SHA)
A_make_batch, A_load = A.make_batch, A.load


class Predictor(A.Predictor):
    """The accepted predictor with a wider first layer: the text of the segment and of the question is added."""

    def __init__(self, cell, meta, cfg):
        super().__init__(cell, meta, cfg)                               # the accepted mask 'keep' is built from CELLS[cell]
        self.proj = nn.Linear(41 + 64 + 2 * TEXT, cfg["width"])

    def forward(self, b):
        x = b["xin"] * self.keep
        mode, prev = b["mode"], b["prev_mode"]
        if not CELLS[self.cell]["S"]:                                   # the span types are structure: the 'unknown' index instead
            mode, prev = torch.full_like(mode, self.K), torch.full_like(prev, self.K)
        h = self.proj(torch.cat((x, self.mode(mode), self.prev(prev), self.solver(b["solver"]), self.bench(b["bench"]), b["text"], b["prompt"]), dim=-1))
        n = h.shape[1]
        h = h + self.position(torch.arange(n, device=h.device))[None]
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=h.device), 1)
        h = self.norm(self.core(h, mask=mask, src_key_padding_mask=b["pad"]))
        p = self.wait(h).view(*h.shape[:-1], 3, 3 + self.K + 1)
        return dict(gate=self.gate(h), dest=self.dest(h), events=self.events(h), length=self.length(h).squeeze(-1), wait_pi=p[..., 0], wait_logscale=p[..., 1].clamp(-3, 16),
                    wait_shape=(F.softplus(p[..., 2]) + .2).clamp(max=5), wait_mark=p[..., 3:], rem=self.rem(h))


def make_batch(d, ids, device, T):
    """The accepted batch plus the text vectors of the same positions: the segment itself and the question, nothing later."""
    b = A_make_batch(d, ids, device, T)
    emb = d["emb"]
    for src, dst in (("text_id", "text"), ("prompt_id", "prompt")):
        idx = torch.from_numpy(d[src][b["index"]].astype(np.int64)).to(emb.device)
        b[dst] = emb[idx].float().to(device)
    return b


A.CELLS, A.Predictor, A.make_batch = CELLS, Predictor, make_batch       # fit() and evaluate() of the accepted code build this class and these batches


def load(v2, inputs, embeddings, device):
    v2 = Path(v2)
    # Numerical configuration and corpus are validated by the packaged loader.
    assert sha(inputs) == INPUTS_SHA, "the targets are not ablation_inputs_endfix_v2.npz"
    assert sha(embeddings) == EMBEDDINGS_SHA, "the text vectors are not the frozen embeddings.npy"
    B, M, d, meta, cfg = A_load(v2, inputs)                             # checks the frozen corpus; applies the END targets of the inputs file
    e = np.load(embeddings, mmap_mode="r")
    assert e.shape == (meta["unique_texts"], TEXT) and e.dtype == np.float16, (e.shape, e.dtype)
    for k in ("text_id", "prompt_id"):
        assert d[k].shape == d["mode"].shape and int(d[k].min()) >= 0 and int(d[k].max()) < len(e), k
    d["emb"] = torch.from_numpy(np.array(e)).to(device)                 # a copy in memory: the table file is never written
    return B, M, d, meta, cfg


def clone(b):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def spread(o1, o2, cut=None):
    return max(float((o1[k][:, :cut] - o2[k][:, :cut]).abs().max()) for k in o1)


def structure(B, M, d, meta, cfg, device):
    """Checks that need no training: one class and one start, identity of the shared columns, closed structure in T,
    reaction to every opened group, no target as an input, no future in the past, gradients."""
    rep = {}
    full = np.flatnonzero((d["chunks"][:, 4] == 0) & (d["chunks"][:, 1] == cfg["context"]))[:4]     # four full training windows of 256 segments
    b = make_batch(d, full, device, cfg["context"])
    assert not bool(b["pad"].any())
    models = {}
    for cell in CELLS:
        torch.manual_seed(5)
        torch.cuda.manual_seed_all(5)
        models[cell] = Predictor(cell, meta, cfg).to(device).eval()
    rep["parameters"] = {c: sum(p.numel() for p in m.parameters()) for c, m in models.items()}
    assert all(v == PARAMETERS for v in rep["parameters"].values()), rep["parameters"]
    sd = {c: {k: v for k, v in m.state_dict().items() if k != "keep"} for c, m in models.items()}
    assert sd["T"].keys() == sd["TES"].keys() and all(torch.equal(sd["T"][k], sd["TES"][k]) for k in sd["T"])
    rep["same_class_and_same_initial_weights"] = True
    rep["keep_mask_open_columns"] = {c: [int(i) for i in torch.nonzero(m.keep).flatten().tolist()] for c, m in models.items()}
    assert rep["keep_mask_open_columns"]["T"] == [0, 1, 2] and rep["keep_mask_open_columns"]["TES"] == list(range(41))
    seen, base = {}, {}
    with torch.no_grad():
        for c, m in models.items():
            hook = m.proj.register_forward_pre_hook(lambda module, args, c=c: seen.__setitem__(c, args[0].detach().clone()))
            base[c] = m(b)
            hook.remove()
        assert seen["T"].shape[-1] == 873 and seen["TES"].shape == seen["T"].shape
        sl = {k: slice(*v) for k, v in COLS.items()}
        rep["first_layer"] = dict(width=873, columns=COLS, shared_groups=SHARED, structure_groups=STRUCTURE)
        rep["first_layer"]["shared_columns_bitwise_equal_in_T_and_TES"] = {k: bool(torch.equal(seen["T"][..., sl[k]], seen["TES"][..., sl[k]])) for k in SHARED}
        assert all(rep["first_layer"]["shared_columns_bitwise_equal_in_T_and_TES"].values())
        m_t, m_s = models["T"], models["TES"]
        tid = torch.from_numpy(d["text_id"][b["index"]].astype(np.int64)).to(d["emb"].device)
        pid = torch.from_numpy(d["prompt_id"][b["index"]].astype(np.int64)).to(d["emb"].device)
        facts = dict(
            text_equals_the_table_row_of_text_id=bool(torch.equal(seen["T"][..., sl["text"]], d["emb"][tid].float().to(device))),
            prompt_equals_the_table_row_of_prompt_id=bool(torch.equal(seen["T"][..., sl["prompt"]], d["emb"][pid].float().to(device))),
            common_equals_the_data=bool(torch.equal(seen["T"][..., sl["common"]], b["xin"][..., 0:3])),
            T_span_and_event_columns_all_zero=bool((seen["T"][..., 3:41] == 0).all()),
            T_span_type_columns_are_the_unknown_index=bool(torch.equal(seen["T"][..., sl["mode"]], m_t.mode.weight[7].expand_as(seen["T"][..., sl["mode"]]))
                                                           and torch.equal(seen["T"][..., sl["prev"]], m_t.prev.weight[7].expand_as(seen["T"][..., sl["prev"]]))),
            TES_span_and_event_columns_equal_the_data=bool(torch.equal(seen["TES"][..., 3:41], b["xin"][..., 3:41])),
            TES_span_type_columns_equal_the_embeddings_of_the_data=bool(torch.equal(seen["TES"][..., sl["mode"]], m_s.mode(b["mode"])) and torch.equal(seen["TES"][..., sl["prev"]], m_s.prev(b["prev_mode"]))))
        rep["first_layer"].update(facts)
        assert all(facts.values()), facts
        # with the structure closed in the data, TES is the same function as T: the only difference of the two paths is E/S
        closed = clone(b)
        closed["xin"][..., 3:41] = 0
        closed["mode"], closed["prev_mode"] = torch.full_like(b["mode"], 7), torch.full_like(b["prev_mode"], 7)
        rep["TES_with_structure_closed_in_the_data_minus_T"] = spread(m_s(closed), base["T"])
        assert rep["TES_with_structure_closed_in_the_data_minus_T"] == 0

        def change(fn):
            b2 = clone(b)
            fn(b2)
            return {c: spread(m(b2), base[c]) for c, m in models.items()}

        def ev(b2):
            b2["xin"][..., 5:41] = torch.rand_like(b2["xin"][..., 5:41])

        def sp(b2):
            b2["xin"][..., 3:5] += 1.0
            b2["mode"], b2["prev_mode"] = (b2["mode"] + 1) % 6, (b2["prev_mode"] + 1) % 6

        def tx(b2):
            b2["text"] = F.normalize(torch.randn_like(b2["text"]), dim=-1)

        def pr(b2):
            b2["prompt"] = F.normalize(torch.randn_like(b2["prompt"]), dim=-1)

        def tg(b2):
            for k in ("gate", "dest", "next_span"):
                b2[k] = (b2[k] + 1).clamp(max=2)
            b2["rem"] += 5
            b2["rem_mask"], b2["wait_observed"], b2["event_mask"], b2["score"] = ~b2["rem_mask"], ~b2["wait_observed"], ~b2["event_mask"], ~b2["score"]
            b2["event_y"] = 1 - b2["event_y"]
            b2["wait_y"] += 3
            b2["length_y"] += 1
            b2["weight"] *= 2

        e, s, t, p, g = change(ev), change(sp), change(tx), change(pr), change(tg)
        rep["max_output_change"] = dict(event_inputs=e, span_inputs=s, segment_text=t, question_text=p, targets_masks_and_weights=g)
        assert e["T"] == 0 and s["T"] == 0 and e["TES"] > 1e-4 and s["TES"] > 1e-4, (e, s)
        assert min(t.values()) > 1e-4 and min(p.values()) > 1e-4 and max(g.values()) == 0, (t, p, g)
        # the future does not reach the past: everything from position 100 on is replaced, text included
        fut = clone(b)
        fut["xin"][:, 100:] += 1.0
        fut["mode"][:, 100:], fut["prev_mode"][:, 100:] = (fut["mode"][:, 100:] + 1) % 6, (fut["prev_mode"][:, 100:] + 1) % 6
        fut["text"][:, 100:] = F.normalize(torch.randn_like(fut["text"][:, 100:]), dim=-1)
        rep["future_change_max_effect_on_positions_before_100"] = {c: spread(m(fut), base[c], 100) for c, m in models.items()}
        rep["future_change_max_effect_on_positions_from_100"] = {c: max(float((m(fut)[k][:, 100:] - base[c][k][:, 100:]).abs().max()) for k in base[c]) for c, m in models.items()}
        assert max(rep["future_change_max_effect_on_positions_before_100"].values()) < 1e-4 and min(rep["future_change_max_effect_on_positions_from_100"].values()) > 1e-4
        # a left prefix alone gives the forecast of the same positions inside the full window (batching and window)
        cut = clone(b)
        for k in ("xin", "mode", "prev_mode", "solver", "bench", "text", "prompt", "pad"):
            cut[k] = cut[k][:, :100]
        rep["left_prefix_of_100_alone_minus_the_full_window"] = {c: spread(m(cut), base[c], 100) for c, m in models.items()}
        assert max(rep["left_prefix_of_100_alone_minus_the_full_window"].values()) < 1e-4
    # one step of the accepted loss: finite, the closed groups get no gradient in T, the text gets one in both
    grads = {}
    for c, m in models.items():
        m.train()
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        loss = A.losses(M, B, m(b), b)
        m.zero_grad(set_to_none=True)
        loss.backward()
        w = m.proj.weight.grad.abs().amax(0)
        grads[c] = dict(loss=float(loss.detach()), first_layer_gradient_abs_max={k: float(w[slice(*v)].max()) for k, v in COLS.items()},
                        span_type_embedding_rows_with_gradient=dict(mode=[int(i) for i in torch.nonzero(m.mode.weight.grad.abs().amax(1)).flatten().tolist()],
                                                                    prev=[int(i) for i in torch.nonzero(m.prev.weight.grad.abs().amax(1)).flatten().tolist()]))
        assert math.isfinite(grads[c]["loss"])
        m.eval()
    rep["one_step_of_the_accepted_loss"] = grads
    g_t, g_s = grads["T"]["first_layer_gradient_abs_max"], grads["TES"]["first_layer_gradient_abs_max"]
    assert g_t["S"] == 0 and g_t["E"] == 0 and grads["T"]["span_type_embedding_rows_with_gradient"] == dict(mode=[7], prev=[7])
    assert min(g_s[k] for k in COLS) > 0 and min(g_t[k] for k in SHARED) > 0 and 7 not in grads["TES"]["span_type_embedding_rows_with_gradient"]["mode"]
    # two real optimiser steps from one seed: TES on data with the structure closed is bit for bit the T model
    # (same initial weights, same batches, same dropout stream, same optimiser); TES on the real data is not
    ids = np.flatnonzero((d["chunks"][:, 4] == 0) & (d["chunks"][:, 1] == cfg["context"]))[4:8]
    end = {}
    for name, cell, closed in (("T", "T", False), ("TES_closed", "TES", True), ("TES", "TES", False)):
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        m = Predictor(cell, meta, cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=.01)
        for step in range(2):
            bb = make_batch(d, ids[step * 2:step * 2 + 2], device, cfg["context"])
            if closed:
                bb["xin"][..., 3:41] = 0
                bb["mode"], bb["prev_mode"] = torch.full_like(bb["mode"], 7), torch.full_like(bb["prev_mode"], 7)
            m.train()
            opt.zero_grad(set_to_none=True)
            loss = A.losses(M, B, m(bb), bb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
        end[name] = ({k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k != "keep"}, float(loss.detach()))
        del m, opt
    diff = lambda x, y: max(float((end[x][0][k] - end[y][0][k]).abs().max()) for k in end[x][0])
    rep["two_optimiser_steps_from_seed_17"] = dict(weights_max_abs_difference_T_vs_TES_with_structure_closed=diff("T", "TES_closed"), weights_max_abs_difference_T_vs_TES=diff("T", "TES"),
                                                   last_loss={k: v[1] for k, v in end.items()})
    assert rep["two_optimiser_steps_from_seed_17"]["weights_max_abs_difference_T_vs_TES_with_structure_closed"] == 0 and end["T"][1] == end["TES_closed"][1]
    assert rep["two_optimiser_steps_from_seed_17"]["weights_max_abs_difference_T_vs_TES"] > 0
    # the targets: what the corrected file gives to the loss
    g, sp_, tr = d["gate"], d["split"], d["trace"]
    last = np.zeros(len(g), bool)
    last[d["offsets"][1:] - 1] = True
    natural = np.zeros(len(g), bool)
    natural[np.isin(tr, tr[last & (g == 2)])] = True
    rep["targets"] = {}
    for k, name in enumerate(("train", "validation", "split2")):
        s_ = sp_ == k
        rep["targets"][name] = dict(traces=int((last & s_).sum()), last_positions_with_END=int((last & s_ & (g == 2)).sum()), last_positions_masked=int((last & s_ & (g == -1)).sum()),
                                    END_anywhere_but_the_last_position=int((~last & s_ & (g == 2)).sum()), masked_anywhere_but_the_last_position=int((~last & s_ & (g == -1)).sum()),
                                    remainder_positions=int((d["rem_mask"] & s_).sum()), remainder_positions_on_traces_without_END=int((d["rem_mask"] & s_ & ~natural).sum()),
                                    positions_of_traces_with_END_without_remainder=int((~d["rem_mask"] & s_ & natural).sum()),
                                    wait_positions_in_the_loss=int(((d["wait_y"] > 0) & s_).sum()), positions_before_the_last_one_without_wait=int(((d["wait_y"] <= 0) & ~last & s_).sum()), wait_observed=int(((d["wait_y"] > 0) & d["wait_observed"] & s_).sum()),
                                    wait_censored=int(((d["wait_y"] > 0) & ~d["wait_observed"] & s_).sum()), wait_censored_on_traces_with_END=int(((d["wait_y"] > 0) & ~d["wait_observed"] & s_ & natural).sum()),
                                    END_as_the_next_boundary_on_traces_without_END=int(((d["next_span"] == 7) & s_ & ~natural).sum()))
        r = rep["targets"][name]
        assert r["END_anywhere_but_the_last_position"] == 0 and r["masked_anywhere_but_the_last_position"] == 0 and r["remainder_positions_on_traces_without_END"] == 0
        assert r["positions_of_traces_with_END_without_remainder"] == 0 and r["wait_censored_on_traces_with_END"] == 0 and r["END_as_the_next_boundary_on_traces_without_END"] == 0
        assert r["last_positions_with_END"] + r["last_positions_masked"] == r["traces"]
    return rep


def timing(B, M, d, meta, cfg):
    """Seconds of a step and memory of the full recipe, input set TES, 60 real steps; nothing of it is a result."""
    train_ids = np.flatnonzero(d["chunks"][:, 4] == 0)
    per_epoch = math.ceil(len(train_ids) / cfg["batch"])
    torch.manual_seed(17)
    model = Predictor("TES", meta, cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.01, fused=True)
    order = np.random.default_rng(17).permutation(train_ids)
    times = []
    for k in range(60):
        t0 = time.time()
        b = make_batch(d, order[k * cfg["batch"]:(k + 1) * cfg["batch"]], "cuda", cfg["context"])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = A.losses(M, B, model(b), b)
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
    fit = per_epoch * cfg["epochs"] * step + (cfg["epochs"] + 1) * val * 1.5
    return dict(train_chunks=int(len(train_ids)), steps_per_epoch=per_epoch, total_steps=per_epoch * cfg["epochs"], seconds_per_step=step, loss_after_60_steps=float(loss.detach()),
                validation_forward_seconds_estimate=val, fit_seconds_estimate=fit, six_fits_seconds_estimate=6 * fit)


def memory():
    return dict(torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20, torch_max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                note="nvidia-smi of the process is the reserved figure plus the CUDA context; it is read by the launcher side, not here")


def probe(B, M, d, meta, cfg, out):
    rep = dict(task=TASK, structure=structure(B, M, d, meta, cfg, "cuda"))
    small = dict(cfg, epochs=1)                                         # a short real pipeline on the fit part: tiny fits with a restart; nothing of it is a result
    limit = time.time() + 900
    a_ = A.fit(B, M, d, meta, small, out / "smoke_straight", "TES", 17, "cuda", limit, [], max_steps=6, tag="x")
    try:
        A.fit(B, M, d, meta, small, out / "smoke_resumed", "TES", 17, "cuda", limit, [], max_steps=6, tag="x", stop_at=3)
    except TimeoutError:
        pass
    assert (out / "smoke_resumed" / "fits" / "x" / "resume.pt").exists() and not (out / "smoke_resumed" / "fits" / "x" / "DONE.json").exists()
    b_ = A.fit(B, M, d, meta, small, out / "smoke_resumed", "TES", 17, "cuda", limit, [], max_steps=6, tag="x")
    t_ = A.fit(B, M, d, meta, small, out / "smoke_text", "T", 17, "cuda", limit, [], max_steps=6, tag="x")
    w1 = torch.load(out / "smoke_straight" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    w2 = torch.load(out / "smoke_resumed" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    w3 = torch.load(out / "smoke_text" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    rep["restart"] = dict(stopped_at_step=3, weights_max_abs_difference_after_6_steps=max(float((w1[k].float() - w2[k].float()).abs().max()) for k in w1),
                          objective_straight=a_["best_validation_objective"], objective_resumed=b_["best_validation_objective"])
    rep["pair_of_one_seed"] = dict(same_start_state_sha256=a_["start_state_sha256"] == t_["start_state_sha256"], start_state_sha256=a_["start_state_sha256"], parameters_T=t_["parameters"], parameters_TES=a_["parameters"],
                                   objective_T_after_6_steps=t_["best_validation_objective"], objective_TES_after_6_steps=a_["best_validation_objective"],
                                   weights_differ_after_6_steps=max(float((w1[k].float() - w3[k].float()).abs().max()) for k in w1 if k != "keep") > 0)
    assert rep["restart"]["weights_max_abs_difference_after_6_steps"] == 0 and a_["best_validation_objective"] == b_["best_validation_objective"]
    assert rep["pair_of_one_seed"]["same_start_state_sha256"] and t_["parameters"] == a_["parameters"] == PARAMETERS and rep["pair_of_one_seed"]["weights_differ_after_6_steps"]
    saved = {k: d[k].copy() for k in ("xin", "mode", "prev_mode")}     # the training loop itself: TES on data with the structure closed must be the T fit
    d["xin"][:, 3:41] = 0
    d["mode"][:], d["prev_mode"][:] = 7, 7
    try:
        c_ = A.fit(B, M, d, meta, small, out / "smoke_closed", "TES", 17, "cuda", limit, [], max_steps=6, tag="x")
    finally:
        for k, v in saved.items():
            d[k] = v
    w4 = torch.load(out / "smoke_closed" / "fits" / "x" / "best.pt", map_location="cpu", weights_only=False)["model"]
    rep["training_loop_TES_with_structure_closed_vs_T"] = dict(weights_max_abs_difference_after_6_steps=max(float((w3[k].float() - w4[k].float()).abs().max()) for k in w3 if k != "keep"),
                                                               objective_T=t_["best_validation_objective"], objective_TES_closed=c_["best_validation_objective"])
    assert rep["training_loop_TES_with_structure_closed_vs_T"]["weights_max_abs_difference_after_6_steps"] == 0 and t_["best_validation_objective"] == c_["best_validation_objective"]
    for sub in ("smoke_straight", "smoke_resumed", "smoke_text", "smoke_closed"):
        (out / sub / "fits" / "x" / "best.pt").unlink()
    rep["timing"] = timing(B, M, d, meta, cfg)
    rep["memory"] = memory()
    rep["device"], rep["torch"] = torch.cuda.get_device_name(0), torch.__version__
    A.atomic_json(out / "PROBE.json", rep)
    print(json.dumps(rep, indent=1))
    print("MATCHED PROBE PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True, help="directory with the frozen base.py, model.py, config.json and data/ of the article model v2")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("cpucheck", "probe", "train"), required=True)
    ap.add_argument("--seconds", type=float, default=7200)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    code = {"matched.py": sha(Path(__file__).resolve()), "ablation.py": ABLATION_SHA}
    if a.mode == "cpucheck":
        assert not torch.cuda.is_available(), "cpucheck is the check without a card"
        B, M, d, meta, cfg = load(a.v2, a.inputs, a.embeddings, "cpu")
        out.mkdir(parents=True, exist_ok=True)
        rep = dict(task=TASK, mode="cpucheck", code=code, inputs=INPUTS_SHA, embeddings=EMBEDDINGS_SHA, torch=torch.__version__, config={k: cfg[k] for k in ("width", "layers", "heads", "context", "batch", "epochs", "lr", "warmup_steps")},
                   structure=structure(B, M, d, meta, cfg, "cpu"))
        A.atomic_json(out / "CPUCHECK.json", rep)
        print(json.dumps(rep, indent=1))
        print("MATCHED CPUCHECK PASSED")
        return
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA device is required; no CPU fallback"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_GIB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    B, M, d, meta, cfg = load(a.v2, a.inputs, a.embeddings, "cuda")
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "probe":
        return probe(B, M, d, meta, cfg, out)
    clock = out / "CLOCK.json"
    if not clock.exists():
        A.atomic_json(clock, dict(first_start=time.time(), deadline=time.time() + a.seconds))
    deadline = json.loads(clock.read_text())["deadline"]                # a restart does not extend the limit
    with open(out / "ATTEMPTS.jsonl", "a") as f:
        f.write(json.dumps(dict(event="start", time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=code, inputs=INPUTS_SHA, embeddings=EMBEDDINGS_SHA, seeds=SEEDS, device=torch.cuda.get_device_name(0))) + "\n")
    man = out / "MANIFEST.json"
    state = json.loads(man.read_text()) if man.exists() else dict(task=TASK, elements={f"{c}_seed{s}": "pending" for s in SEEDS for c in CELLS})
    for seed in SEEDS:                                                  # the two input sets of a seed run one after the other
        for cell in CELLS:
            key = f"{cell}_seed{seed}"
            state["elements"][key] = "running"
            A.atomic_json(man, state)
            try:
                A.fit(B, M, d, meta, cfg, out, cell, seed, "cuda", deadline, [1])
            except BaseException as e:
                state["elements"][key] = f"stopped: {type(e).__name__}: {e}"
                A.atomic_json(man, state)
                raise
            state["elements"][key] = "done"
            A.atomic_json(man, state)
    A.atomic_json(out / "COMPLETE.json", dict(task=TASK, fits=len(SEEDS) * len(CELLS), seeds=SEEDS, cells=list(CELLS), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=code, inputs=INPUTS_SHA, embeddings=EMBEDDINGS_SHA,
                                              config=cfg, memory=memory()))
    print("MATCHED PAIR COMPLETE", flush=True)


if __name__ == "__main__":
    main()
