"""HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013: training of the E+S predictor in hold-out regimes.

The training code is the accepted code of the corrected ablation repeat (ablation.py next to this file, imported
unchanged): same architecture 12 / 512 / 8, same heads, losses, schedule and rule of the choice of the epoch, same
targets of the recovered END statuses.  This file adds only what a hold-out needs:
  * the contribution of the solver and benchmark embeddings is multiplied by zero after the embedding in every regime,
    the reference one included; sizes and the number of parameters stay, the span-type embeddings stay active;
  * a regime file (build_regimes.py) says which traces train, which validate and which are held out; the weights of
    the questions are recomputed from the training part of the regime by the rule of the frozen base.load_data;
  * several regimes and seeds are run by one worker into one output directory under one clock;
  * the own process may not take more than 16 GiB of the card (torch per-process memory fraction).
Every model starts from scratch from the initial state of its seed; no trained weights are loaded.
Usage: python holdout.py --v2 V2 --inputs INPUTS --regimes-dir DIR --regimes a,b,c --seeds 17,29 --out OUT --mode train|check [--seconds N]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from . import ablation as A

MEMORY_LIMIT_GIB = 16


class HPredictor(A.Predictor):
    """The predictor of the ablation with the identity inputs switched off: zero contribution, zero gradient."""

    def forward(self, b):
        x = b["xin"] * self.keep
        ident = torch.cat((self.solver(b["solver"]), self.bench(b["bench"])), dim=-1) * 0.0
        h = self.proj(torch.cat((x, self.mode(b["mode"]), self.prev(b["prev_mode"]), ident), dim=-1))
        n = h.shape[1]
        h = h + self.position(torch.arange(n, device=h.device))[None]
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=h.device), 1)
        h = self.norm(self.core(h, mask=mask, src_key_padding_mask=b["pad"]))
        p = self.wait(h).view(*h.shape[:-1], 3, 3 + self.K + 1)
        return dict(gate=self.gate(h), dest=self.dest(h), events=self.events(h), length=self.length(h).squeeze(-1), wait_pi=p[..., 0], wait_logscale=p[..., 1].clamp(-3, 16),
                    wait_shape=(torch.nn.functional.softplus(p[..., 2]) + .2).clamp(max=5), wait_mark=p[..., 3:], rem=self.rem(h))


A.Predictor = HPredictor                                               # fit() and the checks of ablation.py build this class


def apply_regime(d, meta, frozen, path):
    """Split codes of the regime replace the frozen ones; the question weights follow the training part of the regime."""
    with np.load(path) as z:
        assert sorted(z.files) == ["chunks", "split"]
        split, chunks = z["split"], z["chunks"]
    assert split.shape == frozen["split"].shape and np.array_equal(chunks[:, :4], frozen["chunks"][:, :4])
    assert np.array_equal(chunks[:, 4], split[chunks[:, 0]]), "a chunk and its positions must carry one code"
    moved = split != frozen["split"]
    assert (frozen["split"][moved] < 2).all() and (split[moved] >= 3).all(), "a regime may only take traces out of train and validation"
    d["split"], d["chunks"] = split.astype(frozen["split"].dtype), chunks.astype(frozen["chunks"].dtype)
    counts = np.bincount(d["question"][d["split"] == 0], minlength=len(meta["questions"]))
    w = np.array([1 / max(1, counts[q]) for q in d["question"]], np.float32)
    w *= max(1, counts[counts > 0].mean())                              # the rule of base.load_data, on the training part of this regime
    if not moved.any():
        assert np.array_equal(w, frozen["weight"]), "the weight rule does not reproduce the frozen weights on the full training part"
    d["weight"] = w
    return dict(train_positions=int((split == 0).sum()), validation_positions=int((split == 1).sum()), train_chunks=int((chunks[:, 4] == 0).sum()), train_questions=int((counts > 0).sum()), moved_positions=int(moved.sum()))


def check(B, M, d, meta, cfg, frozen, a, out):
    """Identity inputs give no contribution and no gradient; labels of everything outside train and validation change nothing in the training."""
    res = {}
    torch.manual_seed(5)
    model = HPredictor("ES", meta, cfg).cuda()
    ids = np.flatnonzero(d["chunks"][:, 4] == 0)[:6]
    b = A.make_batch(d, ids, "cuda", cfg["context"])
    model.eval()
    with torch.no_grad():
        o1 = model(b)
        b2 = dict(b)
        b2["solver"] = torch.randint(0, len(meta["solvers"]), b["solver"].shape, device="cuda")
        b2["bench"] = torch.randint(0, len(meta["benchmarks"]), b["bench"].shape, device="cuda")
        o2 = model(b2)
    res["max_output_change_when_every_solver_and_benchmark_index_is_replaced"] = max(float((o1[k] - o2[k]).abs().max()) for k in o1)
    model.train()
    A.losses(M, B, model(b), b).backward()
    res["gradient_of_identity_embeddings_abs_max"] = max(float(model.solver.weight.grad.abs().max()), float(model.bench.weight.grad.abs().max()))
    res["gradient_of_span_type_embeddings_abs_max"] = float(model.mode.weight.grad.abs().max())
    res["parameters"] = sum(p.numel() for p in model.parameters())
    assert res["max_output_change_when_every_solver_and_benchmark_index_is_replaced"] == 0 and res["gradient_of_identity_embeddings_abs_max"] == 0 and res["gradient_of_span_type_embeddings_abs_max"] > 0
    del model
    # labels of the held-out and test positions do not reach the training, the weights or the choice of the epoch
    name = a.regimes.split(",")[0]
    res["regime_of_the_label_check"] = name
    info = apply_regime(d, meta, frozen, Path(a.regimes_dir) / f"regime_{name}.npz")
    r1 = A.fit(B, M, d, meta, cfg, out, "ES", 17, "cuda", time.time() + 600, [], max_steps=6, tag="check_straight")
    s1 = torch.load(out / "fits" / "check_straight" / "best.pt", map_location="cpu", weights_only=False)["model"]
    w_before = d["weight"].copy()
    outside = d["split"] >= 2
    rng = np.random.default_rng(3)
    saved = {k: d[k].copy() for k in ("gate", "dest", "event_y", "wait_y", "rem", "next_span", "length_y", "wait_observed", "rem_mask", "event_mask")}
    d["gate"][outside] = rng.integers(0, 3, int(outside.sum()))
    d["dest"][outside] = rng.integers(0, len(meta["operators"]), int(outside.sum()))
    d["event_y"][outside] = 1 - d["event_y"][outside]
    d["wait_y"][outside] = d["wait_y"][outside] * 3 + 5
    d["rem"][outside] = d["rem"][outside] * 2 + 7
    d["next_span"][outside] = rng.integers(0, len(meta["operators"]), int(outside.sum()))
    d["length_y"][outside] = d["length_y"][outside] + 1
    d["wait_observed"][outside] = ~d["wait_observed"][outside]
    d["rem_mask"][outside] = ~d["rem_mask"][outside]
    d["event_mask"][outside] = ~d["event_mask"][outside]
    info2 = apply_regime(d, meta, frozen, Path(a.regimes_dir) / f"regime_{name}.npz")
    r2 = A.fit(B, M, d, meta, cfg, out, "ES", 17, "cuda", time.time() + 600, [], max_steps=6, tag="check_labels_changed_outside")
    s2 = torch.load(out / "fits" / "check_labels_changed_outside" / "best.pt", map_location="cpu", weights_only=False)["model"]
    for k, v in saved.items():
        d[k] = v
    res["label_check"] = dict(positions_with_changed_labels=int(outside.sum()), weights_max_abs_difference_after_6_steps=max(float((s1[k].float() - s2[k].float()).abs().max()) for k in s1),
                              validation_objective_straight=r1["best_validation_objective"], validation_objective_after_change=r2["best_validation_objective"],
                              question_weights_equal=bool(np.array_equal(w_before, d["weight"])), same_start=r1["start_state_sha256"] == r2["start_state_sha256"], regime=info, regime_again=info2)
    assert res["label_check"]["weights_max_abs_difference_after_6_steps"] == 0 and r1["best_validation_objective"] == r2["best_validation_objective"] and res["label_check"]["question_weights_equal"]
    res["cuda_peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    res["memory_limit"] = f"torch.cuda.set_per_process_memory_fraction: {MEMORY_LIMIT_GIB} GiB of the card for this process; a larger request fails inside the process"
    A.atomic_json(out / "HOLDOUT_CHECK.json", res)
    print(json.dumps(res, indent=1))
    print("HOLDOUT CHECK PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--regimes-dir", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--seeds", default="17,29")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("train", "check"), required=True)
    ap.add_argument("--seconds", type=float, default=7200)
    a = ap.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA device is required; no CPU fallback"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT_GIB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    B, M, d, meta, cfg = A.load(Path(a.v2), a.inputs)
    frozen = dict(split=d["split"].copy(), chunks=d["chunks"].copy(), weight=d["weight"].copy())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "check":
        return check(B, M, d, meta, cfg, frozen, a, out)
    regimes, seeds = a.regimes.split(","), [int(x) for x in a.seeds.split(",")]
    files = {r: Path(a.regimes_dir) / f"regime_{r}.npz" for r in regimes}
    clock = out / "CLOCK.json"
    if not clock.exists():
        A.atomic_json(clock, dict(first_start=time.time(), deadline=time.time() + a.seconds))
    deadline = json.loads(clock.read_text())["deadline"]                # a restart does not extend the limit
    code = {"holdout.py": A.sha(Path(__file__).resolve()), "ablation.py": A.sha(Path(A.__file__).resolve())}
    with open(out / "ATTEMPTS.jsonl", "a") as f:
        f.write(json.dumps(dict(event="start", time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=code, inputs=A.sha(a.inputs), regimes={r: A.sha(p) for r, p in files.items()}, seeds=seeds,
                                device=torch.cuda.get_device_name(0))) + "\n")
    man = out / "MANIFEST.json"
    state = json.loads(man.read_text()) if man.exists() else dict(task_id="HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013", elements={f"{r}__seed{s}": "pending" for r in regimes for s in seeds}, regimes={})
    for r in regimes:
        info = apply_regime(d, meta, frozen, files[r])
        state["regimes"][r] = dict(info, sha256=A.sha(files[r]))
        for s in seeds:
            key = f"{r}__seed{s}"
            state["elements"][key] = "running"
            A.atomic_json(man, state)
            try:
                A.fit(B, M, d, meta, cfg, out, "ES", s, "cuda", deadline, [1], tag=key)
            except BaseException as e:
                state["elements"][key] = f"stopped: {type(e).__name__}: {e}"
                A.atomic_json(man, state)
                raise
            state["elements"][key] = "done"
            A.atomic_json(man, state)
    A.atomic_json(out / "COMPLETE.json", dict(fits=len(regimes) * len(seeds), regimes=regimes, seeds=seeds, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=code, inputs=A.sha(a.inputs), config=cfg))
    print("HOLDOUT PART COMPLETE", flush=True)


if __name__ == "__main__":
    main()
