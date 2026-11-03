"""HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013: evaluation-only export of one fixed checkpoint (no gradients, no choice).

Forecasts of the fixed weights at EVERY stored position of the original test part (code 2) and, for a hold-out regime,
of the held-out traces of the original train part (code 3): also the last stored position of a trace without a
natural end, whose next-step target is absent.  Every metric applies the mask of its own target later, on the CPU.
The validation objective of the regime is recomputed first and must equal the one recorded by the fit.
Usage: python holdout_eval.py --v2 V2 --inputs INPUTS --regimes-dir DIR --regime NAME --seed N --checkpoint best.pt
                              --expect-sha SHA --expect-objective X --out DIR"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from . import holdout as H

A = H.A


@torch.no_grad()
def export(M, model, d, code, cfg, path):
    chunks = np.flatnonzero(d["chunks"][:, 4] == code)
    keep = {k: [] for k in ("index", "gate", "dest", "rem", "wait_nll", "length", "events")}
    model.eval()
    for start in range(0, len(chunks), 128):
        b = A.make_batch(d, chunks[start:start + 128], "cuda", cfg["context"])
        o = model(b)
        m = b["score"].cpu().numpy()
        nll = M.wait_nll(o, b["wait_y"], b["wait_observed"], b["next_span"])
        nll = torch.where(b["wait_y"] > 0, nll, torch.full_like(nll, float("nan")))
        keep["index"].append(b["index"][m])
        keep["gate"].append(F.softmax(o["gate"], -1).float().cpu().numpy()[m])
        keep["dest"].append(F.softmax(o["dest"], -1).float().cpu().numpy()[m])
        keep["rem"].append(np.clip(np.expm1(np.minimum(o["rem"].float().cpu().numpy()[m], 20)), 0, None))
        keep["wait_nll"].append(nll.float().cpu().numpy()[m])
        keep["length"].append(o["length"].float().cpu().numpy()[m])
        keep["events"].append(torch.sigmoid(o["events"]).float().cpu().numpy()[m])
    idx = np.concatenate(keep["index"])
    order = np.argsort(idx)
    assert np.array_equal(idx[order], np.flatnonzero(d["split"] == code)), "a position of the set is missing or repeated"
    tmp = Path(str(path)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, index=idx[order].astype(np.int64), gate=np.concatenate(keep["gate"])[order].astype(np.float32), dest=np.concatenate(keep["dest"])[order].astype(np.float32),
                        rem=np.concatenate(keep["rem"])[order].astype(np.float32), wait_nll=np.concatenate(keep["wait_nll"])[order].astype(np.float32),
                        length=np.concatenate(keep["length"])[order].astype(np.float32), events=np.concatenate(keep["events"])[order].astype(np.float16))
    tmp.replace(path)
    return dict(positions=int(len(idx)), without_next_step_target=int((d["gate"][idx] < 0).sum()), sha256=A.sha(path))


def main():
    ap = argparse.ArgumentParser()
    for k in ("v2", "inputs", "regimes-dir", "regime", "checkpoint", "out", "expect-sha"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--expect-objective", type=float, required=True)
    a = ap.parse_args()
    t0 = time.time()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(H.MEMORY_LIMIT_GIB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    assert A.sha(a.checkpoint) == a.expect_sha, "the checkpoint is not the one fixed in the manifest"
    B, M, d, meta, cfg = A.load(Path(a.v2), a.inputs)
    frozen = dict(split=d["split"].copy(), chunks=d["chunks"].copy(), weight=d["weight"].copy())
    info = H.apply_regime(d, meta, frozen, Path(a.regimes_dir) / f"regime_{a.regime}.npz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    state = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    model = H.HPredictor("ES", meta, cfg).cuda()
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    val = A.evaluate(M, B, model, d, 1, "cuda", cfg)
    assert abs(val["scores"]["selection_objective"] - a.expect_objective) < 2e-3, "the validation objective of the regime is not reproduced"
    rec = dict(regime=a.regime, seed=a.seed, checkpoint_sha256=a.expect_sha, selected_step=state["step"], validation_objective_recorded=a.expect_objective, validation_objective_recomputed=val["scores"]["selection_objective"],
               regime_info=info, sets={})
    for code, name in ((2, "test_part"), (3, "held_out_train_part")):
        if (d["chunks"][:, 4] == code).any():
            rec["sets"][name] = export(M, model, d, code, cfg, out / f"export_{name}.npz")
    rec.update(code={"holdout_eval.py": A.sha(Path(__file__).resolve()), "holdout.py": A.sha(Path(H.__file__).resolve()), "ablation.py": A.sha(Path(A.__file__).resolve())}, inputs=A.sha(a.inputs),
               device=torch.cuda.get_device_name(0), gradient_steps=0, seconds=time.time() - t0)
    A.atomic_json(out / "EVAL_DONE.json", rec)
    print(json.dumps(rec, indent=1))
    print("HOLDOUT EVAL DONE", a.regime, a.seed)


if __name__ == "__main__":
    main()
