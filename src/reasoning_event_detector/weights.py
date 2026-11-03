"""Weight registry: named checkpoints, sha256 verification, local resolution.

Nothing is downloaded. A checkpoint is used only if it is present on disk AND
its sha256 matches the value frozen in REGISTRY. A mismatch is an error, never a
warning: the metrics in MODEL_CARD.md are statements about these exact bytes.

Search order for a checkpoint file:
  1. an explicit path passed by the caller;
  2. $REASONING_EVENT_DETECTOR_WEIGHTS (a directory);
  3. `checkpoints/` inside the installed package (where the files ship);
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_DIR = "REASONING_EVENT_DETECTOR_WEIGHTS"

_PKG_DIR = Path(__file__).resolve().parent
# src/reasoning_event_detector/ -> src/ -> <artifact root>/
_ARTIFACT_ROOT = _PKG_DIR.parent.parent


@dataclass(frozen=True)
class WeightSpec:
    name: str
    filename: str
    sha256: str
    fmt: str                 # "torch" (.pt blob) or "npz" (torch-free export)
    teacher: str
    domain: str
    use_history: bool
    target_density_per_1k: float
    canonical_path: str
    note: str


REGISTRY: Dict[str, WeightSpec] = {
    "M_hist_judged": WeightSpec(
        name="M_hist_judged",
        filename="event_scorer_M_hist_judged.pt",
        sha256="3e2c224d8cd728bfedd25435cea3fe56a84ab6df60a1b8fceb13a34b3f602426",
        fmt="torch",
        teacher="DeepSeek judge labels (gold_deepseek_dual), domain M",
        domain="M",
        use_history=True,
        target_density_per_1k=9.29,
        canonical_path="",
        note="regex channels + own-prediction history (no_internal ablation, "
             "D=40). The checkpoint the project's gated reward runs used. "
             "Identical weights to r2_deepseek_M_hist_td4.pt; only "
             "target_density_per_1k differs (9.29 vs 4.0). Counts agree with "
             "the teacher slightly worse than M_nohist and under-count by 14 "
             "per cent.",
    ),
    "M_nohist": WeightSpec(
        name="M_nohist",
        filename="event_scorer_M_nohist.pt",
        sha256="10bd3d5528fef026d6ee85b73ee9ce211f7f7246aef1148b8275e18df07d9411",
        fmt="torch",
        teacher="DeepSeek judge labels (gold_deepseek_dual), domain M",
        domain="M",
        use_history=False,
        target_density_per_1k=8.0243,
        canonical_path="",
        note="DEFAULT. regex channels only (no_history ablation, D=7). Not "
             "gameable through its own history; the project's headline "
             "count-Spearman 0.781 was measured on this checkpoint, and this "
             "package re-measures 0.778 on the same split.",
    ),
    "M_hist_judged_npz": WeightSpec(
        name="M_hist_judged_npz",
        filename="event_scorer_M_hist_judged.npz",
        sha256="7f5db82a18adc99a3c66459f4027d3201c7449ae33809ebd1cf0eed19e3150c1",
        fmt="npz",
        teacher="DeepSeek judge labels (gold_deepseek_dual), domain M",
        domain="M",
        use_history=True,
        target_density_per_1k=9.29,
        canonical_path="derived from M_hist_judged by tools/export_npz.py",
        note="torch-free export of M_hist_judged; lets the runtime depend on "
             "numpy only.",
    ),
    "M_nohist_npz": WeightSpec(
        name="M_nohist_npz",
        filename="event_scorer_M_nohist.npz",
        sha256="0fa5606445373e62f3d889b8dfd57eff6f79ea0373b665fe04aff330a841b7d6",
        fmt="npz",
        teacher="DeepSeek judge labels (gold_deepseek_dual), domain M",
        domain="M",
        use_history=False,
        target_density_per_1k=8.0243,
        canonical_path="derived from M_nohist by tools/export_npz.py",
        note="torch-free export of M_nohist.",
    ),
}

# M_nohist is the default on measured grounds, not on habit: on the same 658
# held-out traces it beats M_hist_judged on every count metric this artifact
# claims (count-Spearman 0.778 vs 0.755, calibration 1.00 vs 0.865, macro
# per-type 0.428 vs 0.368; validation_counts.json), it cannot be driven by its
# own history, and its streaming pass is a single causal sweep. M_hist_judged
# stays available: it is the checkpoint the project's reward runs used.
DEFAULT_WEIGHTS = "M_nohist"


class WeightsUnavailable(RuntimeError):
    """Checkpoint file not found in any of the search locations."""


class WeightsCorrupted(RuntimeError):
    """Checkpoint found but its sha256 does not match the registry."""


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def search_dirs() -> List[Path]:
    dirs: List[Path] = []
    env = os.environ.get(ENV_DIR)
    if env:
        dirs.append(Path(env))
    dirs.append(_PKG_DIR / "checkpoints")
    dirs.append(_ARTIFACT_ROOT / "checkpoints")
    return dirs


def resolve(name: str = DEFAULT_WEIGHTS, path: Optional[str] = None,
            verify: bool = True) -> Path:
    """Absolute path of a registered checkpoint, sha256-verified."""
    if name not in REGISTRY:
        raise KeyError(f"unknown weights '{name}'; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates += [d / spec.filename for d in search_dirs()]
        canon = Path(spec.canonical_path)
        if canon.is_absolute():
            candidates.append(canon)
    for cand in candidates:
        if cand.is_file():
            if verify and spec.sha256:
                got = sha256_of(cand)
                if got != spec.sha256:
                    raise WeightsCorrupted(
                        f"{cand}: sha256 {got} != registry {spec.sha256}. "
                        "The checkpoint does not match its registered hash.")
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise WeightsUnavailable(
        f"weights '{name}' ({spec.filename}) not found. Tried:\n  {tried}\n"
        f"Set ${ENV_DIR} to a directory holding it.")


def available() -> Dict[str, dict]:
    """Registry entries annotated with whether the file is present and valid."""
    out = {}
    for name, spec in REGISTRY.items():
        row = {"filename": spec.filename, "format": spec.fmt,
               "teacher": spec.teacher, "domain": spec.domain,
               "use_history": spec.use_history,
               "target_density_per_1k": spec.target_density_per_1k,
               "note": spec.note, "path": None, "status": "missing"}
        try:
            p = resolve(name)
            row["path"] = str(p)
            row["status"] = "ok"
        except WeightsCorrupted as exc:
            row["status"] = f"corrupted: {exc}"
        except WeightsUnavailable:
            pass
        out[name] = row
    return out
