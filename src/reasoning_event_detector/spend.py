"""Optional: forecast of how much reasoning is still to come, from text alone.

NO WEIGHTS ARE SHIPPED FOR THIS FUNCTION. Calling `spend_forecast` without
supplying a head raises `SpendWeightsUnavailable`. That is the honest state of
the artifact, not an oversight — see MODEL_CARD.md, section "what is not here".

Background. The project's text-only remaining-budget monitor (variant 23,
internal_signals_poc/latent_monitor/text/) reads the reasoning text with a
frozen Qwen2.5-0.5B, takes hidden states of layers 12 and 24, projects them
896 -> 128 with a fixed Gaussian projection (seed 42), pools with stride 8, and
regresses the remaining budget with a small head. On 311 held-out M traces it
reached R2 0.760 / MAE 659 tokens on the remaining-token task (P2), beating the
event-count baseline bar of 0.703 / 697 — that is, an outside reader of the text
does about as well as a monitor with access to the generating model's logits.
Only the metrics were persisted from that run; the head checkpoint was not.
internal_signals_poc/latent_monitor/checkpoints/ holds the LOGIT variant, which
needs logits and therefore cannot be used on a service model.

Interface, so a retrained head can be dropped in without touching callers:

    encoder(text) -> np.ndarray [n_pooled, d]     any text encoder
    head npz: W1 [h, d], b1 [h], W2 [h], b2 []    2-layer MLP, tanh in between
              plus scalar `scale` and `bias` applied to the output

    spend_forecast(text, head="/path/head.npz", encoder=my_encoder)

The head math is exercised by tests with a stub encoder; the Qwen encoder path
is untested here because no weights exist to test it against, and it is marked
as such at runtime.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from .models import SpendForecast

ENV_HEAD = "REASONING_EVENT_DETECTOR_SPEND_HEAD"
ENV_ENCODER_MODEL = "REASONING_EVENT_DETECTOR_SPEND_ENCODER"

Encoder = Callable[[str], np.ndarray]


class SpendWeightsUnavailable(RuntimeError):
    """No remaining-budget head is configured."""


def head_path(head: Optional[str] = None) -> Optional[str]:
    p = head or os.environ.get(ENV_HEAD)
    return p if (p and os.path.isfile(p)) else None


def available(head: Optional[str] = None) -> bool:
    return head_path(head) is not None


def _apply_head(feats: np.ndarray, blob) -> float:
    """feats [n, d] -> scalar forecast. Mean-pool, MLP, affine rescale."""
    x = np.asarray(feats, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("encoder must return a non-empty [n_pooled, d] array")
    v = x.mean(axis=0)
    h = np.tanh(blob["W1"] @ v + blob["b1"])
    y = float(blob["W2"] @ h + blob["b2"])
    return float(y * float(blob.get("scale", 1.0)) + float(blob.get("bias", 0.0)))


def _qwen_encoder(model_dir: str) -> Encoder:
    """Reference encoder of variant 23. UNTESTED in this package: no head exists
    to validate it end to end, so it is built only when explicitly requested."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    rng = np.random.default_rng(42)

    def enc(text: str) -> np.ndarray:
        with torch.no_grad():
            ids = tok(text, return_tensors="pt", add_special_tokens=False)
            out = mdl(**ids, output_hidden_states=True)
            hs = out.hidden_states[12][0].numpy()          # layer 12, as in v23
        proj = rng.standard_normal((hs.shape[1], 128)) / np.sqrt(hs.shape[1])
        z = hs @ proj
        stride = 8
        n = (len(z) // stride) * stride
        if n == 0:
            return z[:1]
        return z[:n].reshape(-1, stride, z.shape[1]).mean(axis=1)

    return enc


def spend_forecast(text: str, head: Optional[str] = None,
                   encoder: Optional[Encoder] = None) -> SpendForecast:
    """Forecast remaining reasoning tokens from the text seen so far.

    Raises SpendWeightsUnavailable when no head is configured — this package
    ships none.
    """
    hp = head_path(head)
    if hp is None:
        raise SpendWeightsUnavailable(
            "no remaining-budget head is available. This artifact ships none: "
            "the text-monitor run (variant 23, R2 0.760 on held-out M) persisted "
            "metrics but not the checkpoint, and the checkpoint that does exist "
            "in latent_monitor/checkpoints/ is the logit variant, which a service "
            f"model cannot feed. Set ${ENV_HEAD} to a head npz, or pass head=.")
    blob = dict(np.load(hp, allow_pickle=False))
    if encoder is None:
        model_dir = os.environ.get(ENV_ENCODER_MODEL)
        if not model_dir:
            raise SpendWeightsUnavailable(
                f"head {hp} found but no encoder: pass encoder= or set "
                f"${ENV_ENCODER_MODEL} to a local Qwen2.5-0.5B directory.")
        encoder = _qwen_encoder(model_dir)
        note = "encoder=Qwen2.5-0.5B reference path (untested in this package)"
    else:
        note = "encoder supplied by caller"
    feats = encoder(text)
    return SpendForecast(remaining_tokens=_apply_head(feats, blob),
                         consumed_words=len(text.split()),
                         model=os.path.basename(hp), note=note)
