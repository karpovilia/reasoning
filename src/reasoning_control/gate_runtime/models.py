"""Linear / MLP / GRU observers with identical action heads and train-only scaling.

No pretrained weights and no difficulty head. Five action heads predict, for each
of ACTIONS, the probability that the answer obtained after that action is correct
and the cost of that action in the unit named by `cost_unit` (the label file's
column; never renamed to a full cost). Index 0 (`stop`) estimates the CURRENT
elicited answer, not the correctness of each reasoning step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import (ACTIONS, AGGREGATES, CHANNEL_MASKS, FEATURE_NAMES, VARIANTS, VERSION,
                      WIDTH)

ARCHITECTURES = ('linear', 'mlp', 'gru')


class Observer(nn.Module):
    def __init__(self, variant: str, architecture: str, *, width: int = 64,
                 cost_scale: float = 1024., cost_unit: str = 'additional_tokens'):
        super().__init__()
        if variant not in VARIANTS or architecture not in ARCHITECTURES:
            raise ValueError('Unknown representation or architecture')
        if architecture != 'gru' and variant not in AGGREGATES:
            raise ValueError('Aggregate model cannot silently discard a sequence')
        if width <= 0 or not np.isfinite(cost_scale) or cost_scale <= 0 or not cost_unit:
            raise ValueError('Invalid model/cost configuration')
        self.variant, self.architecture = variant, architecture
        self.width, self.cost_scale, self.cost_unit = int(width), float(cost_scale), cost_unit
        self.register_buffer('feature_mean', torch.zeros(WIDTH))
        self.register_buffer('feature_scale', torch.ones(WIDTH))
        self.register_buffer('channel_mask', torch.as_tensor(CHANNEL_MASKS[variant]))
        self.register_buffer('scaling_fitted', torch.zeros((), dtype=torch.bool))
        if architecture == 'linear':
            self.project = nn.Identity()
            hidden_width = WIDTH
        else:
            self.project = nn.Sequential(nn.Linear(WIDTH, width), nn.GELU())
            hidden_width = width
        if architecture == 'mlp':
            self.hidden = nn.Sequential(nn.Linear(width, width), nn.GELU())
        elif architecture == 'gru':
            self.hidden = nn.GRU(width, width, batch_first=True)
        else:
            self.hidden = nn.Identity()
        self.success = nn.Linear(hidden_width, len(ACTIONS))
        self.cost = nn.Linear(hidden_width, len(ACTIONS))

    def fit_scaling(self, sequences):
        """Call ONLY on training inputs; validation/test never touch these buffers."""
        values = np.concatenate(sequences)
        if values.ndim != 2 or values.shape[1] != WIDTH or not np.isfinite(values).all():
            raise ValueError('Invalid training features')
        mean, scale = values.mean(0), values.std(0)
        scale[scale < 1e-6] = 1
        mean[~CHANNEL_MASKS[self.variant]] = 0
        scale[~CHANNEL_MASKS[self.variant]] = 1
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        self.scaling_fitted.fill_(True)

    def forward(self, x, lengths):
        if x.ndim != 3 or x.shape[2] != WIDTH or lengths.shape != (len(x),):
            raise ValueError('Invalid model input shape')
        if (lengths < 1).any() or (lengths > x.shape[1]).any():
            raise ValueError('Invalid sequence lengths')
        if self.architecture != 'gru' and (lengths != 1).any():
            raise ValueError('Aggregate model requires one record')
        if not bool(self.scaling_fitted):
            raise RuntimeError('Train-only scaling has not been fitted')
        if bool((x[:, :, ~self.channel_mask] != 0).any()):
            raise ValueError(f'Input carries channels outside variant {self.variant}')
        z = self.project((x - self.feature_mean) / self.feature_scale)
        if self.architecture == 'gru':
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(),
                                                       batch_first=True, enforce_sorted=False)
            _, h = self.hidden(packed)
            h = h[-1]
        else:
            h = self.hidden(z[:, 0])
        logits = self.success(h)
        return {'embedding': h, 'success_logits': logits,
                'cost': nn.functional.softplus(self.cost(h)) * self.cost_scale}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def metadata(self) -> dict:
        return {'variant': self.variant, 'architecture': self.architecture, 'width': self.width,
                'cost_scale': self.cost_scale, 'cost_unit': self.cost_unit,
                'parameters': self.parameter_count(), 'feature_version': VERSION,
                'feature_names': list(FEATURE_NAMES),
                'active_channels': [n for n, m in zip(FEATURE_NAMES, CHANNEL_MASKS[self.variant]) if m],
                'actions': list(ACTIONS), 'difficulty_head_trained': False,
                'stop_head_estimates': 'current elicited answer, not per-step correctness'}


def save_checkpoint(model: Observer, path, extra: dict | None = None):
    """Weights + scaling buffers + metadata in one file, plus a readable JSON twin."""
    path = Path(path)
    payload = {'state_dict': model.state_dict(), 'metadata': {**model.metadata(), **(extra or {})}}
    torch.save(payload, path)
    with path.with_suffix('.json').open('w') as f:
        json.dump(payload['metadata'], f, ensure_ascii=False, indent=2, default=str)


def load_checkpoint(path) -> tuple[Observer, dict]:
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    meta = payload['metadata']
    if meta.get('feature_version') != VERSION:
        raise ValueError(f'Checkpoint feature version {meta.get("feature_version")} != {VERSION}')
    model = Observer(meta['variant'], meta['architecture'], width=meta['width'],
                     cost_scale=meta['cost_scale'], cost_unit=meta['cost_unit'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, meta
