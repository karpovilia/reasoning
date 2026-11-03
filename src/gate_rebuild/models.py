"""Encoders of an observer state and the action-conditioned heads.

Delta against the historical gate (gate_trainval_2026_09_25/snapshot/runtime/models.py, class Observer):
  - the historical model has five independent linear heads on a width-64 summary of 40 lexical channels; here ONE scorer is
    applied to every action with that action's embedding and its budget features, so the heads are conditioned on the action
    and on the budget, and an action without a real branch is masked instead of being trained as a failure;
  - the cost head predicts the mean AND an upper quantile of the marginal cost of the action in the unit of the cost model;
  - the encoder is chosen by name and consumes a NAMED state source: the summary GRU of the historical gate (the 'simple
    predictor'), a readout of states of a frozen trunk ('process_state'), or one vector ('aggregate');
  - a wide state is first projected on its leading principal directions, fitted on the TRAIN records without any label; the
    same number of directions is used for every wide source;
  - no weight of the historical gate can be loaded.

Review finding GATE-V1-R6. Version 1 had a small transformer trained from nothing as the control "without process pretraining".
It was not the accepted dynamics transformer (no start token, other positions, other inputs, other readout and head width), so
it is REMOVED. The matched control is now a state source: the SAME class, configuration, inputs and state boundary as the
pretrained dynamics transformer, with seeded random weights, computed by the same producer, and read here by the SAME
'process_state' readout, the same principal directions rule and the same heads as the main candidate. One thing differs between
the two rows: where the weights of the trunk came from.
"""
from __future__ import annotations

import torch
from torch import nn

from .contract import ACTIONS, STATE_SOURCES

ENCODERS = ('aggregate', 'gru', 'process_state')


class Encoder(nn.Module):
    def __init__(self, source: str, kind: str, width: int = 64, readout: int = 4, reduce: int = 0):
        super().__init__()
        if source not in STATE_SOURCES or kind not in ENCODERS:
            raise ValueError('unknown state source or encoder')
        spec = STATE_SOURCES[source]
        if kind != 'aggregate' and not spec['sequence']:
            raise ValueError('a sequence encoder needs a sequence source')
        if kind == 'aggregate' and spec['sequence']:
            raise ValueError('an aggregate encoder must not silently discard a sequence')
        self.source, self.kind, self.width, self.readout = source, kind, int(width), int(readout)
        self.register_buffer('mean', torch.zeros(spec['width']))
        self.register_buffer('scale', torch.ones(spec['width']))
        self.register_buffer('scaling_fitted', torch.zeros((), dtype=torch.bool))
        if reduce < 0 or reduce > spec['width']:
            raise ValueError('invalid number of principal directions')
        self.reduce = int(reduce)
        inner = self.reduce or spec['width']
        self.register_buffer('basis', torch.zeros(spec['width'], inner) if self.reduce else torch.zeros(0))
        if kind == 'aggregate':
            self.body = nn.Sequential(nn.Linear(inner, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        elif kind == 'gru':
            self.project = nn.Sequential(nn.Linear(inner, width), nn.GELU())
            self.body = nn.GRU(width, width, batch_first=True)
        else:                                       # 'process_state': the trunk that made the state is NOT here, it stays frozen where it was accepted
            self.body = nn.Sequential(nn.Linear(inner * 2, width), nn.GELU(), nn.LayerNorm(width))

    def fit_scaling(self, arrays):
        """Train inputs only."""
        values = torch.cat([torch.as_tensor(a, dtype=torch.float32) for a in arrays])
        scale = values.std(0)
        scale[scale < 1e-6] = 1.
        self.mean.copy_(values.mean(0))
        self.scale.copy_(scale)
        if self.reduce:                                   # principal directions of the standardized TRAIN records; no label is involved
            standardized = (values - self.mean) / self.scale
            _, _, directions = torch.linalg.svd(standardized, full_matrices=False)
            if len(directions) < self.reduce:
                raise ValueError('fewer train records than principal directions')
            self.basis.copy_(directions[:self.reduce].T)
        self.scaling_fitted.fill_(True)

    def forward(self, x, lengths):
        """x [B, T, W] padded on the right; lengths [B] >= 1. Returns [B, width]. Only records before the decision exist in x."""
        if not bool(self.scaling_fitted):
            raise RuntimeError('train-only scaling has not been fitted')
        if x.ndim != 3 or lengths.shape != (len(x),) or (lengths < 1).any() or (lengths > x.shape[1]).any():
            raise ValueError('invalid encoder input')
        z = (x - self.mean) / self.scale
        if self.reduce:
            z = z @ self.basis
        index = (lengths - 1).clamp_min(0)
        rows = torch.arange(len(x), device=x.device)
        if self.kind == 'aggregate':
            return self.body(z[rows, index])
        if self.kind == 'gru':
            packed = nn.utils.rnn.pack_padded_sequence(self.project(z), lengths.cpu(), batch_first=True, enforce_sorted=False)
            return self.body(packed)[1][-1]
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        recent = valid & (torch.arange(x.shape[1], device=x.device)[None, :] >= (lengths[:, None] - self.readout))     # 'process_state'
        mean = (z * recent[:, :, None]).sum(1) / recent.sum(1, keepdim=True).clamp_min(1)
        return self.body(torch.cat([z[rows, index], mean], dim=1))                  # the last state and the mean of the last `readout` states


class ActionHeads(nn.Module):
    """p_a = P(correct final answer | state, action, budget) and the marginal cost of the action in the unit of the cost model:
    its mean and an upper quantile. cost_scale is a fixed constant of that unit (the cost of 1024 solver tokens), not a fitted value."""

    def __init__(self, width: int, cost_scale: float, quantile: float = 0.9, hidden: int = 128):
        super().__init__()
        if not 0.5 <= quantile < 1 or cost_scale <= 0:
            raise ValueError('invalid cost head configuration')
        self.cost_scale, self.quantile = float(cost_scale), float(quantile)
        self.action = nn.Embedding(len(ACTIONS), 16)
        self.trunk = nn.Sequential(nn.Linear(width + 16 + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.success, self.cost_mean, self.cost_excess = nn.Linear(hidden, 1), nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, state, budget):
        """state [B, width]; budget [B, A, 2]. Returns per action: success logits, cost mean, cost quantile (never below the mean)."""
        if state.ndim != 2 or budget.shape != (len(state), len(ACTIONS), 2):
            raise ValueError('invalid head input')
        actions = self.action(torch.arange(len(ACTIONS), device=state.device))[None].expand(len(state), -1, -1)
        h = self.trunk(torch.cat([state[:, None, :].expand(-1, len(ACTIONS), -1), actions, budget], dim=2))
        mean = nn.functional.softplus(self.cost_mean(h).squeeze(2)) * self.cost_scale
        return {'success_logits': self.success(h).squeeze(2), 'cost': mean,
                'cost_quantile': mean + nn.functional.softplus(self.cost_excess(h).squeeze(2)) * self.cost_scale}


class Gate(nn.Module):
    def __init__(self, source: str, encoder: str, cost_scale: float, width: int = 64, quantile: float = 0.9, **encoder_options):
        super().__init__()
        self.encoder = Encoder(source, encoder, width, **encoder_options)
        self.heads = ActionHeads(width, cost_scale, quantile)
        self.register_buffer('temperature', torch.ones(len(ACTIONS)))      # set on dev by calibration, never on test

    def forward(self, x, lengths, budget):
        out = self.heads(self.encoder(x, lengths), budget)
        out['success_logits'] = out['success_logits'] / self.temperature
        return out

    def metadata(self):
        return {'state_source': self.encoder.source, 'encoder': self.encoder.kind, 'width': self.encoder.width, 'actions': list(ACTIONS),
                'principal_directions': self.encoder.reduce,
                'cost_scale': self.heads.cost_scale, 'cost_quantile': self.heads.quantile, 'parameters': sum(p.numel() for p in self.parameters()),
                'historical_gate_weights_loadable': False}
