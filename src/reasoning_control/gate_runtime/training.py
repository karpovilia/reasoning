"""Masked action targets, question-disjoint fitting, fixed tuning grid, evaluation.

Unknown outcomes are masked, never negative and never dropped with their prefix.
Prediction accepts observations only; test targets never enter fit/tune/predict.
Evaluation is a separate function that receives targets explicitly.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from .features import (ACTIONS, AGGREGATES, VERSION, Observation, batch_sequences, encode)
from .models import ARCHITECTURES, Observer

SEEDS = (25, 26, 27)
# Frozen before any pilot outcome was read (2026-09-25). Four trials per
# architecture; every architecture gets the same number of candidates, epochs
# and seeds. Changes must be documented in the report before test results.
TUNING_GRID = {
    'linear': [{'learning_rate': r, 'weight_decay': d} for r in (.01, .03) for d in (0., .01)],
    'mlp': [{'learning_rate': r, 'weight_decay': d} for r in (.001, .003) for d in (0., .01)],
    'gru': [{'learning_rate': r, 'weight_decay': d} for r in (.001, .003) for d in (0., .01)],
}
EPOCHS = 80
WIDTH_HIDDEN = 64
COST_LOSS_WEIGHT = .1


@dataclass(frozen=True)
class Example:
    """Observation plus metadata and masked targets. Metadata never enters tensors."""
    key: str
    group: str
    split: str
    observation: Observation
    # Fractions over KNOWN sampled action outcomes; np.nan denotes unknown.
    success: np.ndarray
    cost: np.ndarray
    coverage: dict = field(default_factory=dict)


def question_weights(groups):
    if len(groups) == 0:
        raise ValueError('Empty data')
    counts = Counter(groups)
    result = np.array([1. / counts[g] for g in groups], dtype=np.float32)
    return result / result.sum()


def aggregate_draws(correct, cost):
    """Input [action][seed] outcomes (bool/None) and costs (finite/None).

    A prefix with any unknown outcome keeps every known slot: cost of a known
    generation stays a target even when its correctness is unknown. No
    complete-case filtering here, and unknown never becomes False.
    """
    if len(correct) != len(ACTIONS) or len(cost) != len(ACTIONS):
        raise ValueError('Require all action slots, even when labels are unknown')
    means, costs, known, total, known_cost = [], [], [], [], []
    for outcomes, values in zip(correct, cost):
        if not outcomes or len(outcomes) != len(values):
            raise ValueError('Mismatched/missing seed slots')
        if any(y is not None and type(y) not in (bool, np.bool_) for y in outcomes):
            raise ValueError('Outcome must be boolean or unknown, not a prediction')
        if any(v is not None and (not np.isfinite(v) or v < 0) for v in values):
            raise ValueError('Invalid observed cost')
        yy = [float(y) for y in outcomes if y is not None]
        cc = [float(v) for v in values if v is not None]
        means.append(np.mean(yy) if yy else np.nan)
        costs.append(np.mean(cc) if cc else np.nan)
        known.append(len(yy)); total.append(len(outcomes)); known_cost.append(len(cc))
    return (np.array(means, dtype=np.float32), np.array(costs, dtype=np.float32),
            {'known_correctness_draws': known, 'known_cost_draws': known_cost, 'total_draws': total})


def masked_objective(prediction, target_success, target_cost, weights, cost_scale):
    def average(loss, mask):
        per = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
        present = mask.any(1)
        denominator = (weights * present).sum()
        return (per * weights).sum() / denominator.clamp_min(1e-12)

    known_success, known_cost = torch.isfinite(target_success), torch.isfinite(target_cost)
    yy = torch.nan_to_num(target_success, nan=0.)
    cc = torch.nan_to_num(target_cost, nan=0.)
    success_loss = F.binary_cross_entropy_with_logits(prediction['success_logits'], yy,
                                                      reduction='none')
    cost_loss = F.smooth_l1_loss(prediction['cost'] / cost_scale, cc / cost_scale,
                                 reduction='none')
    return (average(success_loss, known_success.float())
            + COST_LOSS_WEIGHT * average(cost_loss, known_cost.float()))


def validate_examples(examples, expected_split):
    if not examples or any(e.split != expected_split for e in examples):
        raise ValueError(f'Expected nonempty {expected_split} only; test targets forbidden in fit')
    if len({e.key for e in examples}) != len(examples):
        raise ValueError('Duplicate prefix key')
    for e in examples:
        y, c = np.asarray(e.success, dtype=np.float32), np.asarray(e.cost, dtype=np.float32)
        if y.shape != (len(ACTIONS),) or c.shape != y.shape or not e.group or not e.key:
            raise ValueError('Invalid example/target shape')
        finite_y, finite_c = y[np.isfinite(y)], c[np.isfinite(c)]
        if np.isinf(y).any() or np.isinf(c).any() or ((finite_y < 0) | (finite_y > 1)).any() \
                or (finite_c < 0).any():
            raise ValueError('Invalid action outcome/cost')
    y, c = np.stack([e.success for e in examples]), np.stack([e.cost for e in examples])
    if not np.isfinite(y).any(axis=0).all() or not np.isfinite(c).any(axis=0).all():
        raise ValueError('At least one action has no known training/validation target')


def permutation_seed(key, seed):
    return int(hashlib.sha256(f'ablation|{seed}|{key}'.encode()).hexdigest()[:16], 16)


def _encoded(examples, variant, seed):
    return [encode(e.observation, variant, permutation_seed=permutation_seed(e.key, seed))
            for e in examples]


def _targets(examples):
    return (torch.tensor(np.stack([e.success for e in examples]), dtype=torch.float32),
            torch.tensor(np.stack([e.cost for e in examples]), dtype=torch.float32),
            torch.tensor(question_weights([e.group for e in examples])))


def coverage_summary(examples):
    y = np.stack([e.success for e in examples]); c = np.stack([e.cost for e in examples])
    return {'questions': len({e.group for e in examples}), 'prefixes': len(examples),
            'known_success_prefixes_per_action': np.isfinite(y).sum(0).tolist(),
            'known_cost_prefixes_per_action': np.isfinite(c).sum(0).tolist(),
            'prefixes_with_any_unknown_success': int((~np.isfinite(y)).any(1).sum()),
            'retained_all_unknown_success_prefixes': int((~np.isfinite(y)).all(1).sum()),
            'unknown_seed_draws_per_action': [int(sum(e.coverage.get('total_draws', [0]*5)[a]
                                                      - e.coverage.get('known_correctness_draws', [0]*5)[a]
                                                      for e in examples)) for a in range(len(ACTIONS))]}


def fit(train, validation, *, variant, architecture, seed=25, width=WIDTH_HIDDEN,
        epochs=EPOCHS, learning_rate=.001, weight_decay=.01, cost_scale=1024.,
        cost_unit='additional_tokens'):
    """Full-batch AdamW; early stopping and epoch choice on validation only."""
    validate_examples(train, 'train')
    validate_examples(validation, 'validation')
    if {e.group for e in train} & {e.group for e in validation} \
            or {e.key for e in train} & {e.key for e in validation}:
        raise ValueError('Question/prefix leakage between train and validation')
    if epochs < 1 or learning_rate <= 0 or not np.isfinite(learning_rate) \
            or weight_decay < 0 or not np.isfinite(weight_decay):
        raise ValueError('Invalid optimizer configuration')
    torch.manual_seed(seed)
    model = Observer(variant, architecture, width=width, cost_scale=cost_scale, cost_unit=cost_unit)
    tx, vx = _encoded(train, variant, seed), _encoded(validation, variant, seed)
    model.fit_scaling(tx)
    tb, vb = batch_sequences(tx), batch_sequences(vx)
    ty, tc, tw = _targets(train); vy, vc, vw = _targets(validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best, best_state, best_epoch, curve = float('inf'), None, None, []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = masked_objective(model(*tb), ty, tc, tw, cost_scale)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); model.eval()
        with torch.no_grad():
            score = float(masked_objective(model(*vb), vy, vc, vw, cost_scale))
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation loss')
        curve.append((float(loss.detach()), score))
        if score < best:
            best, best_epoch, best_state = score, epoch + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    metadata = {**model.metadata(), 'seed': seed, 'epochs': epochs, 'best_epoch': best_epoch,
                'validation_loss': best, 'learning_rate': learning_rate,
                'weight_decay': weight_decay, 'loss_curve_train_validation': curve,
                'selection_uses_test': False,
                'masking_unknown_is_not_missing_at_random_proof': True,
                'splits': {'train': coverage_summary(train),
                           'validation': coverage_summary(validation)}}
    return model, metadata


def predict(model, observations, *, keys=None, permutation=SEEDS[0]):
    """Observations only. No labels, costs, groups or references can enter here."""
    if keys is None:
        keys = [str(i) for i in range(len(observations))]
    if len(keys) != len(observations) or len(set(keys)) != len(keys):
        raise ValueError('Require one unique key per observation')
    sequences = [encode(o, model.variant, permutation_seed=permutation_seed(k, permutation))
                 for o, k in zip(observations, keys)]
    model.eval()
    with torch.no_grad():
        result = model(*batch_sequences(sequences))
    return {'keys': list(keys), 'actions': list(ACTIONS),
            'p_action_correct': result['success_logits'].sigmoid().cpu().numpy(),
            'predicted_cost': result['cost'].cpu().numpy(), 'cost_unit': model.cost_unit,
            'variant': model.variant, 'architecture': model.architecture,
            'feature_version': VERSION}


def candidate_grid(variant, architectures=None):
    """Fixed trials; the caller selects on validation only and keeps every trial."""
    if architectures is None:
        architectures = ARCHITECTURES if variant in AGGREGATES else ('gru',)
    for architecture in architectures:
        if architecture != 'gru' and variant not in AGGREGATES:
            raise ValueError('Sequence variants only take the GRU')
        for setting in TUNING_GRID[architecture]:
            yield {'variant': variant, 'architecture': architecture, **setting}


def tune(train, validation, *, variant, architectures=None, seeds=SEEDS, epochs=EPOCHS,
         width=WIDTH_HIDDEN, cost_unit='additional_tokens', cost_scale=1024.):
    """Every candidate x every seed; selection = lowest mean validation loss.

    Returns all trials (kept, not only the winner), the selection per
    architecture and overall, and the fitted models of the selected candidate.
    """
    trials, models = [], {}
    for candidate in candidate_grid(variant, architectures):
        for seed in seeds:
            model, meta = fit(train, validation, seed=seed, epochs=epochs, width=width,
                              cost_unit=cost_unit, cost_scale=cost_scale, **candidate)
            trials.append({**candidate, 'seed': seed, 'validation_loss': meta['validation_loss'],
                           'best_epoch': meta['best_epoch'], 'parameters': meta['parameters']})
            models[(candidate['architecture'], candidate['learning_rate'],
                    candidate['weight_decay'], seed)] = (model, meta)
    by_candidate = {}
    for t in trials:
        by_candidate.setdefault((t['architecture'], t['learning_rate'], t['weight_decay']), []).append(t)
    summary = [{'architecture': a, 'learning_rate': r, 'weight_decay': d,
                'mean_validation_loss': float(np.mean([t['validation_loss'] for t in ts])),
                'seed_validation_losses': [t['validation_loss'] for t in ts],
                'best_epochs': [t['best_epoch'] for t in ts], 'parameters': ts[0]['parameters']}
               for (a, r, d), ts in by_candidate.items()]
    per_architecture = {}
    for row in sorted(summary, key=lambda s: (s['mean_validation_loss'], s['learning_rate'], s['weight_decay'])):
        per_architecture.setdefault(row['architecture'], row)
    overall = min(per_architecture.values(), key=lambda s: s['mean_validation_loss'])
    selected = {seed: models[(overall['architecture'], overall['learning_rate'],
                              overall['weight_decay'], seed)] for seed in seeds}
    budget = {a: {'candidates': len(TUNING_GRID[a]), 'seeds': len(seeds), 'epochs': epochs,
                  'fits': len(TUNING_GRID[a]) * len(seeds)}
              for a in {t['architecture'] for t in trials}}
    return {'variant': variant, 'trials': trials, 'candidates': summary,
            'selected_per_architecture': per_architecture, 'selected': overall,
            'tuning_budget': budget, 'selection_rule': 'lowest mean validation loss over seeds',
            'models': selected, 'models_per_architecture': {
                a: {seed: models[(a, row['learning_rate'], row['weight_decay'], seed)]
                    for seed in seeds} for a, row in per_architecture.items()}}


def evaluate(prediction, success, cost, groups, *, coverage=None,
             price_grid=(0., 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)):
    """Targets enter here only. Per action Brier/log-loss/cost error, plus a
    replay of the utility gate on a predeclared price grid with bounds for unknown
    outcomes. Development diagnostic, not a closed-loop or population result."""
    p = np.asarray(prediction['p_action_correct'], dtype=float)
    c = np.asarray(prediction['predicted_cost'], dtype=float)
    y = np.asarray(success, dtype=float); k = np.asarray(cost, dtype=float)
    if p.shape != y.shape or c.shape != k.shape or p.shape[1] != len(ACTIONS):
        raise ValueError('Prediction/target shape mismatch')
    # y is a fraction over KNOWN draws, not all scheduled draws. A finite
    # value (e.g. 1 from [True, None]) does not imply complete ascertainment.
    if coverage is None:
        nk = np.isfinite(y).astype(float)
        nt = np.ones_like(y)
        nc = np.isfinite(k).astype(float)
    else:
        nk = np.asarray([v['known_correctness_draws'] for v in coverage], float)
        nt = np.asarray([v['total_draws'] for v in coverage], float)
        nc = np.asarray([v['known_cost_draws'] for v in coverage], float)
        if (nk.shape != y.shape or nt.shape != y.shape or nc.shape != y.shape
                or not np.isfinite(np.stack([nk, nt, nc])).all()
                or (nt <= 0).any() or (nk < 0).any() or (nk > nt).any()
                or (nc < 0).any() or (nc > nt).any()
                or not np.array_equal(nk > 0, np.isfinite(y))
                or not np.array_equal(nc > 0, np.isfinite(k))):
            raise ValueError('Invalid or inconsistent draw coverage')
    known_fraction = nk / nt
    lower = np.nan_to_num(y, nan=0.) * known_fraction
    upper = lower + (1. - known_fraction)
    w = question_weights(list(groups))
    per_action = {}
    for j, action in enumerate(ACTIONS):
        m = np.isfinite(y[:, j]); mc = np.isfinite(k[:, j])
        ww = w[m] / w[m].sum() if m.any() else w[m]
        pj, yj = np.clip(p[m, j], 1e-6, 1 - 1e-6), y[m, j]
        per_action[action] = {
            'known_prefixes': int(m.sum()), 'unknown_prefixes': int((~m).sum()),
            # For Bernoulli draws with fraction y, expected Brier is p^2-2py+y.
            'brier': float(ww @ (pj * pj - 2 * pj * yj + yj)) if m.any() else None,
            'log_loss': float(ww @ (-yj * np.log(pj) - (1 - yj) * np.log(1 - pj))) if m.any() else None,
            'mean_predicted_success': float(ww @ pj) if m.any() else None,
            'mean_observed_success': float(ww @ yj) if m.any() else None,
            'cost_mae': float((w[mc] / w[mc].sum()) @ np.abs(c[mc, j] - k[mc, j])) if mc.any() else None,
            'cost_unit': prediction['cost_unit']}
    grid = []
    for price in price_grid:
        utility = p - price * c
        chosen = utility.argmax(1)
        obs = y[np.arange(len(y)), chosen]; obc = k[np.arange(len(k)), chosen]
        known = np.isfinite(obs)
        ii = np.arange(len(y))
        fraction = known_fraction[ii, chosen]
        known_weight = w * fraction
        cost_fraction = nc[ii, chosen] / nt[ii, chosen]
        complete_cost = bool((cost_fraction == 1).all())
        grid.append({'price_per_' + prediction['cost_unit']: price,
                     'chosen_actions': dict(Counter(ACTIONS[i] for i in chosen)),
                     'observed_success_known_only': float(known_weight @ np.nan_to_num(obs, nan=0.) / known_weight.sum()) if known_weight.sum() else None,
                     'observed_success_lower_bound': float(w @ lower[ii, chosen]),
                     'observed_success_upper_bound': float(w @ upper[ii, chosen]),
                     'unknown_chosen_outcomes': int((nt - nk)[ii, chosen].sum()),
                     'chosen_prefixes_with_unknown_outcomes': int((fraction < 1).sum()),
                     'mean_observed_cost': float(w @ obc) if complete_cost else None,
                     'observed_cost_lower_bound': float(w @ (np.nan_to_num(obc, nan=0.) * cost_fraction)),
                     'unknown_chosen_cost_draws': int((nt - nc)[ii, chosen].sum()),
                     'cost_unit': prediction['cost_unit']})
    return {'questions': len(set(groups)), 'prefixes': len(y), 'per_action': per_action,
            'replay_price_grid': grid, 'question_weighted': True,
            'draw_coverage_supplied': coverage is not None,
            'warning': 'Single-decision paired-branch development diagnostic; '
                       'unknown outcomes are bounded, not dropped.'}
