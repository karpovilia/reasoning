"""Fitting of one gate variant: masked targets, question weights, a fixed grid, selection inside train, calibration on dev.

Kept from the historical trainer (snapshot/runtime/training.py): an unknown outcome is masked, never negative and never dropped
with its prefix; every question has the same total weight; scaling is fitted on train inputs only; the grid, the seeds and the
largest number of epochs are fixed before any outcome is read.

Changes against version 1.
  Cost target (review finding GATE-V1-R1). The mean head is fitted by SQUARED error on the marginal cost of the action in the
  unit of the cost model: the squared error is minimised by the conditional mean, the Huber loss of version 1 is not. The upper
  quantile keeps its pinball loss.
  Selection (decision of the coordinator of 2026-10-12: the 420 train and 60 dev questions stay apart, inner cross-validation
  only inside train). The grid point and the number of epochs are chosen by grouped cross-validation over the TRAIN questions:
  scaling, principal directions and heads are fitted inside each fold, the held fold stops the fit and scores it. The final
  members are then fitted on the whole train part for the chosen number of epochs, without looking at dev. Dev is used for two
  things only: the temperature of each action and, in the stage, the operating point.
No function of this file accepts a test example.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from .contract import ACTIONS, assert_question_disjoint, validate
from .models import Gate

SEEDS = (31, 32, 33, 34, 35)
CV_SEEDS = SEEDS[:2]
GRID = tuple({'learning_rate': rate, 'weight_decay': decay} for rate in (0.001, 0.003) for decay in (0.0, 0.01))
EPOCHS, PATIENCE, BATCH, COST_WEIGHT, QUANTILE_WEIGHT, FOLDS = 80, 10, 32, 0.1, 0.05, 5
FOLD_SALT = 'gate-rebuild-inner-cv-20261012'
TEMPERATURES = tuple(float(v) for v in np.exp(np.linspace(np.log(0.05), np.log(20.), 121)))


def question_weights(groups):
    counts = Counter(groups)
    weights = np.array([1. / counts[g] for g in groups], dtype=np.float32)
    return weights / weights.sum()


def fold_of(groups, folds=FOLDS, salt=FOLD_SALT):
    """{question: fold}: the questions in the order of a salted hash, dealt round-robin. No outcome and no state is read."""
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(('%s|%s' % (salt, g)).encode()).hexdigest())
    if len(ordered) < folds:
        raise ValueError('fewer train questions than folds')
    return {g: i % folds for i, g in enumerate(ordered)}


def batch(examples, source):
    lengths = torch.tensor([len(e.states[source]) for e in examples])
    width = examples[0].states[source].shape[1]
    x = torch.zeros(len(examples), int(lengths.max()), width)
    for i, e in enumerate(examples):
        x[i, :lengths[i]] = torch.as_tensor(e.states[source], dtype=torch.float32)
    stack = lambda name, kind: torch.as_tensor(np.stack([getattr(e, name) for e in examples]), dtype=kind)
    return {'x': x, 'lengths': lengths, 'budget': stack('budget', torch.float32), 'success': stack('success', torch.float32),
            'cost': stack('cost', torch.float32), 'supported': stack('supported', torch.bool),
            'weights': torch.as_tensor(question_weights([e.group for e in examples]))}


def objective(prediction, target, cost_scale, quantile):
    """Masked binary cross-entropy of success, squared error of the mean cost and pinball loss of its quantile; a question counts once."""
    def average(loss, mask):
        per = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
        return (per * target['weights']).sum() / (target['weights'] * mask.any(1)).sum().clamp_min(1e-12)
    known_s = torch.isfinite(target['success']) & target['supported']
    known_c = torch.isfinite(target['cost']) & target['supported']
    y, c = torch.nan_to_num(target['success'], nan=0.), torch.nan_to_num(target['cost'], nan=0.) / cost_scale
    success = F.binary_cross_entropy_with_logits(prediction['success_logits'], y, reduction='none')
    mean = (prediction['cost'] / cost_scale - c) ** 2
    error = c - prediction['cost_quantile'] / cost_scale
    pinball = torch.maximum(quantile * error, (quantile - 1) * error)
    return average(success, known_s.float()) + COST_WEIGHT * average(mean, known_c.float()) + QUANTILE_WEIGHT * average(pinball, known_c.float())


def fit_one(train, held, source, encoder, seed, learning_rate, weight_decay, cost_scale, epochs=EPOCHS, patience=PATIENCE, **options):
    """One member. With `held`: the held examples stop the fit (best held objective, at most `epochs`). Without: exactly
    `epochs` epochs on `train`, nothing else is looked at. Returns the model and a record."""
    torch.manual_seed(seed)
    model = Gate(source, encoder, cost_scale, **options)
    model.encoder.fit_scaling([e.states[source] for e in train])
    data_held = None if held is None else batch(held, source)
    weights = question_weights([e.group for e in train])                # of the whole fitting part: a mini-batch keeps them
    order = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best, best_state, best_epoch, wait, run = float('inf'), None, 0, 0, 0
    for epoch in range(epochs):
        model.train()
        for chunk in torch.randperm(len(train), generator=order).split(BATCH):
            data = batch([train[i] for i in chunk.tolist()], source)
            data['weights'] = torch.as_tensor(weights[chunk.numpy()] / weights[chunk.numpy()].sum())
            optimizer.zero_grad()
            loss = objective(model(data['x'], data['lengths'], data['budget']), data, model.heads.cost_scale, model.heads.quantile)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        run = epoch + 1
        if data_held is None:
            continue
        model.eval()
        with torch.no_grad():
            score = float(objective(model(data_held['x'], data_held['lengths'], data_held['budget']), data_held, model.heads.cost_scale, model.heads.quantile))
        if score < best - 1e-6:
            best, best_epoch, wait = score, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if data_held is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {'seed': seed, 'learning_rate': learning_rate, 'weight_decay': weight_decay, 'epochs_run': run,
                   'held_objective': None if data_held is None else best, 'best_epoch': None if data_held is None else best_epoch}


def calibrate(model, dev, source):
    """One temperature per action on dev success logits, chosen from a fixed grid by the weighted log loss on known labels;
    an action without a known dev label keeps 1. Costs are not rescaled. Returns the temperatures."""
    held = batch(dev, source)
    with torch.no_grad():
        logits = model(held['x'], held['lengths'], held['budget'])['success_logits'] * model.temperature
    known = (torch.isfinite(held['success']) & held['supported']).float()
    target = torch.nan_to_num(held['success'], nan=0.)
    chosen = torch.ones(len(ACTIONS))
    for a in range(len(ACTIONS)):
        if known[:, a].sum() == 0:
            continue
        losses = [float((F.binary_cross_entropy_with_logits(logits[:, a] / t, target[:, a], reduction='none') * known[:, a] * held['weights']).sum()) for t in TEMPERATURES]
        chosen[a] = TEMPERATURES[int(np.argmin(losses))]
    model.temperature.copy_(chosen)
    return chosen.tolist()


def fit(train, dev, source, encoder, cost_scale, seeds=SEEDS, grid=GRID, epochs=EPOCHS, folds=FOLDS, cv_seeds=CV_SEEDS, **options):
    """The whole fitting of one variant. Returns the calibrated members of all seeds and a record. Test examples are refused."""
    validate(train, 'train', [source])
    validate(dev, 'dev', [source])
    assert_question_disjoint(train, dev)
    for name, part in (('train', train), ('dev', dev)):          # an action without one known outcome and cost cannot be fitted or calibrated
        known = np.stack([np.isfinite(e.success) & np.isfinite(e.cost) & e.supported for e in part]).any(0)
        if not known.all():
            raise ValueError('%s has no known outcome and cost for the action %s' % (name, [a for a, k in zip(ACTIONS, known) if not k]))
    assignment = fold_of([e.group for e in train], folds)
    trials = []
    for point in grid:                                           # selection INSIDE train: dev is not read here
        scores, stops = [], []
        for fold in range(folds):
            inner = [e for e in train if assignment[e.group] != fold]
            held = [e for e in train if assignment[e.group] == fold]
            for seed in cv_seeds:
                _, record = fit_one(inner, held, source, encoder, seed, cost_scale=cost_scale, epochs=epochs, **point, **options)
                scores.append(record['held_objective'])
                stops.append(record['best_epoch'])
        trials.append((float(np.mean(scores)), point, int(np.median(stops))))
    score, point, stop = min(trials, key=lambda trial: trial[0])
    runs = [fit_one(train, None, source, encoder, seed, cost_scale=cost_scale, epochs=max(1, stop), **point, **options) for seed in seeds]
    models = [model for model, _ in runs]
    record = {'state_source': source, 'encoder': encoder, 'chosen': point, 'epochs': max(1, stop), 'inner_cv_objective': score,
              'selection': 'grouped %d-fold cross-validation over the train questions; dev only for the temperatures and the operating point' % folds,
              'grid': [{'point': p, 'inner_cv_objective': s, 'median_best_epoch': e} for s, p, e in trials], 'seeds': [r for _, r in runs],
              'train_states': len(train), 'dev_states': len(dev), 'train_questions': len({e.group for e in train}), 'dev_questions': len({e.group for e in dev}),
              'cost_scale': cost_scale}
    record['temperatures'] = [calibrate(model, dev, source) for model in models]
    return models, record


def predict(models, examples, source):
    """Seed-averaged probabilities and costs (in the unit of the cost model). Inputs only: targets of the examples are not read.
    Test examples are refused here; the test entry point is a separate file that does not exist until a test admission is given."""
    if any(e.split == 'test' for e in examples):
        raise ValueError('this function never predicts on test examples')
    data = batch(examples, source)
    with torch.no_grad():
        outs = [model(data['x'], data['lengths'], data['budget']) for model in models]
    mean = lambda name, f=(lambda v: v): torch.stack([f(o[name]) for o in outs]).mean(0).numpy()
    return {'success': mean('success_logits', torch.sigmoid), 'cost': mean('cost'), 'cost_quantile': mean('cost_quantile')}
