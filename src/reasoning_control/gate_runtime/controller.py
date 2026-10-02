"""Frozen seconds-priced observers, question-only baseline and constant mixtures."""
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
from .policy import FrozenObserverPolicy

ACTIONS = ('stop', 'continue_512', 'continue_2048', 'verify_512', 'neutral_512')


def seconds(tokens, prompt_tokens, converter):
    c = np.asarray(tokens, float)
    if not np.isfinite(c).all() or (c < 0).any() or not math.isfinite(prompt_tokens) or prompt_tokens < 0:
        raise ValueError('Invalid predicted cost inputs')
    a, b, d = converter['coefficients']
    if min(a, b, d) < 0 or not np.isfinite([a, b, d]).all():
        raise ValueError('Invalid frozen cost converter')
    return a + b * c + d * prompt_tokens


def choose(p, costs, price):
    p, costs = np.asarray(p, float), np.asarray(costs, float)
    if p.shape != costs.shape or p.ndim != 1 or len(p) not in (3, 5):
        raise ValueError('Action shape mismatch')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(costs).all() or (costs < 0).any() or not math.isfinite(price) or price < 0:
        raise ValueError('Invalid action prediction')
    values = p - price * costs
    k = int(np.argmax(values))
    return ACTIONS[k], float(values[k])


def mixture_action(probabilities, key, seed, mixture_seed=2026092704):
    p = np.asarray(probabilities, float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-12:
        raise ValueError('Invalid constant mixture')
    raw = json.dumps([mixture_seed, key, int(seed)], separators=(',', ':')).encode()
    u = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64
    return ACTIONS[min(int(np.searchsorted(np.cumsum(p), u, side='right')), 2)]


class SecondsObserver:
    needs_presolve = False

    def __init__(self, spec):
        self.spec = spec
        # The checkpoint's cost head remains additional_tokens, unchanged.
        self.base = FrozenObserverPolicy({**spec, 'price': 0.})

    def decide(self, obs):
        result = self.base.decide(obs)
        if result['fallback']:
            return result
        cost = seconds(result['cost'], obs['prompt_tokens'] + obs['now'], self.spec['converter'])
        allowed = self.spec.get('allowed_actions', list(ACTIONS))
        if tuple(allowed) not in (ACTIONS, ACTIONS[:3]):
            raise ValueError('Only the frozen three- or five-action sets are supported')
        n = len(allowed)
        action, utility = choose(result['p_success'][:n], cost[:n], self.spec['price'])
        return {**result, 'action': action, 'utility': utility,
                'predicted_additional_tokens': result['cost'], 'cost': cost.tolist(),
                'cost_unit': 'seconds', 'price_per_cost_unit': self.spec['price']}


class PreSolve:
    needs_presolve = True

    def __init__(self, spec):
        self.spec = spec
        if spec['controller_kind'] == 'prompt':
            from .prompt_budget import predict
            self.predict = predict
            with Path(spec['weights']).open('rb') as stream:
                self.bundle = pickle.load(stream)

    def pre_decide(self, question, key, seed, prompt_tokens):
        if self.spec['controller_kind'] == 'mixture':
            return {'action': mixture_action(self.spec['probabilities'], key, seed),
                    'p_success': None, 'cost': None, 'utility': None, 'fallback': False}
        t0, c0 = time.perf_counter(), time.process_time()
        result = self.predict(self.bundle, [question])
        prediction = {'p_success': result['p_success'][0].tolist(),
                      'predicted_total_generated_tokens': result['predicted_total_generated_tokens'][0].tolist(),
                      'predict_wall_seconds': time.perf_counter() - t0,
                      'predict_cpu_seconds': time.process_time() - c0}
        c = seconds(prediction['predicted_total_generated_tokens'], prompt_tokens, self.spec['converter'])
        action, utility = choose(prediction['p_success'], c, self.spec['price'])
        return {**prediction, 'action': action, 'cost': c.tolist(), 'cost_unit': 'seconds',
                'utility': utility, 'fallback': False, 'price_per_cost_unit': self.spec['price']}

    def decide(self, obs):
        raise RuntimeError('Pre-solve policy must run BEFORE prefix generation')

    def close(self):
        pass


def load(spec):
    return SecondsObserver(spec) if spec['controller_kind'] == 'observer' else PreSolve(spec)
