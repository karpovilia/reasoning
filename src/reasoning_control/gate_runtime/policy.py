"""Hand-off adapter for the end-to-end runner (task 05): load(spec) -> decide(obs).

The runner builds an observation dict {now, segments, confidence,
confidence_available_at} and calls decide(). This adapter converts the dict into
the explicit Observation whitelist, runs the frozen checkpoint, and returns the
five success probabilities, the five predicted costs in the checkpoint's cost
unit, and the action chosen by the runner's price rule
    utility = p_action_correct - price * predicted_cost.
It is not a controller: no generation, no budget logic, no retraining. A missing
or unparsable required input never becomes a confident stop; the spec's
`fallback_action` is returned with an explicit flag.

Spec (JSON) fields:
  name, module (this file), weights (.pt from save_checkpoint), price (float,
  per cost unit), needs_confidence (bool), needs_structure (bool),
  fallback_action (one of ACTIONS), optional allowed_actions (list).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .features import ACTIONS, Confidence, Observation, Segment, VARIANT_CHANNELS
from .models import load_checkpoint
from .training import predict


def choose_action(success, cost, price: float, allowed=None):
    p, c = np.asarray(success, float), np.asarray(cost, float)
    if (p.shape != (len(ACTIONS),) or c.shape != p.shape or not np.isfinite(p).all()
            or not np.isfinite(c).all() or ((p < 0) | (p > 1)).any() or (c < 0).any()
            or not math.isfinite(price) or price < 0):
        raise ValueError('Invalid action estimates')
    mask = np.ones(len(ACTIONS), bool) if allowed is None else np.array([a in allowed for a in ACTIONS])
    if not mask.any():
        raise ValueError('No allowed action')
    utility = np.where(mask, p - price * c, -np.inf)
    index = int(np.argmax(utility))
    return index, float(utility[index])


def observation_from_dict(obs: dict) -> Observation:
    """Whitelist conversion; unknown keys (text, ids, answers) are ignored, never read."""
    segments = tuple(Segment(int(s['token_start']), int(s['token_end']), int(s['available_at']),
                             tuple(float(v) for v in s['events']), int(s['span']))
                     for s in obs.get('segments') or [])
    confidences = ()
    if obs.get('confidence') is not None:
        at = obs.get('confidence_available_at')
        confidences = (Confidence(int(obs['now'] if at is None else at), float(obs['confidence'])),)
    return Observation(now=int(obs['now']), segments=segments, confidences=confidences)


class FrozenObserverPolicy:
    def __init__(self, spec: dict):
        for field in ('name', 'weights', 'price', 'needs_confidence', 'needs_structure', 'fallback_action'):
            if field not in spec:
                raise ValueError(f'Policy spec lacks {field!r}')
        if spec['fallback_action'] not in ACTIONS:
            raise ValueError('fallback_action must be one of ACTIONS')
        self.spec = spec
        self.name = spec['name']
        self.model, self.metadata = load_checkpoint(spec['weights'])
        channels = VARIANT_CHANNELS[self.model.variant]
        uses_confidence = 'confidence' in channels
        uses_structure = any(c.startswith(('event:', 'count:', 'span_last:', 'token_')) for c in channels)
        if bool(spec['needs_confidence']) != uses_confidence or bool(spec['needs_structure']) != uses_structure:
            raise ValueError(f'Spec flags disagree with variant {self.model.variant}: '
                             f'confidence={uses_confidence}, structure={uses_structure}')
        if spec.get('variant') not in (None, self.model.variant):
            raise ValueError('Spec variant differs from the checkpoint variant')
        self.price = float(spec['price'])
        self.allowed = spec.get('allowed_actions')
        self.fallback = spec['fallback_action']

    def decide(self, obs: dict) -> dict:
        base = {'policy': self.name, 'variant': self.model.variant, 'architecture': self.model.architecture,
                'feature_version': self.metadata['feature_version'], 'cost_unit': self.model.cost_unit,
                'price_per_cost_unit': self.price, 'actions': list(ACTIONS)}
        if self.spec['needs_confidence'] and obs.get('confidence') is None:
            return {**base, 'action': self.fallback, 'p_success': None, 'cost': None, 'utility': None,
                    'fallback': True, 'reason': 'confidence missing or unparsed'}
        try:
            observation = observation_from_dict(obs)
            out = predict(self.model, [observation], keys=[str(obs.get('key', 'online'))])
        except (ValueError, TypeError, KeyError) as exc:
            return {**base, 'action': self.fallback, 'p_success': None, 'cost': None, 'utility': None,
                    'fallback': True, 'reason': f'invalid observation: {exc}'}
        p = out['p_action_correct'][0].tolist(); c = out['predicted_cost'][0].tolist()
        index, utility = choose_action(p, c, self.price, self.allowed)
        return {**base, 'action': ACTIONS[index], 'p_success': p, 'cost': c, 'utility': utility,
                'fallback': False, 'reason': None}


def load(spec):
    if not isinstance(spec, dict):
        spec = json.loads(Path(spec).read_text())
    return FrozenObserverPolicy(spec)
