"""Preparation primitives for the selective995 experiment, not a live policy.

Routing has no access to labels. Claims must come from a separately frozen,
audited policy; this module does not certify heads or turn scores into guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from .core import EVENTS, MODES

ROLE_SALT = 'selective995-validation-roles-v1'
ROLES = ('selection', 'tuning', 'audit')


def any_event_target(targets, mask):
    """A direct known positive suffices; absence requires ALL six known zeros."""
    if len(targets) != len(EVENTS) or len(mask) != len(EVENTS):
        raise ValueError('expected six event targets/masks')
    if any(x not in (0, 1, False, True) for x in mask):
        raise ValueError('event mask must be binary')
    known = [t for t, m in zip(targets, mask, strict=True) if m]
    if any(t not in (0, 1, False, True) for t in known):
        raise ValueError('known event targets must be binary')
    if any(known):
        return 1
    return 0 if all(mask) else None


def validation_role(question_key, split):
    """Roles within the existing validation split; never reshuffle the corpus."""
    if split != 'validation' or not isinstance(question_key, str) or not question_key:
        raise ValueError('a nonempty validation question key is required')
    digest = hashlib.sha256((ROLE_SALT + '|' + question_key).encode()).hexdigest()
    return ROLES[int(digest, 16) % len(ROLES)]


def binary_denominators(rows, absence_positive_cost=20., presence_negative_cost=20.):
    """Compute once over the effective batch, including class/question weights."""
    for cost in (absence_positive_cost, presence_negative_cost):
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError('cost must be finite and positive')
    den_no = den_yes = 0.
    for row in rows:
        weight = row['weight']
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('question weight must be finite and nonnegative')
        target = any_event_target(row['event_targets'], row['event_mask'])
        if target is not None:
            den_no += weight * (absence_positive_cost if target else 1.)
            den_yes += weight * (1. if target else presence_negative_cost)
    return den_no, den_yes


def binary_losses(absence_logits, presence_logits, rows, denominators=None,
                  absence_positive_cost=20., presence_negative_cost=20.):
    """Both heads predict any-event. Index known targets before evaluating BCE.

    Returns two independent losses; the proposed trainer weights each by 0.5.
    Unknown-only microbatches still have differentiable, finite zero losses.
    """
    import torch
    from torch.nn import functional as F
    local_den = binary_denominators(rows, absence_positive_cost, presence_negative_cost)
    den_no, den_yes = local_den if denominators is None else denominators
    if any(not math.isfinite(x) or x < 0 for x in (den_no, den_yes)):
        raise ValueError('invalid effective-batch denominator')
    if den_no + 1e-9 < local_den[0] or den_yes + 1e-9 < local_den[1]:
        raise ValueError('effective-batch denominator smaller than microbatch')
    if absence_logits.shape != presence_logits.shape or absence_logits.shape != (len(rows),):
        raise ValueError('one scalar per row and head is required')
    if absence_logits.device != presence_logits.device:
        raise ValueError('head devices differ')
    targets = [any_event_target(r['event_targets'], r['event_mask']) for r in rows]
    known = torch.tensor([t is not None for t in targets], device=absence_logits.device,
                         dtype=torch.bool)
    # Empty indexed sums do not read unknown logits (including NaN sentinels).
    no = absence_logits.float()[known].sum() * 0
    yes = presence_logits.float()[known].sum() * 0
    if known.any():
        y = torch.tensor([t for t in targets if t is not None],
                         device=absence_logits.device, dtype=torch.float32)
        weights = torch.tensor([r['weight'] for r, t in zip(rows, targets, strict=True)
                                if t is not None], device=y.device, dtype=torch.float32)
        if den_no > 0:
            w = weights * torch.where(y.bool(), absence_positive_cost, 1.)
            no = (F.binary_cross_entropy_with_logits(absence_logits.float()[known], y,
                                                     reduction='none') * w).sum() / den_no
        if den_yes > 0:
            w = weights * torch.where(y.bool(), 1., presence_negative_cost)
            yes = (F.binary_cross_entropy_with_logits(presence_logits.float()[known], y,
                                                      reduction='none') * w).sum() / den_yes
    return no, yes


@dataclass(frozen=True)
class BinaryPolicy:
    """None disables a branch; these thresholds alone are not certification."""
    absence_max: float | None = None
    presence_min: float | None = None

    def __post_init__(self):
        for value in (self.absence_max, self.presence_min):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('threshold must be in [0, 1] or disabled')

    def decide(self, absence_probability, presence_probability):
        for value in (absence_probability, presence_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                return 'invalid'
        absent = self.absence_max is not None and absence_probability <= self.absence_max
        present = self.presence_min is not None and presence_probability >= self.presence_min
        if absent and present:
            return 'conflict'
        return 'absent' if absent else 'present' if present else 'abstain'


@dataclass(frozen=True)
class Claims:
    """Only already accepted outputs. No targets, masks or inferred gold labels."""
    source: str
    events: tuple = (None,) * 6
    any_event: bool | None = None
    mode: int | None = None

    def __post_init__(self):
        if not self.source or len(self.events) != len(EVENTS):
            raise ValueError('source and six event outputs are required')
        if any(v is not None and type(v) is not bool for v in (*self.events, self.any_event)):
            raise ValueError('event decisions must be bool or None')
        if self.mode is not None and (type(self.mode) is not int or not 0 <= self.mode < len(MODES)):
            raise ValueError('mode must be a known index or None')


@dataclass(frozen=True)
class Routed:
    events: tuple
    any_event: bool | None
    mode: int | None
    needs_lora: bool
    reason: str
    sources: tuple


def merge_claims(claims):
    """A complete event vector AND mode are needed to bypass the full LoRA call.

    The caller executes stages lazily, rechecking after each stage. Contradiction
    discards all cheap-stage outputs for this focus and requests full fallback.
    Fallback receives the original available prefix, never a filtered history.
    """
    event_sets = [set() for _ in EVENTS]
    unions, modes, sources = set(), set(), []
    for claim in claims:
        sources.append(claim.source)
        for values, decision in zip(event_sets, claim.events, strict=True):
            if decision is not None:
                values.add(decision)
        if claim.any_event is not None:
            unions.add(claim.any_event)
        if claim.mode is not None:
            modes.add(claim.mode)
    if False in unions:
        for values in event_sets:
            values.add(False)
    if any(True in values for values in event_sets):
        unions.add(True)
    if all(values == {False} for values in event_sets):
        unions.add(False)
    if any(len(values) > 1 for values in (*event_sets, unions, modes)):
        return Routed((None,) * len(EVENTS), None, None, True, 'conflict', tuple(sources))
    events = tuple(next(iter(values)) if values else None for values in event_sets)
    union = next(iter(unions)) if unions else None
    mode = next(iter(modes)) if modes else None
    complete = all(v is not None for v in events) and mode is not None
    return Routed(events, union, mode, not complete,
                  'complete' if complete else 'unresolved_outputs', tuple(sources))


def binary_report(decisions, rows, target_precision=.995, min_known=1000, min_questions=50):
    """Descriptive quality of previously computed decisions, not calibration.

    Unknowns count towards stream coverage but never towards verified successes.
    Branch pass is empirical/support-only, not a confidence bound or deployment
    authorization. Question-cluster uncertainty is a separate required step.
    """
    if not 0 < target_precision <= 1 or min_known < 1 or min_questions < 1:
        raise ValueError('invalid evaluation requirements')
    if len(decisions) != len(rows):
        raise ValueError('decision/row alignment differs')
    if any(d not in ('absent', 'present', 'abstain', 'conflict', 'invalid') for d in decisions):
        raise ValueError('unrecognized route decision')
    targets = [any_event_target(r['event_targets'], r['event_mask']) for r in rows]
    known_total = sum(t is not None for t in targets)
    positive_total = sum(t == 1 for t in targets)
    report = {'total': len(rows), 'known_total': known_total,
              'conflicts': decisions.count('conflict'), 'invalid': decisions.count('invalid')}
    for name, truth in (('absent', 0), ('present', 1)):
        accepted = [i for i, d in enumerate(decisions) if d == name]
        known = [i for i in accepted if targets[i] is not None]
        correct = sum(targets[i] == truth for i in known)
        unknown = len(accepted) - len(known)
        questions = len({rows[i]['question_key'] for i in known})
        precision = correct / len(known) if known else None
        report[name] = {
            'accepted': len(accepted), 'known': len(known), 'unknown': unknown,
            'known_questions': questions, 'correct': correct, 'errors': len(known) - correct,
            'precision_known': precision,
            'whole_stream_precision_bounds_from_unknowns':
                [correct / len(accepted), (correct + unknown) / len(accepted)] if accepted else None,
            'stream_coverage': len(accepted) / len(rows) if rows else None,
            'known_coverage': len(known) / known_total if known_total else None,
            'empirical_and_support_pass': precision is not None and precision >= target_precision
                and len(known) >= min_known and questions >= min_questions,
        }
    report['known_positive_recall_lost_to_absence'] = (
        report['absent']['errors'] / positive_total if positive_total else None)
    return report
