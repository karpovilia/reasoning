"""Value of a policy on a bank of real branches: correctness of the final answer against full cost.

Every decision replays the REAL branch of the chosen action from the same prefix; nothing is imputed. A value is computed from
an ATTEMPT TABLE: one row per attempt (a decision state with the chosen action, or an attempt that ended before the decision
position with its own answer and cost). Every question weighs the same; inside a question every attempt weighs the same.

Unknown answer labels (review finding GATE-V1-R3). An unknown label is never a failure and never a success. Three numbers are
kept apart and none replaces another:
  accuracy_known         mean over questions of the accuracy on their known labels. CONDITIONAL: the set of known labels depends
                         on the policy, so a difference of two such numbers is not a difference of population accuracies.
  accuracy_lower_bound   every unknown label counted wrong;  accuracy_upper_bound: every unknown label counted right.
  coverage               question-weighted share of attempts with a known label.
A claim about the whole population (better, or not worse within a margin) uses the worst case of the pair: the lower bound of
the candidate against the upper bound of the comparator. The conditional difference is reported next to it with both coverages.

Tail of the cost (review finding GATE-V1-R2). p90_attempt_cost is the 0.9 quantile of the cost of ONE ATTEMPT under the
question-weighted distribution of attempts (weight 1 / (questions x attempts of the question)); it is not a quantile of
per-question means. Intervals resample questions, also for the tail.

Two populations are reported: 'whole' (all attempts of the split, including those that ended before the decision position) and
'active' (decision states only).

This file computes on any arrays it is given; which split may be passed is decided by the entry points. No entry point for the
test split exists in this package.
"""
from __future__ import annotations

import warnings

import numpy as np

from .contract import ACTIONS

QUANTILE = 0.9
SUMMARY_KEYS = ('accuracy_known', 'accuracy_lower_bound', 'accuracy_upper_bound', 'coverage', 'questions', 'questions_with_known_label', 'attempts',
                'unknown_attempts', 'mean_total_cost', 'p90_attempt_cost')


def attempt_table(actions, success, total_cost, groups, early_success=None, early_cost=None, early_groups=None):
    """actions [N] chosen indices on the decision states; success [N, A] with nan for unknown; total_cost [N, A] full cost of the
    attempt under each action, including what the policy paid for looking. early_*: attempts that ended before the decision."""
    actions = np.asarray(actions)
    rows = np.arange(len(actions))
    y, c, g = np.asarray(success, float)[rows, actions], np.asarray(total_cost, float)[rows, actions], list(groups)
    active, chosen = np.ones(len(rows), bool), actions.astype(int)
    if early_success is not None:
        if not (len(early_success) == len(early_cost) == len(early_groups)):
            raise ValueError('the attempts that ended early are given inconsistently')
        y, c, g = np.concatenate([y, np.asarray(early_success, float)]), np.concatenate([c, np.asarray(early_cost, float)]), g + list(early_groups)
        active, chosen = np.concatenate([active, np.zeros(len(early_groups), bool)]), np.concatenate([chosen, np.full(len(early_groups), -1)])
    if not np.isfinite(c).all():
        raise ValueError('an attempt has no cost: an action without a real branch must be excluded before the choice')
    if np.isinf(y).any() or ((y[np.isfinite(y)] < 0) | (y[np.isfinite(y)] > 1)).any() or (c < 0).any():
        raise ValueError('an attempt has an outcome or a cost outside its range')
    names = sorted(set(g))
    index = {name: i for i, name in enumerate(names)}
    return {'names': names, 'q': np.array([index[name] for name in g], dtype=int), 'm': np.ones(len(g)), 'y': y, 'c': c, 'active': active, 'action': chosen}


def mix(table_a, table_b, share):
    """The randomised policy that follows policy a with probability `share` and policy b otherwise, independently per decision
    state: its EXPECTED value, exactly, without sampling. Both tables must describe the same attempts."""
    if not 0.0 <= share <= 1.0 or table_a['names'] != table_b['names'] or not np.array_equal(table_a['q'], table_b['q']) or not np.array_equal(table_a['active'], table_b['active']):
        raise ValueError('the two policies were not evaluated on the same attempts')
    early = ~table_a['active']
    if not (np.array_equal(table_a['y'][early], table_b['y'][early], equal_nan=True) and np.array_equal(table_a['c'][early], table_b['c'][early])):
        raise ValueError('the attempts that ended early differ between the two policies')
    part = lambda t, rows, factor: {k: (t[k][rows] * factor if k == 'm' else t[k][rows]) for k in ('q', 'm', 'y', 'c', 'active', 'action')}
    pieces = [part(table_a, table_a['active'], share), part(table_b, table_b['active'], 1.0 - share), part(table_a, early, 1.0)]
    return dict({k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}, names=table_a['names'])


def weighted_quantile(values, weights, quantile=QUANTILE):
    """Smallest value whose cumulative weight reaches `quantile` of the total weight."""
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, quantile * cumulative[-1] - 1e-12)])


def _per_question(table, rows):
    q, m, y, c = table['q'][rows], table['m'][rows], table['y'][rows], table['c'][rows]
    count = len(table['names'])
    total = np.bincount(q, m, count)
    present = total > 0
    known = np.isfinite(y)
    share = lambda values: np.where(present, np.bincount(q, m * values, count) / np.where(present, total, 1.0), np.nan)
    known_share = share(known.astype(float))
    correct_known = share(np.where(known, y, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        accuracy = np.where(known_share > 0, correct_known / known_share, np.nan)
    weight = np.where(present[q], m / np.where(present, total, 1.0)[q], 0.0)           # weight of an attempt inside its question
    return {'present': present, 'accuracy_known': accuracy, 'lower': correct_known, 'upper': share(np.where(known, y, 1.0)), 'coverage': known_share,
            'cost': share(c), 'attempt_question': q, 'attempt_weight': weight, 'attempt_cost': c}


def summary(table, population='whole'):
    """Numbers of one policy on one population ('whole' or 'active')."""
    if population not in ('whole', 'active'):
        raise ValueError('unknown population')
    rows = np.ones(len(table['q']), bool) if population == 'whole' else table['active']
    if not rows.any():
        raise ValueError('the population is empty')
    per = _per_question(table, rows)
    present = per['present']
    questions = int(present.sum())
    known_questions = int(np.isfinite(per['accuracy_known'][present]).sum())
    unknown = float((table['m'][rows] * ~np.isfinite(table['y'][rows])).sum())
    return {'population': population,
            'accuracy_known': float(np.nanmean(per['accuracy_known'][present])) if known_questions else float('nan'),
            'accuracy_lower_bound': float(per['lower'][present].mean()), 'accuracy_upper_bound': float(per['upper'][present].mean()),
            'coverage': float(per['coverage'][present].mean()), 'questions': questions, 'questions_with_known_label': known_questions,
            'attempts': float(table['m'][rows].sum()), 'unknown_attempts': unknown,
            'mean_total_cost': float(per['cost'][present].mean()),
            'p90_attempt_cost': weighted_quantile(per['attempt_cost'], per['attempt_weight'] / questions),
            'per_question': per}


def policy_value(actions, success, total_cost, groups, early_success=None, early_cost=None, early_groups=None):
    """Summary of the whole population with the active-only summary, the action shares and the attempt table next to it."""
    table = attempt_table(actions, success, total_cost, groups, early_success, early_cost, early_groups)
    return value_of_table(table)


def value_of_table(table):
    whole, active = summary(table, 'whole'), summary(table, 'active')
    rows = table['active']
    weight = table['m'][rows] / table['m'][rows].sum()
    shares = {a: float(weight[table['action'][rows] == i].sum()) for i, a in enumerate(ACTIONS)}
    return dict(whole, active_only={k: active[k] for k in SUMMARY_KEYS}, action_shares=shares, table=table)


def plain(value):
    """The numbers of a value without its arrays, for a report."""
    return {k: v for k, v in value.items() if k not in ('per_question', 'table')}


def paired_difference(a, b, draws=10000, seed=20261012):
    """Question bootstrap of (policy a - policy b) on the WHOLE population. Both values must come from the same attempts' questions.

    Returns the conditional difference on known labels with both coverages, the population bounds with the interval of the worst
    case (lower bound of a against upper bound of b) and of the best case, the relative mean cost and the tail."""
    pa, pb = a['per_question'], b['per_question']
    if a['table']['names'] != b['table']['names'] or not np.array_equal(pa['present'], pb['present']) or not pa['present'].all():
        raise ValueError('the two policies were not evaluated on the same questions')
    rng = np.random.default_rng(seed)
    n = len(a['table']['names'])
    pick = rng.integers(0, n, size=(draws, n))
    interval = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)             # a draw without any known label gives nan and is counted below
        known = np.nanmean(pa['accuracy_known'][pick], axis=1) - np.nanmean(pb['accuracy_known'][pick], axis=1)
    undefined = int((~np.isfinite(known)).sum())
    worst = pa['lower'][pick].mean(1) - pb['upper'][pick].mean(1)
    best = pa['upper'][pick].mean(1) - pb['lower'][pick].mean(1)
    cost = pa['cost'][pick].mean(1) / pb['cost'][pick].mean(1) - 1.0
    def tail(per):
        order = np.argsort(per['attempt_cost'], kind='stable')
        values, question, weight = per['attempt_cost'][order], per['attempt_question'][order], per['attempt_weight'][order]
        out = np.empty(draws)
        for start in range(0, draws, 500):                           # the questions of a draw carry their multiplicity to their attempts
            counts = np.stack([np.bincount(row, minlength=n) for row in pick[start:start + 500]]).astype(float)
            cumulative = np.cumsum(counts[:, question] * weight[None, :], axis=1)
            out[start:start + 500] = values[(cumulative >= QUANTILE * cumulative[:, -1:] - 1e-12).argmax(1)]
        return out
    tail_difference = tail(pa) - tail(pb)
    return {'accuracy_known_difference': float(a['accuracy_known'] - b['accuracy_known']),
            'accuracy_known_interval': interval(known) if not undefined else None, 'accuracy_known_undefined_draws': undefined,
            'accuracy_known_is': 'conditional on known labels; the known sets of the two policies differ, see coverage',
            'coverage': [a['coverage'], b['coverage']],
            'accuracy_difference_bounds': [float(a['accuracy_lower_bound'] - b['accuracy_upper_bound']), float(a['accuracy_upper_bound'] - b['accuracy_lower_bound'])],
            'worst_case_interval': interval(worst), 'best_case_interval': interval(best),
            'relative_mean_cost': float(a['mean_total_cost'] / b['mean_total_cost'] - 1.0), 'relative_mean_cost_interval': interval(cost),
            'p90_attempt_cost': [a['p90_attempt_cost'], b['p90_attempt_cost']], 'p90_attempt_cost_difference_interval': interval(tail_difference),
            'draws': draws, 'questions': n}


def non_inferior(difference, margin):
    """Quality of the whole population is kept within the margin only when the WHOLE interval of the worst case over the unknown
    labels lies above -margin. A difference that is merely not significant is not non-inferiority, and a conditional difference
    on known labels is not a population difference."""
    return bool(difference['worst_case_interval'][0] > -margin)


def non_inferior_on_known_labels_only(difference, margin):
    """The conditional statement, for the report only: None when the conditional interval is undefined."""
    return None if difference['accuracy_known_interval'] is None else bool(difference['accuracy_known_interval'][0] > -margin)


def superior(difference, gain=0.0):
    """The whole population gains more than `gain` even in the worst case over the unknown labels."""
    return bool(difference['worst_case_interval'][0] > gain)


def permuted_actions(actions, strata, draws=1000, seed=20261012):
    """Diagnostic of the allocation: the chosen actions are permuted between the states of one stratum, frequencies kept; each
    permuted action is then replayed on the recipient's own real branch by policy_value. Returns [draws, N] action indices."""
    actions, strata = np.asarray(actions), np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, len(actions)), dtype=actions.dtype)
    for d in range(draws):
        out[d] = actions
        for name in np.unique(strata):
            members = np.flatnonzero(strata == name)
            out[d, members] = actions[rng.permutation(members)]
    return out
