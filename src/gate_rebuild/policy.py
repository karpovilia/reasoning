"""Choice of one action from the estimates, the selection of the price on dev, and the fixed mixture.

Rule: among the actions that are supported and admissible under the remaining budget, take the one that maximises
p_a - price * c_a, where c_a is the predicted marginal cost of the action IN THE UNIT OF THE COST MODEL (review finding
GATE-V1-R1: tokens are never passed here). What a policy pays for looking is the same for every action of a state, so it does
not enter the argmax; it enters the VALUE of the policy. A missing or non-finite estimate never becomes a confident stop: the
fallback action is returned with a flag.

Selection on dev (review findings GATE-V1-R2 and R3). A price is feasible when the mean full cost AND the 0.9 quantile of the
cost of one attempt are within their ceilings; both ceilings are required arguments. Among the feasible prices the one with the
best value of the DECLARED estimand is taken:
  'lower_bound'        accuracy with every unknown label counted wrong (the default: a policy is never rewarded for choosing
                       actions whose answers could not be graded);
  'known_conditional'  accuracy on known labels; allowed only as a declared alternative, the table keeps the coverage.
Ties go to the cheaper policy, then to the larger price. Every row of the table keeps both bounds, the conditional accuracy
and the coverage.
"""
from __future__ import annotations

import math

import numpy as np

from .contract import ACTIONS, FINAL_ANSWER_CAP, REASONING_ALLOWANCE
from .evaluation import SUMMARY_KEYS, mix, value_of_table

PRICE_GRID = (0.0,) + tuple(float(v) for v in np.logspace(-5, 1, 73))       # per unit of cost
ESTIMANDS = {'lower_bound': 'accuracy_lower_bound', 'known_conditional': 'accuracy_known'}
MIXTURE_SHARES = tuple(round(i / 20., 2) for i in range(21))


def admissible(remaining_tokens):
    """Actions whose reasoning allowance and final answer cap fit into what is left of the total token budget (None: no limit)."""
    if remaining_tokens is None:
        return np.ones(len(ACTIONS), bool)
    return np.array([REASONING_ALLOWANCE[a] + FINAL_ANSWER_CAP <= remaining_tokens for a in ACTIONS])


def choose(success, cost, price, allowed=None, fallback='continue_2048'):
    """(action index, utility, used_fallback). cost is in the unit of the cost model."""
    p, c = np.asarray(success, float), np.asarray(cost, float)
    mask = np.ones(len(ACTIONS), bool) if allowed is None else np.asarray(allowed, bool)
    if p.shape != (len(ACTIONS),) or c.shape != p.shape or mask.shape != p.shape or not math.isfinite(price) or price < 0:
        raise ValueError('invalid input of the action choice')
    if not mask.any():
        raise ValueError('no allowed action')
    if not np.isfinite(p[mask]).all() or not np.isfinite(c[mask]).all() or ((p[mask] < 0) | (p[mask] > 1)).any() or (c[mask] < 0).any():
        index = ACTIONS.index(fallback) if mask[ACTIONS.index(fallback)] else int(np.flatnonzero(mask)[-1])
        return index, float('nan'), True
    utility = np.where(mask, p - price * c, -np.inf)
    index = int(np.argmax(utility))
    return index, float(utility[index]), False


def decide(prediction, price, supported, remaining_tokens=None, fallback='continue_2048'):
    """Vector of chosen action indices for a batch of states."""
    allowed = admissible(remaining_tokens)
    return np.array([choose(prediction['success'][i], prediction['cost'][i], price, supported[i] & allowed, fallback)[0] for i in range(len(supported))])


def _feasible(value, mean_ceiling, tail_ceiling):
    return bool(value['mean_total_cost'] <= mean_ceiling + 1e-9 and value['p90_attempt_cost'] <= tail_ceiling + 1e-9)


def _check(mean_ceiling, tail_ceiling, estimand):
    if estimand not in ESTIMANDS:
        raise ValueError('the selection estimand must be declared: one of %s' % sorted(ESTIMANDS))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or v < 0 for v in (mean_ceiling, tail_ceiling)):
        raise ValueError('both the mean ceiling and the tail ceiling of the full cost are required')


def _order(row, estimand):
    return (row[ESTIMANDS[estimand]], -row['mean_total_cost'])


def select_price(prediction, supported, value_of, mean_ceiling, tail_ceiling, estimand='lower_bound', grid=PRICE_GRID):
    """Price fixed on DEV. value_of(actions) -> a value of evaluation.policy_value on the dev attempts. Returns the price (None
    when no price is feasible: no silent choice) and the table of all prices."""
    _check(mean_ceiling, tail_ceiling, estimand)
    table = []
    for price in grid:
        value = value_of(decide(prediction, price, supported))
        table.append(dict({k: value[k] for k in SUMMARY_KEYS}, price=price, feasible=_feasible(value, mean_ceiling, tail_ceiling)))
    candidates = [row for row in table if row['feasible'] and math.isfinite(row[ESTIMANDS[estimand]])]
    if not candidates:
        return None, table
    best = max(candidates, key=lambda row: _order(row, estimand) + (row['price'],))
    return best['price'], table


def select_mixture(fixed_tables, mean_ceiling, tail_ceiling, estimand='lower_bound', shares=MIXTURE_SHARES):
    """The strongest FIXED policy under the same ceilings: one action for everybody, or a randomised mixture of two actions with a
    share from the grid. fixed_tables: {action name: attempt table of that action for every decision state}. Nothing of a state
    is read. Returns ({'actions': (first, second), 'share_of_first': s, 'value': value} or None, table)."""
    _check(mean_ceiling, tail_ceiling, estimand)
    if sorted(fixed_tables) != sorted(ACTIONS):
        raise ValueError('the mixture needs the table of every fixed action')
    entries = [((action, action), 1.0) for action in ACTIONS]                     # each pure action once
    entries += [((first, second), share) for i, first in enumerate(ACTIONS) for second in ACTIONS[i + 1:] for share in shares if 0.0 < share < 1.0]
    table, values = [], {}
    for (first, second), share in entries:
        value = value_of_table(fixed_tables[first] if first == second else mix(fixed_tables[first], fixed_tables[second], share))
        values[(first, second, share)] = value
        table.append(dict({k: value[k] for k in SUMMARY_KEYS}, actions=[first, second], share_of_first=share, feasible=_feasible(value, mean_ceiling, tail_ceiling)))
    candidates = [row for row in table if row['feasible'] and math.isfinite(row[ESTIMANDS[estimand]])]
    if not candidates:
        return None, table
    best = max(candidates, key=lambda row: _order(row, estimand) + (abs(row['share_of_first'] - 0.5),))       # last tie: the purer mixture
    key = (best['actions'][0], best['actions'][1], best['share_of_first'])
    return {'actions': tuple(best['actions']), 'share_of_first': best['share_of_first'], 'value': values[key]}, table
