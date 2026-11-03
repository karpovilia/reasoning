#!/usr/bin/env python3
"""Evaluation of the rows A, B, C of PROCESS_FORECAST_GATE_20261013 in ACTUAL TOKENS of the saved branches. numpy only; no fit, no
model, no seconds. Every rule below is fixed before any result of the experiment exists.
THIRD version (review of the supervisor, PFG-EVAL-V3-R1; evaluate_v2_returned_by_the_review.py is the second): the directory of
the fits is tied to the ACCEPTED start, not only to itself. The PINS file of the admission and its SHA256 are arguments; the
manifest and COMPLETE.json must name that SHA256, their plan sums and member counts must agree with each other and with 45; the
plan must carry the fixed recipe 0.001 / 0.01 / 7 epochs as constants of this file, be a production plan with the width, rows,
seeds and calls of the experiment, and its sums of the configuration, of every input and of the worker and project code must be
the pinned ones; the members are checked against the constant recipe, not against the recipe of the checked plan.
SECOND version (review of the supervisor, PFG-V1-R3 and R4; evaluate_v1_returned_by_the_review.py is the first):
  input   BEFORE the output directory is made, the directory of the fits must be the published complete one: COMPLETE.json,
          MEMBERS_MANIFEST.json with its sum, exactly the 45 member files with the sums of the manifest and no other file, the
          plan of the directory with the sum of the manifest, the token contract with the sum the plan names, the outer folds of
          the plan recomputed from the contract, and every member passing pfgate.members_check with the dev and held-out states
          of the contract. Anything else is refused and nothing is written;
  seeds   the bounds of a number over the seeds are now SHARP: the three policies of a row are averaged per state first, and
          only then the unknown labels are filled the worst and the best way (the first version averaged the three per-seed
          bounds, which is valid but wider: an unknown label of one branch is one label for the three seeds);
  tail    the 0.99 quantile of the output tokens of an attempt and the share of all output tokens spent by the tenth of the
          attempts that spend most are reported next to the median and the 0.9 quantile.

Population. The fixed cohort of the train questions: decision states (the gate chooses one of three actions) and attempts that
ended before the decision point (one outcome whatever the gate does; they call no gate). dev is the validation part of the bank,
used for the temperatures (in the worker) and for the choice of lambda only. A question weighs 1; an attempt weighs
1 / (attempts of its question in the population).
Tokens of an attempt under an action: the totals of the token contract. The shared prefix is in them exactly once; `stop` includes
its final answer. output = reasoning + answer. Tokens of the observers are not in them.
Correctness: an unknown label is 0 in the lower bound and 1 in the upper bound of an accuracy. The difference of two policies is
bounded SHARPLY per attempt: where both take the same action nothing is added; otherwise an unknown label of an action taken by
exactly one of them contributes min(c, 0) to the lower and max(c, 0) to the upper bound (c = difference of the probabilities of
that action). These are bounds from unknown labels, not confidence intervals.
Decision of a member: argmax over the actions of q_a - lambda * m_a / 1024, q_a the calibrated probability of a correct final
answer, m_a the forecast additional output tokens; a tie goes to the earlier action in the order stop, continue_512, continue_2048.
Choice of lambda, per member (row, seed, outer call), on dev only, for each tolerance 0, 0.01 and 0.02 separately: among the
fixed grid {0} and 10^(-3 + i/10), i = 0..40, the values whose dev policy has a sharp paired LOWER bound of the accuracy difference
against always continue_2048 of at least -tolerance are admissible; the admissible value with the smallest mean output tokens
on dev is taken (a tie: the smaller lambda). No admissible value: the call is marked NO_ADMISSIBLE_LAMBDA and its held-out states
take continue_2048; a row with such a call is never reported as meeting a criterion.
Cross-fitted policy of (row, seed, tolerance): every state takes the action of the member whose call held its question out.
Fixed mixture: for each tolerance, the pair of actions and the share on the grid 0, 0.05, .., 1 chosen by the same rule on dev.
Intervals: 10000 resamples of whole questions (numpy default_rng(20261013)), percentiles 2.5 and 97.5; they describe the fixed
fits and the fixed choices on dev and do not include refitting. Seeds are paired across the rows; a number 'over seeds' is the
mean of the three per-seed numbers, that is the value of the policy that takes each seed's action with probability 1/3; its
bounds are the sharp bounds of that averaged policy, its interval resamples the per-question values of the averaged policy.
  evaluate.py TOKEN_CONTRACT.json FITS_DIR PINS.json PINS_SHA256 OUT_DIR"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import members_check  # noqa: E402

ACTIONS = ('stop', 'continue_512', 'continue_2048')
ROWS, SEEDS, FOLDS = ('A', 'B', 'C'), (17, 29, 43), 5
TOLERANCES = (0.0, 0.01, 0.02)
LAMBDAS = (0.0,) + tuple(float(10 ** (-3 + i / 10)) for i in range(41))
SHARES = tuple(round(0.05 * i, 2) for i in range(21))
DRAWS, SEED = 10000, 20261013
REFERENCE = 2                                                    # always continue_2048: no control
RECIPE = {'learning_rate': 0.001, 'weight_decay': 0.01, 'epochs': 7}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Population:
    """Decision states and early attempts of one part of the bank, with question weights."""

    def __init__(self, attempts):
        states, early = [a for a in attempts if not a['ended_before_the_decision']], [a for a in attempts if a['ended_before_the_decision']]
        self.states, self.early = states, early
        self.questions = sorted({a['group'] for a in attempts})
        index = {q: i for i, q in enumerate(self.questions)}
        self.Q = len(self.questions)
        self.qs, self.qe = np.array([index[a['group']] for a in states], int), np.array([index[a['group']] for a in early], int)
        per = np.bincount(np.concatenate([self.qs, self.qe]), minlength=self.Q)
        self.ws, self.we = 1.0 / per[self.qs], 1.0 / per[self.qe]
        field = lambda rows, name: np.array([[a['actions'][x][name] for x in ACTIONS] for a in rows], float).reshape(len(rows), 3)
        label = lambda rows: np.array([[np.nan if a['actions'][x]['correct'] is None else float(a['actions'][x]['correct']) for x in ACTIONS] for a in rows], float).reshape(len(rows), 3)
        self.Y, self.known = label(states), np.isfinite(label(states))
        self.Y0, self.Y1 = np.where(self.known, self.Y, 0.0), np.where(self.known, self.Y, 1.0)
        self.tokens = {name: field(states, key) for name, key in (('output', 'total_output_tokens'), ('reasoning', 'total_reasoning_tokens'), ('answer', 'answer_tokens'))}
        ye = label(early)[:, 0] if early else np.zeros(0)
        self.early_lower = np.bincount(self.qe, self.we * np.where(np.isfinite(ye), ye, 0.0), self.Q)
        self.early_upper = np.bincount(self.qe, self.we * np.where(np.isfinite(ye), ye, 1.0), self.Q)
        self.early_tokens = {name: (field(early, key)[:, 0] if early else np.zeros(0)) for name, key in (('output', 'total_output_tokens'), ('reasoning', 'total_reasoning_tokens'), ('answer', 'answer_tokens'))}
        self.key_index = {a['key']: i for i, a in enumerate(states)}
        self.benchmark = {q: next(a['benchmark'] for a in attempts if a['group'] == q) for q in self.questions}

    def per_question(self, P):
        """Per question: accuracy lower, upper, mean output, reasoning, answer tokens of the policy P [states, 3]."""
        out = {'lower': np.bincount(self.qs, self.ws * (P * self.Y0).sum(1), self.Q) + self.early_lower, 'upper': np.bincount(self.qs, self.ws * (P * self.Y1).sum(1), self.Q) + self.early_upper}
        for name, table in self.tokens.items():
            out[name] = np.bincount(self.qs, self.ws * (P * table).sum(1), self.Q) + np.bincount(self.qe, self.we * self.early_tokens[name], self.Q)
        return out

    def paired(self, P1, P2):
        """Per question: sharp lower and upper bound of accuracy(P1) - accuracy(P2)."""
        c = P1 - P2
        return (np.bincount(self.qs, self.ws * np.where(self.known, c * self.Y0, np.minimum(c, 0.0)).sum(1), self.Q),
                np.bincount(self.qs, self.ws * np.where(self.known, c * self.Y0, np.maximum(c, 0.0)).sum(1), self.Q))

    def attempt_quantiles(self, P, name='output', levels=(0.5, 0.9, 0.99)):
        """Weighted quantiles of the tokens of an attempt under P (every branch of a state weighs its probability), and the share
        of all tokens spent by the tenth of the attempts, by weight, that spend most."""
        values = np.concatenate([self.tokens[name].ravel(), self.early_tokens[name]])
        weights = np.concatenate([(self.ws[:, None] * P).ravel(), self.we])
        order = np.argsort(values, kind='stable')
        v, w = values[order], weights[order]
        cumulative = np.cumsum(w)
        quantiles = [float(v[np.searchsorted(cumulative, level * cumulative[-1] - 1e-12)]) for level in levels]
        in_top = np.clip(cumulative - 0.9 * cumulative[-1], 0.0, w)            # the weight of every branch that lies in the top tenth
        return quantiles, float((in_top * v).sum() / (w * v).sum())


one_hot = lambda actions: np.eye(3)[np.asarray(actions, int)]
constant = lambda n, action: np.tile(np.eye(3)[action], (n, 1))


def decide(q, m, lam):
    """Actions [n] of a member for the price lam: argmax of q - lam * m / 1024, ties to the earlier action."""
    return np.argmax(np.asarray(q, float) - lam * np.asarray(m, float) / 1024.0, axis=1)


def choose(dev, candidates, tolerance):
    """The admissible candidate with the smallest mean output tokens on dev. candidates: [(label, P on dev)] in the order of
    preference on ties. Returns (index or None, table)."""
    reference = constant(len(dev.states), REFERENCE)
    table, best = [], None
    for i, (label, P) in enumerate(candidates):
        lower = float(dev.paired(P, reference)[0].mean())
        tokens = float(dev.per_question(P)['output'].mean())
        ok = lower >= -tolerance - 1e-12
        table.append({'candidate': label, 'dev_paired_lower_bound': lower, 'dev_mean_output_tokens': tokens, 'admissible': ok})
        if ok and (best is None or tokens < table[best]['dev_mean_output_tokens'] - 1e-9):
            best = i
    return best, table


def summary(pop, P, draws):
    v = pop.per_question(P)
    (median, p90, p99), top = pop.attempt_quantiles(P)
    out = {'accuracy_lower_bound': float(v['lower'].mean()), 'accuracy_upper_bound': float(v['upper'].mean()), 'mean_output_tokens': float(v['output'].mean()), 'mean_reasoning_tokens': float(v['reasoning'].mean()),
           'mean_answer_tokens': float(v['answer'].mean()), 'median_output_tokens_of_an_attempt': median, 'p90_output_tokens_of_an_attempt': p90, 'p99_output_tokens_of_an_attempt': p99,
           'share_of_output_tokens_spent_by_the_top_tenth_of_attempts': top,
           'action_shares_weighted': (pop.ws @ P / pop.ws.sum()).tolist()}
    if draws is not None:
        out['mean_output_tokens_interval_95'] = [float(x) for x in np.quantile(v['output'][draws].mean(1), (0.025, 0.975))]
    return out, v


def contrast(first, second, bounds, draws):
    """first minus second from per-question arrays (already averaged over seeds when needed)."""
    lower, upper = bounds
    saving = lambda name: 1.0 - first[name][draws].mean(1) / second[name][draws].mean(1)
    out = {'accuracy_difference_lower_bound': float(lower.mean()), 'accuracy_difference_upper_bound': float(upper.mean()),
           'lower_bound_2.5_percentile_over_questions': float(np.quantile(lower[draws].mean(1), 0.025)), 'upper_bound_97.5_percentile_over_questions': float(np.quantile(upper[draws].mean(1), 0.975))}
    for name in ('output', 'reasoning'):
        out['saving_of_%s_tokens' % name] = float(1.0 - first[name].mean() / second[name].mean())
        out['saving_of_%s_tokens_interval_95' % name] = [float(x) for x in np.quantile(saving(name), (0.025, 0.975))]
        out['mean_%s_tokens_first' % name], out['mean_%s_tokens_second' % name] = float(first[name].mean()), float(second[name].mean())
    return out


def restricted(pop, arrays, questions):
    return {k: v[questions] for k, v in arrays.items()}


def checked_members(contract_path, contract, fits_dir, pins_path, pins_sha):
    """The 45 parsed members of the published complete directory of the ACCEPTED start, or SystemExit. Writes nothing."""
    fits = Path(fits_dir)
    if not Path(pins_path).is_file() or sha(pins_path) != pins_sha:
        raise SystemExit('the PINS file is not the one named on the command line')
    pins = json.loads(Path(pins_path).read_text())
    files, roles = pins['files'], pins['roles']
    if pins.get('format') != 'pfgate_pins_v1' or pins.get('recipe') != RECIPE or os.path.realpath(pins['out']) != os.path.realpath(str(fits)):
        raise SystemExit('the PINS file is not for this directory of fits with the fixed recipe')
    for role in ('config', 'gate_table_manifest', 'token_contract', 'members_check'):
        if not Path(roles[role]).is_file() or sha(roles[role]) != files[roles[role]]:
            raise SystemExit('the pinned %s changed on the disk' % role)
    if os.path.realpath(members_check.__file__) != os.path.realpath(roles['members_check']) or os.path.realpath(contract_path) != os.path.realpath(roles['token_contract']):
        raise SystemExit('the checker of the members or the token contract is not the pinned file')
    try:
        complete, manifest, plan = (json.loads((fits / name).read_text()) for name in ('COMPLETE.json', 'MEMBERS_MANIFEST.json', 'PLAN.json'))
    except (OSError, ValueError) as error:
        raise SystemExit('the directory of the fits is not a published complete one: %s' % error)
    names = members_check.expected_names()
    plan_sha = sha(fits / 'PLAN.json')
    agree = (complete.get('state') == 'COMPLETE' and complete.get('members_manifest_sha256') == sha(fits / 'MEMBERS_MANIFEST.json') and complete.get('plan_sha256') == plan_sha == manifest.get('plan_sha256')
             and complete.get('pins_sha256') == pins_sha == manifest.get('pins_sha256') and complete.get('members') == manifest.get('count') == len(names) == 45 and sorted(manifest.get('members', {})) == sorted(names)
             and manifest.get('recipe') == RECIPE and manifest.get('members_check_sha256') == files[roles['members_check']] and manifest.get('worker_sha256') == files[roles['worker']])
    if not agree:
        raise SystemExit('COMPLETE.json, MEMBERS_MANIFEST.json and PLAN.json do not agree with each other and with the accepted PINS')
    present = sorted(p.name for p in (fits / 'members').iterdir())
    if present != sorted(n + '.json' for n in names):
        raise SystemExit('the member directory does not hold exactly the 45 member files')
    wrong = [n for n in names if sha(fits / 'members' / (n + '.json')) != manifest['members'][n]]
    if wrong:
        raise SystemExit('member files differ from the manifest: %s' % wrong[:3])
    config = json.loads(Path(roles['config']).read_text())
    table = json.loads(Path(roles['gate_table_manifest']).read_text())
    seen = lambda spans: sum(b - a for a, b in spans)
    columns = {'A': seen(table['rows']['A']), 'B': seen(table['rows']['B']), **{'C_s%d' % s: seen(table['rows']['C'][str(s)]) for s in SEEDS}}
    project = os.path.realpath(config['project_code'])
    code = {'worker': files[roles['worker']]}
    code.update({'project/' + os.path.relpath(os.path.realpath(p), project): v for p, v in files.items() if os.path.realpath(p).startswith(project + os.sep)})
    expected = {'recipe': RECIPE, 'rehearsal_random_outcomes': False, 'width': config['width'], 'rows': list(ROWS), 'seeds': list(SEEDS), 'calls': ['outer_%d' % k for k in range(FOLDS)], 'members': names, 'works': 45,
                'columns_seen_by_a_source': columns, 'code_sha256': code, 'outer_salt': config['outer_salt'], 'outer_folds_sha256': config['expected_outer_folds_sha256'],
                'sums': {'config': files[roles['config']], 'features': files[roles['gate_table']], 'token_contract': files[roles['token_contract']], 'cohort': files[roles['cohort']], 'trainval': files[roles['trainval']],
                         'replay': files[roles['replay']], 'cost_freeze': files[roles['cost_freeze']]}}
    differ = [k for k, v in expected.items() if plan.get(k) != v]
    if differ or config.get('recipe') != RECIPE or config.get('rehearsal_random_outcomes') or table['width'] != config['width']:
        raise SystemExit('the plan of the fits is not the plan of the accepted start; fields that differ: %s' % differ)
    cohort = [a for a in contract['attempts'] if a['cohort']]
    questions = sorted({a['group'] for a in cohort})
    ordered = sorted(questions, key=lambda g: hashlib.sha256(('%s|%s' % (plan['outer_salt'], g)).encode()).hexdigest())
    outer = {g: i % FOLDS for i, g in enumerate(ordered)}
    if hashlib.sha256(json.dumps(sorted(outer.items())).encode()).hexdigest() != plan['outer_folds_sha256'] or any(a['outer_fold'] != outer[a['group']] for a in cohort if not a['ended_before_the_decision']):
        raise SystemExit('the outer folds of the plan are not the folds of the token contract')
    dev_keys = [a['key'] for a in contract['attempts'] if a['split'] == 'validation' and not a['ended_before_the_decision']]
    held = {fold: [a['key'] for a in cohort if not a['ended_before_the_decision'] and a['outer_fold'] == fold] for fold in range(FOLDS)}
    accepted, problems = members_check.check_directory(fits, RECIPE, dev_keys, held, require_all=True)      # the constant recipe, not the recipe of the checked plan
    if problems or accepted != manifest['members']:
        raise SystemExit('members fail the check: %s' % problems[:5])
    return {n: json.loads((fits / 'members' / (n + '.json')).read_text()) for n in names}, manifest, plan


def main(argv):
    contract_path, fits_dir, pins_path, pins_sha, out_dir = argv
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit('the output directory exists')
    contract = json.loads(Path(contract_path).read_text())
    parsed, manifest, plan = checked_members(contract_path, contract, fits_dir, pins_path, pins_sha)      # before anything is written
    out_dir.mkdir()
    train = Population([a for a in contract['attempts'] if a['cohort']])
    dev = Population([a for a in contract['attempts'] if a['split'] == 'validation'])
    fold_of_state = np.array([a['outer_fold'] for a in train.states], int)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, train.Q, size=(DRAWS, train.Q))
    members = {(m['row'], m['seed'], int(m['call'].split('_')[1])): m for m in parsed.values()}
    if len(members) != len(parsed) or set(members) != {(row, seed, fold) for row in ROWS for seed in SEEDS for fold in range(FOLDS)}:
        raise SystemExit('the members are not exactly the 45 expected ones')
    reference_train, reference_dev = constant(len(train.states), REFERENCE), constant(len(dev.states), REFERENCE)
    result = {'format': 'pfgate_evaluation_v1', 'what_this_is': __doc__, 'token_contract_sha256': sha(contract_path), 'this_file_sha256': sha(__file__), 'lambdas': list(LAMBDAS), 'tolerances': list(TOLERANCES),
              'train': {'questions': train.Q, 'decision_states': len(train.states), 'early_attempts': len(train.early)}, 'dev': {'questions': dev.Q, 'decision_states': len(dev.states), 'early_attempts': len(dev.early)},
              'members_manifest_sha256': sha(Path(fits_dir) / 'MEMBERS_MANIFEST.json'), 'plan_sha256': manifest['plan_sha256'], 'pins_sha256': manifest.get('pins_sha256'), 'recipe': plan['recipe'], 'by_tolerance': {}}
    fixed = {'always_' + a: constant(len(train.states), i) for i, a in enumerate(ACTIONS)}
    fixed['uniform_thirds'] = np.full((len(train.states), 3), 1 / 3)
    result['fixed_policies'] = {name: summary(train, P, draws)[0] for name, P in fixed.items()}
    known = np.where(train.known, train.Y, -1.0)
    retrospective = one_hot(np.where((known >= 0).any(1), known.argmax(1), 0))
    result['retrospective_reference_by_the_known_outcomes_not_a_policy'] = summary(train, retrospective, None)[0]
    ref_arrays = train.per_question(reference_train)
    outcomes = []
    for tolerance in TOLERANCES:
        block = {'rows': {}, 'contrasts': {}, 'choices': {}}
        mixtures = [('%.2f %s + %.2f %s' % (w, ACTIONS[i], 1 - w, ACTIONS[j]), (i, j, w)) for i in range(3) for j in range(i + 1, 3) for w in SHARES]
        mix_p = lambda spec, n: np.tile(spec[2] * np.eye(3)[spec[0]] + (1 - spec[2]) * np.eye(3)[spec[1]], (n, 1))
        best, table = choose(dev, [(label, mix_p(spec, len(dev.states))) for label, spec in mixtures], tolerance)
        mixture_train = mix_p(mixtures[best][1], len(train.states)) if best is not None else reference_train
        block['fixed_mixture'] = {'chosen': mixtures[best][0] if best is not None else None, 'dev': table[best] if best is not None else None, 'value': summary(train, mixture_train, draws)[0]}
        mixture_arrays = train.per_question(mixture_train)
        policies, per_seed_arrays = {}, {}
        for row in ROWS:
            block['choices'][row] = []
            for seed in SEEDS:
                actions = np.full(len(train.states), -1, int)
                for fold in range(FOLDS):
                    m = members[(row, seed, fold)]
                    devp, held = m['parts']['dev'], m['parts']['held_out']
                    if sorted(devp['keys']) != sorted(dev.key_index) or len(set(devp['keys'])) != len(devp['keys']):
                        raise SystemExit('the dev predictions of %s are not on exactly the dev states' % m['name'])
                    order = np.argsort([dev.key_index[k] for k in devp['keys']])            # the rows of the member put into the order of the dev states
                    dq, dm = np.asarray(devp['success'], float)[order], np.asarray(devp['cost'], float)[order]
                    best, table = choose(dev, [(lam, one_hot(decide(dq, dm, lam))) for lam in LAMBDAS], tolerance)
                    index = [train.key_index[k] for k in held['keys']]
                    if not (fold_of_state[index] == fold).all() or (actions[index] != -1).any():
                        raise SystemExit('the held-out states of %s are not the states of its fold' % m['name'])
                    actions[index] = decide(held['success'], held['cost'], LAMBDAS[best]) if best is not None else REFERENCE
                    block['choices'][row].append({'seed': seed, 'call': fold, 'status': 'ADMISSIBLE' if best is not None else 'NO_ADMISSIBLE_LAMBDA', 'lambda': LAMBDAS[best] if best is not None else None,
                                                  'dev': table[best] if best is not None else None, 'admissible_lambdas': sum(t['admissible'] for t in table), 'held_out_states': len(index)})
                    for k, i in zip(held['keys'], index):
                        a = train.states[i]
                        outcomes.append([tolerance, row, seed, k, a['group'], a['benchmark'], 'decision state', fold, ACTIONS[actions[i]], a['actions'][ACTIONS[actions[i]]]['correct'],
                                         a['actions'][ACTIONS[actions[i]]]['total_output_tokens'], a['actions'][ACTIONS[actions[i]]]['total_reasoning_tokens'], a['actions'][ACTIONS[actions[i]]]['answer_tokens'],
                                         LAMBDAS[best] if best is not None else '', 'ADMISSIBLE' if best is not None else 'NO_ADMISSIBLE_LAMBDA'])
                if (actions < 0).any():
                    raise SystemExit('a state of the cohort got no decision for %s seed %d' % (row, seed))
                policies[(row, seed)] = one_hot(actions)
                per_seed_arrays[(row, seed)] = train.per_question(policies[(row, seed)])
        mean_policy = {row: np.mean([policies[(row, s)] for s in SEEDS], 0) for row in ROWS}      # each seed's action with probability 1/3
        mean_arrays = {row: train.per_question(mean_policy[row]) for row in ROWS}                 # bounds of the averaged policy: one label per branch for all seeds
        for row in ROWS:
            calls = block['choices'][row]
            by_seed = {str(s): dict(summary(train, policies[(row, s)], draws)[0], against_no_control=contrast(per_seed_arrays[(row, s)], ref_arrays, train.paired(policies[(row, s)], reference_train), draws)) for s in SEEDS}
            over = {'accuracy_lower_bound': float(mean_arrays[row]['lower'].mean()), 'accuracy_upper_bound': float(mean_arrays[row]['upper'].mean()), 'mean_output_tokens': float(mean_arrays[row]['output'].mean()),
                    'mean_reasoning_tokens': float(mean_arrays[row]['reasoning'].mean()), 'mean_answer_tokens': float(mean_arrays[row]['answer'].mean())}
            bounds = train.paired(mean_policy[row], reference_train)
            against = contrast(mean_arrays[row], ref_arrays, bounds, draws)
            mix_bounds = train.paired(mean_policy[row], mixture_train)
            every_call = all(c['status'] == 'ADMISSIBLE' for c in calls)
            block['rows'][row] = {'by_seed': by_seed, 'over_seeds': over, 'against_no_control_over_seeds': against, 'against_the_fixed_mixture_over_seeds': contrast(mean_arrays[row], mixture_arrays, mix_bounds, draws),
                                  'calls_without_an_admissible_lambda': sum(c['status'] != 'ADMISSIBLE' for c in calls), 'every_call_admissible': every_call,
                                  'criterion_10_percent_output_tokens_within_the_tolerance': {
                                      'point': bool(every_call and against['saving_of_output_tokens'] >= 0.10 and against['accuracy_difference_lower_bound'] >= -tolerance),
                                      'with_intervals': bool(every_call and against['saving_of_output_tokens_interval_95'][0] >= 0.10 and against['lower_bound_2.5_percentile_over_questions'] >= -tolerance),
                                      'note': 'the tolerance of this block; a call without an admissible lambda makes both false'},
                                  'by_benchmark': {}}
            for name in sorted(set(train.benchmark.values())):
                qs = np.array([i for i, q in enumerate(train.questions) if train.benchmark[q] == name])
                sub = rng.integers(0, len(qs), size=(2000, len(qs)))
                block['rows'][row]['by_benchmark'][name] = dict(questions=len(qs), **contrast(restricted(train, mean_arrays[row], qs), restricted(train, ref_arrays, qs), tuple(b[qs] for b in bounds), sub))
        for first, second in (('C', 'B'), ('B', 'A'), ('C', 'A')):
            bounds = train.paired(mean_policy[first], mean_policy[second])
            block['contrasts']['%s_minus_%s' % (first, second)] = dict(
                over_seeds=contrast(mean_arrays[first], mean_arrays[second], bounds, draws),
                by_seed={str(s): contrast(per_seed_arrays[(first, s)], per_seed_arrays[(second, s)], train.paired(policies[(first, s)], policies[(second, s)]), draws) for s in SEEDS},
                states_where_the_two_differ_weighted_share_by_seed={str(s): float((train.ws * (policies[(first, s)] != policies[(second, s)]).any(1)).sum() / train.ws.sum()) for s in SEEDS})
        result['by_tolerance']['%.2f' % tolerance] = block
    for a in train.early:                                         # an early attempt: one outcome, no gate, the same in every row
        b = a['actions']['stop']
        outcomes.append(['', 'every row', '', a['key'], a['group'], a['benchmark'], 'early attempt', '', 'none', b['correct'], b['total_output_tokens'], b['total_reasoning_tokens'], b['answer_tokens'], '', ''])
    with open(out_dir / 'POLICY_OUTCOMES.csv', 'x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tolerance', 'row', 'seed', 'attempt', 'question', 'benchmark', 'kind', 'outer_fold', 'action', 'correct', 'output_tokens', 'reasoning_tokens', 'answer_tokens', 'lambda', 'call_status'])
        writer.writerows(outcomes)
    result['policy_outcomes_sha256'] = sha(out_dir / 'POLICY_OUTCOMES.csv')
    with open(out_dir / 'EVALUATION.json', 'x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    main_block = result['by_tolerance']['0.02']
    print(json.dumps({'reference': result['fixed_policies']['always_continue_2048'], 'rows_at_2_points': {r: main_block['rows'][r]['against_no_control_over_seeds'] for r in ROWS},
                      'contrasts_at_2_points': {k: v['over_seeds'] for k, v in main_block['contrasts'].items()}, 'sha256': sha(out_dir / 'EVALUATION.json')}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
