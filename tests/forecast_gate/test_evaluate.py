#!/usr/bin/env python3
"""Checks of pfgate/evaluate.py (second version) on a small artificial bank, numpy only.
Numbers: question weights and early attempts in every number; an unknown label is 0 below and 1 above; identical members give
identical rows and zero contrasts; the price rule on dev; a call without an admissible price falls back to no control and fails
the criterion; the order of the dev rows of a member does not matter.
Seeds (PFG-V1-R4): two rows whose seeds take the same actions in another order over a state with unknown labels have the sharp
contrast [0, 0], not [-2/3, +2/3]; with all labels known the bounds collapse to the plain difference.
Accepted start (PFG-EVAL-V3-R1): a directory that agrees with itself but has one epoch everywhere, another PINS, another table,
configuration or code in its plan, or contradictory fields of COMPLETE.json, is refused before any output.
Input (PFG-V1-R3): a directory without COMPLETE.json, with a plan of another token contract, with a member of another plan, with
one epoch, with a 46th file, with NaN, with a changed member after the manifest: all refused BEFORE the output directory is made."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import evaluate as E, members_check as M  # noqa: E402

SALT = 'test-salt'
RECIPE = {'learning_rate': 0.001, 'weight_decay': 0.01, 'epochs': 7}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fold_of(groups):
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(('%s|%s' % (SALT, g)).encode()).hexdigest())
    return {g: i % 5 for i, g in enumerate(ordered)}


def attempt(key, group, split, early, fold, correct, extra):
    prefix = 600 if early else 1024
    actions = {}
    for i, a in enumerate(E.ACTIONS):
        j = 0 if early else i
        actions[a] = {'correct': correct[j], 'total_output_tokens': prefix + extra[j], 'total_reasoning_tokens': prefix + max(extra[j] - 40, 0), 'answer_tokens': min(extra[j], 40), 'additional_output_tokens': extra[j]}
    return {'key': key, 'group': group, 'benchmark': 'b' + group[-1], 'split': split, 'ended_before_the_decision': early, 'cohort': ('early attempt' if early else 'decision state') if split == 'train' else None,
            'outer_fold': fold if not early and split == 'train' else None, 'actions': actions}


def bank(labels=None):
    rng = np.random.default_rng(0)
    folds = fold_of(['g%d' % q for q in range(20)])
    rows = []
    for q in range(20):
        for s in (1, 2):
            early = (q + s) % 5 == 0
            label = labels or [bool(rng.integers(0, 2)), bool(rng.integers(0, 2)), True if q % 7 else None]
            rows.append(attempt('q%d|%d' % (q, s), 'g%d' % q, 'train', early, folds['g%d' % q], label, [40, 300, 1500]))
    for q in range(20, 28):
        for s in (1, 2):
            rows.append(attempt('q%d|%d' % (q, s), 'g%d' % q, 'validation', (q + s) % 6 == 0, None, [False, s == 1, True], [40, 300, 1500]))
    return {'format': 'process_forecast_gate_token_contract_v1', 'actions': list(E.ACTIONS), 'attempts': rows}


ROWS_OF_THE_TABLE = {'A': [[0, 2], [4, 6]], 'B': [[0, 6]], 'C': {'17': [[0, 7]], '29': [[0, 6], [7, 8]], '43': [[0, 6], [8, 9]]}}


def publish(contract, t, q_of, shuffle=False, change=None, extra_file=False, manifest_first=None, complete=True, plan_change=None, epochs=7, manifest_change=None, complete_change=None, pins_out=None):
    """A published directory of fits for the artificial bank with its PINS file. q_of(row, seed, attempt) -> three probabilities.
    Everything written here is CONSISTENT with itself, also when `epochs` or a sum of the plan is changed: that is the point of
    the checks against the accepted PINS."""
    (t / 'contract.json').write_text(json.dumps(contract))
    fits = t / 'fits'
    (fits / 'members').mkdir(parents=True)
    (t / 'project' / 'gate_rebuild').mkdir(parents=True)
    (t / 'project' / 'gate_rebuild' / 'training.py').write_text('X = 1\n')
    for name in ('table.npz', 'cohort.json', 'trainval.jsonl', 'replay.jsonl', 'cost.json', 'worker.py'):
        (t / name).write_text('content of ' + name)
    (t / 'table.npz.manifest.json').write_text(json.dumps({'rows': ROWS_OF_THE_TABLE, 'width': 9}))
    train = [a for a in contract['attempts'] if a['cohort'] == 'decision state']
    dev = [a for a in contract['attempts'] if a['split'] == 'validation' and not a['ended_before_the_decision']]
    folds = fold_of([a['group'] for a in contract['attempts'] if a['cohort']])
    folds_sha = hashlib.sha256(json.dumps(sorted(folds.items())).encode()).hexdigest()
    (t / 'config.json').write_text(json.dumps({'width': 9, 'project_code': str(t / 'project'), 'outer_salt': SALT, 'expected_outer_folds_sha256': folds_sha, 'recipe': RECIPE, 'features': str(t / 'table.npz')}))
    roles = {'config': t / 'config.json', 'worker': t / 'worker.py', 'members_check': Path(M.__file__).resolve(), 'gate_table': t / 'table.npz', 'gate_table_manifest': t / 'table.npz.manifest.json',
             'token_contract': t / 'contract.json', 'cohort': t / 'cohort.json', 'trainval': t / 'trainval.jsonl', 'replay': t / 'replay.jsonl', 'cost_freeze': t / 'cost.json'}
    files = {str(p): sha(p) for p in list(roles.values()) + [t / 'project' / 'gate_rebuild' / 'training.py']}
    (t / 'pins.json').write_text(json.dumps({'format': 'pfgate_pins_v1', 'files': files, 'roles': {k: str(v) for k, v in roles.items()}, 'out': str(pins_out or fits), 'recipe': RECIPE, 'budget_seconds': 900}))
    pins_sha = sha(t / 'pins.json')
    recipe = dict(RECIPE, epochs=epochs)
    plan = {'recipe': recipe, 'rehearsal_random_outcomes': False, 'width': 9, 'rows': list(E.ROWS), 'seeds': list(E.SEEDS), 'calls': ['outer_%d' % k for k in range(5)], 'members': M.expected_names(), 'works': 45,
            'columns_seen_by_a_source': {'A': 4, 'B': 6, 'C_s17': 7, 'C_s29': 7, 'C_s43': 7}, 'code_sha256': {'worker': sha(t / 'worker.py'), 'project/gate_rebuild/training.py': sha(t / 'project' / 'gate_rebuild' / 'training.py')},
            'outer_salt': SALT, 'outer_folds_sha256': folds_sha,
            'sums': {'config': sha(t / 'config.json'), 'features': sha(t / 'table.npz'), 'token_contract': sha(t / 'contract.json'), 'cohort': sha(t / 'cohort.json'), 'trainval': sha(t / 'trainval.jsonl'),
                     'replay': sha(t / 'replay.jsonl'), 'cost_freeze': sha(t / 'cost.json')}}
    for key, value in (plan_change or {}).items():
        plan[key] = dict(plan[key], **value) if isinstance(value, dict) and isinstance(plan.get(key), dict) else value
    (fits / 'PLAN.json').write_text(json.dumps(plan))
    plan_sha = sha(fits / 'PLAN.json')
    for name in M.expected_names():
        call, _, seed, row = name.split('__')
        fold, s = int(call.split('_')[1]), int(seed[1:])
        held = [a for a in train if a['outer_fold'] == fold]
        d = list(reversed(dev)) if shuffle else dev
        part = lambda rows: {'keys': [a['key'] for a in rows], 'success': [q_of(row, s, a) for a in rows], 'cost': [[40., 300., 1500.]] * len(rows), 'cost_quantile': [[60., 500., 1900.]] * len(rows)}
        m = {'name': name, 'call': call, 'row': row, 'source': row if row != 'C' else 'C_' + seed, 'seed': s, 'plan_sha256': plan_sha, 'recipe': recipe, 'epochs_run': epochs, 'temperature': [1., 1., 1.],
             'parameters': 100, 'held_out_states': len(held), 'parts': {'dev': part(d), 'held_out': part(held)}}
        if change and change[0] == name:
            m = change[1](m)
        (fits / 'members' / (name + '.json')).write_text(m if isinstance(m, str) else json.dumps(m))
    manifest = {'members': {n: sha(fits / 'members' / (n + '.json')) for n in M.expected_names()}, 'count': 45, 'plan_sha256': plan_sha, 'pins_sha256': pins_sha, 'recipe': recipe,
                'members_check_sha256': sha(M.__file__), 'worker_sha256': sha(t / 'worker.py')}
    manifest.update(manifest_change or {})
    (fits / 'MEMBERS_MANIFEST.json').write_text(json.dumps(manifest))
    if manifest_first:
        manifest_first(fits)
    if extra_file:
        (fits / 'members' / 'outer_0__member__s17__A_copy.json').write_text((fits / 'members' / 'outer_0__member__s17__A.json').read_text())
    if complete:
        done = {'state': 'COMPLETE', 'members': 45, 'members_manifest_sha256': sha(fits / 'MEMBERS_MANIFEST.json'), 'plan_sha256': plan_sha, 'pins_sha256': manifest['pins_sha256']}
        done.update(complete_change or {})
        (fits / 'COMPLETE.json').write_text(json.dumps(done))
    return fits


class Evaluate(unittest.TestCase):
    def run_all(self, q_of, contract=None, **options):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        contract = contract or bank()
        fits = publish(contract, t, q_of, **{k: v for k, v in options.items() if k != 'named_pins_sha'})
        self.out = t / 'out'
        E.main([str(t / 'contract.json'), str(fits), str(t / 'pins.json'), options.get('named_pins_sha') or sha(t / 'pins.json'), str(self.out)])
        return contract, json.loads((self.out / 'EVALUATION.json').read_text()), (self.out / 'POLICY_OUTCOMES.csv').read_text().splitlines()

    def test_reference_weights_bounds_and_identical_rows(self):
        contract, r, outcomes = self.run_all(lambda row, s, a: [0.1, 0.5, 0.9])
        train = [a for a in contract['attempts'] if a['cohort']]
        by = {}
        for a in train:
            by.setdefault(a['group'], []).append(a)
        lower = np.mean([np.mean([float(a['actions']['continue_2048']['correct'] is True) for a in rows]) for rows in by.values()])
        upper = np.mean([np.mean([float(a['actions']['continue_2048']['correct'] is not False) for a in rows]) for rows in by.values()])
        tokens = np.mean([np.mean([a['actions']['continue_2048']['total_output_tokens'] for a in rows]) for rows in by.values()])
        ref = r['fixed_policies']['always_continue_2048']
        self.assertAlmostEqual(ref['accuracy_lower_bound'], lower)
        self.assertAlmostEqual(ref['accuracy_upper_bound'], upper)
        self.assertAlmostEqual(ref['mean_output_tokens'], tokens)
        self.assertGreater(upper, lower)
        self.assertIn('p99_output_tokens_of_an_attempt', ref)
        self.assertTrue(0.1 <= ref['share_of_output_tokens_spent_by_the_top_tenth_of_attempts'] <= 1.0)
        block = r['by_tolerance']['0.02']
        for name in ('C_minus_B', 'B_minus_A', 'C_minus_A'):
            c = block['contrasts'][name]['over_seeds']
            self.assertEqual((c['accuracy_difference_lower_bound'], c['accuracy_difference_upper_bound'], c['saving_of_output_tokens']), (0.0, 0.0, 0.0))
        self.assertEqual(len(outcomes), 1 + 3 * 3 * 3 * sum(1 for a in train if not a['ended_before_the_decision']) + sum(1 for a in train if a['ended_before_the_decision']))

    def test_the_bounds_over_the_seeds_are_sharp(self):
        # B: seed 17 stops, seed 29 takes continue_512, seed 43 continue_2048; C: the same three actions in another order of seeds.
        order = {'B': {17: 0, 29: 1, 43: 2}, 'C': {17: 1, 29: 0, 43: 2}, 'A': {17: 2, 29: 2, 43: 2}}
        q_of = lambda row, s, a: [0.9 if i == order[row][s] else 0.1 for i in range(3)]
        unknown = bank(labels=[None, None, True])                         # stop and continue_512 unknown everywhere
        _, r, _ = self.run_all(q_of, contract=unknown, plan_change=None)
        zero = r['by_tolerance']['0.00']                                    # at a tolerance of 0 the rows that lose accuracy fall back; use the widest tolerance below
        del zero
        c = None
        for tolerance in ('0.02', '0.01', '0.00'):
            b = r['by_tolerance'][tolerance]
            if all(b['rows'][x]['every_call_admissible'] for x in 'BC'):
                c = b['contrasts']['C_minus_B']
                break
        if c is None:                                                       # the dev rule refused these members: test the estimand directly
            pop = E.Population([a for a in unknown['attempts'] if a['cohort']])
            n = len(pop.states)
            mean = lambda row: np.mean([E.one_hot(np.full(n, order[row][s])) for s in (17, 29, 43)], 0)
            lower, upper = pop.paired(mean('C'), mean('B'))
            self.assertEqual((float(lower.mean()), float(upper.mean())), (0.0, 0.0))
            per_seed = [pop.paired(E.one_hot(np.full(n, order['C'][s])), E.one_hot(np.full(n, order['B'][s]))) for s in (17, 29, 43)]
            self.assertLess(np.mean([p[0].mean() for p in per_seed]), -0.3)   # the average of per-seed bounds is wide: what the first version reported
        else:
            self.assertEqual((c['over_seeds']['accuracy_difference_lower_bound'], c['over_seeds']['accuracy_difference_upper_bound']), (0.0, 0.0))
            self.assertLess(c['by_seed']['17']['accuracy_difference_lower_bound'], 0.0)
        known = bank(labels=[False, True, True])                            # all labels known: the bounds are the plain difference
        pop = E.Population([a for a in known['attempts'] if a['cohort']])
        n = len(pop.states)
        lower, upper = pop.paired(E.one_hot(np.full(n, 1)), E.one_hot(np.full(n, 0)))
        self.assertAlmostEqual(float(lower.mean()), float(upper.mean()))
        self.assertGreater(float(lower.mean()), 0.0)

    def test_a_call_without_an_admissible_lambda_falls_back_and_fails_the_criterion(self):
        _, r, _ = self.run_all(lambda row, s, a: [0.9, 0.1, 0.1] if row == 'A' else [0.1, 0.5, 0.9])
        for tolerance in ('0.00', '0.01', '0.02'):
            a = r['by_tolerance'][tolerance]['rows']['A']
            self.assertEqual(a['calls_without_an_admissible_lambda'], 15)
            self.assertFalse(a['criterion_10_percent_output_tokens_within_the_tolerance']['point'])
            self.assertAlmostEqual(a['over_seeds']['mean_output_tokens'], r['fixed_policies']['always_continue_2048']['mean_output_tokens'])

    def test_the_order_of_the_dev_rows_does_not_matter(self):
        varied = lambda row, s, a: [0.2, 0.6 if a['key'].endswith('1') else 0.3, 0.7]
        _, first, _ = self.run_all(varied)
        _, second, _ = self.run_all(varied, shuffle=True)
        self.assertEqual(first['by_tolerance'], second['by_tolerance'])

    def refused(self, **options):
        with self.assertRaises(SystemExit):
            self.run_all(lambda row, s, a: [0.1, 0.5, 0.9], **options)
        self.assertFalse(self.out.exists())                                 # nothing was written

    def test_a_directory_that_is_not_the_published_complete_one_is_refused_before_any_output(self):
        name = 'outer_3__member__s29__B'
        self.refused(complete=False)
        self.refused(plan_change={'sums': {'token_contract': '0' * 64}})
        self.refused(plan_change={'rehearsal_random_outcomes': True})
        self.refused(plan_change={'outer_salt': 'another'})
        self.refused(change=(name, lambda m: dict(m, plan_sha256='0' * 64)))
        self.refused(change=(name, lambda m: dict(m, epochs_run=1)))
        self.refused(change=(name, lambda m: json.dumps(m).replace('0.1, 0.5, 0.9', 'NaN, 0.5, 0.9', 1)))
        self.refused(extra_file=True)
        self.refused(manifest_first=lambda fits: (fits / 'members' / (name + '.json')).write_text((fits / 'members' / (name + '.json')).read_text() + ' '))
        self.refused(manifest_first=lambda fits: (fits / 'members' / (name + '.json')).unlink())

    def test_a_consistent_directory_of_another_start_is_refused(self):
        # the counterexample of the review: everything agrees with itself, but it is not the accepted start
        self.refused(epochs=1)                                               # plan, members and manifest all say one epoch
        self.refused(manifest_change={'pins_sha256': 'f' * 64})              # another PINS in the manifest (COMPLETE follows it)
        self.refused(complete_change={'pins_sha256': 'a' * 64})
        self.refused(complete_change={'plan_sha256': '0' * 64})
        self.refused(complete_change={'members': 1})
        self.refused(manifest_change={'count': 44})
        self.refused(named_pins_sha='0' * 64)                                # the PINS file is not the one of the admission
        self.refused(plan_change={'sums': {'features': '1' * 64}})           # another table
        self.refused(plan_change={'sums': {'config': '2' * 64}})             # another configuration
        self.refused(plan_change={'code_sha256': {'worker': '3' * 64}})      # another worker
        self.refused(plan_change={'code_sha256': {'project/gate_rebuild/training.py': '4' * 64}})
        self.refused(plan_change={'width': 10})
        self.refused(plan_change={'seeds': [17, 29, 44]})
        self.refused(plan_change={'columns_seen_by_a_source': {'A': 5}})
        self.refused(manifest_change={'worker_sha256': '5' * 64})
        self.refused(manifest_change={'members_check_sha256': '6' * 64})
        self.refused(pins_out='/somewhere/else')                             # the PINS file is for another directory

    def test_the_price_moves_the_decision_to_the_cheaper_action(self):
        q, m = [[0.5, 0.6, 0.7]], [[40., 300., 1500.]]
        self.assertEqual([int(E.decide(q, m, lam)[0]) for lam in (0.0, 0.05, 0.2, 5.0)], [2, 2, 1, 0])
        self.assertEqual(int(E.decide([[0.5, 0.5, 0.5]], m, 0.0)[0]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
