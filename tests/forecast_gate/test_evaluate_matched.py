#!/usr/bin/env python3
"""Checks of pfgate/evaluate_matched.py on the small artificial bank of test_evaluate.py, numpy only.
  * with identical members everywhere the accepted numbers are reproduced, the rows A and B of the variant runs are reported
    identical and every contrast is zero;
  * the contrast C_matched minus C_text equals the contrast C minus B that the ACCEPTED evaluation gives for one directory
    whose row B holds the members of C_text and whose row C holds the members of C_matched (the new file adds no arithmetic);
  * a changed number of the accepted evaluation, an evaluation of other fits: refused, nothing written;
  * PINS of a variant that do not name the accepted PINS, differ in a fifth file, or whose table has the heads of the other
    cell: refused, nothing written;
  * rows A and B of a variant that differ from the accepted members: reported, and the contrast against the B of the same run
    differs from the contrast against the accepted B."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfgate import evaluate as E, evaluate_matched as EM, members_check as M  # noqa: E402
import test_evaluate as T  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(contract_path, contract, t, q_of, variant=None, accepted=None, pins_change=None, cell=None):
    """A published directory of 45 members with its PINS file, with the roles of the real PINS (a data contract among them).
    A variant shares every file with the accepted directory except the configuration, the data contract and the table."""
    t.mkdir()
    fits = t / 'fits'
    (fits / 'members').mkdir(parents=True)
    shared = accepted['root'] if accepted else t
    if not accepted:
        (t / 'project' / 'gate_rebuild').mkdir(parents=True)
        (t / 'project' / 'gate_rebuild' / 'training.py').write_text('X = 1\n')
        for name in ('cohort.json', 'trainval.jsonl', 'replay.jsonl', 'cost.json', 'worker.py'):
            (t / name).write_text('content of ' + name)
    (t / 'table.npz').write_text('table of %s' % (variant or 'the accepted run'))
    table = {'rows': T.ROWS_OF_THE_TABLE, 'width': 9, 'npz_sha256': sha(t / 'table.npz'), 'format': 'pfgate_gate_table_v2'}
    if variant:
        table.update(format='pfgate_gate_table_v2_variant', heads_cell=cell or {'text': 'T', 'matched': 'TES'}[variant], base_table_sha256=sha(accepted['root'] / 'table.npz'), pes_heads={'17': {}})
    (t / 'table.npz.manifest.json').write_text(json.dumps(table))
    (t / 'data_contract.json').write_text(json.dumps({'gate table': {'npz_sha256': table['npz_sha256']}}))
    train = [a for a in contract['attempts'] if a['cohort'] == 'decision state']
    dev = [a for a in contract['attempts'] if a['split'] == 'validation' and not a['ended_before_the_decision']]
    folds = T.fold_of([a['group'] for a in contract['attempts'] if a['cohort']])
    folds_sha = hashlib.sha256(json.dumps(sorted(folds.items())).encode()).hexdigest()
    (t / 'config.json').write_text(json.dumps({'width': 9, 'project_code': str(shared / 'project'), 'outer_salt': T.SALT, 'expected_outer_folds_sha256': folds_sha, 'recipe': T.RECIPE, 'features': str(t / 'table.npz')}))
    roles = {'config': t / 'config.json', 'data_contract': t / 'data_contract.json', 'worker': shared / 'worker.py', 'members_check': Path(M.__file__).resolve(), 'gate_table': t / 'table.npz',
             'gate_table_manifest': t / 'table.npz.manifest.json', 'token_contract': contract_path, 'cohort': shared / 'cohort.json', 'trainval': shared / 'trainval.jsonl', 'replay': shared / 'replay.jsonl', 'cost_freeze': shared / 'cost.json'}
    files = {str(p): sha(p) for p in list(roles.values()) + [shared / 'project' / 'gate_rebuild' / 'training.py']}
    pins = {'format': 'pfgate_pins_v1', 'files': files, 'roles': {k: str(v) for k, v in roles.items()}, 'out': str(fits), 'recipe': T.RECIPE, 'budget_seconds': 900}
    if variant:
        pins['accepted_pins_sha256'] = accepted['pins_sha']
    if pins_change:
        pins_change(pins, t)
    (t / 'pins.json').write_text(json.dumps(pins))
    pins_sha = sha(t / 'pins.json')
    plan = {'recipe': T.RECIPE, 'rehearsal_random_outcomes': False, 'width': 9, 'rows': list(E.ROWS), 'seeds': list(E.SEEDS), 'calls': ['outer_%d' % k for k in range(5)], 'members': M.expected_names(), 'works': 45,
            'columns_seen_by_a_source': {'A': 4, 'B': 6, 'C_s17': 7, 'C_s29': 7, 'C_s43': 7}, 'code_sha256': {'worker': sha(shared / 'worker.py'), 'project/gate_rebuild/training.py': sha(shared / 'project' / 'gate_rebuild' / 'training.py')},
            'outer_salt': T.SALT, 'outer_folds_sha256': folds_sha,
            'sums': {'config': sha(t / 'config.json'), 'features': sha(t / 'table.npz'), 'token_contract': sha(contract_path), 'cohort': sha(shared / 'cohort.json'), 'trainval': sha(shared / 'trainval.jsonl'),
                     'replay': sha(shared / 'replay.jsonl'), 'cost_freeze': sha(shared / 'cost.json')}}
    (fits / 'PLAN.json').write_text(json.dumps(plan))
    plan_sha = sha(fits / 'PLAN.json')
    for name in M.expected_names():
        call, _, seed, row = name.split('__')
        fold, s = int(call.split('_')[1]), int(seed[1:])
        held = [a for a in train if a['outer_fold'] == fold]
        part = lambda rows: {'keys': [a['key'] for a in rows], 'success': [q_of(row, s, a) for a in rows], 'cost': [[40., 300., 1500.]] * len(rows), 'cost_quantile': [[60., 500., 1900.]] * len(rows)}
        weights = hashlib.sha256(json.dumps([q_of(row, s, a) for a in train[:3]]).encode()).hexdigest()
        m = {'name': name, 'call': call, 'row': row, 'source': row if row != 'C' else 'C_' + seed, 'seed': s, 'plan_sha256': plan_sha, 'recipe': T.RECIPE, 'epochs_run': 7, 'temperature': [1., 1., 1.],
             'parameters': 100, 'held_out_states': len(held), 'final_state_sha256': weights, 'parts': {'dev': part(dev), 'held_out': part(held)}}
        (fits / 'members' / (name + '.json')).write_text(json.dumps(m))
    manifest = {'members': {n: sha(fits / 'members' / (n + '.json')) for n in M.expected_names()}, 'count': 45, 'plan_sha256': plan_sha, 'pins_sha256': pins_sha, 'recipe': T.RECIPE,
                'members_check_sha256': sha(M.__file__), 'worker_sha256': sha(shared / 'worker.py')}
    (fits / 'MEMBERS_MANIFEST.json').write_text(json.dumps(manifest))
    (fits / 'COMPLETE.json').write_text(json.dumps({'state': 'COMPLETE', 'members': 45, 'members_manifest_sha256': sha(fits / 'MEMBERS_MANIFEST.json'), 'plan_sha256': plan_sha, 'pins_sha256': pins_sha}))
    return {'root': t, 'fits': fits, 'pins': t / 'pins.json', 'pins_sha': pins_sha}


def number_and_sample(a):
    q, sample = a['key'][1:].split('|')
    return int(q), int(sample)


def base(row, s, a):
    """Always continue_2048 at any price that keeps dev admissible."""
    return [0.1, 0.2, 0.9]


def other(row, s, a):
    """continue_512 on the first sample of a question (on dev that branch is correct), continue_2048 otherwise."""
    return [0.1, 0.9, 0.9] if number_and_sample(a)[1] == 1 else [0.1, 0.2, 0.9]


def third(row, s, a):
    """As `other`, and continue_512 also on the second sample of the even train questions: cheaper on train, the same on dev."""
    q, sample = number_and_sample(a)
    return [0.1, 0.9, 0.5] if sample == 2 and q < 20 and q % 2 == 0 else other(row, s, a)


class Matched(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.contract = T.bank()
        self.contract_path = self.t / 'contract.json'
        self.contract_path.write_text(json.dumps(self.contract))

    def build(self, accepted_q=base, text_q=base, matched_q=base, text_options=None, matched_options=None):
        self.accepted = publish(self.contract_path, self.contract, self.t / 'accepted', accepted_q)
        E.main([str(self.contract_path), str(self.accepted['fits']), str(self.accepted['pins']), self.accepted['pins_sha'], str(self.t / 'evaluation_accepted')])
        self.accepted_evaluation = self.t / 'evaluation_accepted' / 'EVALUATION.json'
        self.text = publish(self.contract_path, self.contract, self.t / 'text', text_q, 'text', self.accepted, **(text_options or {}))
        self.matched = publish(self.contract_path, self.contract, self.t / 'matched', matched_q, 'matched', self.accepted, **(matched_options or {}))

    def evaluate(self, evaluation=None):
        argv = [str(self.contract_path), str(evaluation or self.accepted_evaluation)]
        for d in (self.accepted, self.text, self.matched):
            argv += [str(d['fits']), str(d['pins']), d['pins_sha']]
        EM.main(argv + [str(self.t / 'out')])
        return json.loads((self.t / 'out' / 'EVALUATION_MATCHED.json').read_text())

    def assert_refused(self, *words, **options):
        with self.assertRaises(SystemExit) as caught:
            self.evaluate(**options)
        self.assertFalse((self.t / 'out').exists(), 'a refusal wrote something')
        for word in words:
            self.assertIn(word, str(caught.exception))

    def test_identical_members_reproduce_and_give_zero_contrasts(self):
        self.build()
        r = self.evaluate()
        self.assertEqual(r['the_accepted_numbers_are_reproduced']['contrasts_checked'], 9)
        for name in ('text', 'matched'):
            self.assertEqual(r['rows_A_and_B_of_the_variant_runs_against_the_accepted_members'][name], {'members_compared': 30, 'members_with_the_same_final_weights': 30, 'largest_absolute_difference_of_a_prediction': 0.0, 'identical': True})
        for tolerance, block in r['by_tolerance'].items():
            for name, c in list(block['contrasts'].items()) + list(block['contrasts_against_the_B_of_the_same_run'].items()):
                self.assertEqual((c['over_seeds']['accuracy_difference_lower_bound'], c['over_seeds']['accuracy_difference_upper_bound'], c['over_seeds']['saving_of_output_tokens']), (0.0, 0.0, 0.0), (tolerance, name))
            self.assertEqual(block['rows']['C_text'], block['rows']['C'])
        self.assertEqual(len((self.t / 'out' / 'POLICY_OUTCOMES_MATCHED.csv').read_text().splitlines()), 1 + 3 * 2 * 3 * len([a for a in self.contract['attempts'] if a['cohort'] == 'decision state']))

    def test_the_main_contrast_equals_the_accepted_arithmetic_on_the_same_members(self):
        text_q = lambda row, s, a: other(row, s, a) if row == 'C' else base(row, s, a)
        matched_q = lambda row, s, a: third(row, s, a) if row == 'C' else base(row, s, a)
        self.build(text_q=text_q, matched_q=matched_q)
        r = self.evaluate()
        # one directory whose row B holds the members of C_text and whose row C the members of C_matched, evaluated by the ACCEPTED file
        mixed_q = lambda row, s, a: {'A': base, 'B': other, 'C': third}[row](row, s, a)
        mixed = publish(self.contract_path, self.contract, self.t / 'mixed', mixed_q)
        E.main([str(self.contract_path), str(mixed['fits']), str(mixed['pins']), mixed['pins_sha'], str(self.t / 'evaluation_mixed')])
        accepted_arithmetic = json.loads((self.t / 'evaluation_mixed' / 'EVALUATION.json').read_text())
        differs = 0
        for tolerance in ('0.00', '0.01', '0.02'):
            mine, theirs = r['by_tolerance'][tolerance]['contrasts']['C_matched_minus_C_text'], accepted_arithmetic['by_tolerance'][tolerance]['contrasts']['C_minus_B']
            self.assertEqual(mine, theirs, tolerance)
            self.assertEqual(r['by_tolerance'][tolerance]['rows']['C_matched']['against_no_control_over_seeds'], accepted_arithmetic['by_tolerance'][tolerance]['rows']['C']['against_no_control_over_seeds'])
            differs += mine['over_seeds']['saving_of_output_tokens'] != 0.0 or mine['over_seeds']['accuracy_difference_lower_bound'] != mine['over_seeds']['accuracy_difference_upper_bound']
        self.assertGreater(differs, 0)                                    # the members do differ: the equality above is not 0 == 0
        self.assertEqual(r['rows_A_and_B_of_the_variant_runs_against_the_accepted_members']['text']['identical'], True)
        self.assertEqual(r['main_contrast'], {'first': 'C_matched', 'second': 'C_text', 'tolerance': '0.02'})

    def test_a_changed_or_foreign_accepted_evaluation_is_refused(self):
        self.build()
        value = json.loads(self.accepted_evaluation.read_text())
        value['by_tolerance']['0.02']['contrasts']['C_minus_B']['over_seeds']['saving_of_output_tokens'] += 1e-9
        changed = self.t / 'changed.json'
        changed.write_text(json.dumps(value))
        self.assert_refused('NOT reproduced', evaluation=changed)
        value = json.loads(self.accepted_evaluation.read_text())
        value['members_manifest_sha256'] = '0' * 64
        changed.write_text(json.dumps(value))
        self.assert_refused('not the evaluation of the accepted fits', evaluation=changed)

    def test_pins_of_a_variant_that_are_not_the_accepted_pins_with_another_table_are_refused(self):
        def fifth(pins, t):
            (t / 'cost.json').write_text('another cost')
            del pins['files'][pins['roles']['cost_freeze']]
            pins['roles']['cost_freeze'] = str(t / 'cost.json')
            pins['files'][str(t / 'cost.json')] = sha(t / 'cost.json')
        cases = {'no name of the accepted PINS': {'pins_change': lambda pins, t: pins.pop('accepted_pins_sha256')}, 'the name of other PINS': {'pins_change': lambda pins, t: pins.update(accepted_pins_sha256='1' * 64)},
                 'the heads of the other cell': {'cell': 'TES'}}
        for label, options in cases.items():
            with self.subTest(label):
                self.setUp()
                self.build(text_options=options)
                self.assert_refused('are not the accepted PINS')

    def test_rows_A_and_B_of_a_variant_that_differ_are_reported_and_both_contrasts_are_given(self):
        text_q = lambda row, s, a: other(row, s, a) if row == 'C' else (third(row, s, a) if row == 'B' else base(row, s, a))
        self.build(text_q=text_q)
        r = self.evaluate()
        report = r['rows_A_and_B_of_the_variant_runs_against_the_accepted_members']
        self.assertEqual((report['text']['identical'], report['text']['members_with_the_same_final_weights'], report['matched']['identical']), (False, 15, True))
        self.assertGreater(report['text']['largest_absolute_difference_of_a_prediction'], 0.1)
        differs = 0
        for block in r['by_tolerance'].values():
            differs += block['contrasts']['C_text_minus_B']['over_seeds'] != block['contrasts_against_the_B_of_the_same_run']['C_text_minus_B_of_the_text_run']['over_seeds']
            self.assertEqual(block['contrasts']['C_matched_minus_B']['over_seeds'], block['contrasts_against_the_B_of_the_same_run']['C_matched_minus_B_of_the_matched_run']['over_seeds'])
        self.assertGreater(differs, 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
