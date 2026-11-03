#!/usr/bin/env python3
"""Checks of pfgate/export_member_weights_v2.py with stand-ins for the model, the fit, torch and the accepted modules: the REAL
main of the exporter, real files and real sums of the pins, no bank, no fit, no torch (review PFG-W6-R1 and R2).
  R1  a member that fails any check is never published: a wrong SHA256 of the weights, other predictions, another temperature
      (nothing of it is saved); a saved file that does not load back as the member, a save that raises (it stays in quarantine
      or does not exist); in every case code 5, FAILED.json, no COMPLETE, and weights/ holds only members that passed;
  R2  before any module of the project is used: a pins file with another sum, a changed exporter, a changed evaluate.py, a
      directory of fits whose MEMBERS_MANIFEST is not the pinned one (with a COMPLETE that agrees with it), pins that do not
      pin a needed file, accepted PINS for another directory: code 4, no stand-in of the project was called, no output.
  the positive control: 45 members published, 45 records, COMPLETE, an empty quarantine removed, the pinned sums in the manifest."""
import hashlib
import importlib.util
import json
import pickle
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2] / 'src/pfgate'
EXPORTER = CODE / 'export_member_weights_v2.py'
NAMES = ['outer_%d__member__s%d__%s' % (f, s, r) for f in range(5) for s in (17, 29, 43) for r in 'ABC']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RECIPE = {'epochs': 7, 'learning_rate': 0.001, 'weight_decay': 0.01}


class Tensor:
    def __init__(self, value):
        self.value = np.array([value], np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Model:
    def __init__(self, value=0.0, shift=0.0):
        self.value, self.shift = value, shift

    def state_dict(self):
        return {'x': Tensor(self.value)}

    def load_state_dict(self, state):
        self.value = float(state['x'].value[0])

    def parameters(self):
        return [types.SimpleNamespace(numel=lambda: 1)]

    def eval(self):
        pass


def value_of(name):
    return float(NAMES.index(name) + 1)


def state_hash(value):
    return hashlib.sha256(np.array([value], np.float32).tobytes()).hexdigest()


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fits = self.root / 'fits_v1'
        self.fits.mkdir()
        (self.root / 'token.json').write_text('{}')
        (self.root / 'accepted_pins.json').write_text(json.dumps({'accepted': 'pins'}))
        (self.fits / 'PLAN.json').write_text(json.dumps({'outer_folds_sha256': 'folds'}))
        (self.fits / 'MEMBERS_MANIFEST.json').write_text(json.dumps({'members': {n: 'member-file-sum' for n in NAMES}, 'plan_sha256': sha(self.fits / 'PLAN.json')}))
        (self.fits / 'COMPLETE.json').write_text(json.dumps({'state': 'COMPLETE', 'members_manifest_sha256': sha(self.fits / 'MEMBERS_MANIFEST.json')}))
        self.train = [types.SimpleNamespace(group='q%d' % i, key='k%d' % i) for i in range(5)]
        self.dev = [types.SimpleNamespace(group='d', key='dev')]
        self.behaviour, self.calls = {}, []
        self.pins = {'format': 'pfgate_member_weights_pins_v1', 'accepted_pins': str(self.root / 'accepted_pins.json'), 'accepted_pins_sha256': sha(self.root / 'accepted_pins.json'), 'accepted_fits': str(self.fits),
                     'files': {str(p): sha(p) for p in [EXPORTER] + [CODE / (m + '.py') for m in ('evaluate', 'launch_fits', 'worker', 'members_check')]
                               + [self.fits / n for n in ('MEMBERS_MANIFEST.json', 'PLAN.json', 'COMPLETE.json')] + [self.root / 'accepted_pins.json']}}
        self.out = self.root / 'member_weights_v1'

    def prediction(self, model, examples):
        return {'success': [[0.25 + model.value / 1000 + model.shift] * 3 for _ in examples]}

    def members(self):
        out = {}
        for name in NAMES:
            fold, model = int(name.split('__')[0].split('_')[1]), Model(value_of(name))
            parts = {part: {'keys': [e.key for e in examples], **self.prediction(model, examples)} for part, examples in (('dev', self.dev), ('held_out', [self.train[fold]]))}
            out[name] = {'final_state_sha256': state_hash(value_of(name)), 'parts': parts, 'temperature': [1.0, 1.0, 1.0], 'parameters': 1}
        return out

    def install(self):
        """Stand-ins of the accepted modules, of the fit and of torch; the files they claim to be are the real pinned files."""
        test, config = self, {'threads': 1, 'expected_outer_folds_sha256': 'folds', 'project_code': '/unused', 'recipe': RECIPE, 'width': 2676}
        current = {}
        E = types.ModuleType('pfgate.evaluate')
        E.__file__ = str(CODE / 'evaluate.py')
        E.checked_members = lambda *a: (test.calls.append('checked_members'), (test.members(), json.loads((test.fits / 'MEMBERS_MANIFEST.json').read_text()), {'outer_folds_sha256': 'folds'}))[1]
        L = types.ModuleType('pfgate.launch_fits')
        L.__file__, L.Refusal = str(CODE / 'launch_fits.py'), ValueError
        L.check_pins = lambda *a: (test.calls.append('check_pins'), ({'roles': {'token_contract': str(test.root / 'token.json')}, 'out': str(test.accepted_out)}, config))[1]
        W = types.ModuleType('pfgate.worker')
        W.__file__, W.COST_SCALE = str(CODE / 'worker.py'), 1024.0
        W.source_of = lambda row, seed: row if row != 'C' else 'C_s%d' % seed
        W.load = lambda c: (test.calls.append('load'), (test.train, test.dev, {'q%d' % i: i for i in range(5)}, None, 'folds', {W.source_of(r, s): 1 for r in 'ABC' for s in (17, 29, 43)}))[1]
        M = types.ModuleType('pfgate.members_check')
        M.__file__, M.expected_names = str(CODE / 'members_check.py'), lambda: NAMES
        package = types.ModuleType('pfgate')
        package.evaluate, package.launch_fits, package.worker, package.members_check = E, L, W, M
        torch = types.ModuleType('torch')
        torch.__version__, torch.set_num_threads = 'STAND-IN', lambda n: None

        def manual_seed(seed):
            current['seed'] = seed
        torch.manual_seed = manual_seed

        def fit_one(fitting, held, source, encoder, seed, **options):
            fold = next(i for i in range(5) if test.train[i] not in fitting)
            row = source[0]
            name = 'outer_%d__member__s%d__%s' % (fold, seed, row)
            current['name'] = name
            how = test.behaviour.get(name)
            test.calls.append('fit ' + name)
            return Model(value_of(name) + (0.5 if how == 'bad_hash' else 0.0), 0.001 if how == 'bad_prediction' else 0.0), {'epochs_run': 7}

        def save(value, path):
            if test.behaviour.get(current['name']) == 'save_raises':
                raise OSError('no space left on the stand-in device')
            with open(path, 'wb') as handle:
                pickle.dump({k: (v if k != 'state_dict' else {'x': float(v['x'].value[0])}) for k, v in value.items()}, handle)

        def load(path, **options):
            with open(path, 'rb') as handle:
                value = pickle.load(handle)
            shift = 0.5 if test.behaviour.get(current['name']) == 'bad_reload' else 0.0
            value['state_dict'] = {'x': Tensor(value['state_dict']['x'] + shift)}
            return value
        torch.save, torch.load = save, load
        models = types.ModuleType('gate_rebuild.models')
        models.Gate = lambda *a: Model()
        training = types.ModuleType('gate_rebuild.training')
        training.fit_one = fit_one
        training.calibrate = lambda model, dev, source: [2.0, 1.0, 1.0] if test.behaviour.get(current['name']) == 'bad_temperature' else [1.0, 1.0, 1.0]
        training.predict = lambda models_, examples, source: test.prediction(models_[0], examples)
        stand_ins = {'pfgate': package, 'pfgate.evaluate': E, 'pfgate.launch_fits': L, 'pfgate.worker': W, 'pfgate.members_check': M, 'torch': torch, 'gate_rebuild': types.ModuleType('gate_rebuild'),
                     'gate_rebuild.models': models, 'gate_rebuild.training': training}
        saved = {name: sys.modules.get(name) for name in stand_ins}
        sys.modules.update(stand_ins)
        self.addCleanup(lambda: [sys.modules.pop(n) if v is None else sys.modules.__setitem__(n, v) for n, v in saved.items()])

    def export(self, pins_sum=None, accepted_out=None):
        self.accepted_out = accepted_out or self.fits
        (self.root / 'export_pins.json').write_text(json.dumps(self.pins))
        self.install()
        spec = importlib.util.spec_from_file_location('exporter_under_test', EXPORTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        timer, signal.setitimer = signal.setitimer, lambda *a: None
        try:
            try:
                return module.main([str(self.root / 'export_pins.json'), pins_sum or sha(self.root / 'export_pins.json'), str(self.out), '600'])
            except SystemExit as stop:
                return stop.code
        finally:
            signal.setitimer = timer

    def published(self):
        return sorted(p.stem for p in (self.out / 'weights').glob('*.pt')) if (self.out / 'weights').exists() else []

    def test_positive_control_all_45_members(self):
        self.assertEqual(self.export(), 0)
        manifest = json.loads((self.out / 'WEIGHTS_MANIFEST.json').read_text())
        self.assertEqual((self.published(), len(list((self.out / 'records').glob('*.json'))), (self.out / 'COMPLETE.json').exists(), (self.out / 'quarantine').exists(), (self.out / 'FAILED.json').exists()),
                         (sorted(NAMES), 45, True, False, False))
        self.assertEqual((manifest['accepted_members_manifest_sha256'], manifest['accepted_plan_sha256'], manifest['accepted_complete_sha256'], manifest['this_file_sha256'], manifest['count']),
                         (sha(self.fits / 'MEMBERS_MANIFEST.json'), sha(self.fits / 'PLAN.json'), sha(self.fits / 'COMPLETE.json'), sha(EXPORTER), 45))
        record = json.loads((self.out / 'records' / (NAMES[7] + '.json')).read_text())
        self.assertEqual((all(record['checks'].values()), len(record['checks']), record['weights_file_sha256']), (True, 6, sha(self.out / 'weights' / (NAMES[7] + '.pt'))))
        self.assertEqual(self.calls[:3], ['check_pins', 'checked_members', 'load'])

    def test_review_case_a_member_that_fails_a_check_is_never_published(self):
        bad = NAMES[2]
        for how, quarantined in (('bad_hash', []), ('bad_prediction', []), ('bad_temperature', []), ('bad_reload', [bad + '.pt']), ('save_raises', [])):
            with self.subTest(how):
                self.setUp()
                self.behaviour = {bad: how}
                self.assertEqual(self.export(), 5)
                failed = json.loads((self.out / 'FAILED.json').read_text())
                self.assertEqual((self.published(), (self.out / 'COMPLETE.json').exists(), (self.out / 'WEIGHTS_MANIFEST.json').exists(), sorted(p.name for p in (self.out / 'quarantine').iterdir())),
                                 (sorted(NAMES[:2]), False, False, quarantined))          # only the two members before it, which passed every check
                self.assertEqual((failed['name'], failed['published_weights_of_this_member'], failed['in_quarantine'], (self.out / 'records' / (bad + '.json')).exists()), (bad, False, quarantined, False))
                self.assertEqual(len([c for c in self.calls if c.startswith('fit ')]), 3)       # the run stopped at the bad member

    def test_review_case_nothing_of_the_project_runs_before_the_pins_of_the_export_hold(self):
        def another_manifest():
            (self.fits / 'MEMBERS_MANIFEST.json').write_text(json.dumps({'members': {n: 'another-member-file' for n in NAMES}, 'plan_sha256': sha(self.fits / 'PLAN.json')}))
            (self.fits / 'COMPLETE.json').write_text(json.dumps({'state': 'COMPLETE', 'members_manifest_sha256': sha(self.fits / 'MEMBERS_MANIFEST.json')}))      # consistent with itself, not with the pins
            self.pins['files'][str(self.fits / 'COMPLETE.json')] = sha(self.fits / 'COMPLETE.json')
        cases = {'a changed exporter': lambda: self.pins['files'].__setitem__(str(EXPORTER), '0' * 64), 'a changed evaluate.py': lambda: self.pins['files'].__setitem__(str(CODE / 'evaluate.py'), '1' * 64),
                 'members and manifest rewritten consistently, another sum than the accepted manifest': another_manifest,
                 'pins that do not pin the worker': lambda: self.pins['files'].pop(str(CODE / 'worker.py')), 'pins that do not pin the accepted manifest': lambda: self.pins['files'].pop(str(self.fits / 'MEMBERS_MANIFEST.json')),
                 'another format': lambda: self.pins.__setitem__('format', 'other'), 'accepted PINS with another sum than named': lambda: self.pins.__setitem__('accepted_pins_sha256', '2' * 64)}
        for label, change in cases.items():
            with self.subTest(label):
                self.setUp()
                change()
                self.assertEqual((self.export(), self.calls, self.out.exists()), (4, [], False))
        self.setUp()
        self.assertEqual((self.export(pins_sum='3' * 64), self.calls, self.out.exists()), (4, [], False))

    def test_accepted_pins_for_another_directory_and_an_existing_output_are_refused_before_any_fit(self):
        other = self.root / 'other_fits'
        other.mkdir()
        self.assertEqual((self.export(accepted_out=other), self.calls, self.out.exists()), (4, ['check_pins'], False))
        self.setUp()
        self.out.mkdir()
        self.assertEqual((self.export(), [c for c in self.calls if c.startswith('fit ')], list(self.out.iterdir())), (4, [], []))


if __name__ == '__main__':
    unittest.main(verbosity=1)
