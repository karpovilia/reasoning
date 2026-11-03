#!/usr/bin/env python3
"""Checks of pfgate/pes_bridge.py, numpy only.
Without arguments: the hard rules (threshold inclusive, the first mode on a tie), the columns on a hand example, and the
causality of every record: a prefix cut after segment j, and a prefix with the same first j + 1 segments and another
continuation, give the same records 0..j (300 random prefixes; span changes, two events in one segment, a one-segment prefix).
With the argument CORPUS V2_DATA_DIR ABLATION_INPUTS.npz: PARITY with the inputs the predictor was trained on. For traces of the
frozen corpus whose known flags are all 1, the bridge is given the corpus's own hard events, span types and characters; its 41
columns and previous span types must equal the corpus arrays (the old columns that carried the span, as the owner's
prepare_ablation.build reads them). This checks names, order, units and scales against the owner's files, not against this
file's own reading of them."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import pes_bridge as B  # noqa: E402

CORPUS = sys.argv[sys.argv.index('CORPUS') + 1:] if 'CORPUS' in sys.argv else []
if CORPUS:
    sys.argv = sys.argv[:sys.argv.index('CORPUS')]
THRESHOLDS = np.array([0.2568320035934448, 0.2436424344778061, 0.36840569972991943, 0.319121390581131, 0.23231014609336853, 0.2991051971912384])


class Rules(unittest.TestCase):
    def test_hard_rules(self):
        p = np.tile(THRESHOLDS, (3, 1))
        p[1] -= 1e-9
        p[2] += 1e-9
        self.assertEqual(B.hard_events(p, THRESHOLDS).tolist(), [[1.] * 6, [0.] * 6, [1.] * 6])
        self.assertEqual(B.hard_modes([[.3, .3, .1, .1, .1, .1], [0, 0, 0, 0, 0, 1], [.1, .2, .2, .2, .2, .1]]).tolist(), [0, 5, 1])
        for bad in ([[0.5] * 5], [[float('nan')] * 6], [[1.5] + [0.] * 5]):
            with self.assertRaises(ValueError):
                B.hard_events(bad, THRESHOLDS)

    def test_columns_on_a_hand_example(self):
        chars = [10, 20, 30, 40]
        ev = np.zeros((4, 6))
        ev[1, 1] = ev[1, 5] = ev[3, 1] = 1                            # verify and commit in segment 1, verify again in segment 3
        xin, mode, prev = B.records(chars, ev, [1, 1, 3, 1])
        self.assertEqual(prev.tolist(), [7, 7, 1, 3])
        np.testing.assert_allclose(xin[:, 0], np.log1p([1, 2, 3, 4]) / 10, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 1], np.log1p(chars) / 10, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 2], np.log1p([10, 30, 60, 100]) / 10, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 3], np.log1p([1, 2, 1, 1]) / 10, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 4], np.log1p([10, 30, 30, 40]) / 10, rtol=1e-6)
        self.assertTrue((xin[:, 11:17] == 1).all())
        d4, d32 = np.exp(-1 / 4), np.exp(-1 / 32)
        np.testing.assert_allclose(xin[:, 17 + 1], np.log1p([0, 1, d4, d4 * d4 + 1]) / 5, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 23 + 1], np.log1p([0, 1, d32, d32 * d32 + 1]) / 5, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 29 + 1], np.log1p([0, 1, 1, 2]) / 5, rtol=1e-6)
        np.testing.assert_allclose(xin[:, 35 + 1], np.log1p([1, 0, 1, 0]) / 10, rtol=1e-6)      # segments since the last verify
        np.testing.assert_allclose(xin[:, 35 + 0], np.log1p([1, 2, 3, 4]) / 10, rtol=1e-6)      # no backtrack yet: since the start
        self.assertEqual(xin.dtype, np.float32)

    def test_every_record_depends_on_its_prefix_alone(self):
        rng = np.random.default_rng(4)
        for _ in range(300):
            n = int(rng.integers(1, 60))
            chars, ep, mp = rng.integers(1, 400, n), rng.uniform(0, 1, (n, 6)), rng.dirichlet(np.ones(6) * 0.3, n)
            xin, mode, prev = B.bridge(chars, ep, mp, THRESHOLDS)
            cut = int(rng.integers(1, n + 1))
            a = B.bridge(chars[:cut], ep[:cut], mp[:cut], THRESHOLDS)
            m = int(rng.integers(0, 30))
            other = B.bridge(np.concatenate([chars[:cut], rng.integers(1, 400, m)]), np.concatenate([ep[:cut], rng.uniform(0, 1, (m, 6))]), np.concatenate([mp[:cut], rng.dirichlet(np.ones(6), m)]), THRESHOLDS)
            for full, short, longer in zip((xin, mode, prev), a, other):
                np.testing.assert_array_equal(full[:cut], short)
                np.testing.assert_array_equal(full[:cut], longer[:cut])
        with self.assertRaises(ValueError):
            B.records([], np.zeros((0, 6)), [])


@unittest.skipUnless(CORPUS, 'only with the argument CORPUS V2_DATA_DIR ABLATION_INPUTS.npz')
class Parity(unittest.TestCase):
    def test_the_bridge_reproduces_the_training_inputs_from_the_corpus_labels(self):
        import json
        data, inputs = Path(CORPUS[0]), CORPUS[1]
        d, a = np.load(data / 'sequences.npz'), np.load(inputs)
        meta = json.loads((data / 'metadata.json').read_text())
        self.assertEqual((meta['events'], meta['operators']), (list(B.EVENTS), list(B.MODES)))
        x, mode, prev, offsets, xin = d['x'], d['mode'], d['prev_mode'], d['offsets'], a['xin']
        rng = np.random.default_rng(1)
        checked = segments = skipped = 0
        worst = 0.0
        for i in rng.permutation(len(offsets) - 1)[:1500]:
            s, z = int(offsets[i]), int(offsets[i + 1])
            if not (x[s:z, 11:17] == 1).all():                   # a trace with an unknown event label: the bridge has no unknown, not comparable
                skipped += 1
                continue
            chars = np.rint(np.expm1(x[s:z, 2].astype(np.float64) * 10))
            mine, my_mode, my_prev = B.records(chars, x[s:z, 5:11], mode[s:z])
            worst = max(worst, float(np.abs(mine - xin[s:z]).max()))
            np.testing.assert_array_equal(my_prev, prev[s:z])
            np.testing.assert_array_equal(my_mode, mode[s:z])
            checked, segments = checked + 1, segments + (z - s)
        print('parity with the corpus: %d traces, %d segments compared, %d traces with unknown labels left out; largest absolute difference of the 41 columns %.3g' % (checked, segments, skipped, worst))
        self.assertGreater(checked, 300)
        self.assertLess(worst, 2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=1)
