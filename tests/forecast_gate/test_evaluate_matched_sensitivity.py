#!/usr/bin/env python3
"""Checks of pfgate/evaluate_matched_sensitivity.py on the artificial bank of the tests of the evaluation, numpy only:
the whole population is reproduced from the file of the outcomes; leaving out a question on which the two rows take different
actions changes the contrast, and the change equals the contrast computed by hand on the kept questions; leaving out a question
on which they agree changes only the weights; a question outside the cohort, a changed file of outcomes: refused, nothing written."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfgate import evaluate as E, evaluate_matched_sensitivity as S  # noqa: E402
import test_evaluate_matched as TM  # noqa: E402


class Sensitivity(TM.Matched):
    def prepare(self):
        text_q = lambda row, s, a: TM.other(row, s, a) if row == 'C' else TM.base(row, s, a)
        matched_q = lambda row, s, a: TM.third(row, s, a) if row == 'C' else TM.base(row, s, a)
        self.build(text_q=text_q, matched_q=matched_q)
        self.evaluation = self.evaluate()
        self.files = [str(self.contract_path), str(self.t / 'out' / 'EVALUATION_MATCHED.json'), str(self.t / 'out' / 'POLICY_OUTCOMES_MATCHED.csv'), str(self.t / 'sensitivity.json')]

    def test_whole_reproduced_and_a_question_left_out(self):
        self.prepare()
        S.main(self.files + ['g2'])                                       # an even train question: the two rows differ on its second sample
        r = json.loads((self.t / 'sensitivity.json').read_text())
        self.assertEqual((r['whole_population_reproduced_from_the_outcomes'], r['questions'], r['questions_left_out']), (True, {'whole': 20, 'without': 19}, ['g2']))
        self.assertEqual(r['whole']['0.02']['C_matched_minus_C_text'], self.evaluation['by_tolerance']['0.02']['contrasts']['C_matched_minus_C_text']['over_seeds'])
        whole, without = r['whole']['0.02']['C_matched_minus_C_text'], r['without_the_questions']['0.02']['C_matched_minus_C_text']
        self.assertNotEqual(whole['mean_output_tokens_first'], without['mean_output_tokens_first'])
        # by hand: the mean over the 19 kept questions of the per-question tokens of the policy that follows the outcomes
        cohort = [a for a in self.contract['attempts'] if a['cohort'] and a['group'] != 'g2']
        taken = {}
        import csv
        for line in csv.DictReader(open(self.files[2])):
            if line['row'] == 'C_matched' and float(line['tolerance']) == 0.02:
                taken.setdefault(line['attempt'], []).append(float(line['output_tokens']))
        per_question = {}
        for a in cohort:
            tokens = np.mean(taken[a['key']]) if not a['ended_before_the_decision'] else a['actions']['stop']['total_output_tokens']
            per_question.setdefault(a['group'], []).append(tokens)
        self.assertAlmostEqual(without['mean_output_tokens_first'], float(np.mean([np.mean(v) for v in per_question.values()])), places=9)

    def test_refusals(self):
        self.prepare()
        for label, argv in (('a question outside the cohort', self.files + ['g25']), ('named twice', self.files + ['g2', 'g2']), ('no question', self.files)):
            with self.subTest(label):
                with self.assertRaises(SystemExit):
                    S.main(argv)
                self.assertFalse((self.t / 'sensitivity.json').exists())
        outcomes = Path(self.files[2])
        outcomes.write_text(outcomes.read_text().replace('continue_512', 'continue_2048', 1))
        with self.assertRaises(SystemExit):
            S.main(self.files + ['g2'])
        self.assertFalse((self.t / 'sensitivity.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=1)
