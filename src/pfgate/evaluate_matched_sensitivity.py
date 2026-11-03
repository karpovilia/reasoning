#!/usr/bin/env python3
"""A descriptive sensitivity of the contrasts of the rows C_text and C_matched: the same policies, without the questions named on
the command line (PROCESS_FORECAST_GATE_20261013; the question whose text shares a long beginning with a training question of the
process predictors). numpy only. NOTHING is fitted or chosen again: the gates, their prices and so the action of every state
are those of the evaluation; a question is only left out of the population, with every one of its attempts. It is not a new
test and it does not show what the result would be had the question been absent from the fitting of the gates.
Before anything is written, the contrasts are computed here from the file of the outcomes for the WHOLE population and must
equal the numbers of EVALUATION_MATCHED.json exactly (the same draws): this shows that the file of the outcomes is read right.
The intervals without the questions use draws of their own generator.
  evaluate_matched_sensitivity.py TOKEN_CONTRACT.json EVALUATION_MATCHED.json POLICY_OUTCOMES_MATCHED.csv OUT.json QUESTION [QUESTION ...]"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import evaluate as E  # noqa: E402

PAIRS = (('C_matched', 'C_text'),)


def policies_of(path, population, rows):
    """{(tolerance, row, seed): one-hot policy [states, 3]} from the file of the outcomes, for the states of the population."""
    actions = {}
    with open(path, newline='') as handle:
        for line in csv.DictReader(handle):
            if line['row'] in rows and line['kind'] == 'decision state' and line['attempt'] in population.key_index:
                key = ('%.2f' % float(line['tolerance']), line['row'], int(line['seed']))
                table = actions.setdefault(key, np.full(len(population.states), -1, int))
                if table[population.key_index[line['attempt']]] != -1:
                    raise SystemExit('a state has two outcomes for %s' % (key,))
                table[population.key_index[line['attempt']]] = E.ACTIONS.index(line['action'])
    if any((table < 0).any() for table in actions.values()) or len(actions) != len(E.TOLERANCES) * len(rows) * len(E.SEEDS):
        raise SystemExit('the file of the outcomes does not hold one action for every state, row, seed and tolerance')
    return {key: E.one_hot(table) for key, table in actions.items()}


def contrasts(population, policies, draws):
    out = {}
    for tolerance in ('%.2f' % t for t in E.TOLERANCES):
        for first, second in PAIRS:
            mean = {row: np.mean([policies[(tolerance, row, s)] for s in E.SEEDS], 0) for row in (first, second)}
            arrays = {row: population.per_question(mean[row]) for row in (first, second)}
            out.setdefault(tolerance, {})['%s_minus_%s' % (first, second)] = E.contrast(arrays[first], arrays[second], population.paired(mean[first], mean[second]), draws)
    return out


def main(argv):
    contract_path, evaluation_path, outcomes_path, out, excluded = argv[0], argv[1], argv[2], argv[3], argv[4:]
    if Path(out).exists() or not excluded:
        raise SystemExit('the output exists, or no question is named')
    contract, evaluation = json.loads(Path(contract_path).read_text()), json.loads(Path(evaluation_path).read_text())
    if evaluation.get('format') != 'pfgate_evaluation_matched_v1' or evaluation['policy_outcomes_sha256'] != E.sha(outcomes_path) or evaluation['token_contract_sha256'] != E.sha(contract_path):
        raise SystemExit('the evaluation, its file of outcomes and the token contract do not belong together')
    cohort = [a for a in contract['attempts'] if a['cohort']]
    absent = [q for q in excluded if not any(a['group'] == q for a in cohort)]
    if absent or len(set(excluded)) != len(excluded):
        raise SystemExit('questions that are not in the cohort, or named twice: %s' % absent)
    rows = sorted({row for pair in PAIRS for row in pair})
    whole = E.Population(cohort)
    draws = np.random.default_rng(E.SEED).integers(0, whole.Q, size=(E.DRAWS, whole.Q))
    recomputed = contrasts(whole, policies_of(outcomes_path, whole, rows), draws)
    for tolerance, block in recomputed.items():
        for name, value in block.items():
            if json.loads(json.dumps(value)) != evaluation['by_tolerance'][tolerance]['contrasts'][name]['over_seeds']:
                raise SystemExit('the contrast %s at %s is not reproduced from the file of the outcomes: nothing was written' % (name, tolerance))
    kept = [a for a in cohort if a['group'] not in excluded]
    reduced = E.Population(kept)
    reduced_draws = np.random.default_rng([E.SEED, len(excluded)]).integers(0, reduced.Q, size=(E.DRAWS, reduced.Q))
    result = {'format': 'pfgate_evaluation_matched_sensitivity_v1', 'what_this_is': __doc__, 'evaluation_sha256': E.sha(evaluation_path), 'policy_outcomes_sha256': E.sha(outcomes_path), 'token_contract_sha256': E.sha(contract_path),
              'this_file_sha256': E.sha(__file__), 'questions_left_out': excluded, 'attempts_left_out': len(cohort) - len(kept), 'decision_states_left_out': len(whole.states) - len(reduced.states),
              'questions': {'whole': whole.Q, 'without': reduced.Q}, 'whole_population_reproduced_from_the_outcomes': True,
              'whole': recomputed, 'without_the_questions': contrasts(reduced, policies_of(outcomes_path, reduced, rows), reduced_draws)}
    with open(out, 'x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    main_whole, main_without = result['whole']['0.02']['C_matched_minus_C_text'], result['without_the_questions']['0.02']['C_matched_minus_C_text']
    print(json.dumps({'left_out': excluded, 'decision_states_left_out': result['decision_states_left_out'], 'whole_at_2_points': main_whole, 'without_at_2_points': main_without, 'sha256': E.sha(out)}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
