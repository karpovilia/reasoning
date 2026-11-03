#!/usr/bin/env python3
"""Evaluation of the rows C_text and C_matched of PROCESS_FORECAST_GATE_20261013 next to the accepted rows B and C, in ACTUAL
TOKENS of the saved branches. numpy only; no fit, no model. Every rule is the accepted one: this file imports the accepted
evaluation (pfgate/evaluate.py) and uses ITS population, ITS choice of lambda on dev, ITS decision rule, ITS sharp bounds and
ITS resamples of questions (the same 10000 draws from the same generator seed); nothing of them is written again here. What
is new is only where the members of a row come from:
  B, C        the accepted directory of the 45 fits (C = B + the forecast heads of P_ES without text)
  C_text      row C of the directory fitted on the variant `text` of the gate table (B + the heads of P_text, cell T)
  C_matched   row C of the directory fitted on the variant `matched` (B + the heads of P_ES_matched, cell TES)
The three directories are each checked by the accepted checked_members against their own PINS file; the PINS of a variant must
name the accepted PINS and differ from it in exactly four files (configuration, data contract, gate table, its manifest), and
the manifest of its table must name the cell of its heads.
Checks made before anything is written:
  1. THE ACCEPTED NUMBERS ARE REPRODUCED. The rows A, B, C are computed here from the accepted directory and every number of
     the contrasts C-B, B-A, C-A and of the rows against no control must equal the accepted EVALUATION.json exactly. This shows
     that the loop below is the accepted computation.
  2. ROWS A AND B OF A VARIANT ARE THE ACCEPTED ROWS. A variant differs from the accepted table only in columns the rows A
     and B do not read; their members are compared with the accepted ones (final weights, predictions). The comparison is
     reported; contrasts against B use the ACCEPTED B, and the same contrasts against the B of the variant's own run are
     reported next to them, so that a run-to-run difference of the fits is visible and not hidden.
Main contrast, fixed before any result: C_matched minus C_text at the tolerance of 2 points (what explicit events and spans in
the predictor add over a predictor that reads the same text). Reported with it: C_text minus B, C_matched minus B, C_matched
minus C, C_text minus C; the tolerances 0 and 1 point; every seed; the benchmarks. This is development on a bank that was
looked at before, not an independent test, and the events and spans the predictors were trained on are retrospective.
  evaluate_matched.py TOKEN_CONTRACT.json ACCEPTED_EVALUATION.json ACCEPTED_FITS ACCEPTED_PINS SHA256 TEXT_FITS TEXT_PINS SHA256 MATCHED_FITS MATCHED_PINS SHA256 OUT_DIR"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import evaluate as E  # noqa: E402

SEEDS, FOLDS, ACTIONS = E.SEEDS, E.FOLDS, E.ACTIONS
NEW = {'C_text': ('text', 'T'), 'C_matched': ('matched', 'TES')}
CONTRASTS = (('C_matched', 'C_text'), ('C_text', 'B'), ('C_matched', 'B'), ('C_matched', 'C'), ('C_text', 'C'))
MAIN = ('C_matched', 'C_text', '0.02')


def by_cell(parsed):
    members = {(m['row'], m['seed'], int(m['call'].split('_')[1])): m for m in parsed.values()}
    if len(members) != len(parsed) or set(members) != {(row, seed, fold) for row in E.ROWS for seed in SEEDS for fold in range(FOLDS)}:
        raise SystemExit('the members are not exactly the 45 expected ones')
    return members


def policy_of(train, dev, fold_of_state, members, row, seed, tolerance):
    """The cross-fitted policy of (row, seed) at a tolerance: exactly the loop of the accepted evaluation. Returns (one-hot
    policy [states, 3], the record of the choice of every call)."""
    actions, calls = np.full(len(train.states), -1, int), []
    for fold in range(FOLDS):
        m = members[(row, seed, fold)]
        devp, held = m['parts']['dev'], m['parts']['held_out']
        if sorted(devp['keys']) != sorted(dev.key_index) or len(set(devp['keys'])) != len(devp['keys']):
            raise SystemExit('the dev predictions of %s are not on exactly the dev states' % m['name'])
        order = np.argsort([dev.key_index[k] for k in devp['keys']])
        dq, dm = np.asarray(devp['success'], float)[order], np.asarray(devp['cost'], float)[order]
        best, table = E.choose(dev, [(lam, E.one_hot(E.decide(dq, dm, lam))) for lam in E.LAMBDAS], tolerance)
        index = [train.key_index[k] for k in held['keys']]
        if not (fold_of_state[index] == fold).all() or (actions[index] != -1).any():
            raise SystemExit('the held-out states of %s are not the states of its fold' % m['name'])
        actions[index] = E.decide(held['success'], held['cost'], E.LAMBDAS[best]) if best is not None else E.REFERENCE
        calls.append({'seed': seed, 'call': fold, 'status': 'ADMISSIBLE' if best is not None else 'NO_ADMISSIBLE_LAMBDA', 'lambda': E.LAMBDAS[best] if best is not None else None,
                      'dev': table[best] if best is not None else None, 'admissible_lambdas': sum(t['admissible'] for t in table), 'held_out_states': len(index)})
    if (actions < 0).any():
        raise SystemExit('a state of the cohort got no decision for %s seed %d' % (row, seed))
    return E.one_hot(actions), calls


def same_members(accepted, variant, rows=('A', 'B')):
    """How the members of the rows A and B of a variant run compare with the accepted members."""
    weights, largest, n = 0, 0.0, 0
    for (row, seed, fold), m in accepted.items():
        if row not in rows:
            continue
        other, n = variant[(row, seed, fold)], n + 1
        weights += m['final_state_sha256'] == other['final_state_sha256']
        for part in ('dev', 'held_out'):
            if m['parts'][part]['keys'] != other['parts'][part]['keys']:
                raise SystemExit('the member %s of a variant is not on the states of the accepted member' % other['name'])
            for field in ('success', 'cost'):
                largest = max(largest, float(np.abs(np.asarray(m['parts'][part][field], float) - np.asarray(other['parts'][part][field], float)).max()))
    return {'members_compared': n, 'members_with_the_same_final_weights': int(weights), 'largest_absolute_difference_of_a_prediction': largest, 'identical': bool(weights == n and largest == 0.0)}


def main(argv):
    contract_path, accepted_evaluation_path, out_dir = argv[0], argv[1], Path(argv[11])
    if out_dir.exists():
        raise SystemExit('the output directory exists')
    contract = json.loads(Path(contract_path).read_text())
    given = {'accepted': argv[2:5], 'text': argv[5:8], 'matched': argv[8:11]}
    members, manifests, pins = {}, {}, {}
    for name, (fits, pins_path, pins_sha) in given.items():                # before anything is written
        parsed, manifests[name], _ = E.checked_members(contract_path, contract, fits, pins_path, pins_sha)
        members[name], pins[name] = by_cell(parsed), json.loads(Path(pins_path).read_text())
    accepted_pins_sha = given['accepted'][2]
    variant_tables = {}
    for row, (name, cell) in NEW.items():
        p, a = pins[name], pins['accepted']
        differ = sorted(k for k in a['roles'] if p['roles'][k] != a['roles'][k])
        shared = sum(1 for path, value in p['files'].items() if a['files'].get(path) == value)
        table = json.loads(Path(p['roles']['gate_table_manifest']).read_text())
        if p.get('accepted_pins_sha256') != accepted_pins_sha or differ != ['config', 'data_contract', 'gate_table', 'gate_table_manifest'] or shared != len(a['files']) - 4 or len(p['files']) != len(a['files']) \
                or table.get('heads_cell') != cell or table.get('base_table_sha256') != a['files'][a['roles']['gate_table']] or table.get('format') != 'pfgate_gate_table_v2_variant':
            raise SystemExit('the PINS of the variant %s are not the accepted PINS with the table of cell %s' % (name, cell))
        variant_tables[row] = {'variant': name, 'cell': cell, 'table_sha256': table['npz_sha256'], 'heads': table['pes_heads'], 'pins_sha256': given[name][2], 'members_manifest_sha256': E.sha(Path(given[name][0]) / 'MEMBERS_MANIFEST.json')}
    train = E.Population([a for a in contract['attempts'] if a['cohort']])
    dev = E.Population([a for a in contract['attempts'] if a['split'] == 'validation'])
    fold_of_state = np.array([a['outer_fold'] for a in train.states], int)
    rng = np.random.default_rng(E.SEED)
    draws = rng.integers(0, train.Q, size=(E.DRAWS, train.Q))               # the draws of the accepted evaluation: the first call of its generator
    reference_train = E.constant(len(train.states), E.REFERENCE)
    ref_arrays = train.per_question(reference_train)
    sources = {'A': ('accepted', 'A'), 'B': ('accepted', 'B'), 'C': ('accepted', 'C'), 'C_text': ('text', 'C'), 'C_matched': ('matched', 'C'), 'B_of_the_text_run': ('text', 'B'), 'B_of_the_matched_run': ('matched', 'B')}
    accepted_evaluation = json.loads(Path(accepted_evaluation_path).read_text())
    if accepted_evaluation.get('members_manifest_sha256') != E.sha(Path(given['accepted'][0]) / 'MEMBERS_MANIFEST.json') or accepted_evaluation.get('token_contract_sha256') != E.sha(contract_path):
        raise SystemExit('the accepted evaluation is not the evaluation of the accepted fits and of this token contract')
    result = {'format': 'pfgate_evaluation_matched_v1', 'what_this_is': __doc__, 'token_contract_sha256': E.sha(contract_path), 'this_file_sha256': E.sha(__file__), 'accepted_evaluate_sha256': E.sha(E.__file__),
              'accepted_evaluation_sha256': E.sha(accepted_evaluation_path), 'accepted_pins_sha256': accepted_pins_sha, 'accepted_members_manifest_sha256': accepted_evaluation['members_manifest_sha256'],
              'new_rows': variant_tables, 'main_contrast': {'first': MAIN[0], 'second': MAIN[1], 'tolerance': MAIN[2]},
              'rows_A_and_B_of_the_variant_runs_against_the_accepted_members': {name: same_members(members['accepted'], members[name]) for name in ('text', 'matched')},
              'train': {'questions': train.Q, 'decision_states': len(train.states), 'early_attempts': len(train.early)}, 'dev': {'questions': dev.Q, 'decision_states': len(dev.states), 'early_attempts': len(dev.early)},
              'by_tolerance': {}}
    outcomes, mismatches = [], []
    for tolerance in E.TOLERANCES:
        tag = '%.2f' % tolerance
        policies, arrays, choices = {}, {}, {}
        for row, (name, source_row) in sources.items():
            choices[row] = []
            for seed in SEEDS:
                policies[(row, seed)], calls = policy_of(train, dev, fold_of_state, members[name], source_row, seed, tolerance)
                arrays[(row, seed)] = train.per_question(policies[(row, seed)])
                choices[row] += calls
        mean_policy = {row: np.mean([policies[(row, s)] for s in SEEDS], 0) for row in sources}
        mean_arrays = {row: train.per_question(mean_policy[row]) for row in sources}

        def between(first, second):
            return dict(over_seeds=E.contrast(mean_arrays[first], mean_arrays[second], train.paired(mean_policy[first], mean_policy[second]), draws),
                        by_seed={str(s): E.contrast(arrays[(first, s)], arrays[(second, s)], train.paired(policies[(first, s)], policies[(second, s)]), draws) for s in SEEDS},
                        states_where_the_two_differ_weighted_share_by_seed={str(s): float((train.ws * (policies[(first, s)] != policies[(second, s)]).any(1)).sum() / train.ws.sum()) for s in SEEDS})
        # check 1: the accepted numbers, from the accepted members, by this loop
        accepted_block = accepted_evaluation['by_tolerance'][tag]
        for first, second in (('C', 'B'), ('B', 'A'), ('C', 'A')):
            mine = between(first, second)
            if json.loads(json.dumps(mine)) != accepted_block['contrasts']['%s_minus_%s' % (first, second)]:
                mismatches.append('%s minus %s at the tolerance %s' % (first, second, tag))
        for row in ('A', 'B', 'C'):
            against = E.contrast(mean_arrays[row], ref_arrays, train.paired(mean_policy[row], reference_train), draws)
            if json.loads(json.dumps(against)) != accepted_block['rows'][row]['against_no_control_over_seeds'] or json.loads(json.dumps(choices[row])) != accepted_block['choices'][row]:
                mismatches.append('row %s at the tolerance %s' % (row, tag))
        block = {'rows': {}, 'contrasts': {}, 'contrasts_against_the_B_of_the_same_run': {}, 'choices': {row: choices[row] for row in ('C_text', 'C_matched', 'B_of_the_text_run', 'B_of_the_matched_run')}}
        for row in ('B', 'C', 'C_text', 'C_matched'):
            against = E.contrast(mean_arrays[row], ref_arrays, train.paired(mean_policy[row], reference_train), draws)
            every_call = all(c['status'] == 'ADMISSIBLE' for c in choices[row])
            block['rows'][row] = {'over_seeds': {'accuracy_lower_bound': float(mean_arrays[row]['lower'].mean()), 'accuracy_upper_bound': float(mean_arrays[row]['upper'].mean()), 'mean_output_tokens': float(mean_arrays[row]['output'].mean()),
                                                 'mean_reasoning_tokens': float(mean_arrays[row]['reasoning'].mean()), 'mean_answer_tokens': float(mean_arrays[row]['answer'].mean())},
                                  'by_seed': {str(s): dict(E.summary(train, policies[(row, s)], draws)[0], against_no_control=E.contrast(arrays[(row, s)], ref_arrays, train.paired(policies[(row, s)], reference_train), draws)) for s in SEEDS},
                                  'against_no_control_over_seeds': against, 'calls_without_an_admissible_lambda': sum(c['status'] != 'ADMISSIBLE' for c in choices[row]), 'every_call_admissible': every_call,
                                  'criterion_10_percent_output_tokens_within_the_tolerance': {
                                      'point': bool(every_call and against['saving_of_output_tokens'] >= 0.10 and against['accuracy_difference_lower_bound'] >= -tolerance),
                                      'with_intervals': bool(every_call and against['saving_of_output_tokens_interval_95'][0] >= 0.10 and against['lower_bound_2.5_percentile_over_questions'] >= -tolerance)},
                                  'by_benchmark': {}}
            bounds = train.paired(mean_policy[row], reference_train)
            for name in sorted(set(train.benchmark.values())):
                qs = np.array([i for i, q in enumerate(train.questions) if train.benchmark[q] == name])
                sub = np.random.default_rng([E.SEED, len(qs)]).integers(0, len(qs), size=(2000, len(qs)))      # a generator of its own: the draws of the main intervals are not touched
                block['rows'][row]['by_benchmark'][name] = dict(questions=len(qs), **E.contrast(E.restricted(train, mean_arrays[row], qs), E.restricted(train, ref_arrays, qs), tuple(b[qs] for b in bounds), sub))
        for first, second in CONTRASTS:
            block['contrasts']['%s_minus_%s' % (first, second)] = between(first, second)
        for row, own in (('C_text', 'B_of_the_text_run'), ('C_matched', 'B_of_the_matched_run')):
            block['contrasts_against_the_B_of_the_same_run']['%s_minus_%s' % (row, own)] = between(row, own)
        result['by_tolerance'][tag] = block
        for row in ('C_text', 'C_matched'):
            for seed in SEEDS:
                taken = policies[(row, seed)].argmax(1)
                lam = {c['call']: c for c in choices[row] if c['seed'] == seed}
                for i, a in enumerate(train.states):
                    b, c = a['actions'][ACTIONS[taken[i]]], lam[int(fold_of_state[i])]
                    outcomes.append([tolerance, row, seed, a['key'], a['group'], a['benchmark'], 'decision state', int(fold_of_state[i]), ACTIONS[taken[i]], b['correct'], b['total_output_tokens'], b['total_reasoning_tokens'],
                                     b['answer_tokens'], c['lambda'] if c['lambda'] is not None else '', c['status']])
    if mismatches:
        raise SystemExit('the accepted numbers are NOT reproduced by this file, nothing was written: %s' % mismatches)
    result['the_accepted_numbers_are_reproduced'] = {'contrasts_checked': 9, 'rows_checked': 9, 'what': 'C-B, B-A, C-A over seeds, by seed and the shares of differing states; rows A, B, C against no control and their choices of lambda; three tolerances'}
    out_dir.mkdir()
    with open(out_dir / 'POLICY_OUTCOMES_MATCHED.csv', 'x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tolerance', 'row', 'seed', 'attempt', 'question', 'benchmark', 'kind', 'outer_fold', 'action', 'correct', 'output_tokens', 'reasoning_tokens', 'answer_tokens', 'lambda', 'call_status'])
        writer.writerows(outcomes)
    result['policy_outcomes_sha256'] = E.sha(out_dir / 'POLICY_OUTCOMES_MATCHED.csv')
    with open(out_dir / 'EVALUATION_MATCHED.json', 'x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    main_block = result['by_tolerance'][MAIN[2]]
    print(json.dumps({'rows_A_B_of_the_variant_runs': result['rows_A_and_B_of_the_variant_runs_against_the_accepted_members'], 'rows_at_2_points': {r: main_block['rows'][r]['against_no_control_over_seeds'] for r in main_block['rows']},
                      'contrasts_at_2_points': {k: v['over_seeds'] for k, v in main_block['contrasts'].items()}, 'sha256': E.sha(out_dir / 'EVALUATION_MATCHED.json')}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
