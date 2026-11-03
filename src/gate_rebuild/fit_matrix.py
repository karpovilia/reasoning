"""Entry point of the fitting stage: the trained rows of the first matrix on the train and dev parts of the bank, the fixed
policies, and the dev report.

Reads the bank files of the train extension, the FROZEN cohort, the state caches through their trusted manifest digests and the
cost model; writes into a NEW directory the weights of every row and seed, the fit records and DEV_REPORT.json. It never reads a
test example and has no option to do so. It runs no observer: a row whose state cache or digest is absent is reported as not
run, never substituted.

    python -B -m gate_rebuild.fit_matrix --trainval BANK/trainval_ext.jsonl --replay BANK/replay_ext.jsonl \\
        --cohort COHORT.json --cohort-sha256 H --panel-sha256 H --caches CACHE_DIR --cache-digests DIGESTS.json \\
        --cost-model COST_MODEL.json --out NEW_DIR [--rows G00,G11] [--estimand lower_bound]

What the report holds for every row and operating point (review findings GATE-V1-R1, R2, R3, R4, R5, R7):
  - one cohort for all rows, built from the full roster; the same dev attempts under every row;
  - full cost in the unit of the cost model with its parts: common, looking (segmentation, observer, every fitted member of the
    gate), action; the gate is charged to every trained row, also to a row without an observer;
  - the mean ceiling AND the ceiling of the 0.9 quantile of the cost of one attempt, both applied in the selection;
  - the declared selection estimand, both accuracy bounds, the conditional accuracy and the coverage of every price;
  - the whole population and the decision states only;
  - the strongest fixed policy under the same ceilings: a pure action or a mixture of two, selected by the same rule;
  - the paired comparison of the row with that fixed policy. These are DEV numbers that were used for selection: they are
    development diagnostics, not evidence;
  - the SHA256 of every input, of the cohort and of every cache manifest, with the binding of each cache.

Version 3 (review of the second package). The bank files must have the SHA256 the cohort was built from, and the bank must hold
EXACTLY the attempts of the cohort; every cache must have been made from the pinned panel. The tail ceiling is an explicit
allowance: the 0.9 quantile of the cost of one attempt may be up to TAIL_TOLERANCE times that of always continue_2048. This is
NOT "the tail is unchanged". The same selection is therefore repeated with the allowance 1.00 on the same inputs, prices and
controls and reported next to the first; a point that is infeasible there is reported as infeasible. The allowance is fixed in
the code before any outcome is read and is not chosen by the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import VERSION
from .cohort import apply_cohort, check_bank_files
from .contract import ACTIONS, STATE_SOURCES
from .cost import CostModel
from .data import bank_binding, file_sha256, load_trainext, read_state_cache, with_states
from .evaluation import non_inferior, non_inferior_on_known_labels_only, paired_difference, plain, policy_value, superior
from .policy import ESTIMANDS, decide, select_mixture, select_price
from .training import CV_SEEDS, FOLDS, GRID, SEEDS, fit, predict

# row -> state source, encoder and options. The first required matrix of the coordinator's scope decision of 2026-10-12.
ROWS = {
    'G00': ('old_lexical', 'gru', {}),
    'G00_aggregate': ('old_lexical_aggregate', 'aggregate', {}),       # the input form of the best historical gate, refitted; weights of that gate are not used
    'G10': ('lora_outputs', 'gru', {}),
    'G11': ('lora_process_state', 'process_state', {'reduce': 64}),
    'C_matched_random_dynamics': ('random_process_state', 'process_state', {'reduce': 64}),
    'C_text': ('text_only', 'process_state', {'reduce': 64}),
    'C_question_length_budget': ('question_length_budget', 'aggregate', {'reduce': 64}),
    'C_length_budget': ('length_budget', 'aggregate', {}),
}
POINTS = ('continue_512', 'midpoint', 'ninety_percent_of_continue_2048')
MARGIN, GAIN = 0.02, 0.03
TAIL_TOLERANCE = 1.10                    # accepted for the exploratory train and dev protocol: an explicit allowance of ten percent above always continue_2048
TAIL_SENSITIVITY = 1.00                  # the same selection with no allowance, always reported next to it


def ceilings(fixed, tolerance):
    """(mean ceiling, tail ceiling) of the three operating points, from the fixed actions on the same dev attempts. The mean
    ceiling moves with the point. The tail ceiling is one number for all points: `tolerance` times the 0.9 quantile of the cost of
    one attempt under always continue_2048."""
    low, high = fixed['continue_512'], fixed['continue_2048']
    tail = tolerance * high['p90_attempt_cost']
    return {'continue_512': (low['mean_total_cost'], tail), 'midpoint': ((low['mean_total_cost'] + high['mean_total_cost']) / 2, tail),
            'ninety_percent_of_continue_2048': (0.9 * high['mean_total_cost'], tail)}


def cost_table(examples, extras, looking):
    """[N, A] full cost of the attempt under each action: common + looking (one number per state, 0 for a fixed policy) + action."""
    table = np.empty((len(examples), len(ACTIONS)))
    for i, (e, extra) in enumerate(zip(examples, extras)):
        table[i] = extra['common_cost'] + looking[i] + np.asarray(e.cost, float)
    return table


def comparison(value, reference):
    """Paired difference of a row against a reference policy on the same dev attempts, with the three declared verdicts."""
    difference = paired_difference(value, reference)
    return dict(difference, margin=MARGIN, gain=GAIN, non_inferior_population=non_inferior(difference, MARGIN),
                non_inferior_on_known_labels_only=non_inferior_on_known_labels_only(difference, MARGIN), superior_population_by_gain=superior(difference, GAIN))


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--trainval', '--replay', '--cohort', '--cohort-sha256', '--panel-sha256', '--caches', '--cache-digests', '--cost-model', '--out'):
        p.add_argument(name, required=True)
    p.add_argument('--rows', default=','.join(ROWS))
    p.add_argument('--estimand', default='lower_bound', choices=sorted(ESTIMANDS))
    p.add_argument('--epochs', type=int, default=None)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        sys.exit('refusing to write into the existing %s' % out)
    rows = args.rows.split(',')
    if any(row not in ROWS for row in rows):
        sys.exit('unknown row')
    if file_sha256(args.cohort) != args.cohort_sha256:
        sys.exit('the cohort file is not the frozen one')
    cost_model = CostModel(**json.loads(Path(args.cost_model).read_text()))
    cohort, digests = json.loads(Path(args.cohort).read_text()), json.loads(Path(args.cache_digests).read_text())
    try:
        check_bank_files(cohort, {'trainval': args.trainval, 'replay': args.replay})
    except ValueError as exc:
        sys.exit(str(exc))
    active, early = load_trainext(args.trainval, args.replay, cost_model)
    if any(e.split == 'test' for e, _ in active) or any(e['split'] == 'test' for e in early):
        sys.exit('the bank holds a test example: this stage does not read test')
    active, early, counted = apply_cohort(cohort, active, early)
    extras = {e.key: extra for e, extra in active}
    examples = [e for e, _ in active]
    base_train, base_dev = [e for e in examples if e.split == 'train'], [e for e in examples if e.split == 'dev']
    early_dev = [e for e in early if e['split'] == 'dev']
    dev_extras = [extras[e.key] for e in base_dev]
    success, groups, supported = np.stack([e.success for e in base_dev]), [e.group for e in base_dev], np.stack([e.supported for e in base_dev])
    early_kw = dict(early_success=np.array([e['success'] for e in early_dev]), early_cost=np.array([e['cost'] for e in early_dev]), early_groups=[e['group'] for e in early_dev])
    free = cost_table(base_dev, dev_extras, np.zeros(len(base_dev)))
    fixed = {name: policy_value(np.full(len(base_dev), a), success, free, groups, **early_kw) for a, name in enumerate(ACTIONS)}
    plans = {}
    for label, tolerance in (('primary', TAIL_TOLERANCE), ('sensitivity', TAIL_SENSITIVITY)):
        points = ceilings(fixed, tolerance)
        plans[label] = {'tolerance': tolerance, 'points': points,
                        'mixtures': {name: select_mixture({a: fixed[a]['table'] for a in ACTIONS}, mean, tail, args.estimand)[0] for name, (mean, tail) in points.items()}}

    def points_report(plan):
        return {name: {'mean_ceiling': mean, 'tail_ceiling': tail,
                       'best_fixed_policy': None if plan['mixtures'][name] is None else {'actions': list(plan['mixtures'][name]['actions']), 'share_of_first': plan['mixtures'][name]['share_of_first'],
                                                                                         'value': plain(plan['mixtures'][name]['value'])}}
                for name, (mean, tail) in plan['points'].items()}
    out.mkdir(parents=True)
    report = {'version': VERSION, 'cost_unit': cost_model.unit, 'kv_cache': cost_model.kv_cache, 'selection_estimand': args.estimand, 'tail_tolerance': TAIL_TOLERANCE,
              'tail_is': 'an allowance of growth of the 0.9 quantile of the cost of one attempt over always continue_2048, fixed before any outcome; not an unchanged tail',
              'panel_sha256': args.panel_sha256,
              'inputs_sha256': {name: file_sha256(getattr(args, name)) for name in ('trainval', 'replay', 'cohort', 'cache_digests', 'cost_model')},
              'cohort': dict(counted, counts=cohort['counts'], train_questions=len({e.group for e in base_train}), dev_questions_with_a_decision_state=len(set(groups)),
                             dev_attempts_that_ended_early=len(early_dev)),
              'what_these_numbers_are': 'dev numbers used for selection: development diagnostics, not evidence',
              'fixed_actions': {name: plain(value) for name, value in fixed.items()},
              'operating_points': points_report(plans['primary']),
              'tail_sensitivity': {'tolerance': TAIL_SENSITIVITY, 'operating_points': points_report(plans['sensitivity'])},
              'caches': {}, 'rows': {}, 'not_run': {}}
    for row in rows:
        source, encoder, options = ROWS[row]
        train, dev, segments = base_train, base_dev, {e.key: 0 for e in examples}
        if source != 'length_budget':
            directory = Path(args.caches) / source
            if source not in digests or not (directory / ('%s.manifest.json' % source)).is_file():
                report['not_run'][row] = 'no state cache of %s in %s or no trusted digest for it' % (source, directory)
                continue
            states, manifest = read_state_cache(directory, source, {e.key: bank_binding(e, extras[e.key]) for e in examples}, digests[source])
            if manifest['binding']['panel_sha256'] != args.panel_sha256:
                sys.exit('the state cache of %s was made from another panel' % source)
            report['caches'][source] = {'manifest_sha256': digests[source], 'npz_sha256': manifest['npz_sha256'], 'binding': manifest['binding'], 'states': manifest['states']}
            segments = {key: manifest['records'][key]['segments'] for key in states}
            train, dev = with_states(base_train, source, states), with_states(base_dev, source, states)
        torch.manual_seed(0)
        models, record = fit(train, dev, source, encoder, cost_scale=cost_model.seconds_per_solver_token * 1024., seeds=SEEDS, grid=GRID, folds=FOLDS, cv_seeds=CV_SEEDS,
                             **({'epochs': args.epochs} if args.epochs else {}), **options)
        for seed, model in zip(record['seeds'], models):
            torch.save({'state_dict': model.state_dict(), 'metadata': model.metadata(), 'row': row}, out / ('%s_seed%d.pt' % (row, seed['seed'])))
        prediction = predict(models, dev, source)
        observer = source if STATE_SOURCES[source]['observer'] is not None else None
        looking = np.array([cost_model.looking(observer, segments[e.key], len(models)) for e in dev])       # every trained row pays for its gate
        own = cost_table(dev, dev_extras, looking)
        value_of = lambda actions: policy_value(actions, success, own, groups, **early_kw)
        entry = {'fit': record, 'state_source': source, 'looking_cost_per_decision_state': {'mean': float(looking.mean()), 'max': float(looking.max())}}
        for label, key in (('primary', 'operating_points'), ('sensitivity', 'tail_sensitivity')):
            plan, entry[key] = plans[label], {}
            for name, (mean_ceiling, tail_ceiling) in plan['points'].items():
                price, table = select_price(prediction, supported, value_of, mean_ceiling, tail_ceiling, args.estimand)
                point = {'price': price, 'feasible': price is not None, 'feasible_prices': sum(1 for r in table if r['feasible']), 'prices': table, 'dev_value': None,
                         'against_best_fixed_policy': None}
                if price is not None:
                    chosen = decide(prediction, price, supported)
                    value = value_of(chosen)
                    without = policy_value(chosen, success, free, groups, **early_kw)            # the same actions without paying for looking
                    point['dev_value'] = dict(plain(value), looking_part_of_mean_total_cost=value['mean_total_cost'] - without['mean_total_cost'],
                                              p90_relative_to_always_continue_2048=value['p90_attempt_cost'] / fixed['continue_2048']['p90_attempt_cost'])
                    if plan['mixtures'][name] is not None:
                        point['against_best_fixed_policy'] = comparison(value, plan['mixtures'][name]['value'])
                entry[key][name] = point
        report['rows'][row] = entry
    (out / 'DEV_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + '\n')
    print(json.dumps({'rows_fitted': sorted(report['rows']), 'not_run': report['not_run']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
