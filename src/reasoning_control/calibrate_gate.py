"""Frozen-grid validation-only selection of seconds-priced controls. No test I/O."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
from scipy.optimize import nnls

from .gate_runtime.features import VARIANT_CHANNELS
from .gate_runtime.controller import ACTIONS, choose, seconds

PRICES = [0.] + np.logspace(-5, 1, 61).tolist()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(variant):
    channels = VARIANT_CHANNELS[variant]
    return {'needs_confidence': 'confidence' in channels,
            'needs_structure': any(c.startswith(('event:', 'count:', 'span_last:', 'token_')) for c in channels)}


def quantile(values, weights, q=.9):
    v, w = np.asarray(values), np.asarray(weights)
    order = np.argsort(v, kind='stable')
    return float(v[order[min(np.searchsorted(np.cumsum(w[order]) / w.sum(), q - 1e-12), len(v) - 1)]])


def weights_for(records):
    groups = defaultdict(set)
    counts = Counter(r['group'] for r in records)
    for r in records:
        groups[r['benchmark']].add(r['group'])
    return np.array([1 / len(groups) / len(groups[r['benchmark']]) / counts[r['group']] for r in records])


def summarize(rs, weights=None):
    w = weights_for(rs) if weights is None else np.asarray(weights)
    w = w / w.sum()
    c = np.array([r['seconds'] for r in rs])
    lower = sum(x * (r['correct'] is True) for x, r in zip(w, rs))
    unknown = sum(x * (r['correct'] is None) for x, r in zip(w, rs))
    return {'accuracy_lower': float(lower), 'accuracy_upper': float(lower + unknown),
            'unknown_mass': float(unknown), 'mean_seconds': float(w @ c),
            'p90_seconds': quantile(c, w), 'mean_tokens': float(w @ [r['tokens'] for r in rs]),
            'actions': dict(Counter(r['action'] for r in rs))}


def fit_converter(x, y):
    coef, _ = nnls(np.asarray(x, float), np.asarray(y, float))
    return {'coefficients': coef.tolist(), 'features': ['intercept', 'predicted_tokens', 'available_prompt_tokens'],
            'unit': 'seconds', 'fit': 'nonnegative least squares on validation observed runtime',
            'fit_mae_seconds': float(np.mean(np.abs(np.asarray(x) @ coef - y)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    HERE, out = args.data_root, args.out
    if out.exists():
        raise FileExistsError(out)
    records = rows(HERE / 'validation_content_v2/records.jsonl')
    assert len(records) == 300 and all(r['split'] == 'validation' for r in records)
    states = defaultdict(dict)
    for r in records:
        state = (r['question_key'], r['seed'])
        states[state][r['policy'].removeprefix('profile_')] = r
    assert len(states) == 60 and all(set(s) == set(ACTIONS) for s in states.values())
    predictions = {(r['question_key'], r['seed'], r['variant']): r
                   for r in rows(HERE / 'observer_cpu_results_v1/predictions_timings.jsonl')}
    pp = {r['group']: r for r in rows(HERE / 'prompt_budget_v1/validation_predictions.jsonl')}
    prompt_timing = json.loads((HERE / 'prompt_runtime_check_v1.json').read_text())
    assert prompt_timing['passed'] and prompt_timing['test_seen'] is False
    overhead = prompt_timing['median_roundtrip_seconds']
    x, y, xp, yp = [], [], [], []
    for (key, seed), state in states.items():
        for action, r in state.items():
            a, p = r['action_result'], r['prefix']
            if not r['ended_before_checkpoint']:
                x.append([1, a['sampled_tokens'] + a['injected_tokens'], p['initial_prompt_tokens'] + r['observed_at']])
                y.append(a['wall_seconds'])
            if action in ACTIONS[:3]:
                xp.append([1, a['sampled_tokens'] + p['sampled_tokens'], p['initial_prompt_tokens']])
                yp.append(a['wall_seconds'] + p['wall_seconds'])
    converter, prompt_converter = fit_converter(x, y), fit_converter(xp, yp)

    def evaluate(selector, variant=None, prompt=False):
        result = []
        for (key, seed), state in sorted(states.items()):
            base = state['stop']
            action = selector(key, seed, base)
            r = state[action]
            cost = r['prefix']['wall_seconds'] + r['action_result']['wall_seconds']
            tokens = r['prefix']['sampled_tokens'] + r['action_result']['sampled_tokens']
            if prompt:
                cost += overhead
            if variant and not r['ended_before_checkpoint']:
                f, comp = flags(variant), r['components']
                if f['needs_confidence']:
                    cost += comp['confidence']['wall_seconds']
                    tokens += comp['confidence']['sampled_tokens']
                if f['needs_structure']:
                    cost += comp['structure']['tagger_round_trip_seconds'] + comp['structure']['segmentation_wall_seconds']
                cost += predictions[(key, seed, variant)]['median_wall_seconds']
            result.append({'group': r['group'], 'benchmark': r['benchmark'], 'seed': seed,
                           'action': 'natural_end_shared_final' if r['ended_before_checkpoint'] else action,
                           'correct': r['content_correct'], 'seconds': cost, 'tokens': tokens})
        return result

    fixed_rows = {a: evaluate(lambda k, s, r, a=a: a) for a in ACTIONS}
    fixed = {a: summarize(rs) for a, rs in fixed_rows.items()}
    baseline = min(ACTIONS, key=lambda a: (-fixed[a]['accuracy_lower'], fixed[a]['mean_seconds'], ACTIONS.index(a)))
    bounds = {'mean_seconds': .9 * fixed[baseline]['mean_seconds'], 'p90_seconds': fixed[baseline]['p90_seconds']}

    def feasible(m):
        return m['mean_seconds'] <= bounds['mean_seconds'] + 1e-9 and m['p90_seconds'] <= bounds['p90_seconds'] + 1e-9

    def select(candidates):
        ok = [r for r in candidates if feasible(r)]
        if not ok:
            # Keep every requested ablation visible, but never call this feasible.
            return {**candidates[-1], 'feasible': False, 'selection_rule': 'diagnostic fixed maximum lambda; no feasible grid point'}
        r = min(ok, key=lambda r: (-r['accuracy_lower'], r['mean_seconds'], r['p90_seconds'], r.get('price', 0)))
        return {**r, 'feasible': True}

    trials, selected = {}, {}
    for policy_name in list(VARIANT_CHANNELS) + [v + '__a3' for v in VARIANT_CHANNELS]:
        variant = policy_name.removesuffix('__a3')
        n_actions = 3 if policy_name.endswith('__a3') else 5
        candidates = []
        for price in PRICES:
            def selector(key, seed, row):
                if row['ended_before_checkpoint']:
                    return 'stop'
                pred = predictions[(key, seed, variant)]['prediction']
                if pred.get('fallback'):
                    return 'continue_2048'
                c = seconds(pred['cost'], row['prefix']['initial_prompt_tokens'] + row['observed_at'], converter)
                return choose(pred['p_success'][:n_actions], c[:n_actions], price)[0]
            candidates.append({'price': price, **summarize(evaluate(selector, variant=variant))})
        trials[policy_name] = candidates; selected[policy_name] = select(candidates)
    trials['prompt_budget'] = []
    for price in PRICES:
        def selector(key, seed, row):
            pred = pp[row['group']]
            c = seconds(pred['predicted_total_generated_tokens'], row['prefix']['initial_prompt_tokens'], prompt_converter)
            return choose(pred['p_success'], c, price)[0]
        trials['prompt_budget'].append({'price': price, **summarize(evaluate(selector, prompt=True))})
    selected['prompt_budget'] = select(trials['prompt_budget'])
    mixtures = []
    for i in range(21):
        for j in range(21 - i):
            probs = [i / 20, j / 20, (20 - i - j) / 20]
            rr, ww = [], []
            for a, prob in zip(ACTIONS, probs):
                if prob:
                    rr.extend(fixed_rows[a]); ww.extend(weights_for(fixed_rows[a]) * prob)
            mixtures.append({'probabilities': probs, **summarize(rr, ww)})
    ok = [m for m in mixtures if feasible(m)]
    best_mix = min(ok, key=lambda m: (-m['accuracy_lower'], m['mean_seconds'], m['p90_seconds'])) if ok else None
    baseline_candidates = {f'fixed_{a}': v for a, v in fixed.items() if feasible(v)}
    if selected['prompt_budget']['feasible']:
        baseline_candidates['prompt_budget'] = selected['prompt_budget']
    if best_mix is not None:
        baseline_candidates['mixture_selected'] = best_mix
    primary = min(baseline_candidates, key=lambda n: (-baseline_candidates[n]['accuracy_lower'], baseline_candidates[n]['mean_seconds'], n)) if baseline_candidates else None
    summary = {'state': 'SELECTED_BEFORE_TEST; FULL-RUNTIME VALIDATION GATE REQUIRED',
               'test_seen': False, 'validation_questions': 30, 'question_seed_states': 60,
               'prices_per_second': PRICES, 'converter': converter, 'prompt_converter': prompt_converter,
               'budget_reference_fixed_action': baseline, 'bounds': bounds, 'fixed': fixed,
               'selected': selected, 'best_mixture': best_mix, 'primary_no_state_comparator': primary,
               'prompt_runtime_roundtrip_seconds': overhead,
               'unknowns': 'included as accuracy interval; select by lower bound, never relabel unknown false',
               'cost_scope': 'historical component timing with all actual observation costs; confirm expanded timer overhead <=5% before test',
               'infeasible': 'retain diagnostic lambda=10; report constraint violation, never relax bounds',
               'source_sha256': {str(p): sha(p) for p in [Path(__file__), HERE / 'PROTOCOL.md', HERE / 'validation_content_v2/records.jsonl',
                                  HERE / 'observer_cpu_results_v1/predictions_timings.jsonl', HERE / 'prompt_budget_v1/validation_predictions.jsonl',
                                  HERE / 'prompt_runtime_check_v1.json']}}
    out.mkdir()
    for name, value in [('SUMMARY.json', summary), ('all_prices.json', trials), ('all_mixtures.json', mixtures)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({k: summary[k] for k in ('bounds', 'budget_reference_fixed_action', 'primary_no_state_comparator', 'selected')}, indent=2))


if __name__ == '__main__':
    main()
