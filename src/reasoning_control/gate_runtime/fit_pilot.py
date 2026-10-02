"""Train gate observers on saved, question-disjoint train/validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from . import features as F
from .features import ACTIONS, VARIANTS
from . import labels as L
from .models import save_checkpoint
from .training import EPOCHS, SEEDS, TUNING_GRID, evaluate, predict, tune

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
CODE = ('features.py', 'models.py', 'training.py', 'labels.py', 'fit_pilot.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observer', required=True)
    ap.add_argument('--confidence', required=True)
    ap.add_argument('--labels', required=True, help='branch-level JSONL: key, action, seed, correctness, cost')
    ap.add_argument('--correctness-field', default='correct')
    ap.add_argument('--cost-field', default='additional_tokens')
    ap.add_argument('--cost-unit', default=None, help='defaults to the cost field name; never renamed')
    ap.add_argument('--restrict-keys-to', default=None, help='second observer JSONL; keep only shared keys')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--trainval-only', action='store_true',
                    help='Freeze weights using train+validation only; refuse any test observations/unused label keys')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    cost_unit = args.cost_unit or args.cost_field

    observations, provenance = L.load_observations(args.observer, args.confidence, require_confidence=True)
    restriction = None
    if args.restrict_keys_to:
        other, other_prov = L.load_observations(args.restrict_keys_to, args.confidence, require_confidence=True)
        shared, exclusions = L.common_keys(observations, other)
        restriction = {'path': args.restrict_keys_to, 'sha256': other_prov['observer_sha256'],
                       'shared_keys': len(shared), 'exclusions': exclusions}
        observations = {k: observations[k] for k in shared}
    labels = L.load_branch_labels(args.labels, correctness_field=args.correctness_field,
                                  cost_field=args.cost_field)
    examples, info = L.build_examples(observations, labels)
    parts = L.split(examples)
    manifest = {
        'feature_contract': F.describe(), 'tuning_grid': TUNING_GRID, 'seeds': list(SEEDS),
        'epochs': args.epochs, 'cost_unit': cost_unit, 'correctness_field': args.correctness_field,
        'cost_field': args.cost_field, 'command': ' '.join(sys.argv),
        'started': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'inputs': {'observer': {'path': args.observer, **provenance},
                   'labels': {k: v for k, v in labels.items() if k != 'index'},
                   'restriction': restriction},
        'code_sha256': {name: L.sha256(HERE / name) for name in CODE},
        'label_source_kind': L.label_source_kind(observations),
        'examples': info,
        'splits': {s: {'questions': len({e.group for e in v}), 'prefixes': len(v),
                       'by_now': {str(n): sum(e.observation.now == n for e in v) for n in sorted({e.observation.now for e in v})},
                       'known_success_prefixes_per_action': np.isfinite(np.stack([e.success for e in v])).sum(0).tolist() if v else [],
                       'known_cost_prefixes_per_action': np.isfinite(np.stack([e.cost for e in v])).sum(0).tolist() if v else []}
                   for s, v in parts.items()},
        'not_an_efficacy_result': True, 'labels_are_llm_not_human': True,
        'selection_uses_test': False, 'results': {},
    }
    if args.trainval_only and (parts['test'] or info['label_keys_not_used']):
        raise ValueError('Trainval-only freeze refuses test observations or labels outside the train/validation input')
    manifest['trainval_only'] = args.trainval_only
    if not parts['train'] or not parts['validation'] or (not args.trainval_only and not parts['test']):
        raise ValueError(f'Need all three splits, got {manifest["splits"]}')
    test = parts['test']
    if test:
        ty, tc, tg = np.stack([e.success for e in test]), np.stack([e.cost for e in test]), [e.group for e in test]
    predictions = []
    table = []
    for variant in args.variants.split(','):
        t0 = time.time()
        result = tune(parts['train'], parts['validation'], variant=variant, epochs=args.epochs,
                      cost_unit=cost_unit)
        row = {'variant': variant, 'seconds': round(time.time() - t0, 1), 'selected': result['selected'],
               'selected_per_architecture': result['selected_per_architecture'],
               'candidates': result['candidates'], 'trials': result['trials'],
               'tuning_budget': result['tuning_budget'], 'test': {}}
        for arch, models in result['models_per_architecture'].items():
            per_seed = []
            for seed, (model, meta) in models.items():
                if test:
                    pred = predict(model, [e.observation for e in test], keys=[e.key for e in test], permutation=seed)
                    ev = evaluate(pred, ty, tc, tg, coverage=[e.coverage for e in test])
                    per_seed.append({'seed': seed, 'validation_loss': meta['validation_loss'],
                                     'best_epoch': meta['best_epoch'], 'parameters': meta['parameters'],
                                     'evaluation': ev})
                save_checkpoint(model, out / f'{variant}_{arch}_s{seed}.pt',
                                {'tuning': result['selected_per_architecture'][arch], 'labels_sha256': labels['sha256'],
                                 'observer_sha256': provenance['observer_sha256'], 'split_rule': L.FOLD_SPLITS,
                                 'correctness_field': args.correctness_field})
                for i, e in enumerate(test):
                    predictions.append({'variant': variant, 'architecture': arch, 'seed': seed, 'key': e.key,
                                        'group': e.group, 'now': e.observation.now, 'split': 'test',
                                        'p_action_correct': pred['p_action_correct'][i].tolist(),
                                        'predicted_cost': pred['predicted_cost'][i].tolist(), 'cost_unit': cost_unit,
                                        'observed_success_fraction': [None if not np.isfinite(v) else float(v) for v in e.success],
                                        'observed_cost': [None if not np.isfinite(v) else float(v) for v in e.cost],
                                        'coverage': e.coverage})
            if test:
                row['test'][arch] = per_seed
        # Frozen deployment rule: selected candidate, then the seed with the lower validation loss.
        sel = row['selected']['architecture']
        best_seed = min(result['models'], key=lambda seed: (result['models'][seed][1]['validation_loss'], seed))
        deployed = out / f'{variant}_{sel}_s{best_seed}.pt'
        row['deployed_checkpoint'] = {'architecture': sel, 'seed': best_seed, 'path': str(deployed),
                                      'sha256': L.sha256(deployed),
                                      'rule': 'selected candidate by mean validation loss over seeds; '
                                              'then the seed with the lowest validation loss'}
        manifest['results'][variant] = row
        table.append(row)
        print(variant, row['selected']['architecture'], 'val', round(row['selected']['mean_validation_loss'], 4),
              {a: round(float(np.mean([np.mean([s['evaluation']['per_action'][x]['brier'] for x in ACTIONS]) for s in ps])), 4)
               for a, ps in row['test'].items()}, row['seconds'], 's', flush=True)
        with (out / 'manifest.json').open('w') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1, allow_nan=False, default=float)
    with (out / 'test_predictions.jsonl').open('x') as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False, default=float) + '\n')
    if args.trainval_only:
        (out / 'summary.md').write_text(
            '# Train/validation freeze only\n\nNo test observations or predictions were used.\n\n'
            + '\n'.join(f'- {r["variant"]}: {r["deployed_checkpoint"]["architecture"]}, '
                        f'seed {r["deployed_checkpoint"]["seed"]}, '
                        f'sha256 {r["deployed_checkpoint"]["sha256"]}' for r in table) + '\n')
        return
    s = manifest['splits']
    lines = ['# Обучение на пилоте: сводка', '',
             'Техническая проверка на 18 исторически отобранных вопросах, не доказательство эффекта.', '',
             f'Слой наблюдений {args.observer}; метки {args.labels}, колонка правильности {args.correctness_field}, стоимость {args.cost_field} в единицах {cost_unit}.',
             f'Вопросов train/validation/test {s["train"]["questions"]}/{s["validation"]["questions"]}/{s["test"]["questions"]}, префиксов {s["train"]["prefixes"]}/{s["validation"]["prefixes"]}/{s["test"]["prefixes"]}.',
             f'Известных целей правильности на тесте по действиям {s["test"]["known_success_prefixes_per_action"]} из {s["test"]["prefixes"]}; известных стоимостей {s["test"]["known_cost_prefixes_per_action"]}.', '',
             '| Вариант | Архитектура по валидации | Валидационная потеря, среднее 3 seed | Тест: Brier по 5 действиям, среднее по seed, linear / mlp / gru | Тест: Brier для stop, gru или выбранная | Тест: ошибка стоимости, среднее по действиям, выбранная | Параметров выбранной |',
             '|---|---|---:|---|---:|---:|---:|']
    for row in table:
        te = row['test']; sel = row['selected']['architecture']

        def mean_brier(ps):
            return float(np.mean([np.mean([x['evaluation']['per_action'][a]['brier'] for a in ACTIONS]) for x in ps]))
        briers = ' / '.join(f'{mean_brier(te[a]):.4f}' if a in te else '—' for a in ('linear', 'mlp', 'gru'))
        stop = float(np.mean([x['evaluation']['per_action']['stop']['brier'] for x in te[sel]]))
        mae = float(np.mean([np.mean([x['evaluation']['per_action'][a]['cost_mae'] for a in ACTIONS]) for x in te[sel]]))
        lines.append(f'| {row["variant"]} | {sel} | {row["selected"]["mean_validation_loss"]:.4f} | {briers} | {stop:.4f} | {mae:.0f} | {te[sel][0]["parameters"]} |')
    lines += ['', 'Brier для доли исходов y по двум seed считается как p^2 - 2py + y, ожидание квадрата ошибки на розыгрыше. Ошибка стоимости в единицах ' + cost_unit + '.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
