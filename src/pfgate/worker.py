#!/usr/bin/env python3
"""The worker of the 45 outer fits of PROCESS_FORECAST_GATE_20261013: rows A, B, C x seeds 17, 29, 43 x the five accepted outer
folds of the train questions. It is the accepted worker of the GRU matrix (grubatch/worker.py) with four named differences:

  1. COST IN TOKENS. The cost of an action is its ADDITIONAL OUTPUT TOKENS (sampled reasoning + answer tokens after the decision
     point) from the token contract of the saved branches; no cost model, no seconds. The cost scale is 1024 tokens. The unknown
     correctness of a branch stays unknown (masked in the loss by the accepted objective), never a failure.
  2. ONE FIXED RECIPE, NO TRIALS. No inner folds and no grid: every member is fitted with the learning rate, weight decay and
     number of epochs written in the configuration before any result of this experiment existed.
  3. ROWS WITH A SEED-DEPENDENT BLOCK. A row is a mask of the columns of one table; the row C of the member with seed s also sees
     the forecast heads of the predictor P_ES with the same seed s, and of no other seed.
  4. OUTER CALLS ONLY: outer_0 .. outer_4; no fit on all train questions.
Unchanged and imported: gate_rebuild.models.Gate with the encoder 'gru' (train-only standardisation, projection width -> 64, one
GRU layer of width 64, the accepted heads of success and of cost), gate_rebuild.training.fit_one / calibrate / predict /
fold_of. A switched-off column is zero in every record, so after the train-only standardisation it is exactly zero: the rows are
the same network with the same number of parameters, and for one seed the same initial weights and order of batches.
A member: the fit on the fitting part of its call, its temperatures on dev (accepted calibrate), its calibrated predictions on
dev and on the held-out states. Every member is one file, linked to its name once complete; PLAN.json lists every expected
work and is compared at every start; the seconds of every start count against the cap; a finished work is never repeated.

  python -m pfgate.worker init CONFIG.json OUT_DIR
  python -m pfgate.worker run  CONFIG.json OUT_DIR [ONLY_FIRST_N_WORKS]      (the last argument is for the timing probe alone)"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ACTIONS = ('stop', 'continue_512', 'continue_2048')
SEEDS, ROWS = (17, 29, 43), ('A', 'B', 'C')
COST_SCALE = 1024.0
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def link_once(path, value):
    """A result file appears under its name only complete, and only once."""
    tmp = Path(str(path) + '.tmp.%d' % os.getpid())
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(value, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(tmp, path)
    os.unlink(tmp)


def source_of(row, seed):
    return row if row != 'C' else 'C_s%d' % seed


def token_costs(contract_path, keys):
    """{key: [3] additional output tokens of the actions} for the decision states, from the token contract."""
    contract = json.loads(Path(contract_path).read_text())
    if contract['format'] != 'process_forecast_gate_token_contract_v1' or contract['actions'] != list(ACTIONS):
        raise ValueError('the token contract is not of the expected format')
    by_key = {a['key']: a for a in contract['attempts']}
    out = {}
    for key in keys:
        attempt = by_key[key]
        if attempt['ended_before_the_decision']:
            raise ValueError('%s is not a decision state in the token contract' % key)
        out[key] = np.array([attempt['actions'][a]['additional_output_tokens'] for a in ACTIONS], np.float32)
    return out


def load(config):
    """(train examples, dev examples, outer fold of a train question, sums, sha of the outer folds). The bank is read by the
    accepted loader and cut to the accepted cohort; the inputs come from the gate table; the costs from the token contract."""
    sys.path.insert(0, config['project_code'])
    from gate_rebuild import contract
    from gate_rebuild.cohort import apply_cohort, check_bank_files
    from gate_rebuild.contract import Example
    from gate_rebuild.cost import CostModel
    from gate_rebuild.data import load_trainext, with_states
    from gate_rebuild.training import FOLDS, fold_of
    cost_model = CostModel(**json.loads(Path(config['cost_freeze']).read_text())['cost_model'])      # needed by the accepted loader only; its costs are replaced below
    active, early = load_trainext(config['trainval'], config['replay'], cost_model)
    cohort = json.loads(Path(config['cohort']).read_text())
    check_bank_files(cohort, {'trainval': config['trainval'], 'replay': config['replay']})
    active, early, _ = apply_cohort(cohort, active, early)
    table = np.load(config['features'])
    manifest = json.loads(Path(config['features'] + '.manifest.json').read_text())
    if sha(config['features']) != manifest['npz_sha256'] or manifest['width'] != config['width']:
        raise ValueError('the gate table is not the one its manifest describes')
    where = {str(k): tuple(o) for k, o in zip(table['keys'], table['offsets'])}
    examples = [e for e, _ in active]
    if sorted(where) != sorted(e.key for e in examples):
        raise ValueError('the gate table does not hold exactly the decision states of the cohort')
    tokens = token_costs(config['token_contract'], [e.key for e in examples])
    examples = [Example(e.key, e.group, e.split, e.states, e.budget, e.success, tokens[e.key], e.tokens, e.supported, e.prompt_tokens, e.prefix_tokens) for e in examples]
    x, extra = table['x'], manifest['decision_context_columns_filled_by_the_worker']
    info_of = {e.key: info for e, info in active}
    masks = {}
    for row in ROWS:
        for seed in SEEDS:
            name = source_of(row, seed)
            if name in masks:
                continue
            spans = manifest['rows'][row] if row != 'C' else manifest['rows']['C'][str(seed)]
            keep = np.zeros(config['width'], np.float32)
            for first, last in spans:
                keep[first:last] = 1.
            masks[name] = keep
    for name, keep in masks.items():
        contract.STATE_SOURCES[name] = {'width': config['width'], 'sequence': True, 'observer': 'see the manifest of the gate table', 'records': 'row %s of PROCESS_FORECAST_GATE_20261013' % name}
        states = {}
        for e in examples:
            offset, length = where[e.key]
            value = np.array(x[offset:offset + length], np.float32)
            value[:, extra['length_budget'][0]:extra['length_budget'][1]] = np.asarray(e.states['length_budget'], np.float32)[0]
            value[:, extra['confidence']] = float(info_of[e.key]['lexical_observation']['confidence'])
            states[e.key] = value * keep
        examples = with_states(examples, name, states)
    if config.get('rehearsal_random_outcomes'):                                    # a technical probe: no real outcome and no real cost reaches a fit
        rng = np.random.default_rng(0)
        examples = [Example(e.key, e.group, e.split, e.states, e.budget, rng.integers(0, 2, 3).astype(np.float32), rng.uniform(1., 2000., 3).astype(np.float32), e.tokens, e.supported, e.prompt_tokens, e.prefix_tokens) for e in examples]
    train, dev = [e for e in examples if e.split == 'train'], [e for e in examples if e.split == 'dev']
    questions = sorted({e.group for e in train} | {e['group'] for e in early if e['split'] == 'train'})
    outer = fold_of(questions, FOLDS, config['outer_salt'])
    sums = {k: sha(config[k]) for k in ('trainval', 'replay', 'cohort', 'cost_freeze', 'features', 'token_contract')}
    return train, dev, outer, sums, hashlib.sha256(json.dumps(sorted(outer.items())).encode()).hexdigest(), {name: int(keep.sum()) for name, keep in masks.items()}


def plan_of(config, sums, outer_sha, columns):
    from gate_rebuild.training import BATCH, FOLDS
    calls = ['outer_%d' % k for k in range(FOLDS)]
    members = ['%s__member__s%d__%s' % (call, s, row) for call in calls for s in SEEDS for row in ROWS]
    code = Path(config['project_code'])
    return {'format': 'pfgate_plan_v1', 'task': config['task'], 'width': config['width'], 'calls': calls, 'members': members, 'works': len(members), 'rows': list(ROWS), 'seeds': list(SEEDS),
            'columns_seen_by_a_source': columns, 'recipe': config['recipe'], 'batch': BATCH, 'cost_scale_tokens': COST_SCALE, 'rehearsal_random_outcomes': bool(config.get('rehearsal_random_outcomes')),
            'cap_seconds': config['cap_seconds'], 'sums': sums, 'outer_folds_sha256': outer_sha, 'outer_salt': config['outer_salt'],
            'code_sha256': {'worker': sha(__file__), **{'project/%s' % p.relative_to(code): sha(p) for p in sorted((code / 'gate_rebuild').glob('*.py'))}}}


def main(argv=None):
    argv = argv or sys.argv[1:]
    command, config_path, out = argv[:3]
    only = int(argv[3]) if len(argv) > 3 else None
    config, out = json.loads(Path(config_path).read_text()), Path(out)
    recipe = config['recipe']
    if sorted(recipe) != ['epochs', 'learning_rate', 'weight_decay'] or isinstance(recipe['epochs'], bool) or not isinstance(recipe['epochs'], int) or recipe['epochs'] < 1:
        raise ValueError('the recipe must be exactly a learning rate, a weight decay and a whole number of epochs')
    if config.get('expected_outer_folds_sha256') is None:
        raise ValueError('the configuration must name the SHA256 of the accepted outer folds')
    import torch
    torch.set_num_threads(int(config.get('threads', 1)))
    train, dev, outer, sums, outer_sha, columns = load(config)
    if outer_sha != config['expected_outer_folds_sha256']:
        raise ValueError('the outer folds are not the accepted ones: %s' % outer_sha)
    from gate_rebuild.training import calibrate, fit_one, predict
    plan = plan_of(config, dict(sums, config=sha(config_path)), outer_sha, columns)
    if command == 'init':
        os.makedirs(out / 'members')
        atomic(out / 'PLAN.json', plan)
        atomic(out / 'CLOCK.json', {'spent_seconds': 0.0, 'starts': 0, 'open_since': None})
        print(json.dumps({'plan_sha256': sha(out / 'PLAN.json'), 'works': plan['works'], 'train_states': len(train), 'dev_states': len(dev), 'train_questions': len({e.group for e in train}), 'columns_seen_by_a_source': columns}))
        return 0
    if json.loads((out / 'PLAN.json').read_text()) != plan:
        print('refused: the inputs, the configuration or the code are not the ones of the plan of this directory')
        return 2
    plan_sha = sha(out / 'PLAN.json')
    clock = json.loads((out / 'CLOCK.json').read_text())
    if clock['open_since'] is not None:                              # an earlier start did not close its clock: its time is charged up to now
        clock['spent_seconds'] += max(time.time() - clock['open_since'], 0.0)
    clock.update(starts=clock['starts'] + 1, open_since=time.time())
    atomic(out / 'CLOCK.json', clock)
    began, base = time.time(), clock['spent_seconds']
    spent = lambda: base + time.time() - began
    journal = open(out / 'JOURNAL.jsonl', 'a')
    say = lambda **row: (journal.write(json.dumps(dict(row, at=time.time(), spent_seconds=spent())) + '\n'), journal.flush())
    say(event='start', start=clock['starts'])
    done, stopped = 0, None
    for name in plan['members']:
        target = out / 'members' / (name + '.json')
        if target.exists():
            continue
        if only is not None and done >= only:
            stopped = 'probe_limit'
            break
        if spent() >= config['cap_seconds']:
            stopped = 'cap_reached'
            say(event='cap_reached', next_work=name)
            break
        call, _, seed, row = name.split('__')
        seed, fold = int(seed[1:]), int(call.split('_')[1])
        source = source_of(row, seed)
        fitting, held_out = [e for e in train if outer[e.group] != fold], [e for e in train if outer[e.group] == fold]
        t0 = time.time()
        torch.manual_seed(seed)
        model, record = fit_one(fitting, None, source, 'gru', seed, cost_scale=COST_SCALE, epochs=recipe['epochs'], learning_rate=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
        temperature = calibrate(model, dev, source)
        state_sha = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for _, v in sorted(model.state_dict().items()))).hexdigest()
        parts = {}
        for part, examples in (('dev', dev), ('held_out', held_out)):
            prediction = predict([model], examples, source)
            parts[part] = {'keys': [e.key for e in examples], **{k: np.asarray(v, float).tolist() for k, v in prediction.items()}}
        link_once(target, dict(record, name=name, call=call, row=row, source=source, plan_sha256=plan_sha, seconds=time.time() - t0, fitting_states=len(fitting), fitting_questions=len({e.group for e in fitting}),
                               held_out_states=len(held_out), recipe=recipe, temperature=temperature, final_state_sha256=state_sha, parameters=sum(p.numel() for p in model.parameters()), parts=parts))
        done += 1
        say(event='member', name=name, seconds=time.time() - t0)
    atomic(out / 'CLOCK.json', dict(clock, spent_seconds=spent(), open_since=None))
    missing = [n for n in plan['members'] if not (out / 'members' / (n + '.json')).exists()]
    atomic(out / 'FIT_STATE.json', {'state': 'COMPLETE' if not missing else 'INCOMPLETE', 'missing': len(missing), 'first_missing': missing[:3], 'stopped_by': stopped, 'spent_seconds': spent(), 'plan_sha256': plan_sha, 'at': time.time()})
    say(event='end', state='COMPLETE' if not missing else 'INCOMPLETE', missing=len(missing))
    print(json.dumps({'state': 'COMPLETE' if not missing else 'INCOMPLETE', 'missing': len(missing), 'done_in_this_start': done, 'spent_seconds': spent()}))
    return 0 if not missing else 3


if __name__ == '__main__':
    sys.exit(main())
