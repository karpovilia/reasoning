#!/usr/bin/env python3
"""Checks of the PUBLISHED members of PROCESS_FORECAST_GATE_20261013 (review of the supervisor, PFG-V1-R3). Standard library only.
A member file is accepted only when everything below holds; a file that fails is reported, never repaired and never overwritten:
  its name is the expected one and its fields name, call, row, source agree with it; plan_sha256 is the SHA256 of the PLAN.json of
  the directory; the recipe is the fixed one and exactly that many epochs were run; the dev part holds exactly the dev states,
  the held-out part exactly the held-out states of its fold, each key once; success, cost and cost_quantile are [n, 3] tables of
  finite numbers, success in [0, 1], cost >= 0, cost_quantile >= cost; three finite positive temperatures; a positive parameter
  count that is the same for every member.
The directory is complete when exactly the expected member files exist, every one passes, and no other *.json is there."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROWS, SEEDS, FOLDS = ('A', 'B', 'C'), (17, 29, 43), 5
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_names():
    return ['outer_%d__member__s%d__%s' % (fold, seed, row) for fold in range(FOLDS) for seed in SEEDS for row in ROWS]


def _table(values, n, low=None, high=None):
    if not isinstance(values, list) or len(values) != n:
        return False
    for line in values:
        if not isinstance(line, list) or len(line) != 3:
            return False
        for v in line:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (low is not None and v < low) or (high is not None and v > high):
                return False
    return True


def check_member(member, name, plan_sha, recipe, dev_keys, held_keys):
    """Problems of one parsed member file (an empty list: accepted)."""
    if not isinstance(member, dict):
        return ['not an object']
    found = []
    call, _, seed, row = name.split('__')
    source = row if row != 'C' else 'C_%s' % seed
    for key, value in (('name', name), ('call', call), ('row', row), ('source', source), ('plan_sha256', plan_sha), ('recipe', recipe), ('epochs_run', recipe['epochs']), ('seed', int(seed[1:]))):
        if member.get(key) != value:
            found.append('%s is %r, expected %r' % (key, member.get(key), value))
    parts = member.get('parts')
    if not isinstance(parts, dict) or sorted(parts) != ['dev', 'held_out']:
        return found + ['the parts are not exactly dev and held_out']
    for part, keys in (('dev', dev_keys), ('held_out', held_keys)):
        p = parts[part]
        if not isinstance(p, dict) or not isinstance(p.get('keys'), list) or len(set(map(str, p['keys']))) != len(p['keys']) or sorted(p['keys']) != sorted(keys):
            found.append('the keys of %s are not exactly its %d states, each once' % (part, len(keys)))
            continue
        n = len(keys)
        if not _table(p.get('success'), n, 0.0, 1.0) or not _table(p.get('cost'), n, 0.0) or not _table(p.get('cost_quantile'), n, 0.0):
            found.append('%s: success, cost or cost_quantile is not a finite [%d, 3] table in its range' % (part, n))
        elif any(q < c - 1e-6 for qs, cs in zip(p['cost_quantile'], p['cost']) for q, c in zip(qs, cs)):
            found.append('%s: a cost quantile is below its mean' % part)
    t = member.get('temperature')
    if not isinstance(t, list) or len(t) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in t):
        found.append('the temperatures are not three finite positive numbers')
    if isinstance(member.get('parameters'), bool) or not isinstance(member.get('parameters'), int) or member['parameters'] <= 0:
        found.append('the parameter count is not a positive integer')
    if member.get('held_out_states') != len(held_keys):
        found.append('held_out_states is %r, expected %d' % (member.get('held_out_states'), len(held_keys)))
    return found


def check_directory(out, recipe, dev_keys, held_keys_of_fold, require_all):
    """(accepted {name: sha256}, problems [..]) of OUT/members. With require_all every expected member must exist."""
    out = Path(out)
    plan_sha = sha(out / 'PLAN.json')
    names = expected_names()
    present = sorted(p.name[:-5] for p in (out / 'members').glob('*.json'))
    problems = ['an unexpected member file: %s' % n for n in present if n not in names]
    accepted, parameters = {}, set()
    for name in names:
        path = out / 'members' / (name + '.json')
        if not path.exists():
            if require_all:
                problems.append('missing member: %s' % name)
            continue
        try:
            member = json.loads(path.read_text(), parse_constant=lambda c: float('nan'))
        except ValueError:
            problems.append('%s: not JSON' % name)
            continue
        found = check_member(member, name, plan_sha, recipe, dev_keys, held_keys_of_fold[int(name.split('__')[0].split('_')[1])])
        if found:
            problems += ['%s: %s' % (name, f) for f in found]
        else:
            accepted[name] = sha(path)
            parameters.add(member['parameters'])
    if len(parameters) > 1:
        problems.append('the members do not have one parameter count: %s' % sorted(parameters))
    return accepted, problems
