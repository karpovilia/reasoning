#!/usr/bin/env python3
"""The ONE entry point of the 45 fits of PROCESS_FORECAST_GATE_20261013 (review of the supervisor, PFG-V1-R1, R2, R3).

R1  Before anything of the project is imported and before any outcome of the bank is read, every file the fits depend on is
    compared with the SHA256 written in the PINS file, and the PINS file with the SHA256 named on the command line. The sums of
    the PINS file come from the accepted delivery; nothing is recomputed from the disk and trusted. Pinned: the configuration, this
    launcher, the worker, the checker of the members, every file of gate_rebuild that the worker imports, the token contract, the
    cohort, both bank files, the cost freeze, the gate table AND its manifest (the masks of the rows live there), and the data
    contract, with which the table, the token contract and the bank are cross-checked. The configuration must be a production
    one (no random outcomes) with the fixed recipe. The same check runs at every start, also at a continuation.
R2  One finite positive limit covers the whole start: loading, init, the fits, the check of the members and the closing records.
    It is a kernel timer of THIS process (setitimer): when it expires the process is killed, whatever happened to the shell that
    started it. The lock of the run is held by this same process, which does the work itself (no child process): while it lives
    no second start can open the directory, and when it dies the lock is free. The budget of all starts together is fixed in
    the PINS file; a start is charged its whole limit unless it closed its record itself, then its measured seconds rounded up;
    a start whose limit exceeds what is left is refused, and so is a limit of 0. A first start refuses a directory with a history;
    a continuation must name the number of starts so far, so that a blind repetition of a command line starts nothing.
R3  A published member is checked (pfgate.members_check) BEFORE it is skipped as done; a member file that fails the check stops
    the start and is not overwritten. COMPLETE.json and MEMBERS_MANIFEST.json are written only after all 45 members passed.
Exit codes: 0 complete; 3 incomplete inside the limit (cap of the worker or a stop); 4 refused before any work; 5 a published
member or the plan is wrong. A start killed by the timer has no exit record: its limit is charged in full.
  launch_fits.py PINS.json PINS_SHA256 first LIMIT_SECONDS
  launch_fits.py PINS.json PINS_SHA256 continue LIMIT_SECONDS STARTS_SO_FAR"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
RECIPE = {'learning_rate': 0.001, 'weight_decay': 0.01, 'epochs': 7}


class Refusal(Exception):
    pass


def durable_append(path, record):
    with open(path, 'a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_write(path, value):
    tmp = str(path) + '.tmp.%d' % os.getpid()
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(os.path.dirname(str(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_pins(pins_path, pins_sha):
    """The pinned set, checked against the disk. Returns the pins. Reads no outcome and imports nothing of the project."""
    if not os.path.isfile(pins_path) or sha(pins_path) != pins_sha:
        raise Refusal('the PINS file is not the one named on the command line')
    pins = json.loads(Path(pins_path).read_text())
    if pins.get('format') != 'pfgate_pins_v1' or not isinstance(pins.get('files'), dict) or not pins['files']:
        raise Refusal('the PINS file is not of the expected format')
    wrong = []
    for path, expected in sorted(pins['files'].items()):
        got = sha(path) if os.path.isfile(path) else 'absent'
        if got != expected:
            wrong.append('%s is %s, pinned %s' % (path, got, expected))
    if wrong:
        raise Refusal('pinned files differ:\n' + '\n'.join(wrong))
    for role in ('config', 'launcher', 'worker', 'members_check', 'data_contract', 'gate_table', 'gate_table_manifest', 'token_contract', 'cohort', 'trainval', 'replay', 'cost_freeze'):
        if pins['roles'].get(role) not in pins['files']:
            raise Refusal('the role %s names no pinned file' % role)
    if os.path.realpath(__file__) != os.path.realpath(pins['roles']['launcher']):
        raise Refusal('this launcher is not the pinned launcher')
    budget = pins.get('budget_seconds')
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise Refusal('the budget of the PINS file is not a positive integer')
    config = json.loads(Path(pins['roles']['config']).read_text())
    if config.get('rehearsal_random_outcomes') or config.get('recipe') != RECIPE or pins.get('recipe') != RECIPE:
        raise Refusal('the configuration is not the production one with the fixed recipe')
    for role, key in (('gate_table', 'features'), ('token_contract', 'token_contract'), ('cohort', 'cohort'), ('trainval', 'trainval'), ('replay', 'replay'), ('cost_freeze', 'cost_freeze')):
        if os.path.realpath(config[key]) != os.path.realpath(pins['roles'][role]):
            raise Refusal('the configuration reads another file than the pinned %s' % role)
    if os.path.realpath(config['features'] + '.manifest.json') != os.path.realpath(pins['roles']['gate_table_manifest']):
        raise Refusal('the manifest of the gate table is not the pinned one')
    project = os.path.realpath(config['project_code'])
    listed = sorted(p for p in pins['files'] if os.path.realpath(p).startswith(project + os.sep + 'gate_rebuild' + os.sep))
    on_disk = sorted(str(p) for p in (Path(project) / 'gate_rebuild').glob('*.py'))
    if [os.path.realpath(p) for p in listed] != [os.path.realpath(p) for p in on_disk]:
        raise Refusal('the pinned files of gate_rebuild are not exactly the files of the imported package')
    contract = json.loads(Path(pins['roles']['data_contract']).read_text())
    table_manifest = json.loads(Path(pins['roles']['gate_table_manifest']).read_text())
    files = pins['files']
    checks = (('gate table', contract['gate table']['npz_sha256'], files[pins['roles']['gate_table']]), ('gate table manifest', table_manifest['npz_sha256'], files[pins['roles']['gate_table']]),
              ('token contract', contract['sources']['token contract']['sha256'], files[pins['roles']['token_contract']]), ('cohort', contract['sources']['cohort']['COHORT_V3.json'], files[pins['roles']['cohort']]),
              ('replay', contract['sources']['collected bank']['replay_ext.jsonl'], files[pins['roles']['replay']]), ('trainval', contract['sources']['collected bank']['trainval_ext.jsonl'], files[pins['roles']['trainval']]))
    for name, in_contract, pinned in checks:
        if in_contract != pinned:
            raise Refusal('the pinned %s is not the one of the data contract' % name)
    if table_manifest['rows'] != contract['gate table']['rows'] or table_manifest['width'] != config['width']:
        raise Refusal('the rows of the gate table are not those of the data contract')
    return pins, config


def open_start(pins, mode, limit, starts_so_far):
    """Take the lock, check the history and the budget, record the start, arm the timer. Returns (lock, starts path, number)."""
    out = Path(pins['out'])
    state = Path(str(out) + '.launch')
    state.mkdir(parents=True, exist_ok=True)
    lock = open(state / 'run.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise Refusal('another start holds the lock of the run')
    starts_path = state / 'STARTS.jsonl'
    history = [json.loads(line) for line in starts_path.read_text().splitlines()] if starts_path.exists() else []
    begun = [r for r in history if r['event'] == 'begin']
    ended = {r['start']: r for r in history if r['event'] == 'end'}
    if mode == 'first' and (begun or out.exists()):
        raise Refusal('the run has a history: a first start is refused')
    if mode == 'continue' and (not begun or starts_so_far != len(begun)):
        raise Refusal('a continuation must name the number of starts so far, which is %d' % len(begun))
    charged = 0
    for r in begun:
        end = ended.get(r['start'])
        seconds = end.get('seconds') if end else None
        ok = end is not None and isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and 0 <= seconds <= r['limit']
        charged += math.ceil(seconds) if ok else r['limit']
    left = pins['budget_seconds'] - charged
    if limit > left:
        raise Refusal('the limit of %d s exceeds the %d s left of the budget of %d s' % (limit, left, pins['budget_seconds']))
    number = len(begun) + 1
    durable_append(str(starts_path), {'event': 'begin', 'start': number, 'limit': limit, 'pid': os.getpid(), 'budget_left_before': left,
                                     'boottime': time.clock_gettime(time.CLOCK_BOOTTIME), 'wall_clock_for_reading_only': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    signal.setitimer(signal.ITIMER_REAL, limit)                 # no handler: the kernel ends this process when the limit is over
    return lock, starts_path, number


def fits(pins, config, pins_sha):
    """Init when needed, check what is published, fit what is missing, check all, publish the manifest. Returns an exit code."""
    code_dir = str(Path(pins['roles']['worker']).resolve().parents[1])
    sys.path.insert(0, code_dir)
    from pfgate import members_check, worker
    out, config_path = Path(pins['out']), pins['roles']['config']
    if not out.exists():
        if worker.main(['init', config_path, str(out)]) != 0:
            return 5
    train, dev, outer, _, _, _ = worker.load(config)
    dev_keys = [e.key for e in dev]
    held = {fold: [e.key for e in train if outer[e.group] == fold] for fold in range(members_check.FOLDS)}
    accepted, problems = members_check.check_directory(out, config['recipe'], dev_keys, held, require_all=False)
    if problems:                                                # a published member that is wrong: nothing is fitted over it
        print('REFUSED, published members fail the check:\n' + '\n'.join(problems[:20]))
        return 5
    code = worker.main(['run', config_path, str(out)])
    if code == 2:
        return 5
    accepted, problems = members_check.check_directory(out, config['recipe'], dev_keys, held, require_all=True)
    if problems:
        print('NOT COMPLETE:\n' + '\n'.join(problems[:20]))
        return 5 if code == 0 else 3
    manifest = {'format': 'pfgate_members_manifest_v1', 'members': accepted, 'count': len(accepted), 'plan_sha256': sha(out / 'PLAN.json'), 'pins_sha256': pins_sha,
                'recipe': config['recipe'], 'members_check_sha256': sha(members_check.__file__), 'worker_sha256': sha(worker.__file__)}
    durable_write(out / 'MEMBERS_MANIFEST.json', manifest)
    durable_write(out / 'COMPLETE.json', {'state': 'COMPLETE', 'members': len(accepted), 'members_manifest_sha256': sha(out / 'MEMBERS_MANIFEST.json'), 'plan_sha256': manifest['plan_sha256'], 'pins_sha256': pins_sha})
    return 0


def main(argv, work=fits):
    try:
        if len(argv) not in (4, 5) or argv[2] not in ('first', 'continue') or (argv[2] == 'first') != (len(argv) == 4):
            raise Refusal(__doc__.split('Exit codes')[1])
        pins_path, pins_sha, mode = argv[:3]
        if not argv[3].isdigit() or int(argv[3]) <= 0 or (mode == 'continue' and not argv[4].isdigit()):
            raise Refusal('the limit must be a positive whole number of seconds')
        limit = int(argv[3])
        pins, config = check_pins(pins_path, pins_sha)
        lock, starts_path, number = open_start(pins, mode, limit, int(argv[4]) if mode == 'continue' else None)
    except Refusal as refusal:
        print('REFUSED: %s' % refusal)
        return 4
    began = time.monotonic()
    code = work(pins, config, pins_sha)
    durable_append(str(starts_path), {'event': 'end', 'start': number, 'seconds': time.monotonic() - began, 'exit': code})
    signal.setitimer(signal.ITIMER_REAL, 0)
    lock.close()
    print(json.dumps({'start': number, 'exit': code, 'seconds': time.monotonic() - began}))
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
