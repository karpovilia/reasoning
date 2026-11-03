#!/usr/bin/env python3
"""The weights of the 45 accepted members of PROCESS_FORECAST_GATE_20261013 (rows A, B, C), recovered by fitting them AGAIN and
proved to be the accepted members. Second edition (review of the supervisor, PFG-W6-R1 and R2; export_member_weights.py is the
first and is not used).
  R1  NOTHING THAT FAILED A CHECK IS PUBLISHED. A member is first compared with the accepted member (SHA256 of the final weights,
      predictions on dev and on the held-out states, temperatures, number of parameters) BEFORE anything is saved; then its
      weights are saved into quarantine/NAME.pt and synced, loaded from there into a NEW network and compared again; only then
      the file is linked to weights/NAME.pt and its record is written. On any difference or exception the run stops with code
      5: FAILED.json says what happened, no weights/NAME.pt of that member exists, a saved file stays in quarantine/, and no
      COMPLETE is written. The directories are synced before the record of a member and before COMPLETE.
  R2  THE REFERENCE AND THE NEW ENTRY POINT ARE PINNED FROM OUTSIDE. The first thing this file does, with the standard library
      only and before any module of the project is imported, is to compare the pins file of the export with the SHA256 named on
      the command line, and every file it lists with its SHA256: this file itself, evaluate.py, launch_fits.py, worker.py,
      members_check.py, and the accepted MEMBERS_MANIFEST.json, PLAN.json and COMPLETE.json of the accepted fits. So the
      result that is recovered is the accepted one by its known sum, not whatever the directory says of itself. After the
      import the loaded modules must be exactly the pinned files. The exact command compares the SHA256 of this file with
      sha256sum before python is started; that comparison and the whole run are under one external limit.
Nothing is chosen and nothing new is fitted: for every member the lines of the accepted worker are run again on the pinned
inputs of the accepted start (accepted PINS, checked by the accepted launcher), in the accepted order.
Written: weights/NAME.pt (torch.save of {'state_dict', 'constructor', ...}), records/NAME.json, WEIGHTS_MANIFEST.json with the
SHA256 of every file and of every pinned reference, COMPLETE.json.
How to rebuild a member from its file: register the source as pfgate.worker.load does (contract.STATE_SOURCES[source], width
2676, sequence), then gate_rebuild.models.Gate(source, 'gru', cost_scale) and load_state_dict; the train-only standardisation
and the three temperatures are buffers of the state.
  export_member_weights_v2.py EXPORT_PINS.json ITS_SHA256 OUT_DIR LIMIT_SECONDS"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

PINS_FORMAT = 'pfgate_member_weights_pins_v1'
PINNED_MODULES = ('evaluate', 'launch_fits', 'worker', 'members_check')
PINNED_REFERENCE = ('MEMBERS_MANIFEST.json', 'PLAN.json', 'COMPLETE.json')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_export_pins(pins_path, pins_sha, me=__file__):
    """The pins of the export, checked against the disk with the standard library only. Returns the pins or raises SystemExit(4)."""
    def refuse(text):
        print('REFUSED: %s' % text)
        raise SystemExit(4)
    if not os.path.isfile(pins_path) or sha(pins_path) != pins_sha:
        refuse('the pins file of the export is not the one named on the command line')
    pins = json.loads(Path(pins_path).read_text())
    files = pins.get('files') if isinstance(pins, dict) else None
    if not isinstance(files, dict) or pins.get('format') != PINS_FORMAT or not all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in files.items()):
        refuse('the pins file of the export is not of the expected format')
    wrong = ['%s is %s, pinned %s' % (path, sha(path) if os.path.isfile(path) else 'absent', expected) for path, expected in sorted(files.items()) if not os.path.isfile(path) or sha(path) != expected]
    if wrong:
        refuse('pinned files of the export differ:\n' + '\n'.join(wrong))
    here = os.path.dirname(os.path.realpath(me))
    real = {os.path.realpath(path) for path in files}
    needed = [os.path.realpath(me)] + [os.path.join(here, name + '.py') for name in PINNED_MODULES] + [os.path.join(os.path.realpath(pins['accepted_fits']), name) for name in PINNED_REFERENCE] \
        + [os.path.realpath(pins['accepted_pins'])]
    missing = [path for path in needed if path not in real]
    if missing:
        refuse('the pins file of the export does not pin: %s' % missing)
    if files[pins['accepted_pins']] != pins['accepted_pins_sha256']:
        refuse('the accepted PINS file is not pinned with its named sum')
    return pins


def sync_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable(path, value):
    tmp = str(path) + '.tmp.%d' % os.getpid()
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(tmp, path)                                            # never over an existing file
    os.unlink(tmp)
    sync_file(os.path.dirname(str(path)))


def state_sha(model):
    """The SHA256 of the final weights exactly as pfgate.worker computes it."""
    return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for _, v in sorted(model.state_dict().items()))).hexdigest()


def main(argv):
    if len(argv) != 4 or not argv[3].isdigit() or int(argv[3]) <= 0:
        print(__doc__)
        return 4
    pins_path, pins_sha, out, limit = argv[0], argv[1], Path(argv[2]), int(argv[3])
    export_pins = check_export_pins(pins_path, pins_sha)          # before any module of the project is imported
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pfgate import evaluate as E, launch_fits as L, members_check, worker as W
    here = os.path.dirname(os.path.realpath(__file__))
    for name, module in (('evaluate', E), ('launch_fits', L), ('worker', W), ('members_check', members_check)):
        if os.path.realpath(module.__file__) != os.path.join(here, name + '.py'):
            print('REFUSED: the imported module %s is not the pinned file' % name)
            return 4
    try:
        pins, config = L.check_pins(export_pins['accepted_pins'], export_pins['accepted_pins_sha256'])      # every input and code file of the accepted start
    except L.Refusal as refusal:
        print('REFUSED: %s' % refusal)
        return 4
    if os.path.realpath(pins['out']) != os.path.realpath(export_pins['accepted_fits']):
        print('REFUSED: the accepted PINS are not for the pinned directory of the accepted fits')
        return 4
    if out.exists():
        print('REFUSED: the output directory exists')
        return 4
    lock = open(str(out) + '.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('REFUSED: another start holds the lock')
        return 4
    signal.setitimer(signal.ITIMER_REAL, limit)                   # no handler: the kernel ends this process when the limit is over
    began = time.monotonic()
    contract_path = pins['roles']['token_contract']
    accepted, manifest, plan = E.checked_members(contract_path, json.loads(Path(contract_path).read_text()), pins['out'], export_pins['accepted_pins'], export_pins['accepted_pins_sha256'])
    reference = {name: export_pins['files'][os.path.join(export_pins['accepted_fits'], name)] for name in PINNED_REFERENCE}
    if sha(Path(pins['out']) / 'MEMBERS_MANIFEST.json') != reference['MEMBERS_MANIFEST.json'] or manifest['plan_sha256'] != reference['PLAN.json']:
        print('REFUSED: the directory of the fits is not the accepted one by its pinned sums')
        return 4
    import torch
    torch.set_num_threads(int(config.get('threads', 1)))
    train, dev, outer, _, outer_sha, columns = W.load(config)
    if outer_sha != config['expected_outer_folds_sha256'] or outer_sha != plan['outer_folds_sha256']:
        print('REFUSED: the outer folds are not the accepted ones')
        return 4
    sys.path.insert(0, config['project_code'])
    from gate_rebuild.models import Gate
    from gate_rebuild.training import calibrate, fit_one, predict
    recipe = config['recipe']
    for directory in ('weights', 'records', 'quarantine'):
        (out / directory).mkdir(parents=True)
    files = {}

    def failed(name, why, **detail):
        print('STOPPED: the member %s is not the accepted member: %s' % (name, why))
        durable(out / 'FAILED.json', dict(detail, name=name, why=why, published_weights_of_this_member=(out / 'weights' / (name + '.pt')).exists(),
                                          in_quarantine=sorted(p.name for p in (out / 'quarantine').iterdir())))
        return 5

    for name in members_check.expected_names():
        member = accepted[name]
        call, _, seed, row = name.split('__')
        seed, fold = int(seed[1:]), int(call.split('_')[1])
        source = W.source_of(row, seed)
        fitting, held_out = [e for e in train if outer[e.group] != fold], [e for e in train if outer[e.group] == fold]
        t0 = time.time()
        try:
            torch.manual_seed(seed)                               # the lines of the accepted worker, unchanged
            model, record = fit_one(fitting, None, source, 'gru', seed, cost_scale=W.COST_SCALE, epochs=recipe['epochs'], learning_rate=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
            temperature = calibrate(model, dev, source)
            recomputed = state_sha(model)
            parts = {}
            for part, examples in (('dev', dev), ('held_out', held_out)):
                prediction = predict([model], examples, source)
                parts[part] = {'keys': [e.key for e in examples], **{k: np.asarray(v, float).tolist() for k, v in prediction.items()}}
            checks = {'final_state_sha256_equal_to_the_accepted_member': recomputed == member['final_state_sha256'],
                      'predictions_equal_to_the_accepted_member': parts == member['parts'], 'temperature_equal_to_the_accepted_member': temperature == member['temperature'],
                      'parameters_equal_to_the_accepted_member': sum(p.numel() for p in model.parameters()) == member['parameters']}
            if not all(checks.values()):                          # nothing of this member is saved at all
                return failed(name, 'the fitted member differs from the accepted one', checks=checks, final_state_sha256={'accepted': member['final_state_sha256'], 'recomputed': recomputed})
            constructor = {'class': 'gate_rebuild.models.Gate', 'source': source, 'encoder': 'gru', 'cost_scale': W.COST_SCALE, 'source_width': config['width'], 'columns_seen_by_the_source': columns[source]}
            held = out / 'quarantine' / (name + '.pt')
            torch.save({'format': 'pfgate_member_weights_v1', 'name': name, 'row': row, 'seed': seed, 'call': call, 'state_dict': model.state_dict(), 'constructor': constructor,
                        'temperature': temperature, 'final_state_sha256': recomputed, 'recipe': recipe}, str(held))
            sync_file(held)
            loaded = torch.load(str(held), map_location='cpu', weights_only=False)
            again = Gate(source, 'gru', W.COST_SCALE)
            again.load_state_dict(loaded['state_dict'])
            again.eval()
            reloaded = predict([again], dev, source)
            checks['the_saved_file_gives_the_same_weights_in_a_new_network'] = state_sha(again) == member['final_state_sha256']
            checks['the_saved_file_gives_the_same_dev_predictions'] = {k: np.asarray(v, float).tolist() for k, v in reloaded.items()} == {k: member['parts']['dev'][k] for k in reloaded}
            if not all(checks.values()):                          # the file stays in quarantine and is never given its final name
                return failed(name, 'the saved file does not give the accepted member back', checks=checks)
        except Exception as error:                                # a fit, a save or a load that raised: the same stop
            return failed(name, 'an exception: %s: %s' % (type(error).__name__, error))
        target = out / 'weights' / (name + '.pt')
        os.link(held, target)                                     # published only now, after every check
        os.unlink(held)
        sync_file(out / 'weights')
        durable(out / 'records' / (name + '.json'), {'format': 'pfgate_member_weights_record_v1', 'name': name, 'row': row, 'seed': seed, 'call': call, 'source': source, 'weights_file': 'weights/%s.pt' % name,
                                                     'weights_file_sha256': sha(target), 'weights_file_bytes': target.stat().st_size, 'final_state_sha256': recomputed, 'accepted_member_file_sha256': manifest['members'][name],
                                                     'checks': checks, 'constructor': constructor, 'temperature': temperature, 'recipe': recipe, 'epochs_run': record['epochs_run'], 'seconds': time.time() - t0})
        files[name] = {'weights_sha256': sha(target), 'record_sha256': sha(out / 'records' / (name + '.json')), 'final_state_sha256': recomputed}
    if sorted(files) != sorted(accepted) or len(files) != 45 or any((out / 'quarantine').iterdir()):
        return failed('all', 'not exactly the 45 members were published, or the quarantine is not empty')
    os.rmdir(out / 'quarantine')
    durable(out / 'WEIGHTS_MANIFEST.json', {'format': 'pfgate_member_weights_manifest_v2', 'members': files, 'count': len(files), 'export_pins_sha256': pins_sha, 'accepted_pins_sha256': export_pins['accepted_pins_sha256'],
                                             'accepted_members_manifest_sha256': reference['MEMBERS_MANIFEST.json'], 'accepted_plan_sha256': reference['PLAN.json'], 'accepted_complete_sha256': reference['COMPLETE.json'],
                                             'this_file_sha256': sha(__file__), 'evaluate_sha256': sha(E.__file__), 'worker_sha256': sha(W.__file__), 'torch': torch.__version__, 'threads': int(config.get('threads', 1)),
                                             'every_member': 'fitted again by the lines of the accepted worker; published only with the final weights, the predictions, the temperatures and the reloaded file equal to the accepted member'})
    sync_file(out / 'weights')
    sync_file(out / 'records')
    durable(out / 'COMPLETE.json', {'state': 'COMPLETE', 'members': len(files), 'weights_manifest_sha256': sha(out / 'WEIGHTS_MANIFEST.json'), 'seconds': time.monotonic() - began})
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({'state': 'COMPLETE', 'members': len(files), 'weights_manifest_sha256': sha(out / 'WEIGHTS_MANIFEST.json'), 'seconds': time.monotonic() - began}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
