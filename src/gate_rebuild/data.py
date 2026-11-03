"""Examples from the existing bank files and from state caches.

Bank rows are read as they were written by the historical collection (gate_trainext_2026_09_27/ext_v2: trainval_ext.jsonl and
replay_ext.jsonl): the three actions of the a3 set, one real continuation per action, `success` from the blind content review
with null for unknown, additional solver tokens per action. Nothing is regenerated and no label is recomputed here. The marginal
cost of an action is computed from its tokens by the declared cost model, so the targets, the predictions, the price and the
values are all in ONE unit (review finding GATE-V1-R1).

State caches (review finding GATE-V1-R5). States of the observers are NOT in the bank. A cache is one .npz and one manifest per
source, written by a separately admitted producer. The consumer trusts a cache only through the SHA256 of its manifest, which is
part of the frozen inputs of the fitting stage, and checks it against the bank:
  - the manifest names its producer (code files with SHA256), its observer (bundle id, manifest, weights), the panel of prefixes
    it read (SHA256), the decision boundary and the segmentation, and the environment;
  - every record is bound to one prefix: question group, split, the prefix hash of the bank, the decision position, the number
    of completed segments, offset and length in the table;
  - the reader refuses another manifest digest, another schema, repeated or unsorted keys, offsets that do not tile the table,
    a length that contradicts the source, a non-finite value, and any requested prefix whose group, split, prefix hash or
    decision position differs from the bank row. A cache of another version, another panel or a longer prefix does not pass
    by having the same keys.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .contract import ACTIONS, DECISION_POSITION, QUESTION_HASH_WIDTH, STATE_SOURCES, Example, budget_features

CACHE_FORMAT = 'gate_rebuild_state_cache_v2'
SPLIT_NAMES = {'train': 'train', 'validation': 'dev', 'dev': 'dev', 'test': 'test'}
MANIFEST_KEYS = ('binding', 'format', 'keys', 'npz_sha256', 'records', 'rows', 'source', 'states', 'width')
BINDING_KEYS = ('boundary', 'environment', 'observer', 'panel_sha256', 'producer')
BOUNDARY_KEYS = ('as_of', 'decision_position', 'segmentation')
RECORD_KEYS = ('decision_position', 'group', 'length', 'offset', 'prefix_hash', 'segments', 'split')
BOUND_FIELDS = ('group', 'split', 'prefix_hash', 'decision_position')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def length_budget_state(now, segments, prompt_tokens):
    """What is known without reading the question or the reasoning. No benchmark label, no question id."""
    count = len(segments)
    mean = (sum(s['token_end'] - s['token_start'] for s in segments) / count) if count else 0.0
    return np.array([[np.log1p(prompt_tokens) / 10., now / 2048., np.log1p(count) / 5., np.log1p(mean) / 10.]], dtype=np.float32)


def question_features(question):
    """[1, QUESTION_HASH_WIDTH] hashed counts of the character 3-, 4- and 5-grams of the question text, log-scaled. A stable hash
    (not the per-process hash of the interpreter); no model, no vocabulary fitted on any split, no benchmark label."""
    text = re.sub(r'\s+', ' ', question).strip().casefold()
    if not text:
        raise ValueError('empty question')
    counts = np.zeros(QUESTION_HASH_WIDTH, dtype=np.float32)
    for size in (3, 4, 5):
        for start in range(max(1, len(text) - size + 1)):
            digest = hashlib.blake2b(('%d|%s' % (size, text[start:start + size])).encode(), digest_size=8).digest()
            counts[int.from_bytes(digest, 'big') % QUESTION_HASH_WIDTH] += 1.0
    return np.log1p(counts)[None, :]


def load_trainext(trainval_path, replay_path, cost_model):
    """Decision states of the train extension as examples with the 'length_budget' state (with what the cost of the attempt and
    the lexical records need kept aside), and the attempts that ended before the decision position (they stay in every
    whole-population number). Costs are in the unit of `cost_model`."""
    replay = {}
    with open(replay_path) as handle:
        for line in handle:
            row = json.loads(line)
            key = '%s|%s' % (row['question_key'], row['seed'])
            if key in replay:
                raise ValueError('the bank repeats the attempt %s' % key)
            replay[key] = row
    active, early, seen = [], [], set()
    with open(trainval_path) as handle:
        for line in handle:
            row = json.loads(line)
            source = replay[row['key']]
            if row['key'] in seen or source['group'] != row['group'] or bool(source['ended_before_checkpoint']):
                raise ValueError('bank row %s is repeated or does not match its replay record' % row['key'])
            seen.add(row['key'])
            success = np.array([np.nan if v is None else float(v) for v in row['success']], dtype=np.float32)
            tokens = np.array([np.nan if v is None else float(v) for v in row['cost']], dtype=np.float32)
            if success.shape != (len(ACTIONS),) or [b['action'] for b in source['branches']] != list(ACTIONS):
                raise ValueError('bank row %s does not carry exactly the three supported actions' % row['key'])
            observation, prompt = row['observation'], int(source['prompt_tokens'])
            if int(observation['now']) != DECISION_POSITION:
                raise ValueError('bank row %s is not a decision at position %d' % (row['key'], DECISION_POSITION))
            cost = np.array([np.nan if not np.isfinite(t) else cost_model.action(float(t), prompt, DECISION_POSITION) for t in tokens], dtype=np.float32)
            active.append((Example(key=row['key'], group=row['group'], split=SPLIT_NAMES[row['split']],
                                   states={'length_budget': length_budget_state(observation['now'], observation['segments'], prompt)},
                                   budget=budget_features(), success=success, cost=cost, tokens=tokens, supported=np.ones(len(ACTIONS), bool),
                                   prompt_tokens=prompt, prefix_tokens=DECISION_POSITION),
                           {'prefix_hash': source['prefix_hash'], 'benchmark': row['benchmark'], 'lexical_observation': observation,
                            'common_cost': cost_model.common(prompt, DECISION_POSITION)}))
    for key, row in replay.items():
        if bool(row['ended_before_checkpoint']):
            if not row['branches']:
                raise ValueError('the attempt %s ended early and has no answer record' % key)
            branch, prompt = row['branches'][0], int(row['prompt_tokens'])
            early.append({'key': key, 'group': row['group'], 'split': SPLIT_NAMES[row['split']],
                          'success': np.nan if branch.get('content') is None else float(branch['content']),
                          'tokens': float(branch['tokens']), 'prompt_tokens': prompt, 'cost': cost_model.common(prompt, float(branch['tokens']))})
        elif key not in seen:
            raise ValueError('the attempt %s reached the decision but has no decision row' % key)
    return active, early


def bank_binding(example, extra):
    """What a cache record of this prefix must say."""
    return {'group': example.group, 'split': example.split, 'prefix_hash': extra['prefix_hash'], 'decision_position': DECISION_POSITION}


def write_state_cache(directory, source, states, binding, records):
    """states: {prefix key: array [T, width]}; binding: producer, observer, panel, boundary, environment; records: {prefix key:
    {'group', 'split', 'prefix_hash', 'decision_position', 'segments'}}. Create-only. Returns (manifest, SHA256 of the manifest
    file): the digest is what a consumer pins."""
    directory = Path(directory)
    path, manifest_path = directory / ('%s.npz' % source), directory / ('%s.manifest.json' % source)
    if path.exists() or manifest_path.exists():
        raise ValueError('a cache of %s already exists in %s' % (source, directory))
    spec = STATE_SOURCES[source]
    if tuple(sorted(binding)) != BINDING_KEYS or tuple(sorted(binding['boundary'])) != BOUNDARY_KEYS or not re.fullmatch(r'[0-9a-f]{64}', str(binding['panel_sha256'])) \
            or not binding['producer'] or not binding['environment'] or (spec['observer'] is not None and not binding['observer']):
        raise ValueError('the binding of the cache is incomplete')
    keys = sorted(states)
    if not keys or sorted(records) != keys:
        raise ValueError('states and records do not describe the same prefixes')
    table, offset = {}, 0
    for key in keys:
        value, record = np.asarray(states[key]), records[key]
        if value.ndim != 2 or value.shape[1] != spec['width'] or len(value) < 1 or not np.isfinite(value).all() or (not spec['sequence'] and len(value) != 1):
            raise ValueError('the state of %s has the wrong shape or a non-finite value' % key)
        if tuple(sorted(record)) != tuple(k for k in RECORD_KEYS if k not in ('offset', 'length')) or isinstance(record['segments'], bool) or int(record['segments']) != record['segments'] \
                or record['segments'] < 0:
            raise ValueError('the record of %s is incomplete' % key)
        table[key] = dict(record, offset=offset, length=int(len(value)))
        offset += len(value)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(path, values=np.concatenate([np.asarray(states[key], np.float32) for key in keys]))
    manifest = {'format': CACHE_FORMAT, 'source': source, 'width': spec['width'], 'binding': binding, 'keys': keys, 'records': table, 'states': len(keys),
                'rows': int(offset), 'npz_sha256': file_sha256(path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    return manifest, file_sha256(manifest_path)


def read_state_cache(directory, source, expected, trusted_manifest_sha256):
    """({key: array}, manifest) for exactly the prefixes of `expected` = {key: bank_binding(...)}; see the module text for what is
    refused. trusted_manifest_sha256 comes from the frozen inputs of the stage, never from the cache directory itself."""
    directory = Path(directory)
    manifest_path, path = directory / ('%s.manifest.json' % source), directory / ('%s.npz' % source)
    if not re.fullmatch(r'[0-9a-f]{64}', str(trusted_manifest_sha256)) or file_sha256(manifest_path) != trusted_manifest_sha256:
        raise ValueError('the manifest of the state cache of %s is not the trusted one' % source)
    manifest = json.loads(manifest_path.read_text())
    spec = STATE_SOURCES[source]
    if not isinstance(manifest, dict) or tuple(sorted(manifest)) != MANIFEST_KEYS or manifest['format'] != CACHE_FORMAT or manifest['source'] != source \
            or manifest['width'] != spec['width'] or tuple(sorted(manifest['binding'])) != BINDING_KEYS or manifest['npz_sha256'] != file_sha256(path):
        raise ValueError('the state cache of %s is not the one its manifest describes' % source)
    keys, records = manifest['keys'], manifest['records']
    if keys != sorted(set(keys)) or sorted(records) != keys or manifest['states'] != len(keys):
        raise ValueError('the state cache of %s has repeated, unsorted or unrecorded prefixes' % source)
    values = np.load(path, allow_pickle=False)['values']
    cursor = 0
    for key in keys:
        record = records[key]
        if tuple(sorted(record)) != RECORD_KEYS or record['offset'] != cursor or isinstance(record['length'], bool) or record['length'] < 1 \
                or (not spec['sequence'] and record['length'] != 1):
            raise ValueError('the record of %s in the state cache of %s does not tile the table' % (key, source))
        cursor += record['length']
    if values.ndim != 2 or values.shape != (cursor, spec['width']) or manifest['rows'] != cursor or not np.isfinite(values).all():
        raise ValueError('the table of the state cache of %s has another shape than its records or a non-finite value' % source)
    if int(manifest['binding']['boundary']['decision_position']) != DECISION_POSITION:
        raise ValueError('the state cache of %s was made for another decision position' % source)
    out = {}
    for key, wanted in expected.items():
        record = records.get(key)
        if record is None:
            raise ValueError('the state cache of %s lacks the prefix %s' % (source, key))
        differing = [field for field in BOUND_FIELDS if record[field] != wanted[field]]
        if differing:
            raise ValueError('the record of %s in the state cache of %s differs from the bank in %s' % (key, source, differing))
        out[key] = values[record['offset']:record['offset'] + record['length']]
    return out, manifest


def with_states(examples, source, states):
    """Copies of the examples with one more state source."""
    return [Example(e.key, e.group, e.split, dict(e.states, **{source: states[e.key]}), e.budget, e.success, e.cost, e.tokens, e.supported, e.prompt_tokens, e.prefix_tokens)
            for e in examples]
