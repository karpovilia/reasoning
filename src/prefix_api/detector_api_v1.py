"""detector_api_v1: frozen causal API over the native HistoryOnlineDetector.

The wrapper never changes model computation. It adds:
  * full bundle verification: every file, the recomputed bundle id, the EXECUTING copy of this
    module, the runtime modules actually imported, the pinned base-model files actually resolved;
  * named, ordered event/mode axes and validation-only thresholds;
  * explicit primary (+0, immutable) vs revision (lag>=1) outputs with native identities;
  * resumable sessions published as immutable generations behind an atomically switched pointer.

Causal contract: observe(segment) (native name push) is called once per segment, in order, only
after the segment boundary is confirmed. The answer for sid uses the question, at most 16 earlier
whole segments that fit 2048 tokens of BOTH frozen tokenizers, the whole current segment, and the
detector's OWN earlier probabilities. No labels, masks, future text, trace length or final
correctness enter any input.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from copy import deepcopy
from pathlib import Path

API_VERSION = 'detector_api_v1'
EVENTS = ['backtrack', 'verify', 'branch', 'subgoal_done', 'failed_attempt', 'commit']
MODES = ['SETUP', 'DERIVING', 'EXPLORING', 'VERIFYING', 'REVISING', 'CONCLUDING']
# Top-level bundle entries -> file group. Anything else in the bundle root is refused.
GROUP_OF = {'checkpoint': 'checkpoint', 'tokenizers': 'tokenizers', 'code': 'code', 'calibration': 'calibration',
            'api': 'api', 'API_CONTRACT.json': 'contract', 'schema.json': 'contract', 'provenance.json': 'contract',
            'LIMITATIONS.json': 'contract'}
GROUPS = ('checkpoint', 'tokenizers', 'code', 'calibration', 'api', 'contract')
LAYOUT = {'checkpoint': 'checkpoint', 'tokenizers': 'tokenizers', 'code_root': 'code', 'api_root': 'api',
          'thresholds': 'calibration/thresholds.json', 'revision_thresholds': 'calibration/revision_thresholds.json',
          'dynamics_thresholds': 'calibration/dynamics_thresholds.json', 'api_contract': 'API_CONTRACT.json',
          'schema': 'schema.json', 'provenance': 'provenance.json', 'limitations': 'LIMITATIONS.json'}
REQUIRED = {'checkpoint': {'checkpoint/heads.pt', 'checkpoint/detector_config.json', 'checkpoint/backbone/adapter_config.json',
                           'checkpoint/backbone/adapter_model.safetensors', 'checkpoint/CHECKPOINT_COMPLETE.json'},
            'tokenizers': {'tokenizers/encoder/tokenizer.json', 'tokenizers/lora4b/tokenizer.json'},
            'calibration': {'calibration/thresholds.json', 'calibration/revision_thresholds.json',
                            'calibration/dynamics_thresholds.json'},
            'api': {'api/detector_api_v1.py', 'api/schema_check.py'},
            'contract': {'API_CONTRACT.json', 'schema.json', 'provenance.json', 'LIMITATIONS.json'}}
RUNTIME_MODULES = ('clean_detectors.core', 'clean_detectors.model', 'clean_detectors.history_rollout',
                   'clean_detectors.history_inference', 'clean_detectors.spans', 'clean_detectors.dynamics')
IDENTITY = ('target_sid', 'target_sample_id', 'as_of_sid', 'as_of_sample_id', 'as_of_input_sha256', 'revision_id', 'lag')
_fault = None  # test hook: callable(step_name) raising to emulate an interruption


def _step(name):
    if _fault is not None:
        _fault(name)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def tree_hashes(root, skip_cache=True):
    """{relative path: sha256}. skip_cache ignores __pycache__ (source trees); bundles are hashed strictly."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob('*')):
        if skip_cache and '__pycache__' in p.parts:
            continue
        if p.is_symlink():
            raise ValueError('symlink in hashed tree: %s' % p)
        if p.is_file():
            out[str(p.relative_to(root))] = file_sha(p)
    return out


def fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def file_table(root):
    """{group: {relative path: sha256}} of EVERYTHING in the bundle except manifest.json.

    Strict: bytecode caches are files too. A bundle is used read-only or with PYTHONDONTWRITEBYTECODE=1.
    """
    root = Path(root)
    table = {g: {} for g in GROUPS}
    for entry in sorted(root.iterdir()):
        if entry.name == 'manifest.json':
            continue
        if entry.name not in GROUP_OF or entry.is_symlink():
            raise ValueError('unexpected bundle entry: %s' % entry.name)
        group = GROUP_OF[entry.name]
        if entry.is_dir():
            table[group].update({str(Path(entry.name) / k): v for k, v in tree_hashes(entry, skip_cache=False).items()})
        else:
            table[group][entry.name] = file_sha(entry)
    return table


def bundle_id_of(table):
    return hashlib.sha256(canonical(table).encode()).hexdigest()


def _vector(values, name):
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError('calibration vector %s must have 6 entries' % name)
    for v in values:
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1.000001):
            raise ValueError('calibration vector %s has an invalid threshold' % name)


def verify_bundle(manifest_path):
    """Full static verification. Returns (manifest, root, contract, provenance)."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != {'api_version', 'bundle_id', 'status', 'files'} or manifest['api_version'] != API_VERSION:
        raise ValueError('manifest keys or API version are not detector_api_v1')
    actual = file_table(root)
    listed = manifest['files']
    if actual != listed:
        flat_a = {k: v for g in actual.values() for k, v in g.items()}
        flat_l = {k: v for g in listed.values() for k, v in g.items()} if isinstance(listed, dict) else {}
        raise ValueError(('bundle verification failed', {
            'missing': sorted(set(flat_l) - set(flat_a))[:10], 'unlisted': sorted(set(flat_a) - set(flat_l))[:10],
            'sha_mismatch': sorted(k for k in set(flat_a) & set(flat_l) if flat_a[k] != flat_l[k])[:10],
            'group_mismatch': [g for g in GROUPS if set(actual[g]) != set(listed.get(g, {}))]}))
    if manifest['bundle_id'] != bundle_id_of(actual):
        raise ValueError('bundle verification failed: bundle_id is not the digest of the file table')
    for group, required in REQUIRED.items():
        if not required <= set(actual[group]):
            raise ValueError(('bundle verification failed: required files absent', group, sorted(required - set(actual[group]))))
    if actual['api']['api/detector_api_v1.py'] != file_sha(__file__):
        raise ValueError('bundle verification failed: the executing detector_api_v1.py is not the bundled API file')
    contract = json.loads((root / LAYOUT['api_contract']).read_text())
    provenance = json.loads((root / LAYOUT['provenance']).read_text())
    if contract.get('api_version') != API_VERSION or contract.get('layout') != LAYOUT or \
            contract.get('axes') != {'events': EVENTS, 'modes': MODES}:
        raise ValueError('bundle verification failed: contract version/layout/axes differ from the API')
    code = {k[len('code/'):]: v for k, v in actual['code'].items()}
    if not code or code != provenance.get('training_code_sha256'):
        raise ValueError('bundle verification failed: runtime code differs from the frozen bundle runtime')
    base = provenance.get('base_model', {})
    files = base.get('files') or {}
    if not base.get('model_id') or 'config.json' not in files or not any(k.endswith('.safetensors') for k in files):
        raise ValueError('bundle verification failed: pinned base-model file list is empty or lacks weights')
    cal = json.loads((root / LAYOUT['thresholds']).read_text())
    rev = json.loads((root / LAYOUT['revision_thresholds']).read_text())
    dyn = json.loads((root / LAYOUT['dynamics_thresholds']).read_text())
    _vector(cal.get('f1'), 'thresholds.f1')
    _vector(dyn.get('f1'), 'dynamics_thresholds.f1')
    if not isinstance(rev.get('by_exact_lag'), dict) or not rev['by_exact_lag']:
        raise ValueError('revision thresholds are empty')
    for lag, value in rev['by_exact_lag'].items():
        _vector(value.get('f1'), 'revision_thresholds.' + lag)
    config = json.loads((root / LAYOUT['checkpoint'] / 'detector_config.json').read_text())
    if config['model']['model_id'] != base['model_id'] or config['model'].get('revision') != base.get('revision'):
        raise ValueError('bundle verification failed: checkpoint base model differs from pinned provenance')
    return manifest, root, contract, provenance


def import_runtime(code_root):
    """Bind clean_detectors to the verified bundle runtime; refuse a copy imported from elsewhere."""
    code_root = Path(code_root).resolve()

    def foreign():
        bad = []
        for name, module in list(sys.modules.items()):
            if name == 'clean_detectors' or name.startswith('clean_detectors.'):
                origin = getattr(module, '__file__', None)
                if origin is None or code_root not in Path(origin).resolve().parents:
                    bad.append(name)
        return bad
    if foreign():
        raise RuntimeError(('clean_detectors is already imported from outside the verified bundle', sorted(foreign())[:8]))
    sys.dont_write_bytecode = True  # never add files to a verified bundle
    if str(code_root) in sys.path:
        sys.path.remove(str(code_root))
    sys.path.insert(0, str(code_root))
    import importlib
    modules = {name: importlib.import_module(name) for name in RUNTIME_MODULES}
    if foreign():
        raise RuntimeError(('clean_detectors resolved outside the verified bundle', sorted(foreign())[:8]))
    return modules


def resolve_base_snapshot(base):
    """The directory the native loader will read for model_id/revision under the current HF environment."""
    if Path(base['model_id']).is_dir():
        return Path(base['model_id'])
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base['model_id'], revision=base['revision'], local_files_only=True))


def verify_base_model(base):
    snapshot = resolve_base_snapshot(base)
    bad = [name for name, digest in base['files'].items()
           if not (snapshot / name).is_file() or file_sha(snapshot / name) != digest]
    if bad:
        raise ValueError(('base model files differ from pinned SHA256', bad))
    return str(snapshot)


def _threshold(value):
    # 1.000001 is the native sentinel for "no validation positives": never fires.
    return None if value is None or value > 1 else float(value)


class DetectorV1:
    """One trace at a time. Not thread-safe; one instance per concurrent trace."""

    def __init__(self, manifest_path, device='cuda', verify_base=True):
        self.manifest, self.root, self.contract, self.provenance = verify_bundle(manifest_path)
        self.bundle_id = self.manifest['bundle_id']
        self.api_sha256 = file_sha(__file__)
        self.base_snapshot = verify_base_model(self.provenance['base_model']) if verify_base else None
        runtime = import_runtime(self.root / LAYOUT['code_root'])
        core = runtime['clean_detectors.core']
        if list(core.EVENTS) != EVENTS or list(core.MODES) != MODES:
            raise ValueError('axis order in the runtime differs from the API contract')
        self.events, self.modes = list(EVENTS), list(MODES)
        cal = json.loads((self.root / LAYOUT['thresholds']).read_text())
        rev = json.loads((self.root / LAYOUT['revision_thresholds']).read_text())
        self.primary_thresholds = [_threshold(v) for v in cal['f1']]
        self.revision_thresholds = {lag: [_threshold(v) for v in value['f1']]
                                    for lag, value in rev['by_exact_lag'].items() if lag != 'latest'}
        # HistoryOnlineDetector reads <frozen>/tokenizers/{encoder,lora4b}; the bundle root has that layout.
        self.inner = runtime['clean_detectors.history_inference'].HistoryOnlineDetector(
            self.root / LAYOUT['checkpoint'], self.root, device)

    # ---------------------------------------------------------------- session
    def reset(self, trace_id, question):
        """New trace: empty segments, primary/latest records, own history and dynamics state."""
        self.inner.reset(trace_id, question)

    def observe(self, segment):
        """Feed ONE completed segment; returns its +0 answer and revisions of earlier sids."""
        return self.wrap(self.inner.push(segment))

    push = observe  # native name; both refer to the same single-step operation

    def wrap(self, result):
        """Pure reformatting of one native push() result (no model call, no state change)."""
        sid = result['sid']
        out = {'api_version': API_VERSION, 'bundle_id': self.bundle_id, 'trace_id': self.inner.trace_id,
               'sid': sid, 'status': result['status'], 'primary': None, 'dynamics_prior': None, 'revisions': []}
        if result['status'] != 'eligible':
            # The segment does not fit 2048 tokens even without history: no answer, own history reset.
            out['history_reset'] = True
            out['history_used'] = []
            return out
        versions = result['versions']
        primary = versions[0]
        if primary['lag'] != 0 or primary['target_sid'] != sid or primary['as_of_sid'] != sid or primary['revision_id'] != 0:
            raise RuntimeError('native primary version out of order')
        out['history_reset'] = not primary['history_used']
        out['history_used'] = deepcopy(primary['history_used'])
        out['primary'] = self._named(primary, self.primary_thresholds, 'validation F1 thresholds, primary +0')
        if 'dynamics_prior' in primary:
            prior = primary['dynamics_prior']
            out['dynamics_prior'] = {
                'scope': 'forecast for target_sid from own past outputs only, before reading this segment; already fused into primary',
                'as_of_sid': prior['as_of_sid'], 'target_sid': prior['target_sid'],
                'event_probabilities': dict(zip(self.events, prior['event_probabilities'], strict=True)),
                'mode_probabilities': dict(zip(self.modes, prior['mode_probabilities'], strict=True)),
                'any_event_probability': prior['any_event_probability']}
        for v in versions[1:]:
            if v['as_of_sid'] != sid or v['lag'] != sid - v['target_sid'] or v['lag'] < 1 or v['revision_id'] < 1:
                raise RuntimeError('native revision identity inconsistent')
            thresholds = self.revision_thresholds.get(str(v['lag']))
            out['revisions'].append(self._named(v, thresholds, 'validation F1 thresholds for this exact lag' if thresholds
                                                else 'no exact-lag thresholds; probabilities only'))
        return out

    def _named(self, version, thresholds, scope):
        ev = version['event_probabilities']
        mo = version['mode_probabilities']
        rec = {k: version[k] for k in IDENTITY}
        rec.update({'event_probabilities': dict(zip(self.events, ev, strict=True)),
                    'mode_probabilities': dict(zip(self.modes, mo, strict=True)),
                    'mode_argmax': self.modes[max(range(len(mo)), key=mo.__getitem__)],
                    'thresholds_scope': scope,
                    'event_decisions': None if thresholds is None else
                    {name: (None if t is None else bool(p >= t)) for name, p, t in zip(self.events, ev, thresholds, strict=True)}})
        return rec

    def view(self, end_of_trace=False):
        """Observational as-of view (NOT resumable state). Spans from the LATEST available mode per sid.

        end_of_trace=True only when the caller has observed the real end of the trace; it changes
        span closing metadata only and is not a model prediction.
        """
        snap = self.inner.snapshot(end_of_trace)
        spans = [dict(s, mode=self.modes[s['mode']]) for s in snap['spans']]
        return {'api_version': API_VERSION, 'bundle_id': self.bundle_id, 'kind': 'observational_view_not_resumable_state',
                'trace_id': snap['trace_id'], 'as_of_sid': snap['as_of_sid'], 'end_of_trace': end_of_trace, 'spans': spans,
                'spans_scope': 'latest revised mode argmax per sid, may move old boundaries; primary +0 answers are returned by observe()',
                'same_label_boundaries': 'not identifiable: adjacent segments with the same mode form one span'}

    snapshot = view

    def primary_history(self):
        return deepcopy(self.inner.primary)

    # ------------------------------------------------------- resumable state
    def _binding(self):
        return {'api_version': API_VERSION, 'bundle_id': self.bundle_id, 'api_sha256': self.api_sha256}

    def save_session(self, path):
        """Publish a new immutable generation under directory `path`, then switch ACCEPTED.json to it.

        An interruption at any point leaves ACCEPTED.json pointing at a complete generation (the
        previous one or the new one). The previous accepted generation is kept.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        pointer = root / 'ACCEPTED.json'
        previous = json.loads(pointer.read_text())['generation'] if pointer.exists() else None
        name = 'gen_' + uuid.uuid4().hex
        stage = root / (name + '.staging')
        stage.mkdir()
        self.inner.save_session(stage / 'session.pt')  # native: session.pt + session.pt.COMPLETE.json
        _step('after_native_session')
        files = {n: file_sha(stage / n) for n in ('session.pt', 'session.pt.COMPLETE.json')}
        atomic_write(stage / 'BINDING.json', canonical({**self._binding(), 'trace_id': self.inner.trace_id,
                                                       'segments': len(self.inner.segments), 'files': files}))
        for p in stage.iterdir():
            with p.open('rb') as f:
                os.fsync(f.fileno())
        fsync_dir(stage)
        _step('after_binding')
        os.replace(stage, root / name)
        fsync_dir(root)
        _step('after_generation_publish')
        atomic_write(pointer, canonical({'generation': name, 'previous': previous,
                                         'binding_sha256': file_sha(root / name / 'BINDING.json')}))
        _step('after_pointer_switch')
        for old in root.iterdir():  # only now drop generations that are neither accepted nor previous
            if old.is_dir() and old.name not in (name, previous) and (old.name.startswith('gen_')):
                shutil.rmtree(old)
        return name

    def restore_session(self, path):
        root = Path(path)
        pointer = json.loads((root / 'ACCEPTED.json').read_text())
        generation = root / pointer['generation']
        if file_sha(generation / 'BINDING.json') != pointer['binding_sha256']:
            raise ValueError('session binding changed after acceptance')
        binding = json.loads((generation / 'BINDING.json').read_text())
        if {k: binding.get(k) for k in self._binding()} != self._binding():
            raise ValueError('session was saved by a different bundle/API')
        for name, digest in binding['files'].items():
            if file_sha(generation / name) != digest:
                raise ValueError('session file changed after bundle binding')
        self.inner.restore_session(generation / 'session.pt')
        if self.inner.trace_id != binding['trace_id'] or len(self.inner.segments) != binding['segments']:
            raise ValueError('restored session differs from its binding')
        return pointer['generation']
