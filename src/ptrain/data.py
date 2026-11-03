"""Packed data contract for the predictor trainer.

A data package directory contains
  PACK_MANIFEST.json   schema, sha256 of every file, roster sha, role counts, es policy, solver vocab
  units.json           one row per physical trace: unit_id, physical_id, question_component, role,
                       offset, n, solver_ID, benchmark_audit_only, es_class, end_status, text_sha (B)
  arrays.npz           concatenated per-position arrays (see ARRAY_SPEC)
  embeddings.npy       (R,384) float16 MiniLM rows referenced by text_row/prompt_row (A only)
Windows are derived deterministically here: blocks of 64 scored positions with up to 256 segments
of left context (same rule as V10.3 WINDOWS_V1), so every target is scored exactly once.
Only roles TRAIN and DEV may be present; any other role is refused.
"""
import hashlib, json, os
from pathlib import Path

ARRAY_SPEC = {
    # inputs
    'xin': ('f4', (41,)), 'mode': ('i1', ()), 'prev_mode': ('i1', ()), 'solver': ('i2', ()),
    'text_row': ('i4', ()), 'prompt_row': ('i4', ()), 'es_ok': ('?', ()),
    # V9 targets
    'gate': ('i1', ()), 'dest': ('i1', ()), 'event_y': ('f4', (6,)), 'length_y': ('f4', ()),
    'wait_y': ('f4', ()), 'wait_observed': ('?', ()), 'next_span': ('i1', ()), 'rem': ('f4', (3,)),
    'v_step_outcome': ('?', ()), 'v_next_type': ('?', ()), 'v_length': ('?', ()), 'v_events': ('?', (6,)),
    'v_wait_observed': ('?', ()), 'v_wait_censored': ('?', ()), 'v_remainder': ('?', (3,)),
    # group-1 targets
    'g1_next_span': ('i1', ()), 'g1_ev_y': ('f4', (6,)), 'g1_ev_known': ('?', (6,)),
}
INPUT_KEYS = ('xin', 'mode', 'prev_mode', 'solver', 'text_row', 'prompt_row', 'es_ok')
VALID_KEYS = ('step_outcome', 'next_type', 'length', 'events', 'wait_observed', 'wait_censored', 'remainder')
BLOCK = 64


def ns_spec(ns):
    E = ns['E']
    spec = {'gate': ('i1', ()), 'dest': ('i1', ()), 'ev': ('f4', (E,)), 'vstep': ('?', ()), 'vnt': ('?', ()), 'vev': ('?', (E,))}
    if ns.get('group1'):
        # G2: first next span type (K = natural END, -1 unknown) and events inside it with own known mask
        spec.update({'g1n': ('i1', ()), 'g1e': ('f4', (E,)), 'g1k': ('?', (E,))})
    if ns.get('group3'):
        # G3: native label of the observed segment j at a masked position (type -1 = no target), events with known mask
        spec.update({'g3m': ('i1', ()), 'g3e': ('f4', (E,)), 'g3k': ('?', (E,))})
    return spec


DIAG_KEYS = ('xin', 'mode', 'prev_mode')   # fixed diagnostic DEV inputs (G3 masked / G4 corrupted), never used for selection


def hierarchical_weights(units, eligible):
    """w_p = 1/(eligible positions of its trace) / (eligible traces of its question); 0 if not eligible.
    Then sum over a question's positions = 1: mean over positions, then traces, then equal questions."""
    import numpy as np
    w = np.zeros(len(eligible), 'f4'); tr_per_q = {}
    for u in units:
        if eligible[u['offset']:u['offset'] + u['n']].any():
            tr_per_q[u['question_component']] = tr_per_q.get(u['question_component'], 0) + 1
    for u in units:
        e = eligible[u['offset']:u['offset'] + u['n']]; c = int(e.sum())
        if c:
            w[u['offset']:u['offset'] + u['n']] = e.astype('f4') / c / tr_per_q[u['question_component']]
    return w
CONTEXT = 256


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_windows(units, roles=('TRAIN', 'DEV'), block=BLOCK, context=CONTEXT):
    out = []
    for ui, u in enumerate(units):
        if u['role'] not in roles:
            continue
        n = u['n']
        for hi in range(min(block, n), n + block, block):
            hi = min(hi, n)
            score_lo = ((hi - 1) // block) * block
            lo = max(0, hi - context)
            out.append((ui, lo, hi, score_lo))
            if hi == n:
                break
    return out


class PackedData:
    def __init__(self, root, verify=True, allow_roles=('TRAIN', 'DEV'), need_embeddings=True):
        import numpy as np
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'PACK_MANIFEST.json').read_text())
        if verify:
            for name, expected in self.manifest['files'].items():
                if sha256_file(self.root / name) != expected:
                    raise ValueError('PACK_FILE_SHA:' + name)
        self.units = json.loads((self.root / 'units.json').read_text())
        roles = {u['role'] for u in self.units}
        if roles - set(allow_roles):
            raise ValueError('PROTECTED_ROLE_IN_PACK:' + ','.join(sorted(roles - set(allow_roles))))
        for key in ('physical_id', 'question_component'):
            tr = {u[key] for u in self.units if u['role'] == 'TRAIN'}; dv = {u[key] for u in self.units if u['role'] == 'DEV'}
            if tr & dv:
                raise ValueError('TRAIN_DEV_OVERLAP:' + key)
        if len({u['physical_id'] for u in self.units}) != len(self.units):
            raise ValueError('DUPLICATE_PHYSICAL_ID')
        z = np.load(self.root / 'arrays.npz', allow_pickle=False)
        self.a = {k: z[k] for k in ARRAY_SPEC}
        P = sum(u['n'] for u in self.units)
        for k, (dt, shp) in ARRAY_SPEC.items():
            if self.a[k].shape != (P,) + shp:
                raise ValueError('ARRAY_SHAPE:' + k)
        self.namespaces = self.manifest.get('namespaces', [])
        for i, ns in enumerate(self.namespaces):
            for k, (dt, shp) in ns_spec(ns).items():
                arr = z[f'ns{i}_{k}']
                if arr.shape != (P,) + shp:
                    raise ValueError('NS_ARRAY_SHAPE:%d:%s' % (i, k))
                self.a[f'ns{i}_{k}'] = arr
        self.a['w_pos'] = z['w_pos'] if 'w_pos' in z.files else None
        self.input_variant = None
        self.has_g4 = bool(self.manifest.get('group4_targets'))
        if self.has_g4:
            for k in ('g4b', 'g4k'):
                if z[k].shape != (P,):
                    raise ValueError('G4_ARRAY_SHAPE:' + k)
                self.a[k] = z[k]
        self.has_diag = bool(self.manifest.get('diag_inputs'))
        if self.has_diag:
            for k in DIAG_KEYS:
                arr = z['diag_' + k]
                if arr.shape != self.a[k].shape:
                    raise ValueError('DIAG_ARRAY_SHAPE:' + k)
                self.a['diag_' + k] = arr
        if self.namespaces and self.a['w_pos'] is None:
            raise ValueError('W_POS_REQUIRED_WITH_NAMESPACES')
        self.embedding = None
        if need_embeddings:
            self.embedding = np.load(self.root / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
            if self.embedding.ndim != 2 or self.embedding.shape[1] != 384:
                raise ValueError('EMBEDDING_SHAPE')
        self.windows = make_windows(self.units)
        qpos = {}
        for u in self.units:
            qpos[u['question_component']] = qpos.get(u['question_component'], 0) + u['n']
        self.qpos = qpos
        self.nsolvers = len(self.manifest['solver_vocab'])

    def ids(self, role):
        return [i for i, w in enumerate(self.windows) if self.units[w[0]]['role'] == role]

    def use_inputs(self, variant):
        """None = pack inputs (TRAIN as trained, DEV natural); 'diag' = fixed diagnostic DEV inputs (G3 masked / G4 corrupted)."""
        if variant == 'diag' and not self.has_diag:
            raise ValueError('PACK_HAS_NO_DIAG_INPUTS')
        self.input_variant = variant

    def make_batch(self, ids, device, T=CONTEXT):
        import numpy as np
        import torch
        N = len(ids)
        def full(shape, dtype, fill=0):
            return torch.full(shape, fill, dtype=dtype)
        b = dict(xin=full((N, T, 41), torch.float32), mode=full((N, T), torch.long, 7), prev_mode=full((N, T), torch.long, 7),
                 solver=full((N, T), torch.long), es_ok=full((N, T), torch.bool, False),
                 text=full((N, T, 384), torch.float32), prompt=full((N, T, 384), torch.float32),
                 pad=full((N, T), torch.bool, True), score=full((N, T), torch.bool, False), weight=full((N, T), torch.float32))
        tg = {'gate': ((), torch.long, -1), 'dest': ((), torch.long, -1), 'event_y': ((6,), torch.float32, 0),
              'length_y': ((), torch.float32, 0), 'wait_y': ((), torch.float32, 0), 'wait_observed': ((), torch.bool, False),
              'next_span': ((), torch.long, -1), 'rem': ((3,), torch.float32, 0),
              'g1_next_span': ((), torch.long, -1), 'g1_ev_y': ((6,), torch.float32, 0), 'g1_ev_known': ((6,), torch.bool, False)}
        b.update({k: full((N, T) + s, d, f) for k, (s, d, f) in tg.items()})
        b['valid'] = {k: full((N, T, 6) if k == 'events' else ((N, T, 3) if k == 'remainder' else (N, T)), torch.bool, False) for k in VALID_KEYS}
        fill = {'i1': (torch.long, -1), 'f4': (torch.float32, 0), '?': (torch.bool, False)}
        b['ns'] = [{k: full((N, T) + shp, *fill[dt]) for k, (dt, shp) in ns_spec(ns).items()} for ns in self.namespaces]
        if self.has_g4:
            b['g4b'] = full((N, T), torch.long, -1); b['g4k'] = full((N, T), torch.long, -1)
        b['_question_ids'] = []; b['_unit_ids'] = []; b['_positions'] = []
        a = self.a
        for i, wi in enumerate(ids):
            ui, lo, hi, score_lo = self.windows[wi]; u = self.units[ui]; n = hi - lo
            if not (0 <= lo <= score_lo < hi <= u['n']) or n > T:
                raise ValueError('WINDOW_COORDINATE')
            s0 = u['offset'] + lo; s1 = u['offset'] + hi
            for k in INPUT_KEYS:
                if k in ('text_row', 'prompt_row'):
                    continue
                src = 'diag_' + k if (self.input_variant == 'diag' and k in DIAG_KEYS) else k
                b[k][i, :n] = torch.from_numpy(np.ascontiguousarray(a[src][s0:s1])).to(b[k].dtype)
            for k in tg:
                b[k][i, :n] = torch.from_numpy(np.ascontiguousarray(a[k][s0:s1])).to(b[k].dtype)
            for k in VALID_KEYS:
                b['valid'][k][i, :n] = torch.from_numpy(np.ascontiguousarray(a['v_' + k][s0:s1]))
            if self.embedding is not None:
                for src, dst in (('text_row', 'text'), ('prompt_row', 'prompt')):
                    rows = a[src][s0:s1]
                    if rows.min() < 0 or rows.max() >= len(self.embedding):
                        raise ValueError('EMBEDDING_ROW')
                    b[dst][i, :n] = torch.from_numpy(np.asarray(self.embedding[rows], dtype=np.float32))
            b['pad'][i, :n] = False
            b['score'][i, score_lo - lo:n] = True
            if a['w_pos'] is not None:
                b['weight'][i, :n] = torch.from_numpy(np.ascontiguousarray(a['w_pos'][s0:s1]))
            else:
                b['weight'][i, :n] = 1.0 / self.qpos[u['question_component']]
            if self.has_g4:
                for k in ('g4b', 'g4k'):
                    b[k][i, :n] = torch.from_numpy(np.ascontiguousarray(a[k][s0:s1])).to(torch.long)
            for j, ns in enumerate(self.namespaces):
                for k, (dt, shp) in ns_spec(ns).items():
                    b['ns'][j][k][i, :n] = torch.from_numpy(np.ascontiguousarray(a[f'ns{j}_{k}'][s0:s1])).to(b['ns'][j][k].dtype)
            b['_question_ids'].append(u['question_component']); b['_unit_ids'].append(u['unit_id'])
            b['_positions'].append((lo, score_lo, hi))
        return to_device(b, device)


def to_device(b, device):
    out = {}
    for k, v in b.items():
        if k.startswith('_'):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [{kk: vv.to(device, non_blocking=True) for kk, vv in d.items()} for d in v]
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def write_pack(root, units, arrays, solver_vocab, extra_manifest, embeddings=None):
    """Write a package atomically-enough (files first, manifest last)."""
    import numpy as np
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    off = 0
    for u in units:
        u['offset'] = off; off += u['n']
    for k, (dt, shp) in ARRAY_SPEC.items():
        arr = arrays[k]
        if arr.shape != (off,) + shp:
            raise ValueError('WRITE_SHAPE:' + k)
        arrays[k] = arr.astype(dt, copy=False)
    for i, ns in enumerate(extra_manifest.get('namespaces', [])):
        for k, (dt, shp) in ns_spec(ns).items():
            arr = arrays[f'ns{i}_{k}']
            if arr.shape != (off,) + shp:
                raise ValueError('WRITE_NS_SHAPE:%d:%s' % (i, k))
            arrays[f'ns{i}_{k}'] = arr.astype(dt, copy=False)
    (root / 'units.json').write_text(json.dumps(units, ensure_ascii=False))
    np.savez(root / 'arrays.npz', **arrays)
    files = ['units.json', 'arrays.npz']
    if embeddings is not None:
        np.save(root / 'embeddings.npy', embeddings.astype(np.float16)); files.append('embeddings.npy')
    roles = {}
    for u in units:
        r = roles.setdefault(u['role'], {'units': 0, 'positions': 0, 'questions': set()})
        r['units'] += 1; r['positions'] += u['n']; r['questions'].add(u['question_component'])
    man = dict(schema='predictor_pack_v1', solver_vocab=solver_vocab,
               role_counts={k: {'units': v['units'], 'positions': v['positions'], 'questions': len(v['questions'])} for k, v in roles.items()},
               files={f: sha256_file(root / f) for f in files})
    man.update(extra_manifest)
    tmp = root / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1)); os.replace(tmp, root / 'PACK_MANIFEST.json')
    return man
