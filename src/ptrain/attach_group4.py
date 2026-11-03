"""G4 (localization of injected annotation errors), COORDINATOR_G3_G4_DESIGN_DECISION_V1 (873f7855...): pack delta, CTES_RETRO only.

Source: a retrospective_annotation pack (pack_v3_retro). Edits apply to the observed E/S input stream of a trace
(input type 0..5 / UNKNOWN 7, input events obs/known 6), never to next-step targets.
Plan, deterministic from package seed, group tag 'G4', canonical physical UID and operation tag:
  affected traces: within each primary native namespace, traces ranked by sha256(seed|G4|uid|affect); the first floor(n/2)
    are affected, the rest are unchanged controls. TRAIN seed 20261021 edits the pack inputs; DEV keeps natural inputs
    and gets a separate fixed corrupted copy in diag_* (seed 20261022).
  budget per affected trace: floor(0.05 * N_known) actually changed positions (N_known = positions with a known input type
    or a known input event); budget 0 -> no edit (recorded).
  plan: strictly causal, see edit_plan (COORDINATOR_G4_CAUSAL_PLAN_DECISION_V2 c559eec2..., amending 873f7855): one
    left-to-right pass, decisions from labels <= j and independent keyed draws; causal quota floor(0.05 * K_j) at every
    prefix; TYPE = partial replacement possible (quota stop), BOUNDARY = delayed boundary with possible suppression of a
    short right span; EVENT flips one known cell. q per type fixed once from TRAIN affected traces (calibrate), recorded in
    the manifest, identical for A/B, seeds and the diagnostic DEV. Audit flags stay in the plan file, never in model inputs.
After edits S2, prev and E36 are recomputed sequentially from the edited stream.
Targets per position (g4b, g4k): scored only where the clean input has a known type or a known event; g4b = 1 iff an
observed input label value of this position changed, g4k = edit type (1 TYPE, 2 BOUNDARY, 3 EVENT) or 0; -1 unscored.
Later positions whose counters differ are not labelled corrupted. Synthetic errors only: no claim about real errors.
"""
import argparse, collections, hashlib, json, math, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import sha256_file, hierarchical_weights  # noqa: E402
from ptrain import namespace_policy  # noqa: E402
from ptrain.attach_retro_es import unit_index, read_unit, link_copy, COPY_FILES, observed_run, independent_E, UNKNOWN  # noqa: E402

TRAIN_SEED, DEV_SEED, RATE = 20261021, 20261022, 0.05
TYPE, BOUNDARY, EVENT = 1, 2, 3


def h64(*parts):
    return int(hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def spans_of(mode, span_start=None):
    """Spans of known type: (start, end_excl, type); a new span starts at a type change, next to an unknown label, or at an
    explicit native span_start."""
    n = len(mode); out = []; j = 0
    while j < n:
        if mode[j] == UNKNOWN:
            j += 1; continue
        s = j; j += 1
        while j < n and mode[j] == mode[s] and not (span_start is not None and span_start[j]):
            j += 1
        out.append((s, j, mode[s]))
    return out


def u01(key, *parts):
    """Independent uniform draw in [0, 1) from the trace key and the candidate's own coordinates (no global candidate list)."""
    return h64(key, *parts) / float(1 << 64)


def edit_plan(mode, ev, kn, key, events_ok, q, span_start=None, rate=RATE):
    """Strictly causal G4 plan (PTV6-001/PTV7-001 under G4 causal decision): one left-to-right pass; every decision at
    position j uses only clean labels at positions <= j, independent draws keyed by (trace key, kind, position) and the
    decisions already taken. Causal quota: changed positions up to j <= floor(rate * known positions up to j), hence also
    <= floor(rate * N_known) for the trace.
      TYPE      at a span start s (draw < q['TYPE']): new type r != t written at s, s+1, ... while the position stays in the
                same clean span and the quota allows; a quota stop is recorded (truncated_by_quota).
      BOUNDARY  at a boundary b between known spans of different types whose position b-1 is unedited
                (draw < q['BOUNDARY']): d in {1,2,3}; the left type (observed at b-1) is written at b, b+1, ... while the
                position stays in the right clean span, fewer than d positions were written and the quota allows; a right
                span shorter than d is absorbed and recorded (absorbed_right_span).
      EVENT     each known cell of an unedited position (draw < q['EVENT']) when the namespace has certified negatives:
                the cell is flipped, at most one cell per position.
    Returns (mode', ev', kind[j], ops, cap_final)."""
    n = len(mode); m2 = list(mode); e2 = [list(r) for r in ev]; kind = [0] * n; ops = []
    K = 0; changed = 0; act = None

    def new_span(j):
        return mode[j] != UNKNOWN and (j == 0 or mode[j - 1] == UNKNOWN or mode[j] != mode[j - 1] or bool(span_start and span_start[j]))

    for j in range(n):
        K += int(mode[j] != UNKNOWN or any(kn[j])); cap = int(math.floor(rate * K))
        if act is not None:                                     # continue an ongoing TYPE / BOUNDARY edit at j
            same = mode[j] == act['span_type'] and not new_span(j)
            more = same and (act['type'] == 'TYPE' or len(act['positions']) < act['shift'])
            if more and changed < cap:
                m2[j] = act['value']; kind[j] = act['kind']; act['positions'].append(j); changed += 1
            else:
                if more:
                    act['truncated_by_quota'] = True
                elif act['type'] == 'BOUNDARY' and not same and len(act['positions']) < act['shift']:
                    act['absorbed_right_span'] = True
                act = None
        if act is None and not kind[j] and mode[j] != UNKNOWN and new_span(j):
            fired = []
            if u01(key, 'TYPE', j) < q['TYPE']:
                fired.append(TYPE)
            if (j > 0 and mode[j - 1] != UNKNOWN and mode[j - 1] != mode[j] and not kind[j - 1]
                    and u01(key, 'BOUNDARY', j) < q['BOUNDARY']):
                fired.append(BOUNDARY)
            if fired and changed < cap:
                t = fired[int(u01(key, 'PICK', j) * len(fired))]
                if t == TYPE:
                    others = [x for x in range(6) if x != mode[j]]
                    act = dict(type='TYPE', kind=TYPE, span_type=mode[j], value=others[int(u01(key, 'NEW', j) * 5)], start=j,
                               old=mode[j], positions=[j])
                else:
                    act = dict(type='BOUNDARY', kind=BOUNDARY, span_type=mode[j], value=mode[j - 1], boundary=j,
                               shift=1 + int(u01(key, 'SHIFT', j) * 3), inserted_from=j - 1, positions=[j])
                m2[j] = act['value']; kind[j] = act['kind']; changed += 1
                ops.append(act)
        if events_ok and not kind[j] and changed < cap:
            for x in range(6):
                if kn[j][x] and u01(key, 'EVENT', j, x) < q['EVENT']:
                    e2[j][x] = 0.0 if e2[j][x] > 0.5 else 1.0; kind[j] = EVENT; changed += 1
                    ops.append(dict(type='EVENT', position=j, event=x, new=e2[j][x]))
                    break
    for op in ops:                                               # final bookkeeping for the plan record
        if op['type'] == 'TYPE':
            op['new'] = op.pop('value'); op['end'] = op['positions'][-1] + 1; op.pop('span_type'); op.pop('kind')
        elif op['type'] == 'BOUNDARY':
            op.pop('value'); op.pop('span_type'); op.pop('kind')
    return m2, e2, kind, ops, int(math.floor(rate * K))


def calibrate(streams, rate=RATE):
    """q per edit type from TRAIN affected traces only (once, before any plan): each type aims at rate/3 of the known
    positions, using mean written positions per fired candidate (span length for TYPE, min(d, right span) for BOUNDARY)."""
    K = nb = ns = span_pos = bnd_pos = cells = 0
    for mode, kn, events_ok, ss in streams:
        n = len(mode)
        K += sum(1 for j in range(n) if mode[j] != UNKNOWN or any(kn[j]))
        sp = spans_of(mode, ss); ns += len(sp); span_pos += sum(e - s for s, e, _ in sp)
        for a, b in zip(sp, sp[1:]):
            if a[1] == b[0] and a[2] != b[2]:
                nb += 1; bnd_pos += sum(min(d, b[1] - b[0]) for d in (1, 2, 3)) / 3
        if events_ok:
            cells += sum(sum(1 for v in r if v) for r in kn)
    tgt = rate / 3 * K
    q = dict(TYPE=min(1.0, tgt / max(span_pos, 1) * 1.0), BOUNDARY=min(1.0, tgt / max(bnd_pos, 1)), EVENT=min(1.0, tgt / max(cells, 1)))
    return q, dict(known=K, spans=ns, span_positions=span_pos, boundaries=nb, boundary_positions=round(bnd_pos, 1), event_cells=cells)


def rebuild(xin, m2, e2, kn):
    import numpy as np
    chars = [math.expm1(10 * float(z[1])) for z in xin]
    s2, prev = observed_run(m2, chars); ee = independent_E(e2, kn)
    out = np.array(xin, 'f4').copy(); out[:, 3:5] = np.array(s2, 'f4'); out[:, 5:41] = np.array(ee, 'f4')
    return out, np.array(m2), np.array(prev)


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--units', required=True); ap.add_argument('--roster', required=True); ap.add_argument('--roster-sha256', required=True)
    ap.add_argument('--decision', required=True)
    ap.add_argument('--decision-amendment', action='append', default=[], help='later coordinator decisions amending the design (pinned by sha)')
    a = ap.parse_args(argv)
    t0 = time.time()
    src = Path(a.pack); out = Path(a.out); udir = Path(a.units)
    man = json.loads((src / 'PACK_MANIFEST.json').read_text())
    for name, h in man['files'].items():
        if sha256_file(src / name) != h:
            raise ValueError('SOURCE_PACK_FILE_SHA:' + name)
    if sha256_file(a.roster) != a.roster_sha256 or man['roster']['sha256'] != a.roster_sha256:
        raise ValueError('ROSTER_SHA')
    if 'G4' not in json.loads(Path(a.decision).read_text()):
        raise ValueError('DECISION_WITHOUT_G4')
    if (man.get('es_input_source') or {}).get('kind') != 'retrospective_annotation':
        raise ValueError('G4_REQUIRES_RETROSPECTIVE_ANNOTATION_PACK')
    if man.get('diag_inputs') or man.get('group4_targets') or any(ns.get('group1') or ns.get('group3') for ns in man['namespaces']):
        raise ValueError('PACK_ALREADY_HAS_AUX_GROUP')
    byk = unit_index(man, udir, a.roster)
    units = json.loads((src / 'units.json').read_text())
    z = np.load(src / 'arrays.npz', allow_pickle=False); arrays = {k: z[k] for k in z.files}
    P = len(arrays['xin'])
    for k in ('xin', 'mode', 'prev_mode'):
        arrays['diag_' + k] = arrays[k].copy()
    arrays['g4b'] = np.full(P, -1, 'i1'); arrays['g4k'] = np.full(P, -1, 'i1')
    # affected traces: within each (role, primary namespace), first floor(n/2) by stable hash
    docs = {}
    groups = collections.defaultdict(list)
    for u in units:
        d = read_unit(byk, udir, u); docs[u['index_v2_id']] = (d.get('namespace_id'), (d.get('native_labels') or {}).get(d.get('namespace_id')))
        seed = TRAIN_SEED if u['role'] == 'TRAIN' else DEV_SEED
        groups[(u['role'], d.get('namespace_id'))].append((h64(seed, 'G4', u['physical_id'], 'affect'), u['index_v2_id']))
    affected = set()
    for k, lst in groups.items():
        lst.sort(); affected.update(i for _, i in lst[:len(lst) // 2])
    def stream(u):
        o, n = u['offset'], u['n']; ns, lab = docs[u['index_v2_id']]
        x = arrays['xin'][o:o + n]; mode = [int(v) for v in arrays['mode'][o:o + n]]
        ev = [[float(v) for v in r[5:11]] for r in x]; kn = [[bool(v > .5) for v in r[11:17]] for r in x]
        ss = lab.get('span_start') if lab and '|SYS_R' not in str(ns) and lab.get('span_start') and len(lab['span_start']) == n else None
        return x, mode, ev, kn, ss, (namespace_policy.event_supervision_eligible(ns) if ns else False)
    # q calibrated once on TRAIN affected traces only (no DEV), constants for every trace
    q, qstats = calibrate((lambda t: (t[1], t[3], t[5], t[4]))(stream(u)) for u in units if u['role'] == 'TRAIN' and u['index_v2_id'] in affected)
    stats = {r: collections.Counter() for r in ('TRAIN', 'DEV')}
    plan = []
    for u in units:
        o, n = u['offset'], u['n']; role = u['role']; seed = TRAIN_SEED if role == 'TRAIN' else DEV_SEED
        x, mode, ev, kn, ss, ev_ok = stream(u)
        known = np.array([mode[j] != UNKNOWN or any(kn[j]) for j in range(n)])
        c = stats[role]; c['traces'] += 1; c['scored_positions'] += int(known.sum())
        g4b = np.where(known, 0, -1).astype('i1'); g4k = np.where(known, 0, -1).astype('i1')
        if u['index_v2_id'] in affected:
            c['affected_traces'] += 1
            m2, e2, kind, ops, budget = edit_plan(mode, ev, kn, f"{seed}|G4|{u['physical_id']}|edit", ev_ok, q, ss)
            changed = np.array([m2[j] != mode[j] or e2[j] != ev[j] for j in range(n)])
            if (changed != (np.array(kind) > 0)).any():
                raise AssertionError('G4_EDIT_KIND_MISMATCH:' + u['index_v2_id'])
            c['budget'] += budget; c['changed_positions'] += int(changed.sum()); c['no_edit_traces'] += int(not ops)
            for op in ops:
                c['op_' + op['type']] += 1
                c['truncated_by_quota'] += int(bool(op.get('truncated_by_quota'))); c['absorbed_right_span'] += int(bool(op.get('absorbed_right_span')))
            if ops:
                xi, md, pv = rebuild(x, m2, e2, kn)
                tgt = ('xin', 'mode', 'prev_mode') if role == 'TRAIN' else ('diag_xin', 'diag_mode', 'diag_prev_mode')
                arrays[tgt[0]][o:o + n] = xi; arrays[tgt[1]][o:o + n] = md; arrays[tgt[2]][o:o + n] = pv
                g4b[changed] = 1; g4k[changed] = np.array(kind)[changed]
            plan.append(dict(uid=u['physical_id'], role=role, affected=True, budget=budget, ops=ops, events_ok=bool(ev_ok),
                             span_start_positions=[j for j in range(n) if ss and ss[j]] if ss else None))
        else:
            plan.append(dict(uid=u['physical_id'], role=role, affected=False))
        arrays['g4b'][o:o + n] = g4b; arrays['g4k'][o:o + n] = g4k
    nss = man['namespaces']
    elig = np.zeros(P, bool)
    for i, ns in enumerate(nss):
        if ns['status'] == 'accepted':
            elig |= arrays[f'ns{i}_vstep'] | arrays[f'ns{i}_vnt']
            if namespace_policy.event_supervision_eligible(ns['id']):
                elig |= arrays[f'ns{i}_vev'].any(-1)
    elig |= arrays['g4b'] >= 0
    w_old = arrays['w_pos']; arrays['w_pos'] = hierarchical_weights(units, elig)
    out.mkdir(parents=True, exist_ok=True)
    link_copy(src, out, COPY_FILES)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    pf = out / 'G4_EDIT_PLAN.jsonl'
    with open(pf, 'w') as f:
        for r in plan:
            f.write(json.dumps(r) + '\n')
    new = dict(man)
    new['files'] = {f: sha256_file(out / f) for f in man['files'] if (out / f).exists()}
    new['files']['G4_EDIT_PLAN.jsonl'] = sha256_file(pf)
    new['diag_inputs'] = True
    new['group4_targets'] = dict(group='G4_ANNOTATION_ERROR', decision=dict(path=str(Path(a.decision).resolve()), sha256=sha256_file(a.decision)),
                                 decision_amendments=[dict(path=str(Path(x).resolve()), sha256=sha256_file(x)) for x in a.decision_amendment],
                                 train_seed=TRAIN_SEED, dev_diag_seed=DEV_SEED, rate=RATE, affected_fraction='floor(n/2) per (role, primary namespace)',
                                 plan='strictly causal left-to-right (PTV6-001/PTV7-001); causal quota floor(rate*K_j)', q=q, q_calibration_train=qstats,
                                 attach_sha256=sha256_file(Path(__file__)), source_pack_manifest_sha256=sha256_file(src / 'PACK_MANIFEST.json'),
                                 support={r: dict(c) for r, c in stats.items()},
                                 w_pos='recomputed over next-step OR G4 scored positions; positions changed: %d' % int((w_old != arrays['w_pos']).sum()),
                                 dev='selection on natural DEV inputs; G4 metrics only on diag_* inputs (fixed DEV corruption)',
                                 coefficients='losses.G4_COEF changed BCE 0.25, edit type CE (4 classes incl. clean) 0.25',
                                 interpretation='synthetic annotation-error localization only; G4 vs G1_CTES_RETRO = whole corruption+auxiliary condition')
    new['build_seconds_attach_group4'] = round(time.time() - t0, 1)
    tmp = out / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1)); os.replace(tmp, out / 'PACK_MANIFEST.json')
    print(json.dumps(dict(support={r: dict(c) for r, c in stats.items()}, seconds=new['build_seconds_attach_group4'])))


if __name__ == '__main__':
    main()
