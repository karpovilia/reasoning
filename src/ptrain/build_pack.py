"""Build the training pack from the frozen 80/10/10 roster and materialized V10-schema units.

Rules
  * Role, question component and exact raw solver come from the roster (index_v2_id / physical_id).
    A unit whose own role differs from the roster is refused (no silent re-assignment).
  * Only TRAIN and DEV rows are read. TEST/HOLD units are never opened (their paths are not even
    resolved); they are only counted from the roster.
  * Teacher-derived E/S inputs (V10 xin[3:41], mode, prev_mode) are NOT copied into the pack:
    they are written as zeros / UNKNOWN with es_ok=False. Causal E/S inputs enter only from an
    explicitly given prefix-detector export (--es-features), then es_ok marks covered positions.
  * Teacher labels are used only for targets: V9 targets as materialized; group-1 targets derived
    from the teacher mode/event labels of each segment and the bound end status.
  * Solver vocabulary is TRAIN-only over exact raw solver IDs (no gpt-oss-20b/openai merge); index 0
    is the fixed unknown for DEV solvers absent from TRAIN.
  * Every labelled roster row of TRAIN/DEV gets a disposition in DISPOSITION.jsonl:
    INCLUDED or NOT_MATERIALIZED with the producer's reason.
Writes PACK/units.json, arrays.npz, text_rows.json, DISPOSITION.jsonl, PACK_MANIFEST.json.
Embeddings (embed.py) and Qwen tokens (model_b.tokenize_pack) are added afterwards.
"""
import argparse, collections, gzip, hashlib, io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import ARRAY_SPEC, write_pack, sha256_file, ns_spec, hierarchical_weights  # noqa: E402
from ptrain import targets_native, namespace_policy  # noqa: E402
from ptrain.targets_group1 import derive  # noqa: E402


def read_roster(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r['index_v2_id']] = r
    return rows


def unit_end_status(d):
    t = d.get('termination_receipt') or {}
    s = t.get('end_status')
    if s in ('natural', 'confirmed_truncated'):
        return s
    return 'unknown'


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--roster', required=True); ap.add_argument('--roster-sha256', required=True)
    ap.add_argument('--units', action='append', required=True, help='dir with UNIT_MANIFEST*.json and units/ (repeatable)')
    ap.add_argument('--manifest-name', default='UNIT_MANIFEST_V1.json')
    ap.add_argument('--out', required=True); ap.add_argument('--es-features', default=None)
    ap.add_argument('--production-admissible', action='store_true')
    ap.add_argument('--producer-ledger', default=None, help='producer versioned row ledger (jsonl with index_v2_id, disposition, reason)')
    ap.add_argument('--producer-ledger-sha256', default=None)
    ap.add_argument('--coverage-note', default=None, help='explicit coverage status of the producer package, e.g. INTERMEDIATE with pending counts')
    ap.add_argument('--component-changes', default=None,
                    help="aligned ROLE_CHANGES json: a unit built under the previous roster may carry component_v1 of its own row")
    ap.add_argument('--limit', type=int, default=None, help='debug only; marks pack non-admissible')
    a = ap.parse_args(argv)
    t0 = time.time()
    if sha256_file(a.roster) != a.roster_sha256:
        raise ValueError('ROSTER_SHA')
    roster = read_roster(a.roster)
    pledger = {}
    if a.producer_ledger:
        if sha256_file(a.producer_ledger) != a.producer_ledger_sha256:
            raise ValueError('PRODUCER_LEDGER_SHA')
        with open(a.producer_ledger) as f:
            for line in f:
                r = json.loads(line); pledger[r['index_v2_id']] = r
    renames = {}
    if a.component_changes:
        for r in json.loads(Path(a.component_changes).read_text()):
            renames[r['index_v2_id']] = (r['component_v1'], r['component_v2'])
    renamed_used = []
    phys_to_idx = {r['physical_id']: k for k, r in roster.items() if r.get('physical_id')}
    if a.es_features:
        raise NotImplementedError('detector feature join is added when the detector export format is fixed')
    # collect materialized units
    cand = {}
    for udir in a.units:
        udir = Path(udir); man = json.loads((udir / a.manifest_name).read_text())
        for r in man:
            key = r.get('roster_id') or r.get('index_v2_id') or r['physical_id']
            if key not in roster:
                key = phys_to_idx.get(r['physical_id'], key)
            if key not in roster:
                raise ValueError('UNIT_NOT_IN_ROSTER:' + r['physical_id'])
            rr = roster[key]
            if rr['role'] not in ('TRAIN', 'DEV'):
                continue                                   # never open TEST/HOLD payloads
            if key in cand:
                raise ValueError('DUPLICATE_UNIT_FOR_ROSTER_ROW:' + key)
            cand[key] = (udir, r)
    cols = {k: [] for k in ARRAY_SPEC}
    # native namespace table (union over unit dirs, identical rows required)
    ns_rows = {}
    for udir in a.units:
        f = Path(udir) / 'NAMESPACE_TABLE.json'
        if f.exists():
            for r in json.loads(f.read_text()):
                if r['namespace_id'] in ns_rows and ns_rows[r['namespace_id']] != r:
                    raise ValueError('NAMESPACE_TABLE_CONFLICT:' + r['namespace_id'])
                ns_rows[r['namespace_id']] = r
    ns_ids = sorted(ns_rows); ns_index = {k: i for i, k in enumerate(ns_ids)}
    namespaces = [dict(id=k, K=len(ns_rows[k]['mode_vocab']), E=len(ns_rows[k]['event_vocab']), status=ns_rows[k]['status'],
                       mode_vocab=ns_rows[k]['mode_vocab'], event_vocab=ns_rows[k]['event_vocab'],
                       annotation_version=ns_rows[k].get('annotation_version')) for k in ns_ids]
    for i, ns in enumerate(namespaces):
        for k in ns_spec(ns):
            cols[f'ns{i}_{k}'] = []
    ns_support = {k: collections.Counter() for k in ns_ids}
    # vocabulary over TRAIN raw solvers
    train_solvers = sorted({roster[k]['solver'] for k in cand if roster[k]['role'] == 'TRAIN'})
    vocab = ['__OOV__'] + train_solvers; vidx = {s: i for i, s in enumerate(vocab)}
    units = []; text_rows = {}; disp = []
    g1_explicit = collections.Counter(); ends = collections.Counter()
    keys = sorted(cand)
    if a.limit:
        keys = keys[:a.limit]
    for key in keys:
        udir, r = cand[key]; rr = roster[key]
        raw = (udir / r['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != r['sha256']:
            raise ValueError('UNIT_SHA:' + key)
        d = json.loads(gzip.decompress(raw))
        if d.get('question_component') != rr['question_component']:
            rn = renames.get(key)
            if not (rn and rn[0] == d.get('question_component') and rn[1] == rr['question_component']):
                raise ValueError('COMPONENT_MISMATCH:' + key)      # old unit role is ignored; the roster role applies
            renamed_used.append(key)
        x = d['inputs']; t = d['targets']; v = t['target_valid']; n = len(x['mode'])
        xin = np.asarray(x['xin'], 'f4')
        if xin.shape != (n, 41):
            raise ValueError('XIN_SHAPE:' + key)
        mode_lab = list(x['mode'])                                    # teacher mode of segment j: target source only
        ev_lab = [[int(o > 0) for o in row[5:11]] for row in x['xin']]
        ev_known = [[bool(k > 0.5) for k in row[11:17]] for row in x['xin']]
        end = unit_end_status(d); ends[end] += 1
        g1, g1y, g1k, explicit = derive(mode_lab, ev_lab, ev_known, end, span_start=d.get('span_start'))
        g1_explicit[explicit] += 1
        xin_in = xin.copy(); xin_in[:, 3:] = 0.0                      # C kept; teacher S/E never enter as input
        sol = vidx.get(rr['solver'], 0)
        rows_t = [text_rows.setdefault(h, len(text_rows)) for h in x['text_sha256']]
        rows_p = [text_rows.setdefault(h, len(text_rows)) for h in x['prompt_sha256']]
        col = dict(xin=xin_in, mode=np.full(n, 7), prev_mode=np.full(n, 7), solver=np.full(n, sol), text_row=np.array(rows_t), prompt_row=np.array(rows_p),
                   es_ok=np.zeros(n, bool), gate=np.array(t['gate']), dest=np.array(t['dest']), event_y=np.array(t['event_y'], 'f4'),
                   length_y=np.array(t['length_y'], 'f4'), wait_y=np.array(t['wait_y'], 'f4'), wait_observed=np.array(t['wait_observed'], bool),
                   next_span=np.array(t['next_span']), rem=np.array(t['rem'], 'f4'),
                   v_step_outcome=np.array(v['step_outcome'], bool), v_next_type=np.array(v['next_type'], bool), v_length=np.array(v['length'], bool),
                   v_events=np.array(v['events'], bool), v_wait_observed=np.array(v['wait_observed'], bool), v_wait_censored=np.array(v['wait_censored'], bool),
                   v_remainder=np.array(v['remainder'], bool), g1_next_span=np.array(g1), g1_ev_y=np.array(g1y, 'f4'), g1_ev_known=np.array(g1k, bool))
        nl = d.get('native_labels') or {}
        unknown_ns = set(nl) - set(ns_index)
        if unknown_ns:
            raise ValueError('NAMESPACE_NOT_IN_TABLE:' + ','.join(sorted(unknown_ns)))
        for ni, ns in enumerate(namespaces):
            lab = nl.get(ns['id'])
            if lab is None:
                tt = dict(gate=[-1] * n, dest=[-1] * n, ev=[[0] * ns['E']] * n, vstep=[False] * n, vnt=[False] * n, vev=[[False] * ns['E']] * n)
            else:
                if len(lab['mode']) != n or len(lab['event']) != n or len(lab['event_known']) != n:
                    raise ValueError('NATIVE_LENGTH:' + key + ':' + ns['id'])
                if any(m >= ns['K'] or m < -1 for m in lab['mode']) or any(len(e) != ns['E'] for e in lab['event']):
                    raise ValueError('NATIVE_VOCAB:' + key + ':' + ns['id'])
                tt = targets_native.derive(lab['mode'], lab['event'], lab['event_known'], end, span_start=lab.get('span_start'))
                ns_support[ns['id']][rr['role'] + '|traces'] += 1
                ns_support[ns['id']][rr['role'] + '|step_known'] += sum(tt['vstep'])
                ns_support[ns['id']][rr['role'] + '|switch_known'] += sum(tt['vnt'])
            for k in ns_spec(ns):
                col[f'ns{ni}_{k}'] = np.array(tt[k])
        for k in cols:
            cols[k].append(col[k])
        units.append(dict(unit_id=r['unit_id'], physical_id=r['physical_id'], index_v2_id=key, question_component=rr['question_component'], role=rr['role'],
                          n=n, solver_ID=rr['solver'], benchmark_audit_only=rr.get('benchmark'), es_class='none_primary_unknown_mask',
                          end_status=end, annotation_version=rr.get('annotation_version'), prompt_sha256=x['prompt_sha256'][0],
                          text_sha256=x['text_sha256']))
        disp.append(dict(index_v2_id=key, role=rr['role'], disposition='INCLUDED', positions=n))
    included = {d['index_v2_id'] for d in disp}
    for k, rr in roster.items():
        if rr['role'] in ('TRAIN', 'DEV') and rr.get('labelled') and k not in included:
            pr = pledger.get(k)
            disp.append(dict(index_v2_id=k, role=rr['role'], disposition='NOT_MATERIALIZED',
                             producer_disposition=pr.get('disposition') if pr else None,
                             reason=(pr.get('reason') if pr and pr.get('reason') else None) or rr.get('not_in_V6_reason')
                             or ('producer ledger row without reason' if pr else 'not in producer ledger')))
    arrays = {k: np.concatenate(vv) if vv else np.zeros((0,) + ARRAY_SPEC[k][1]) for k, vv in cols.items()}
    # SR-10 (COORDINATOR_POSITIVE_ONLY_EVENT_MASK_DECISION_V1): event supervision eligibility is a frozen namespace table
    # (ptrain/namespace_policy.py), not a count rule. Native values and known masks stay unchanged; TRAIN axis support is
    # recorded descriptively.
    tr_mask = np.concatenate([np.full(u['n'], u['role'] == 'TRAIN') for u in units]) if units else np.zeros(0, bool)
    for i, ns in enumerate(namespaces):
        ev = arrays[f'ns{i}_ev']; vk = arrays[f'ns{i}_vev']
        ns['event_supervision_eligible'] = namespace_policy.event_supervision_eligible(ns['id'])
        ns['event_axis_support_train'] = [dict(axis=ax, name=ns['event_vocab'][ax],
                                               positive=int((vk[:, ax] & tr_mask & (ev[:, ax] > 0.5)).sum()),
                                               negative=int((vk[:, ax] & tr_mask & (ev[:, ax] < 0.5)).sum())) for ax in range(ns['E'])]
    if namespaces:
        off = 0
        for u in units:
            u['offset'] = off; off += u['n']
        elig = np.zeros(off, bool)
        for i, ns in enumerate(namespaces):
            if ns['status'] == 'accepted':
                elig |= arrays[f'ns{i}_vstep'] | arrays[f'ns{i}_vnt']
                if ns['event_supervision_eligible']:
                    elig |= arrays[f'ns{i}_vev'].any(-1)
        arrays['w_pos'] = hierarchical_weights(units, elig)
        no_target = {u['index_v2_id'] for u in units if not elig[u['offset']:u['offset'] + u['n']].any()}
        for d0 in disp:
            if d0['index_v2_id'] in no_target:
                d0['disposition'] = 'INCLUDED_NO_ADMITTED_TARGET'   # kept for inputs/accounting, contributes no loss
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'text_rows.json').write_text(json.dumps(sorted(text_rows, key=text_rows.get)))
    with open(out / 'DISPOSITION.jsonl', 'w') as f:
        for d in disp:
            f.write(json.dumps(d) + '\n')
    lab = collections.Counter((rr['role'], bool(rr.get('labelled'))) for rr in roster.values())
    if a.production_admissible and str(a.coverage_note or '').startswith('INTERMEDIATE'):
        raise ValueError('INTERMEDIATE_PACK_CANNOT_BE_PRODUCTION_ADMISSIBLE')
    extra = dict(synthetic=False, production_admissible=bool(a.production_admissible and not a.limit),
                 roster=dict(path=str(Path(a.roster).resolve()), sha256=a.roster_sha256),
                 unit_sources=[dict(dir=str(Path(u).resolve()), manifest_sha256=sha256_file(Path(u) / a.manifest_name)) for u in a.units],
                 roster_labelled_by_role={f'{r}|{l}': c for (r, l), c in sorted(lab.items())},
                 dispositions={f"{d0}|{d1}": c for (d0, d1), c in collections.Counter((d['role'], d['disposition']) for d in disp).items()},
                 end_status=dict(ends), group1_boundaries_explicit=dict((str(k), v) for k, v in g1_explicit.items()),
                 es_input='none: teacher E/S removed from inputs; es_ok False everywhere until causal detector features are joined',
                 text_rows=len(text_rows), build_seconds=round(time.time() - t0, 1),
                 coverage_note=a.coverage_note,
                 producer_ledger=dict(path=str(Path(a.producer_ledger).resolve()), sha256=a.producer_ledger_sha256) if a.producer_ledger else None,
                 producer_counts={str(Path(u).resolve()): json.loads((Path(u) / 'ACTUAL_COUNTS_V10.json').read_text()).get('dispositions')
                                  for u in a.units if (Path(u) / 'ACTUAL_COUNTS_V10.json').exists()},
                 component_changes=dict(path=str(Path(a.component_changes).resolve()), sha256=sha256_file(a.component_changes),
                                        applied=len(renamed_used)) if a.component_changes else None,
                 namespace_policy=namespace_policy.DECISION,
                 namespaces=namespaces, namespace_support={k: dict(v) for k, v in ns_support.items()},
                 target_policy='native namespaces routed to own heads (COORDINATOR_NATIVE_TARGET_NAMESPACES_DECISION_V2); V9 targets kept for the reproducible original variant')
    man = write_pack(out, units, arrays, vocab, extra)
    man['files']['text_rows.json'] = sha256_file(out / 'text_rows.json'); man['files']['DISPOSITION.jsonl'] = sha256_file(out / 'DISPOSITION.jsonl')
    (out / 'PACK_MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(dict(role_counts=man['role_counts'], dispositions=man['dispositions'], text_rows=len(text_rows), seconds=man['build_seconds'])))


if __name__ == '__main__':
    main()
