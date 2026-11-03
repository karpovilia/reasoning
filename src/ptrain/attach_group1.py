"""G2 targets (event -> first next span -> events inside it) per native namespace, attached to an existing pack.

Adds ns{i}_g1n / ns{i}_g1e / ns{i}_g1k (ptrain/targets_native.derive_group1) from the accepted native labels of the same
producer units (unit sha and exact text sequence checked), sets namespaces[i].group1 = True and recomputes w_pos over the
union of next-step and G2 target positions (same hierarchical rule). Inputs, C/E/S, texts, embeddings, tokens and all
next-step targets stay byte-identical, so the G2 pack of CT and the G2 pack of CTES_RETRO pair exactly (attach this to
pack_v3 and to the retro pack, or attach_retro_es to this pack). Future labels enter only these targets.
"""
import argparse, collections, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import sha256_file, hierarchical_weights  # noqa: E402
from ptrain import targets_native, namespace_policy  # noqa: E402
from ptrain.attach_retro_es import unit_index, read_unit, link_copy, COPY_FILES  # noqa: E402
from ptrain.build_pack import unit_end_status  # noqa: E402


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--units', required=True); ap.add_argument('--roster', required=True); ap.add_argument('--roster-sha256', required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    src = Path(a.pack); out = Path(a.out); udir = Path(a.units)
    man = json.loads((src / 'PACK_MANIFEST.json').read_text())
    for name, h in man['files'].items():
        if sha256_file(src / name) != h:
            raise ValueError('SOURCE_PACK_FILE_SHA:' + name)
    if sha256_file(a.roster) != a.roster_sha256 or man['roster']['sha256'] != a.roster_sha256:
        raise ValueError('ROSTER_SHA')
    nss = man['namespaces']
    if any(ns.get('group1') for ns in nss):
        raise ValueError('PACK_ALREADY_HAS_GROUP1')
    byk = unit_index(man, udir, a.roster)
    units = json.loads((src / 'units.json').read_text())
    z = np.load(src / 'arrays.npz', allow_pickle=False); arrays = {k: z[k] for k in z.files}
    P = len(arrays['xin'])
    for i, ns in enumerate(nss):
        arrays[f'ns{i}_g1n'] = np.full(P, -1, 'i1'); arrays[f'ns{i}_g1e'] = np.zeros((P, ns['E']), 'f4'); arrays[f'ns{i}_g1k'] = np.zeros((P, ns['E']), bool)
    idx = {ns['id']: i for i, ns in enumerate(nss)}
    support = {ns['id']: collections.Counter() for ns in nss}
    for u in units:
        d = read_unit(byk, udir, u)
        end = unit_end_status(d)
        if end != u['end_status']:
            raise ValueError('END_STATUS_DIFFERS_FROM_PACK:' + u['index_v2_id'])
        o, n = u['offset'], u['n']
        for nid, lab in (d.get('native_labels') or {}).items():
            if lab is None:
                continue
            i = idx[nid]; K = nss[i]['K']
            t = targets_native.derive_group1(lab['mode'], lab['event'], lab['event_known'], end, K, span_start=lab.get('span_start'))
            g1n = np.array(t['g1n']); arrays[f'ns{i}_g1n'][o:o + n] = g1n
            arrays[f'ns{i}_g1e'][o:o + n] = np.array(t['g1e'], 'f4').reshape(n, -1)
            arrays[f'ns{i}_g1k'][o:o + n] = np.array(t['g1k'], bool).reshape(n, -1)
            c = support[nid]; r = u['role']
            c[r + '|next_span_known'] += int((g1n >= 0).sum()); c[r + '|next_span_END'] += int((g1n == K).sum())
    tr = np.concatenate([np.full(u['n'], u['role'] == 'TRAIN') for u in units])
    elig = np.zeros(P, bool)
    for i, ns in enumerate(nss):
        ns['group1'] = True
        ev_ok = namespace_policy.event_supervision_eligible(ns['id'])
        if ev_ok != ns.get('event_supervision_eligible', ev_ok):
            raise ValueError('EVENT_POLICY_DIFFERS:' + ns['id'])
        g, e, k = arrays[f'ns{i}_g1n'], arrays[f'ns{i}_g1e'], arrays[f'ns{i}_g1k']
        ns['group1_span_event_support_train'] = [dict(axis=ax, name=ns['event_vocab'][ax] if 'event_vocab' in ns else ax,
                                                      positive=int((k[:, ax] & tr & (e[:, ax] > .5)).sum()), negative=int((k[:, ax] & tr & (e[:, ax] < .5)).sum()))
                                                 for ax in range(ns['E'])]
        if ns['status'] == 'accepted':
            elig |= arrays[f'ns{i}_vstep'] | arrays[f'ns{i}_vnt'] | (g >= 0)
            if ev_ok:
                elig |= arrays[f'ns{i}_vev'].any(-1)
    w_old = arrays['w_pos']
    arrays['w_pos'] = hierarchical_weights(units, elig)
    out.mkdir(parents=True, exist_ok=True)
    link_copy(src, out, COPY_FILES)
    np.savez_compressed(out / 'arrays.npz', **arrays)          # compressed: pod /tmp budget; np.load reads both
    new = dict(man)
    new['files'] = {f: sha256_file(out / f) for f in man['files'] if (out / f).exists()}
    new['namespaces'] = nss
    new['group1_targets'] = dict(group='G2_EVENT_SPAN_LINK', derivation='ptrain/targets_native.derive_group1',
                                 targets_native_sha256=sha256_file(Path(__file__).parent / 'targets_native.py'), attach_sha256=sha256_file(Path(__file__)),
                                 source_pack_manifest_sha256=sha256_file(src / 'PACK_MANIFEST.json'), support=dict((k, dict(v)) for k, v in support.items()),
                                 w_pos='recomputed over next-step OR G2 target positions (hierarchical rule); positions changed: %d' % int((w_old != arrays['w_pos']).sum()),
                                 coefficients='losses.G1_COEF next_span 0.5, span_events 0.25; DEV selection stays the next-step objective')
    new['build_seconds_attach_group1'] = round(time.time() - t0, 1)
    tmp = out / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1)); os.replace(tmp, out / 'PACK_MANIFEST.json')
    print(json.dumps(dict(support={k: dict(v) for k, v in support.items()}, seconds=new['build_seconds_attach_group1'])))


if __name__ == '__main__':
    main()
