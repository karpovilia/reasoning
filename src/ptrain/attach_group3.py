"""G3 (masked annotation reconstruction), COORDINATOR_G3_G4_DESIGN_DECISION_V1 (873f7855...): pack delta.

Plan (frozen, deterministic from package seed, group tag 'G3', canonical physical UID and operation tag 'mask'; independent of
iteration order and fit seed; identical for CT and CTES_RETRO and for A and B):
  eligible position j: at least one accepted native namespace with an enabled reconstruction target known at j
    (type known, or an event known in a namespace whose event supervision is enabled);
  per trace quota floor(0.15 * N_eligible); blocks of 1..8 consecutive eligible positions (a block never crosses a
    non-eligible position), starts drawn among unmasked eligible positions, the last block clipped to the remaining quota,
    no overlap; quota 0 -> no mask, the trace still trains the next step.
  TRAIN uses seed 20261021 (written into the pack inputs of TRAIN units); DEV keeps natural inputs for selection and gets
  a separate fixed diagnostic mask with seed 20261022 in diag_* arrays.
Inputs (CTES_RETRO packs only): at masked positions the whole E/S input becomes unknown (type UNKNOWN, event obs/known 0),
  then S2, prev and E36 of the unit are recomputed sequentially (ctes rule = producer rule, 0 differences on 2093 units).
  CT packs keep their inputs (no E/S); the mask only defines the targets.
Targets per namespace at masked positions: native type of segment j (-1 if unknown) and its events with the native known mask
  (positive-only namespaces keep their masks; the loss disables their event term). Nothing after j enters.
"""
import argparse, collections, hashlib, json, math, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import sha256_file, hierarchical_weights  # noqa: E402
from ptrain import namespace_policy  # noqa: E402
from ptrain.attach_retro_es import unit_index, read_unit, link_copy, COPY_FILES, observed_run, independent_E, UNKNOWN  # noqa: E402

TRAIN_SEED, DEV_SEED, RATE, MAX_LEN = 20261021, 20261022, 0.15, 8


def plan_rng(seed, group, uid, op):
    return random.Random(int(hashlib.sha256(f'{seed}|{group}|{uid}|{op}'.encode()).hexdigest()[:16], 16))


def mask_plan(eligible, rng, rate=RATE, max_len=MAX_LEN):
    n = len(eligible); el = [j for j in range(n) if eligible[j]]; quota = int(math.floor(rate * len(el)))
    masked = [False] * n; done = 0; tries = 0
    while done < quota and tries < 50 * quota + 50:
        tries += 1
        s = rng.choice(el)                                  # start among eligible positions; an already masked start is skipped
        if masked[s]:
            continue
        L = rng.randint(1, max_len); j = s
        while j < n and j < s + L and eligible[j] and not masked[j] and done < quota:
            masked[j] = True; done += 1; j += 1
    return masked, quota


def masked_inputs(xin, mode, masked):
    """Inputs of one unit with E/S unknown at masked positions and dependent features recomputed sequentially."""
    import numpy as np
    m2 = [UNKNOWN if h else int(m) for m, h in zip(mode, masked)]
    ev = [[0.0] * 6 if h else [float(v) for v in z[5:11]] for z, h in zip(xin, masked)]
    kn = [[False] * 6 if h else [bool(v > .5) for v in z[11:17]] for z, h in zip(xin, masked)]
    chars = [math.expm1(10 * float(z[1])) for z in xin]
    s2, prev = observed_run(m2, chars); ee = independent_E(ev, kn)
    out = np.array(xin, 'f4').copy(); out[:, 3:5] = np.array(s2, 'f4'); out[:, 5:41] = np.array(ee, 'f4')
    return out, np.array(m2), np.array(prev)


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--units', required=True); ap.add_argument('--roster', required=True); ap.add_argument('--roster-sha256', required=True)
    ap.add_argument('--decision', required=True, help='COORDINATOR_G3_G4_DESIGN_DECISION_V1.json')
    a = ap.parse_args(argv)
    t0 = time.time()
    src = Path(a.pack); out = Path(a.out); udir = Path(a.units)
    man = json.loads((src / 'PACK_MANIFEST.json').read_text())
    for name, h in man['files'].items():
        if sha256_file(src / name) != h:
            raise ValueError('SOURCE_PACK_FILE_SHA:' + name)
    if sha256_file(a.roster) != a.roster_sha256 or man['roster']['sha256'] != a.roster_sha256:
        raise ValueError('ROSTER_SHA')
    if 'G3' not in json.loads(Path(a.decision).read_text()):
        raise ValueError('DECISION_WITHOUT_G3')
    nss = man['namespaces']
    if any(ns.get('group3') or ns.get('group1') for ns in nss) or man.get('diag_inputs'):
        raise ValueError('PACK_ALREADY_HAS_AUX_GROUP')
    retro = (man.get('es_input_source') or {}).get('kind') == 'retrospective_annotation'
    byk = unit_index(man, udir, a.roster)
    units = json.loads((src / 'units.json').read_text())
    z = np.load(src / 'arrays.npz', allow_pickle=False); arrays = {k: z[k] for k in z.files}
    P = len(arrays['xin'])
    for k in ('xin', 'mode', 'prev_mode'):
        arrays['diag_' + k] = arrays[k].copy()
    for i, ns in enumerate(nss):
        arrays[f'ns{i}_g3m'] = np.full(P, -1, 'i1'); arrays[f'ns{i}_g3e'] = np.zeros((P, ns['E']), 'f4'); arrays[f'ns{i}_g3k'] = np.zeros((P, ns['E']), bool)
    idx = {ns['id']: i for i, ns in enumerate(nss)}
    ev_ok = {ns['id']: namespace_policy.event_supervision_eligible(ns['id']) for ns in nss}
    acc = {ns['id']: ns['status'] == 'accepted' for ns in nss}
    stats = {r: collections.Counter() for r in ('TRAIN', 'DEV')}
    plan = []
    for u in units:
        d = read_unit(byk, udir, u); o, n = u['offset'], u['n']; role = u['role']
        labs = {k: v for k, v in (d.get('native_labels') or {}).items() if v is not None and acc.get(k)}
        elig = [False] * n
        for k, lab in labs.items():
            for j in range(n):
                if lab['mode'][j] >= 0 or (ev_ok[k] and any(lab['event_known'][j])):
                    elig[j] = True
        uid = u['physical_id']
        masked, quota = mask_plan(elig, plan_rng(TRAIN_SEED if role == 'TRAIN' else DEV_SEED, 'G3', uid, 'mask'))
        c = stats[role]; c['traces'] += 1; c['eligible'] += sum(elig); c['masked'] += sum(masked); c['zero_quota_traces'] += int(quota == 0)
        if not any(masked):
            plan.append(dict(uid=uid, role=role, masked=[]))
            continue
        plan.append(dict(uid=uid, role=role, masked=[j for j in range(n) if masked[j]]))
        for k, lab in labs.items():
            i = idx[k]
            for j in range(n):
                if masked[j]:
                    arrays[f'ns{i}_g3m'][o + j] = lab['mode'][j]
                    arrays[f'ns{i}_g3e'][o + j] = [float(v) for v in lab['event'][j]]
                    arrays[f'ns{i}_g3k'][o + j] = [bool(v) for v in lab['event_known'][j]]
                    c['type_known|' + k] += int(lab['mode'][j] >= 0)
        if retro:
            xi, md, pv = masked_inputs(arrays['xin'][o:o + n], arrays['mode'][o:o + n], masked)
            tgt = ('xin', 'mode', 'prev_mode') if role == 'TRAIN' else ('diag_xin', 'diag_mode', 'diag_prev_mode')
            arrays[tgt[0]][o:o + n] = xi; arrays[tgt[1]][o:o + n] = md; arrays[tgt[2]][o:o + n] = pv
    elig_all = np.zeros(P, bool)
    for i, ns in enumerate(nss):
        ns['group3'] = True
        if ns['status'] == 'accepted':
            elig_all |= arrays[f'ns{i}_vstep'] | arrays[f'ns{i}_vnt'] | (arrays[f'ns{i}_g3m'] >= 0)
            if ev_ok[ns['id']]:
                elig_all |= arrays[f'ns{i}_vev'].any(-1) | arrays[f'ns{i}_g3k'].any(-1)
    w_old = arrays['w_pos']; arrays['w_pos'] = hierarchical_weights(units, elig_all)
    out.mkdir(parents=True, exist_ok=True)
    link_copy(src, out, COPY_FILES)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    pf = out / 'G3_MASK_PLAN.jsonl'
    with open(pf, 'w') as f:
        for r in plan:
            f.write(json.dumps(r) + '\n')
    new = dict(man)
    new['files'] = {f: sha256_file(out / f) for f in man['files'] if (out / f).exists()}
    new['files']['G3_MASK_PLAN.jsonl'] = sha256_file(pf)
    new['namespaces'] = nss; new['diag_inputs'] = True
    new['group3_targets'] = dict(group='G3_MASKED_ANNOTATION', decision=dict(path=str(Path(a.decision).resolve()), sha256=sha256_file(a.decision)),
                                 train_seed=TRAIN_SEED, dev_diag_seed=DEV_SEED, rate=RATE, max_block=MAX_LEN, inputs_masked=retro,
                                 attach_sha256=sha256_file(Path(__file__)), source_pack_manifest_sha256=sha256_file(src / 'PACK_MANIFEST.json'),
                                 support={r: dict(c) for r, c in stats.items()},
                                 w_pos='recomputed over next-step OR G3 target positions; positions changed: %d' % int((w_old != arrays['w_pos']).sum()),
                                 dev='selection on natural DEV inputs; G3 metrics only on diag_* inputs (fixed DEV mask)',
                                 coefficients='losses.G3_COEF type 0.5, events 0.25')
    new['build_seconds_attach_group3'] = round(time.time() - t0, 1)
    tmp = out / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1)); os.replace(tmp, out / 'PACK_MANIFEST.json')
    print(json.dumps(dict(support={r: dict(c) for r, c in stats.items()}, inputs_masked=retro, seconds=new['build_seconds_attach_group3'])))


if __name__ == '__main__':
    main()
