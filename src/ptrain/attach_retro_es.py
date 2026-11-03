"""Explicit retrospective_annotation E/S input mode (RETRO_TES_EXECUTION_DECISION_V1): CT pack -> CTES_RETRO pack.

Only the input fields xin[:, 3:41] (S2 + E36), mode, prev_mode and es_ok change. Targets, masks, weights, roles, C,
texts, embeddings, tokens and windows stay byte-identical to the source pack, so CT/CTES pairing is exact.
Source of E/S: the accepted V10.7 producer units (same unit files as the source pack; unit file sha and the per-position
text_sha256 sequence are checked against the source pack). Values at position j are those of the producer recipe
'C3 textonly/S2 observedknownrun/E36 mode-independent windows8/64 and EWM4/32': mode/prev/S2 of the observed run up to j,
event obs/known of segment j, E36 counters over positions <= j (audited by tools/check_retro_inputs.py).
Native closures: where the primary native namespace of a non-R unit is closed (decision 48850f65, HLE positions inside
accepted v4.6 windows; no fallback), the input is closed too: mode -> UNKNOWN, event known -> False, then S2 / prev and E36
of that unit are recomputed sequentially from the closed label stream. GROUNDING (R op 6) is already UNKNOWN in inputs.
es_ok is True on all positions of included units: unknown values are encoded inside the inputs (mode UNKNOWN, known 0).
Interpretation: retrospectively annotated structure, not an online detector measurement.
"""
import argparse, collections, gzip, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import sha256_file  # noqa: E402

KIND = 'retrospective_annotation'
UNKNOWN = 7
COPY_FILES = ('units.json', 'text_rows.json', 'embeddings.npy', 'tokens_qwen3.npz', 'TOKENS_MANIFEST.json', 'DISPOSITION.jsonl')
INTERPRETATION = ('retrospectively annotated structure of observed segments; the annotator could see the continuation, so this is '
                  'not an online detector measurement and not proof of causal availability of labels during generation')


def independent_E(positive, known):
    """Verbatim from owner_review_v9/src/feature_recipe_v7.py (the producer's E36 recipe)."""
    histories = [collections.deque() for _ in range(6)]; tot8 = [0] * 6; tot64 = [0] * 6; mem4 = [0.] * 6; mem32 = [0.] * 6; out = []
    for yy, kk in zip(positive, known):
        obs = [float(bool(y) and bool(k)) for y, k in zip(yy, kk)]
        for a, v in enumerate(obs):
            h = histories[a]; h.append(v); tot8[a] += v; tot64[a] += v
            if len(h) > 8:
                tot8[a] -= h[-9]
            if len(h) > 64:
                tot64[a] -= h.popleft()
            mem4[a] = mem4[a] * math.exp(-.25) + v; mem32[a] = mem32[a] * math.exp(-1 / 32) + v
        out.append(obs + [float(k) for k in kk] + [math.log1p(v) / 5 for v in tot8 + tot64 + mem4 + mem32])
    return out


def observed_run(mode, chars):
    """S2 and prev of the observed run (compile_native_process_inputs_v3 rule): a run restarts at j=0, at a mode change
    and next to an unknown mode; S2 = log1p(age)/10, log1p(run chars)/10, zero where the mode is unknown."""
    n = len(mode); s2 = []; prev = []; age = sc = 0; previous = UNKNOWN
    for j in range(n):
        if j == 0 or mode[j] == UNKNOWN or mode[j - 1] == UNKNOWN or mode[j] != mode[j - 1]:
            previous = mode[j - 1] if j and mode[j] != UNKNOWN else UNKNOWN; age = sc = 0
        age += 1; sc += chars[j]
        s2.append([math.log1p(age) / 10, math.log1p(sc) / 10] if mode[j] != UNKNOWN else [0.0, 0.0]); prev.append(previous)
    return s2, prev


def unit_inputs(d):
    """(xin41, mode, prev, closed_count) of a producer unit with native closures applied."""
    x = d['inputs']; xin = [list(map(float, z)) for z in x['xin']]; mode = list(x['mode']); prev = list(x['prev_mode']); n = len(mode)
    ns = d.get('namespace_id'); nl = (d.get('native_labels') or {}).get(ns)
    closed = 0
    if nl is not None and '|SYS_R' not in str(ns):
        ev = [z[5:11] for z in xin]; kn = [[bool(v) for v in z[11:17]] for z in xin]
        for j in range(n):
            if mode[j] != UNKNOWN and nl['mode'][j] == -1:
                mode[j] = UNKNOWN; closed += 1
            for e in range(6):
                if kn[j][e] and not nl['event_known'][j][e]:
                    kn[j][e] = False; ev[j][e] = 0.0; closed += 1
        if closed:
            chars = [math.expm1(10 * z[1]) for z in xin]
            s2, prev = observed_run(mode, chars); ee = independent_E(ev, kn)
            xin = [z[:3] + s + e for z, s, e in zip(xin, s2, ee)]
    return xin, mode, prev, closed


def unit_index(man, udir, roster_path):
    """index_v2_id -> producer manifest row for TRAIN/DEV units of the source pack's own unit source (TEST/HOLD never opened)."""
    umf = Path(udir) / 'UNIT_MANIFEST_V1.json'
    if not any(s['manifest_sha256'] == sha256_file(umf) for s in man['unit_sources']):
        raise ValueError('UNIT_SOURCE_NOT_THE_SOURCE_PACK_SOURCE')
    roster = {}
    for line in open(roster_path):
        r = json.loads(line); roster[r['index_v2_id']] = r
    phys = {r['physical_id']: k for k, r in roster.items() if r.get('physical_id')}
    byk = {}
    for r in json.loads(umf.read_text()):
        k = r.get('roster_id') or r.get('index_v2_id') or r['physical_id']
        k = k if k in roster else phys.get(r['physical_id'], k)
        if k in roster and roster[k]['role'] in ('TRAIN', 'DEV'):
            byk[k] = r
    return byk


def read_unit(byk, udir, u):
    """Producer unit of a pack unit; unit file sha and the exact per-position text sequence must match (pairing)."""
    r = byk.get(u['index_v2_id'])
    if r is None:
        raise ValueError('UNIT_NOT_IN_SOURCE:' + u['index_v2_id'])
    raw = (Path(udir) / r['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != r['sha256']:
        raise ValueError('UNIT_SHA:' + u['index_v2_id'])
    d = json.loads(gzip.decompress(raw))
    if d['inputs']['text_sha256'] != u['text_sha256'] or len(d['inputs']['mode']) != u['n']:
        raise ValueError('PAIRING_TEXT_SEQUENCE:' + u['index_v2_id'])
    return d


def link_copy(src, out, names):
    for f in names:
        if (src / f).exists():
            if (out / f).exists():
                (out / f).unlink()
            try:
                os.link(src / f, out / f)
            except OSError:
                shutil.copy2(src / f, out / f)


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--units', required=True, help='producer unit dir of the source pack (UNIT_MANIFEST_V1.json + units/)')
    ap.add_argument('--roster', required=True); ap.add_argument('--roster-sha256', required=True)
    ap.add_argument('--decision', required=True, help='coordinator decision admitting retrospective_annotation inputs')
    ap.add_argument('--audit', default=None, help='RETRO_INPUT_AUDIT json of tools/check_retro_inputs.py (pinned in the manifest)')
    a = ap.parse_args(argv)
    t0 = time.time()
    src = Path(a.pack); out = Path(a.out); udir = Path(a.units)
    man = json.loads((src / 'PACK_MANIFEST.json').read_text())
    for name, h in man['files'].items():
        if sha256_file(src / name) != h:
            raise ValueError('SOURCE_PACK_FILE_SHA:' + name)
    if sha256_file(a.roster) != a.roster_sha256 or man['roster']['sha256'] != a.roster_sha256:
        raise ValueError('ROSTER_SHA')
    dec = json.loads(Path(a.decision).read_text())
    if KIND not in json.dumps(dec.get('input_contract', {})) and KIND not in str(dec.get('decision', '')).lower():
        raise ValueError('DECISION_DOES_NOT_ADMIT_RETROSPECTIVE_ANNOTATION')
    umf = udir / 'UNIT_MANIFEST_V1.json'
    byk = unit_index(man, udir, a.roster)
    units = json.loads((src / 'units.json').read_text())
    z = np.load(src / 'arrays.npz', allow_pickle=False); arrays = {k: z[k] for k in z.files}
    xin = arrays['xin'].copy(); mode = arrays['mode'].copy(); prev = arrays['prev_mode'].copy(); es_ok = arrays['es_ok'].copy()
    cnt = collections.Counter(); cov = {r: dict(positions=0, es_known=0, s_known=0, e_known=0) for r in ('TRAIN', 'DEV')}
    for u in units:
        d = read_unit(byk, udir, u)
        o, n = u['offset'], u['n']
        xi, md, pv, closed = unit_inputs(d)
        xi = np.asarray(xi, 'f4')
        if not np.allclose(xi[:, :3], arrays['xin'][o:o + n, :3], atol=1e-6):
            raise ValueError('C_DIFFERS_FROM_SOURCE_PACK:' + u['index_v2_id'])
        xin[o:o + n, 3:] = xi[:, 3:]; mode[o:o + n] = md; prev[o:o + n] = pv; es_ok[o:o + n] = True
        c = cov[u['role']]; c['positions'] += n; c['es_known'] += n
        c['s_known'] += int((np.asarray(md) != UNKNOWN).sum()); c['e_known'] += int((xi[:, 11:17] > 0.5).any(-1).sum())
        cnt['units'] += 1; cnt['closed_cells'] += closed; cnt['units_with_closures'] += int(closed > 0)
    arrays.update(xin=xin.astype('f4'), mode=mode.astype(arrays['mode'].dtype), prev_mode=prev.astype(arrays['prev_mode'].dtype), es_ok=es_ok)
    out.mkdir(parents=True, exist_ok=True)
    link_copy(src, out, COPY_FILES)
    np.savez_compressed(out / 'arrays.npz', **arrays)          # compressed: pod /tmp budget; np.load reads both
    new = dict(man)
    new['files'] = {f: sha256_file(out / f) for f in man['files'] if (out / f).exists()}
    new['es_input'] = KIND
    new['es_input_source'] = dict(kind=KIND, producer_unit_manifest=dict(path=str(umf.resolve()), sha256=sha256_file(umf)),
                                  feature_recipe='C3 textonly/S2 observedknownrun/E36 mode-independent windows8/64 and EWM4/32',
                                  native_closures='primary non-R namespace closed cells -> input UNKNOWN, S2/prev/E36 recomputed',
                                  attach_sha256=sha256_file(Path(__file__)), source_pack_manifest_sha256=sha256_file(src / 'PACK_MANIFEST.json'),
                                  audit=dict(path=str(Path(a.audit).resolve()), sha256=sha256_file(a.audit)) if a.audit else None,
                                  interpretation=INTERPRETATION)
    new['es_admission'] = dict(accepted=True, kind=KIND, acceptance_path=str(Path(a.decision).resolve()), acceptance_sha256=sha256_file(a.decision))
    new['es_coverage'] = cov; new['es_attach_counts'] = dict(cnt)
    new['production_admissible'] = man.get('production_admissible', False)
    new['build_seconds_attach_retro'] = round(time.time() - t0, 1)
    tmp = out / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1)); os.replace(tmp, out / 'PACK_MANIFEST.json')
    print(json.dumps(dict(es_coverage=cov, counts=dict(cnt), seconds=new['build_seconds_attach_retro'])))


if __name__ == '__main__':
    main()
