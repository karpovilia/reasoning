"""Explicit retrospective_annotation input mode (ptrain/attach_retro_es.py) and the runner gate for E/S input kinds."""
import gzip, hashlib, json, random, shutil, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import synthetic, attach_retro_es as R, model_a, runner, losses as L  # noqa: E402
from ptrain.data import PackedData, sha256_file  # noqa: E402

SMALL = dict(width=32, heads=4, layers=1, ff_width=64, dropout=0.0, max_segments=256)
NS = 'syn_flash'


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def producer_unit(xc, rng, n, closed=()):
    """V10.7-like unit: producer recipe inputs from labels; `closed` positions are closed in the primary native labels."""
    mode = [7 if rng.random() < .05 else rng.randrange(6) for _ in range(n)]
    for j in range(1, n):
        if rng.random() < .8:
            mode[j] = mode[j - 1]
    ev = [[float(rng.random() < .1) for _ in range(6)] for _ in range(n)]
    kn = [[rng.random() < .8 for _ in range(6)] for _ in range(n)]
    chars = [float(np.expm1(10 * np.float32(z[1]))) for z in xc]
    s2, prev = R.observed_run(mode, chars); ee = R.independent_E(ev, kn)
    xin = [list(map(float, c)) + s + e for c, s, e in zip(xc, s2, ee)]
    nm = [m if m != 7 else -1 for m in mode]; nev = [[int(v) for v in r] for r in ev]; nk = [list(r) for r in kn]
    for j in closed:
        nm[j] = -1; nk[j] = [False] * 6
    rm = [rng.randrange(7) if rng.random() < .9 else -1 for _ in range(n)]
    for j in range(1, n):
        if rng.random() < .7:
            rm[j] = rm[j - 1]
    sysr = dict(mode=rm, event=[[int(rng.random() < .2) for _ in range(4)] for _ in range(n)], event_known=[[rng.random() < .7 for _ in range(4)] for _ in range(n)])
    return dict(namespace_id=NS, native_labels={NS: dict(mode=nm, event=nev, event_known=nk), 'syn_sysR': sysr},
                inputs=dict(xin=xin, mode=mode, prev_mode=prev, text_sha256=None))


def make_env(tmp, closed_unit=None, closed=()):
    pk = tmp / 'pk'; synthetic.build(pk, n_train=10, n_dev=4, seed=7, max_len=80, emb_rows=60, native=True)
    units = json.loads((pk / 'units.json').read_text()); z = dict(np.load(pk / 'arrays.npz'))
    crng = np.random.default_rng(2)
    for u in units:                                                         # realistic C3: position, own chars, cumulative chars
        n = u['n']; ch = crng.integers(5, 400, n); o = u['offset']
        z['xin'][o:o + n, 0] = np.log1p(np.arange(1, n + 1)) / 10; z['xin'][o:o + n, 1] = np.log1p(ch) / 10; z['xin'][o:o + n, 2] = np.log1p(np.cumsum(ch)) / 10
    np.savez(pk / 'arrays.npz', **z)
    ud = tmp / 'units_src'; (ud / 'units').mkdir(parents=True); rng = random.Random(1); recs = []; roster = []
    for i, u in enumerate(units):
        u['index_v2_id'] = 'idx-%d' % i; u['text_sha256'] = [h('%d-%d' % (i, j)) for j in range(u['n'])]
        d = producer_unit(z['xin'][u['offset']:u['offset'] + u['n'], :3], rng, u['n'], closed if i == closed_unit else ())
        d['inputs']['text_sha256'] = u['text_sha256']; d['termination_receipt'] = dict(end_status=u['end_status'])
        raw = gzip.compress(json.dumps(d).encode(), mtime=0); p = ud / 'units' / ('u%d.json.gz' % i); p.write_bytes(raw)
        recs.append(dict(physical_id=u['physical_id'], index_v2_id=u['index_v2_id'], path='units/' + p.name, sha256=hashlib.sha256(raw).hexdigest()))
        roster.append(dict(index_v2_id=u['index_v2_id'], physical_id=u['physical_id'], role=u['role']))
    roster.append(dict(index_v2_id='idx-test', physical_id='p-test', role='TEST'))
    (ud / 'UNIT_MANIFEST_V1.json').write_text(json.dumps(recs))
    rp = tmp / 'roster.jsonl'; rp.write_text(''.join(json.dumps(r) + '\n' for r in roster))
    (pk / 'units.json').write_text(json.dumps(units))
    man = json.loads((pk / 'PACK_MANIFEST.json').read_text())
    man['files']['units.json'] = sha256_file(pk / 'units.json'); man['files']['arrays.npz'] = sha256_file(pk / 'arrays.npz')
    man.update(roster=dict(path=str(rp), sha256=sha256_file(rp)), unit_sources=[dict(dir=str(ud), manifest_sha256=sha256_file(ud / 'UNIT_MANIFEST_V1.json'))])
    (pk / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    dec = tmp / 'decision.json'; dec.write_text(json.dumps(dict(input_contract=dict(mode='retrospective_annotation'))))
    return pk, ud, rp, dec, units


def attach(pk, ud, rp, dec, out):
    R.main(['--pack', str(pk), '--out', str(out), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp), '--decision', str(dec)])


def test_only_es_inputs_change_and_pairing_exact(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    attach(pk, ud, rp, dec, tmp_path / 'retro')
    a = np.load(pk / 'arrays.npz'); b = np.load(tmp_path / 'retro' / 'arrays.npz')
    assert {k for k in a.files if not np.array_equal(a[k], b[k])} <= {'xin', 'mode', 'prev_mode', 'es_ok'}
    assert np.array_equal(a['xin'][:, :3], b['xin'][:, :3]) and b['es_ok'].all()
    m0 = json.loads((pk / 'PACK_MANIFEST.json').read_text()); m1 = json.loads((tmp_path / 'retro' / 'PACK_MANIFEST.json').read_text())
    assert all(m1['files'][f] == m0['files'][f] for f in m0['files'] if f != 'arrays.npz')
    assert m1['es_input_source']['kind'] == 'retrospective_annotation' == m1['es_admission']['kind']
    assert 'not an online detector' in m1['es_input_source']['interpretation']
    for i, u in enumerate(units):                                           # values are the producer's own inputs
        d = json.loads(gzip.open(ud / 'units' / ('u%d.json.gz' % i)).read())
        o, n = u['offset'], u['n']
        assert np.allclose(b['xin'][o:o + n, 3:], np.asarray(d['inputs']['xin'], 'f4')[:, 3:], atol=1e-6)
        assert list(b['mode'][o:o + n]) == d['inputs']['mode']
    PackedData(tmp_path / 'retro')


def test_closed_native_cells_close_inputs_and_recompute_run(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path, closed_unit=0, closed=(2, 3))
    attach(pk, ud, rp, dec, tmp_path / 'retro')
    b = np.load(tmp_path / 'retro' / 'arrays.npz'); u = units[0]; o = u['offset']
    assert b['mode'][o + 2] == R.UNKNOWN and b['mode'][o + 3] == R.UNKNOWN
    assert (b['xin'][o + 2:o + 4, 3:5] == 0).all() and (b['xin'][o + 2:o + 4, 11:17] == 0).all()
    if b['mode'][o + 4] != R.UNKNOWN:
        assert abs(b['xin'][o + 4, 3] - np.log1p(1) / 10) < 1e-6              # run restarts after the closed positions
    m1 = json.loads((tmp_path / 'retro' / 'PACK_MANIFEST.json').read_text())
    assert m1['es_attach_counts']['units_with_closures'] == 1


def test_recompute_is_prefix_invariant():
    rng = random.Random(3); n = 40
    ch = np.random.default_rng(0).integers(5, 400, n); xc = np.stack([np.log1p(np.arange(1, n + 1)) / 10, np.log1p(ch) / 10, np.log1p(np.cumsum(ch)) / 10], 1).astype('f4')
    d = producer_unit(xc, rng, n, closed=(5,))
    x1, m1, p1, _ = R.unit_inputs(d)
    d2 = json.loads(json.dumps(d)); j0 = 20
    for j in range(j0 + 1, n):                                              # change only future labels and inputs
        d2['inputs']['mode'][j] = (d2['inputs']['mode'][j] + 1) % 6; d2['inputs']['xin'][j][5:17] = [1.0] * 12
        d2['native_labels'][NS]['mode'][j] = -1
    x2, m2, p2, _ = R.unit_inputs(d2)
    assert np.allclose(np.asarray(x1)[:j0 + 1], np.asarray(x2)[:j0 + 1]) and m1[:j0 + 1] == m2[:j0 + 1] and p1[:j0 + 1] == p2[:j0 + 1]


def test_refusals(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    bad = tmp_path / 'bad.json'; bad.write_text(json.dumps(dict(decision='CT_FIRST')))
    with pytest.raises(ValueError, match='DECISION_DOES_NOT_ADMIT'):
        attach(pk, ud, rp, bad, tmp_path / 'o1')
    p = ud / 'units' / 'u1.json.gz'; d = json.loads(gzip.open(p).read()); d['inputs']['text_sha256'][0] = h('other')
    raw = gzip.compress(json.dumps(d).encode(), mtime=0); p.write_bytes(raw)
    with pytest.raises(ValueError, match='UNIT_SHA'):
        attach(pk, ud, rp, dec, tmp_path / 'o2')
    recs = json.loads((ud / 'UNIT_MANIFEST_V1.json').read_text()); recs[1]['sha256'] = hashlib.sha256(raw).hexdigest()
    (ud / 'UNIT_MANIFEST_V1.json').write_text(json.dumps(recs))
    man = json.loads((pk / 'PACK_MANIFEST.json').read_text()); man['unit_sources'][0]['manifest_sha256'] = sha256_file(ud / 'UNIT_MANIFEST_V1.json')
    (pk / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(ValueError, match='PAIRING_TEXT_SEQUENCE'):
        attach(pk, ud, rp, dec, tmp_path / 'o3')


def test_ct_identical_ctes_sees_retro_inputs(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    attach(pk, ud, rp, dec, tmp_path / 'retro')
    d0 = PackedData(pk); d1 = PackedData(tmp_path / 'retro'); ids = d0.ids('TRAIN')[:4]
    b0 = d0.make_batch(ids, 'cpu'); b1 = d1.make_batch(ids, 'cpu')
    torch.manual_seed(0); ct = model_a.build('CT', d0.nsolvers, SMALL, namespaces=d0.namespaces).eval()
    torch.manual_seed(0); ce = model_a.build('CTES', d0.nsolvers, SMALL, namespaces=d0.namespaces).eval()
    with torch.no_grad():
        assert torch.allclose(ct(b0)['ns'][0]['gate'], ct(b1)['ns'][0]['gate'])
        assert not torch.allclose(ce(b0)['ns'][0]['gate'], ce(b1)['ns'][0]['gate'])


def _prod(pack, tmp, name, cell='CTES'):
    cfg = dict(arch='A', cell=cell, model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1., warmup_updates=2,
               cosine_floor=.1), effective_batch=8, micro_batch=4, passes=1, task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3)
    cp = tmp / (name + '.json'); cp.write_text(json.dumps(cfg))
    pr = tmp / (name + '_pass.json')
    pr.write_text(json.dumps(dict(verdict='PASS', code_hashes=runner.code_hashes(), pack_manifest_sha256=[sha256_file(pack / 'PACK_MANIFEST.json')],
                                  config_sha256=[sha256_file(cp)])))
    return runner.main(['--config', str(cp), '--pack', str(pack), '--out', str(tmp / ('o_' + name)), '--seed', '1', '--device', 'cpu',
                        '--max-seconds', '100000', '--purpose', 'production', '--max-updates', '1', '--pass-record', str(pr)])


def test_runner_gate_requires_explicit_matching_kind(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    rt = tmp_path / 'retro'; attach(pk, ud, rp, dec, rt)
    for p in (pk, rt):
        m = json.loads((p / 'PACK_MANIFEST.json').read_text()); m.update(synthetic=False, production_admissible=True)
        (p / 'PACK_MANIFEST.json').write_text(json.dumps(m))
    assert _prod(rt, tmp_path, 'ok') == runner.EXIT_RESUMABLE                # explicit retro kind + accepted decision
    for name, mut in (('nokind', lambda m: m['es_input_source'].pop('kind')),
                      ('mismatch', lambda m: m['es_admission'].update(kind='causal_detector')),
                      ('notaccepted', lambda m: m['es_admission'].update(accepted=False)),
                      ('changeddecision', lambda m: m['es_admission'].update(acceptance_sha256='0' * 64))):
        q = tmp_path / ('pk_' + name); shutil.copytree(rt, q)
        m = json.loads((q / 'PACK_MANIFEST.json').read_text()); mut(m); (q / 'PACK_MANIFEST.json').write_text(json.dumps(m))
        with pytest.raises(ValueError, match='CTES_PRODUCTION_REQUIRES_ACCEPTED_ES_ADMISSION'):
            _prod(q, tmp_path, name)
    with pytest.raises(ValueError, match='CTES_PRODUCTION_REQUIRES_ACCEPTED_ES_ADMISSION'):
        _prod(pk, tmp_path, 'ctpack')                                      # CT pack has no E/S source
