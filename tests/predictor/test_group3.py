"""G3 masked annotation reconstruction (ptrain/attach_group3.py): frozen plan, input masking, targets, loss, runner, predict."""
import json, random, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import attach_group3 as G3, attach_retro_es as R, runner, losses as L, predict  # noqa: E402
from ptrain.data import PackedData, sha256_file  # noqa: E402
from test_retro_es import make_env, attach, SMALL  # noqa: E402


def test_mask_plan_rules():
    el = [True] * 10 + [False] + [True] * 30 + [False] * 3 + [True] * 7
    m1, q = G3.mask_plan(el, G3.plan_rng(1, 'G3', 'u', 'mask'))
    assert q == int(0.15 * sum(el)) and sum(m1) == q                 # exact quota, clipped last block
    assert all(el[j] for j in range(len(el)) if m1[j])                # only eligible positions
    m2, _ = G3.mask_plan(el, G3.plan_rng(1, 'G3', 'u', 'mask'))
    assert m1 == m2                                                    # deterministic from (seed, group, uid, op)
    m3, _ = G3.mask_plan(el, G3.plan_rng(2, 'G3', 'u', 'mask'))
    assert m3 != m1
    m0, q0 = G3.mask_plan([True] * 6, G3.plan_rng(1, 'G3', 'v', 'mask'))
    assert q0 == 0 and not any(m0)                                     # quota 0: no forced mask
    # blocks never cross a non-eligible position: every masked run lies inside one eligible run
    runs = []; j = 0
    while j < len(el):
        if m1[j]:
            s = j
            while j < len(el) and m1[j]:
                j += 1
            runs.append((s, j))
        j += 1
    assert all(all(el[s:e]) for s, e in runs)


@pytest.fixture
def env(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    rt = tmp_path / 'retro'; attach(pk, ud, rp, dec, rt)
    d3 = tmp_path / 'g3dec.json'; d3.write_text(json.dumps(dict(G3=dict(status='IMPLEMENT_NOW'))))
    out = {}
    for name, src in (('ct', pk), ('retro', rt)):
        out[name] = tmp_path / ('g3_' + name)
        G3.main(['--pack', str(src), '--out', str(out[name]), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp), '--decision', str(d3)])
    return tmp_path, pk, rt, out, units, ud


def test_plan_targets_and_inputs(env):
    tmp, pk, rt, out, units, ud = env
    plans = {n: [json.loads(l) for l in open(out[n] / 'G3_MASK_PLAN.jsonl')] for n in out}
    assert plans['ct'] == plans['retro'] and any(p['masked'] for p in plans['ct'])    # same plan for CT and CTES
    a_ct = np.load(pk / 'arrays.npz'); g_ct = np.load(out['ct'] / 'arrays.npz')
    for k in ('xin', 'mode', 'prev_mode'):
        assert np.array_equal(a_ct[k], g_ct[k]) and np.array_equal(a_ct[k], g_ct['diag_' + k])   # CT inputs untouched
    a = np.load(rt / 'arrays.npz'); g = np.load(out['retro'] / 'arrays.npz')
    for k in a.files:
        if k not in ('xin', 'mode', 'prev_mode', 'w_pos'):
            assert np.array_equal(a[k], g[k]), k                       # next-step targets etc. unchanged
    for p, u in zip(plans['retro'], units):
        o, n = u['offset'], u['n']; mk = np.zeros(n, bool); mk[p['masked']] = True
        main_x, main_m = g['xin'][o:o + n], g['mode'][o:o + n]
        dx, dm = g['diag_xin'][o:o + n], g['diag_mode'][o:o + n]
        if u['role'] == 'TRAIN':
            x, m = main_x, main_m
            assert np.array_equal(dx, a['xin'][o:o + n])               # TRAIN diag = clean (unused)
        else:
            x, m = dx, dm
            assert np.array_equal(main_x, a['xin'][o:o + n]) and np.array_equal(main_m, a['mode'][o:o + n])  # natural DEV kept
        if mk.any():
            assert (m[mk] == R.UNKNOWN).all() and (x[mk][:, 3:17] == 0).all()           # nothing of the label survives
            f = int(np.argmax(mk))
            assert np.allclose(x[:f], a['xin'][o:o + f])                                 # prefix before the first mask unchanged
            ev = [list(r[5:11]) for r in x]; kn = [[bool(v > .5) for v in r[11:17]] for r in x]
            assert np.allclose(x[:, 5:41], np.array(R.independent_E(ev, kn)), atol=1e-6)  # counters from the masked stream only
        d = json.loads(__import__('gzip').open(ud / 'units' / ('u%d.json.gz' % units.index(u))).read())
        lab = d['native_labels']['syn_flash']
        man = json.loads((out['retro'] / 'PACK_MANIFEST.json').read_text()); i = [ns['id'] for ns in man['namespaces']].index('syn_flash')
        tm = g[f'ns{i}_g3m'][o:o + n]
        assert (tm[~mk] == -1).all()
        assert all(tm[j] == lab['mode'][j] for j in p['masked'])           # target = native label of the same segment j


def _cfg(tmp, name, cell, **kw):
    cfg = dict(arch='A', cell=cell, model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1., warmup_updates=2,
               cosine_floor=.1), effective_batch=8, micro_batch=4, passes=1, task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3, **kw)
    p = tmp / (name + '.json'); p.write_text(json.dumps(cfg)); return p


def test_runner_predict_and_guards(env):
    tmp, pk, rt, out, units, ud = env
    with pytest.raises(ValueError, match='G3_CONFIG_REQUIRES'):
        runner.main(['--config', str(_cfg(tmp, 'a', 'CTES', group3=True)), '--pack', str(rt), '--out', str(tmp / 'oa'), '--seed', '1', '--device', 'cpu',
                     '--max-seconds', '1000', '--purpose', 'test'])
    with pytest.raises(ValueError, match='G3_PACK_HAS_MASKED_TRAIN_INPUTS'):
        runner.main(['--config', str(_cfg(tmp, 'b', 'CTES')), '--pack', str(out['retro']), '--out', str(tmp / 'ob'), '--seed', '1', '--device', 'cpu',
                     '--max-seconds', '1000', '--purpose', 'test'])
    for name, cell in (('retro', 'CTES'), ('ct', 'CT')):
        o = tmp / ('run_' + name)
        assert runner.main(['--config', str(_cfg(tmp, 'c' + name, cell, group3=True)), '--pack', str(out[name]), '--out', str(o), '--seed', '1',
                            '--device', 'cpu', '--max-seconds', '100000', '--purpose', 'test']) == 0
        dev = [json.loads(l) for l in open(o / 'metrics.jsonl')][-1]['dev']
        assert 'ns0_g3_type' in dev and dev['ns0_g3_type_cells'] > 0
        predict.main(['--run', str(o), '--device', 'cpu'])
        M = json.loads((o / 'predictions_dev' / 'METRICS.json').read_text())
        assert 'g3_diag_dev' in M['namespaces']['syn_flash'] and 'type' in M['namespaces']['syn_flash']['g3_diag_dev']['all']


def test_selection_ignores_g3_and_natural_dev(env):
    """Selection = next-step objective on natural DEV inputs: the same weights give the same selection score on the
    retro pack and on its G3 pack (DEV inputs and next-step targets identical; TRAIN masking and G3 heads do not enter)."""
    tmp, pk, rt, out, units, ud = env
    d0 = PackedData(rt); d1 = PackedData(out['retro'])
    ids = d0.ids('DEV')[:6]
    b0 = d0.make_batch(ids, 'cpu'); b1 = d1.make_batch(ids, 'cpu')
    from ptrain import model_a
    torch.manual_seed(0); m = model_a.build('CTES', d0.nsolvers, SMALL, namespaces=d1.namespaces, group3=True).eval()
    with torch.no_grad():
        o0 = m(b0); o1 = m(b1)
        acc = [ns['status'] == 'accepted' for ns in d1.namespaces]
        l0 = L.native_position_terms(o0, b0, acc, group1=False)[0]; l1 = L.native_position_terms(o1, b1, acc, group1=False, group3=False)[0]
    assert torch.allclose(l0, l1)
    d1.use_inputs('diag'); b2 = d1.make_batch(ids, 'cpu'); d1.use_inputs(None)
    assert not torch.equal(b2['xin'], b1['xin']) or not torch.equal(b2['mode'], b1['mode'])
