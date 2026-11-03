"""G4 injected annotation errors (ptrain/attach_group4.py), CTES_RETRO only: plan rules, pack delta, loss, runner, predict."""
import collections, json, random, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import attach_group4 as G4, attach_retro_es as R, runner, losses as L, predict  # noqa: E402
from ptrain.data import sha256_file  # noqa: E402
from test_retro_es import make_env, attach, SMALL  # noqa: E402


def stream(n, seed):
    r = random.Random(seed); mode = []; m = 0
    for j in range(n):
        if r.random() < .12:
            m = r.randrange(6)
        mode.append(7 if r.random() < .04 else m)
    ev = [[float(r.random() < .1) for _ in range(6)] for _ in range(n)]
    kn = [[r.random() < .7 for _ in range(6)] for _ in range(n)]
    return mode, ev, kn


Q = dict(TYPE=0.15, BOUNDARY=0.4, EVENT=0.02)


def full_inputs(mode, ev, kn, key, events_ok, q=Q, ss=None):
    """Plan + every dependent input of one stream, as attach_group4 writes them (chars fixed per position)."""
    m2, e2, kind, ops, cap = G4.edit_plan(mode, ev, kn, key, events_ok, q, ss)
    chars = [10 + (7 * j) % 50 for j in range(len(mode))]
    s2, prev = R.observed_run(m2, chars); ee = R.independent_E(e2, kn)
    return m2, e2, kind, ops, s2, prev, ee


@pytest.mark.parametrize('seed', range(12))
def test_edit_plan_rules(seed):
    mode, ev, kn = stream(400, seed)
    ss = [j == 0 or random.Random(seed * 7 + j).random() < .03 for j in range(400)]
    m2, e2, kind, ops, cap = G4.edit_plan(mode, ev, kn, f'k{seed}', True, Q, ss)
    changed = [m2[j] != mode[j] or e2[j] != ev[j] for j in range(400)]
    assert changed == [k > 0 for k in kind]                                            # no identity edit, kind = change
    K = 0; c = 0
    for j in range(400):                                                               # causal quota at every position
        K += int(mode[j] != 7 or any(kn[j])); c += int(changed[j])
        assert c <= int(0.05 * K)
    for op in ops:
        if op['type'] == 'TYPE':
            assert op['new'] != op['old'] and all(mode[j] == op['old'] for j in op['positions'])
        if op['type'] == 'BOUNDARY':
            b = op['boundary']
            assert 1 <= op['shift'] <= 3 and op['positions'][0] == b and len(op['positions']) <= op['shift']
            assert all(m2[j] == mode[b - 1] and mode[j] == mode[b] for j in op['positions'])   # past type, inside right span
        if op['type'] == 'EVENT':
            assert kn[op['position']][op['event']]
    assert all(m2[j] == 7 for j in range(400) if mode[j] == 7)
    assert G4.edit_plan(mode, ev, kn, f'k{seed}', True, Q, ss)[:3] == (m2, e2, kind)
    assert not any(o['type'] == 'EVENT' for o in G4.edit_plan(mode, ev, kn, f'k{seed}', False, Q, ss)[3])


@pytest.mark.parametrize('seed', range(40))
def test_ptv6_001_suffix_rebuild_noninterference(seed):
    """PTV6-001 (reviewer): same prefix, any other suffix -> plan, labels and all dependent inputs on the prefix identical.
    Rebuilds both streams from scratch (planner and features), suffix with the same length and known masks."""
    r = random.Random(1000 + seed)
    n = r.randint(30, 300); mode, ev, kn = stream(n, seed)
    for t in sorted(r.sample(range(n - 1), min(6, n - 1))):
        fm = list(mode); fe = [list(x) for x in ev]
        for j in range(t + 1, n):
            if fm[j] != 7:
                fm[j] = r.randrange(6)
            fe[j] = [float(r.random() < .2) for _ in range(6)]
        a = full_inputs(mode, ev, kn, f's{seed}', True); b = full_inputs(fm, fe, kn, f's{seed}', True)
        for k, (x, y) in enumerate(zip(a, b)):
            if k == 3:
                x = [o for o in x if o.get('start', o.get('boundary', o.get('position'))) <= t]
                y = [o for o in y if o.get('start', o.get('boundary', o.get('position'))) <= t]
                x = [{kk: (vv[:sum(1 for p in vv if p <= t)] if kk == 'positions' else vv) for kk, vv in o.items()
                      if kk not in ('end', 'truncated_by_quota', 'absorbed_right_span')} for o in x]
                y = [{kk: (vv[:sum(1 for p in vv if p <= t)] if kk == 'positions' else vv) for kk, vv in o.items()
                      if kk not in ('end', 'truncated_by_quota', 'absorbed_right_span')} for o in y]
                assert x == y, (seed, t)
            else:
                assert x[:t + 1] == y[:t + 1], (seed, t, k)


def test_reviewer_examples_v7():
    """V7 counterexamples: (a) A=[0]*20+[1]*20 vs B=[0]*20+[1]*10+[2]*10 share inputs on 0..29 for every key;
    (b) a right span shorter than the shift is absorbed and recorded, never silently."""
    for key in range(200):
        ev = [[0.0] * 6 for _ in range(40)]; kn = [[False] * 6 for _ in range(40)]
        A = full_inputs([0] * 20 + [1] * 20, ev, kn, f'v{key}', False); B = full_inputs([0] * 20 + [1] * 10 + [2] * 10, ev, kn, f'v{key}', False)
        assert A[0][:30] == B[0][:30] and A[4][:30] == B[4][:30] and A[5][:30] == B[5][:30]
    seen = 0
    for key in range(400):
        mode = [0] * 20 + [1] + [2] * 19
        m2, e2, kind, ops, _ = G4.edit_plan(mode, [[0.0] * 6] * 40, [[False] * 6] * 40, f'w{key}', False, dict(TYPE=0, BOUNDARY=1.0, EVENT=0))
        for op in ops:
            if op['type'] == 'BOUNDARY' and op['boundary'] == 20 and op['shift'] > 1 and op['positions'] == [20]:
                assert op.get('absorbed_right_span') is True; seen += 1
    assert seen > 0


def test_edit_plan_zero_quota_start():
    mode, ev, kn = stream(15, 1)
    m2, e2, kind, ops, cap = G4.edit_plan(mode, ev, kn, 'z', True, dict(TYPE=1.0, BOUNDARY=1.0, EVENT=1.0))
    assert cap == 0 and not ops and m2 == mode                                     # fewer than 20 known positions: quota 0


def test_calibrate_targets_rate():
    streams = [(stream(300, s)[0], stream(300, s)[2], True, None) for s in range(20)]
    q, st = G4.calibrate(streams)
    assert 0 < q['TYPE'] < 1 and 0 < q['BOUNDARY'] <= 1 and 0 < q['EVENT'] < 1 and st['known'] > 0


@pytest.fixture
def env(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    rt = tmp_path / 'retro'; attach(pk, ud, rp, dec, rt)
    d4 = tmp_path / 'g4dec.json'; d4.write_text(json.dumps(dict(G4=dict(status='IMPLEMENT_NOW_RETRO_CTES_ONLY'))))
    with pytest.raises(ValueError, match='G4_REQUIRES_RETROSPECTIVE'):
        G4.main(['--pack', str(pk), '--out', str(tmp_path / 'x'), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp), '--decision', str(d4)])
    g = tmp_path / 'g4'
    G4.main(['--pack', str(rt), '--out', str(g), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp), '--decision', str(d4)])
    return tmp_path, rt, g, units


def test_pack_delta(env):
    tmp, rt, g, units = env
    a = np.load(rt / 'arrays.npz'); b = np.load(g / 'arrays.npz')
    for k in a.files:
        if k not in ('xin', 'mode', 'prev_mode', 'w_pos'):
            assert np.array_equal(a[k], b[k]), k
    plan = [json.loads(l) for l in open(g / 'G4_EDIT_PLAN.jsonl')]
    per = collections.Counter((p['role'], p['affected']) for p in plan)
    for role in ('TRAIN', 'DEV'):
        n = per[(role, True)] + per[(role, False)]
        assert per[(role, True)] == n // 2                                   # one primary namespace in the fixture
    for p, u in zip(plan, units):
        o, n = u['offset'], u['n']; yb = b['g4b'][o:o + n]
        x = b['xin'][o:o + n] if u['role'] == 'TRAIN' else b['diag_xin'][o:o + n]
        m = b['mode'][o:o + n] if u['role'] == 'TRAIN' else b['diag_mode'][o:o + n]
        if u['role'] == 'DEV':
            assert np.array_equal(b['xin'][o:o + n], a['xin'][o:o + n])      # natural DEV kept for selection
        changed = (m != a['mode'][o:o + n]) | (np.abs(x[:, 5:11] - a['xin'][o:o + n, 5:11]) > .5).any(-1)
        assert np.array_equal(changed, yb == 1)                              # positive iff the observed label changed
        if not (p['affected'] and p.get('ops')):
            assert not changed.any() and (yb != 1).all()                     # controls unchanged, scored clean
        known = (a['mode'][o:o + n] != R.UNKNOWN) | (a['xin'][o:o + n, 11:17] > .5).any(-1)
        assert np.array_equal(yb >= 0, known)                                # unknown positions are not certified clean
        ev = [list(r[5:11]) for r in x]; kn = [[bool(v > .5) for v in r[11:17]] for r in x]
        assert np.allclose(x[:, 5:41], np.array(R.independent_E(ev, kn)), atol=1e-6)
        assert np.array_equal(kn, a['xin'][o:o + n, 11:17] > .5)             # known masks never altered


def _cfg(tmp, name, cell, **kw):
    cfg = dict(arch='A', cell=cell, model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1., warmup_updates=2,
               cosine_floor=.1), effective_batch=8, micro_batch=4, passes=1, task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3, **kw)
    p = tmp / (name + '.json'); p.write_text(json.dumps(cfg)); return p


def test_runner_predict_guards(env):
    tmp, rt, g, units = env
    for name, cell, kw, err in (('ct', 'CT', dict(group4=True), 'G4_ONLY_WITH_RETRO'), ('plain', 'CTES', {}, 'G4_PACK_HAS_CORRUPTED'),
                                ('nopack', 'CTES', dict(group4=True), None)):
        pack = rt if name == 'nopack' else g
        with pytest.raises(ValueError, match=err or 'G4_CONFIG_REQUIRES'):
            runner.main(['--config', str(_cfg(tmp, name, cell, **kw)), '--pack', str(pack), '--out', str(tmp / ('o' + name)), '--seed', '1',
                         '--device', 'cpu', '--max-seconds', '1000', '--purpose', 'test'])
    o = tmp / 'run'
    assert runner.main(['--config', str(_cfg(tmp, 'ok', 'CTES', group4=True)), '--pack', str(g), '--out', str(o), '--seed', '1', '--device', 'cpu',
                        '--max-seconds', '100000', '--purpose', 'test']) == 0
    dev = [json.loads(l) for l in open(o / 'metrics.jsonl')][-1]['dev']
    assert dev['g4_changed_cells'] > 0 and 'g4_kind' in dev
    tl = [json.loads(l) for l in open(o / 'train_log.jsonl')]
    assert any('loss_g4_changed' in t for t in tl)
    predict.main(['--run', str(o), '--device', 'cpu'])
    M = json.loads((o / 'predictions_dev' / 'METRICS.json').read_text())
    a = M['g4_diag_dev']['all']
    assert a['changed'] > 0 and 'control_false_positive_rate_p05' in a and set(a['by_edit_type']) == {'TYPE', 'BOUNDARY', 'EVENT'}


def test_g4_loss_masks_unscored():
    b = dict(score=torch.ones(1, 4, dtype=torch.bool), g4b=torch.tensor([[-1, 0, 1, 0]]), g4k=torch.tensor([[-1, 0, 2, 0]]))
    o = dict(g4_changed=torch.zeros(1, 4, requires_grad=True), g4_kind=torch.zeros(1, 4, 4, requires_grad=True))
    bce, ce, Ib = L.g4_terms(o, b)
    assert Ib.tolist() == [[False, True, True, True]]
    (bce * Ib).sum().backward()
    assert o['g4_changed'].grad[0, 0] == 0                                 # unknown position gets no gradient


def test_ptv6_002_average_precision_ties():
    from ptrain.predict import average_precision
    import itertools
    y = [1, 0, 0, 0]; p = [0.5] * 4
    assert all(abs(average_precision([y[i] for i in o], p) - 0.25) < 1e-12 for o in itertools.permutations(range(4)))
    y = [1, 0, 1, 0, 1]; p = [0.9, 0.9, 0.5, 0.5, 0.1]
    vals = {round(average_precision([y[i] for i in o], [p[i] for i in o]), 12) for o in itertools.permutations(range(5))}
    assert len(vals) == 1
    # distinct scores: classic AP
    assert abs(average_precision([1, 0, 1], [0.9, 0.8, 0.7]) - (1 / 2 * 1 + 1 / 2 * 2 / 3)) < 1e-12
    assert average_precision([0, 0], [0.1, 0.2]) is None


@pytest.mark.parametrize('seed', range(30))
def test_ptv6_001_suffix_length_and_known_changes(seed):
    """Decision V2: suffix may differ in length and in known masks; the prefix plan and all prefix inputs stay identical."""
    r = random.Random(5000 + seed)
    n = r.randint(40, 250); mode, ev, kn = stream(n, seed)
    t = r.randrange(5, n - 5)
    for variant in ('shorter', 'longer', 'known'):
        if variant == 'shorter':
            m2_, e2_, k2_ = mode[:t + 1 + r.randrange(0, 4)], ev[:t + 1 + 0], kn[:t + 1]
            L = len(m2_); e2_ = (ev + ev)[:L]; k2_ = (kn + kn)[:L]
        elif variant == 'longer':
            extra = stream(r.randint(1, 60), seed + 77)
            m2_, e2_, k2_ = mode + extra[0], ev + extra[1], kn + extra[2]
        else:
            m2_ = list(mode); e2_ = [list(x) for x in ev]; k2_ = [list(x) for x in kn]
            for j in range(t + 1, n):
                k2_[j] = [r.random() < .5 for _ in range(6)]
                m2_[j] = 7 if r.random() < .3 else r.randrange(6)
        a = full_inputs(mode, ev, kn, f'q{seed}', True)
        L = len(m2_); chars = [10 + (7 * j) % 50 for j in range(L)]
        bm, be, bk, bops, _ = G4.edit_plan(m2_, e2_, k2_, f'q{seed}', True, Q)
        bs2, bprev = R.observed_run(bm, chars); bee = R.independent_E(be, k2_)
        assert a[0][:t + 1] == bm[:t + 1] and a[1][:t + 1] == be[:t + 1] and a[2][:t + 1] == bk[:t + 1], (seed, variant)
        assert a[4][:t + 1] == bs2[:t + 1] and a[5][:t + 1] == bprev[:t + 1] and a[6][:t + 1] == bee[:t + 1], (seed, variant)


def test_quota_stop_is_final_and_recorded():
    """A quota stop ends the active edit for good (no late resumption) and is recorded."""
    mode = [0] * 40 + [1] * 60; ev = [[0.0] * 6] * 100; kn = [[False] * 6] * 100
    for key in range(300):
        m2, e2, kind, ops, _ = G4.edit_plan(mode, ev, kn, f'r{key}', False, dict(TYPE=1.0, BOUNDARY=0.0, EVENT=0.0))
        for op in ops:
            if op.get('truncated_by_quota'):
                last = op['positions'][-1]
                assert all(kind[j] != G4.TYPE or j in op['positions'] for j in range(op['start'], len(mode)) if mode[j] == op['old'])
                assert op['positions'] == list(range(op['start'], last + 1))
                return
    raise AssertionError('no quota stop observed')
