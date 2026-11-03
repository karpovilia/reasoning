"""Native annotation namespace heads and the V2 reduction (COORDINATOR_NATIVE_TARGET_NAMESPACES_DECISION_V2)."""
import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import model_a, losses as L, synthetic, runner, targets_native  # noqa: E402
from ptrain.data import PackedData, hierarchical_weights  # noqa: E402

SMALL = dict(width=64, heads=4, layers=2, ff_width=128, dropout=0.0, max_segments=256)


@pytest.fixture(scope='module')
def npack(tmp_path_factory):
    d = tmp_path_factory.mktemp('npack')
    synthetic.build(d, n_train=30, n_dev=9, seed=3, max_len=250, emb_rows=300, native=True)
    return d


def test_native_targets_no_bridge_and_end():
    mode = [0, 0, -1, 1, 1]
    ev = [[0, 0], [1, 0], [0, 1], [0, 0], [1, 1]]; kn = [[True, True], [True, False], [True, True], [True, True], [True, True]]
    t = targets_native.derive(mode, ev, kn, 'natural')
    assert t['gate'] == [0, -1, -1, 0, 2] and t['vstep'] == [True, False, False, True, True]
    assert t['ev'][0] == [1, 0] and t['vev'][0] == [True, False]          # next-segment events with their own known mask
    assert t['vev'][4] == [False, False]                                  # nothing after the last segment
    t = targets_native.derive(mode, ev, kn, 'unknown')
    assert t['gate'][4] == -1                                             # window edge / unknown end is not END
    t = targets_native.derive([2, 2, 3], [[0]] * 3, [[True]] * 3, 'natural', span_start=[True, True, False])
    assert t['gate'][:2] == [1, 1] and t['dest'][0] == 2                  # explicit same-type boundary


def test_namespace_shapes_and_mixed_batch(npack):
    data = PackedData(npack)
    assert [(n['K'], n['E'], n['status']) for n in data.namespaces] == [(6, 6, 'accepted'), (7, 4, 'accepted'), (6, 6, 'candidate')]
    torch.manual_seed(0)
    m = model_a.build('CT', data.nsolvers, SMALL, namespaces=data.namespaces).eval()
    assert not hasattr(m, 'gate')
    b = data.make_batch(data.ids('TRAIN')[:6], 'cpu')
    o = m(b)
    acc = [n['status'] == 'accepted' for n in data.namespaces]
    Lp, elig, comps = L.native_position_terms(o, b, acc)
    # manual recomputation at positions with two available namespaces
    av = []; Ls = []
    for i, (t, out) in enumerate(zip(b['ns'], o['ns'])):
        c = comps[i]
        Ij = c['joint'][1]; Id = c['destination'][1]; Ie = c['event'][1]
        Ls.append(Ij * c['joint'][0] + .5 * Id * c['destination'][0] + .25 * Ie * c['event'][0]); av.append((Ij | Id | Ie) & acc[i])
    two = av[0] & av[1]
    assert two.any(), 'synthetic data must contain positions with two namespaces'
    assert torch.allclose(Lp[two], ((Ls[0] + Ls[1]) / 2)[two], atol=1e-6)
    one = av[0] & ~av[1]
    assert torch.allclose(Lp[one], Ls[0][one], atol=1e-6)
    assert not (elig & ~(av[0] | av[1])).any()                            # candidate namespace never makes a position eligible


def test_candidate_namespace_does_not_change_loss(npack):
    data = PackedData(npack)
    torch.manual_seed(0); m = model_a.build('CT', data.nsolvers, SMALL, namespaces=data.namespaces).eval()
    b = data.make_batch(data.ids('TRAIN')[:6], 'cpu'); acc = [True, True, False]
    l1, _ = L.native_total_loss(m(b), b, acc)
    b['ns'][2]['gate'][:] = 1; b['ns'][2]['vstep'][:] = True; b['ns'][2]['vev'][:] = True
    l2, _ = L.native_total_loss(m(b), b, acc)
    assert torch.allclose(l1, l2)


def test_position_weight_not_multiplied_by_versions(npack):
    data = PackedData(npack)
    w = data.a['w_pos']
    per_q = {}
    for u in data.units:
        per_q[u['question_component']] = per_q.get(u['question_component'], 0.) + float(w[u['offset']:u['offset'] + u['n']].sum())
    vals = [v for v in per_q.values() if v > 0]
    assert np.allclose(vals, 1.0, atol=1e-4)                              # each question sums to 1, independent of namespace count
    units = [dict(question_component='q', offset=0, n=2), dict(question_component='q', offset=2, n=3)]
    ww = hierarchical_weights(units, np.array([True, True, True, False, True]))
    assert np.allclose(ww, [.25, .25, .25, 0, .25])


def test_native_disabled_heads_and_parity(npack):
    data = PackedData(npack)
    torch.manual_seed(5); a = model_a.build('CT', data.nsolvers, SMALL, namespaces=data.namespaces)
    torch.manual_seed(5); c = model_a.build('CTES', data.nsolvers, SMALL, namespaces=data.namespaces)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), c.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    b = data.make_batch(data.ids('TRAIN')[:4], 'cpu')
    loss, _ = L.native_total_loss(a(b), b, [True, True, False]); loss.backward()
    assert all(p.grad is None for _, p in runner.disabled_head_params(a))
    assert a.ns_heads[2].weight.grad is None or float(a.ns_heads[2].weight.grad.abs().sum()) == 0.0   # candidate head untrained
    assert a.ns_heads[0].weight.grad is not None


def _run(pack, out, mu, budget=None):
    cfg = dict(arch='A', cell='CTES', model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
               warmup_updates=2, cosine_floor=.1), effective_batch=8, micro_batch=4, passes=2, task_weights=L.STAGE1_WEIGHTS,
               checkpoint_every_updates=3, log_every=1)
    if budget:
        cfg['fit_budget_seconds'] = budget
    cp = Path(out).parent / (Path(out).name + '.json'); cp.write_text(json.dumps(cfg))
    args = ['--config', str(cp), '--pack', str(pack), '--out', str(out), '--seed', '42', '--device', 'cpu', '--max-seconds', '100000', '--purpose', 'test']
    if mu:
        args += ['--max-updates', str(mu)]
    return runner.main(args)


def test_native_resume_equals_continuous_and_selection(npack, tmp_path):
    torch.use_deterministic_algorithms(True)
    assert _run(npack, tmp_path / 'c', None) == 0
    assert _run(npack, tmp_path / 'r', 5) == runner.EXIT_RESUMABLE
    assert _run(npack, tmp_path / 'r', None) == 0
    a = torch.load(tmp_path / 'c/checkpoint.pt', weights_only=False); b = torch.load(tmp_path / 'r/checkpoint.pt', weights_only=False)
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    ma = (tmp_path / 'c/metrics.jsonl').read_text(); mb = (tmp_path / 'r/metrics.jsonl').read_text()
    assert ma == mb
    rows = [json.loads(x) for x in ma.splitlines()]
    assert len(rows) == 2 and all('ns0_joint' in r['dev'] and r['dev']['questions'] > 0 for r in rows)
    best = min(rows, key=lambda r: (r['selection'], r['epoch']))
    done = json.loads((tmp_path / 'c/COMPLETE.json').read_text())
    assert done['best_epoch'] == best['epoch']
    att = sorted((tmp_path / 'r/attempts').glob('attempt_*.json'))
    assert len(att) == 2 and done['attempts'] == 1


def test_cumulative_budget_stops_and_is_not_reset(npack, tmp_path):
    out = tmp_path / 'bud'
    out.mkdir(); (out / 'attempts').mkdir()
    (out / 'attempts/attempt_0001.json').write_text(json.dumps(dict(elapsed_seconds=10_000)))
    assert _run(npack, out, None, budget=10_050) == runner.EXIT_BUDGET   # earlier attempts already used the budget


def test_predict_outputs_metrics(npack, tmp_path):
    from ptrain import predict
    assert _run(npack, tmp_path / 'p', None) == 0
    predict.main(['--run', str(tmp_path / 'p'), '--device', 'cpu'])
    m = json.loads((tmp_path / 'p/predictions_dev/METRICS.json').read_text())
    ns = m['namespaces']
    assert 'syn_flash' in ns and 'all' in ns['syn_flash']['groups']
    g = ns['syn_flash']['groups']['all']
    assert g['gate']['positions'] > 0 and 'gate_STAY' in g and g['questions'] > 0
    z = np.load(tmp_path / 'p/predictions_dev/predictions.npz')
    assert np.allclose(z['ns0_gate_p'].sum(1), 1, atol=1e-5)


def test_accumulation_equals_one_batch(npack):
    """SR-1: micro-batches with the common denominator give the same gradient as one effective batch."""
    data = PackedData(npack); acc = [True, True, False]
    ids = data.ids('TRAIN')[:8]
    torch.manual_seed(1); m = model_a.build('CT', data.nsolvers, SMALL, namespaces=data.namespaces).train()
    b = data.make_batch(ids, 'cpu'); l, _ = L.native_total_loss(m(b), b, acc); l.backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    parts = [data.make_batch(ids[:3], 'cpu'), data.make_batch(ids[3:], 'cpu')]
    den = sum(L.native_eligible_weight(x, acc) for x in parts)
    for x in parts:
        l, _ = L.native_total_loss(m(x), x, acc, denominator=den); l.backward()
    for n, p in m.named_parameters():
        if n in g1:
            assert torch.allclose(g1[n], p.grad, atol=1e-6, rtol=1e-4), n


def test_ctes_production_refused_without_causal_es(npack, tmp_path):
    """SR-3 / PREDICTOR_CTES_UNAPPROVED_COVERAGE_GATE_001: production CTES needs an accepted E/S admission record;
    full es_ok coverage alone does not admit it."""
    import shutil
    p2 = tmp_path / 'pk'; shutil.copytree(npack, p2)
    man = json.loads((p2 / 'PACK_MANIFEST.json').read_text()); man.update(synthetic=False, production_admissible=True)
    (p2 / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    z = dict(np.load(p2 / 'arrays.npz')); z['es_ok'][:] = True; np.savez(p2 / 'arrays.npz', **z)
    man['files']['arrays.npz'] = __import__('ptrain.data', fromlist=['x']).sha256_file(p2 / 'arrays.npz'); (p2 / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    cfg = dict(arch='A', cell='CTES', model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
               warmup_updates=2, cosine_floor=.1), effective_batch=8, micro_batch=4, passes=1, task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3)
    cp = tmp_path / 'c.json'; cp.write_text(json.dumps(cfg))
    cp2 = tmp_path / 'c2.json'; cfg2 = dict(cfg, cell='CT'); cp2.write_text(json.dumps(cfg2))
    from ptrain.data import sha256_file as _sha
    pr = tmp_path / 'pass.json'   # a valid PASS record, so that the CTES admission gate itself is what refuses
    pr.write_text(json.dumps(dict(verdict='PASS', code_hashes=runner.code_hashes(), pack_manifest_sha256=[_sha(p2 / 'PACK_MANIFEST.json')],
                                  config_sha256=[_sha(cp), _sha(cp2)])))
    with pytest.raises(ValueError, match='CTES_PRODUCTION_REQUIRES_ACCEPTED_ES_ADMISSION'):
        runner.main(['--config', str(cp), '--pack', str(p2), '--out', str(tmp_path / 'o'), '--seed', '1', '--device', 'cpu', '--max-seconds', '100',
                     '--purpose', 'production', '--pass-record', str(pr)])
    assert runner.main(['--config', str(cp2), '--pack', str(p2), '--out', str(tmp_path / 'o2'), '--seed', '1', '--device', 'cpu', '--max-seconds', '100000',
                        '--purpose', 'production', '--max-updates', '1', '--pass-record', str(pr)]) == runner.EXIT_RESUMABLE


def test_prior_floor_from_mirror(npack, tmp_path):
    """SR-4: with an empty pod directory the plato-mirrored busy time still counts."""
    cfg = dict(arch='A', cell='CT', model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
               warmup_updates=2, cosine_floor=.1), effective_batch=8, micro_batch=8, passes=1, task_weights=L.STAGE1_WEIGHTS,
               checkpoint_every_updates=3, fit_budget_seconds=1000)
    cp = tmp_path / 'c.json'; cp.write_text(json.dumps(cfg))
    assert runner.main(['--config', str(cp), '--pack', str(npack), '--out', str(tmp_path / 'o'), '--seed', '1', '--device', 'cpu',
                        '--max-seconds', '100000', '--purpose', 'test', '--prior-seconds-floor', '990']) == runner.EXIT_BUDGET


def test_positive_only_namespaces_flagged_not_rewritten(tmp_path):
    """SR-10 / PREDICTOR_POSITIVE_ONLY_EVENT_SUPERVISION_001: frozen table sets event_supervision_eligible=False for
    process_V6_unit (one of the four); native known masks are NOT rewritten; other namespaces keep events."""
    import gzip, hashlib
    from ptrain import build_pack
    udir = tmp_path / 'u'; (udir / 'units').mkdir(parents=True)
    roster = []; man = []
    for i, role in enumerate(['TRAIN', 'TRAIN', 'DEV']):
        n = 6
        x = dict(xin=[[0.0] * 41 for _ in range(n)], mode=[0] * n, prev_mode=[7] * n, solver=[1] * n,
                 text_sha256=['t%d' % j for j in range(n)], prompt_sha256=['p'] * n)
        t = dict(gate=[0] * n, dest=[-1] * n, event_y=[[0] * 6] * n, length_y=[0.0] * n, wait_y=[0.0] * n, wait_observed=[False] * n,
                 next_span=[-1] * n, rem=[[0, 0, 0]] * n,
                 target_valid=dict(step_outcome=[True] * n, next_type=[False] * n, length=[True] * n, events=[[True] * 6] * n,
                                   wait_observed=[False] * n, wait_censored=[False] * n, remainder=[[False] * 3] * n))
        nl = {'process_V6_unit': dict(mode=[0, 0, 1, 1, 1, 1], event=[[1, 0]] * n, event_known=[[True, False]] * n),
              'cloud_flash|x': dict(mode=[0, 0, 0, 1, 1, 1], event=[[j % 2, 0] for j in range(n)], event_known=[[True, True]] * n)}
        d = dict(question_component='q%d' % (i if role == 'TRAIN' else 9), inputs=x, targets=t, native_labels=nl,
                 termination_receipt=dict(end_status='natural'))
        raw = gzip.compress(json.dumps(d).encode()); (udir / f'units/u{i}.json.gz').write_bytes(raw)
        man.append(dict(unit_id='u%d' % i, physical_id='p%d' % i, path=f'units/u{i}.json.gz', sha256=hashlib.sha256(raw).hexdigest()))
        roster.append(dict(index_v2_id='p%d' % i, physical_id='p%d' % i, role=role, question_component=d['question_component'], solver='s', labelled=True))
    (udir / 'UNIT_MANIFEST_V1.json').write_text(json.dumps(man))
    (udir / 'NAMESPACE_TABLE.json').write_text(json.dumps([dict(namespace_id=k, status='accepted', mode_vocab=['a', 'b'], event_vocab=['e0', 'e1'])
                                                           for k in ('cloud_flash|x', 'process_V6_unit')]))
    rp = tmp_path / 'r.jsonl'; rp.write_text(''.join(json.dumps(r) + '\n' for r in roster))
    build_pack.main(['--roster', str(rp), '--roster-sha256', hashlib.sha256(rp.read_bytes()).hexdigest(), '--units', str(udir), '--out', str(tmp_path / 'pk')])
    m = json.loads((tmp_path / 'pk/PACK_MANIFEST.json').read_text())
    ns = {n['id']: n for n in m['namespaces']}
    assert ns['process_V6_unit']['event_supervision_eligible'] is False and ns['cloud_flash|x']['event_supervision_eligible'] is True
    assert m['namespace_policy']['sha256'].startswith('2c7c834d') and m['namespace_policy']['replaces']['sha256'].startswith('2e3ab65e')
    z = np.load(tmp_path / 'pk/arrays.npz'); i = [n['id'] for n in m['namespaces']].index('process_V6_unit')
    assert z[f'ns{i}_vev'][:, 0].any()                                     # native known masks kept
    assert ns['process_V6_unit']['event_axis_support_train'][0]['negative'] == 0


def test_event_loss_disabled_namespace_gets_no_event_gradient(npack):
    data = PackedData(npack); acc = [True, True, False]; ev_ok = [True, False, True]
    torch.manual_seed(0); m = model_a.build('CT', data.nsolvers, SMALL, namespaces=data.namespaces)
    b = data.make_batch(data.ids('TRAIN')[:6], 'cpu')
    Lp, elig, comps = L.native_position_terms(m(b), b, acc, ev_ok)
    assert not comps[1]['event'][1].any()                                 # event term exactly off for ns1
    assert torch.isfinite(Lp).all()
    loss, _ = L.native_total_loss(m(b), b, acc, event_eligible=ev_ok); loss.backward()
    K, E = m.ns_dims[1]
    g = m.ns_heads[1].weight.grad
    assert float(g[3 + K:].abs().sum()) == 0.0 and float(g[:3 + K].abs().sum()) > 0   # event rows untouched, gate/type trained
    assert float(m.ns_heads[0].weight.grad[3 + m.ns_dims[0][0]:].abs().sum()) > 0
