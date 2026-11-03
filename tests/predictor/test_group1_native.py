"""G2 (event -> first next span -> events inside it) on native namespaces: targets, pack delta, heads, loss, runner, predict."""
import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import targets_native as TN, attach_group1 as G, attach_retro_es as R, model_a, runner, losses as L, predict  # noqa: E402
from ptrain.data import PackedData, sha256_file  # noqa: E402
from test_retro_es import make_env, attach, SMALL  # noqa: E402


def test_derive_group1_rules():
    E1 = [[0, 0]]; K1 = [[True, True]]
    mode = [0, 0, 1, 1, 2]
    ev = [[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]]; kn = [[True, True]] * 5
    t = TN.derive_group1(mode, ev, kn, 'natural', K=3)
    assert t['g1n'] == [1, 1, 2, 2, 3]                                   # next span type; last span -> natural END (=K)
    assert t['g1e'][0] == [1.0, 0.0] and t['g1k'][0] == [True, True]     # whole next span observed: absence known
    assert t['g1k'][4] == [False, False]                                 # END: events masked
    t = TN.derive_group1(mode, ev, kn, 'confirmed_truncated', K=3)
    assert t['g1n'][4] == -1 and t['g1n'][2] == 2
    assert t['g1e'][2] == [0.0, 1.0] and t['g1k'][2] == [False, True]    # next span cut by the trace stop: only positives known
    t = TN.derive_group1([0, -1, 1, 1], [[0, 0]] * 4, [[True, True]] * 4, 'natural', K=3)
    assert t['g1n'][0] == -1 and t['g1n'][1] == -1                       # unknown label: no bridging
    t = TN.derive_group1([0, 0, 0], [[0, 0], [0, 0], [0, 0]], [[True, True]] * 3, 'natural', K=2, span_start=[True, False, True])
    assert t['g1n'][:2] == [0, 0]                                        # explicit same-type boundary
    t = TN.derive_group1([0, 0, 1, 1], [[0, 0], [1, 0], [0, 0], [0, 0]], [[True, True]] * 4, 'natural', K=2)
    assert t['g1n'][0] == 1                                              # an event inside a span never creates a boundary
    t2 = TN.derive_group1([0, 0, 1, 1], [[0, 0], [1, 0], [0, 0], [0, 1]], [[True, True]] * 4, 'natural', K=2)
    assert t2['g1n'][:2] == t['g1n'][:2] and t2['g1e'][0] == [0.0, 1.0]  # targets change only through the future span
    assert E1 and K1


def g2_env(tmp):
    pk, ud, rp, dec, units = make_env(tmp)
    man = json.loads((pk / 'PACK_MANIFEST.json').read_text())
    for ns in man['namespaces']:
        ns['event_supervision_eligible'] = True; ns['id'] = 'cloud_flash|prompt:test'
    man['namespaces'] = man['namespaces'][:1] if len(man['namespaces']) else man['namespaces']
    return pk, ud, rp, dec, units


@pytest.fixture
def env(tmp_path):
    pk, ud, rp, dec, units = make_env(tmp_path)
    g2 = tmp_path / 'g2'
    G.main(['--pack', str(pk), '--out', str(g2), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp)])
    return tmp_path, pk, ud, rp, dec, units, g2


def test_attach_group1_keeps_everything_else(env):
    tmp, pk, ud, rp, dec, units, g2 = env
    a = np.load(pk / 'arrays.npz'); b = np.load(g2 / 'arrays.npz')
    for k in a.files:
        if k != 'w_pos':
            assert np.array_equal(a[k], b[k]), k
    m = json.loads((g2 / 'PACK_MANIFEST.json').read_text())
    assert all(ns['group1'] for ns in m['namespaces']) and m['group1_targets']['group'] == 'G2_EVENT_SPAN_LINK'
    d = PackedData(g2); bt = d.make_batch(d.ids('TRAIN')[:3], 'cpu')
    assert 'g1n' in bt['ns'][0] and bt['ns'][0]['g1n'].dtype == torch.long
    # CTES_RETRO + G2 = retro attach on top of the G2 pack; next-step and G2 targets identical to CT + G2
    rg = tmp / 'retro_g2'; attach(g2, ud, rp, dec, rg)
    c = np.load(rg / 'arrays.npz')
    for k in b.files:
        if k not in ('xin', 'mode', 'prev_mode', 'es_ok'):
            assert np.array_equal(b[k], c[k]), k
    with pytest.raises(ValueError, match='PACK_ALREADY_HAS_GROUP1'):
        G.main(['--pack', str(g2), '--out', str(tmp / 'x'), '--units', str(ud), '--roster', str(rp), '--roster-sha256', sha256_file(rp)])


def test_span_event_loss_reads_only_true_next_type_and_masks(env):
    tmp, pk, ud, rp, dec, units, g2 = env
    d = PackedData(g2); b = d.make_batch(d.ids('TRAIN')[:4], 'cpu')
    torch.manual_seed(0); m = model_a.build('CTES', d.nsolvers, SMALL, group1=True, namespaces=d.namespaces)
    o = m(b); acc = [True] * len(d.namespaces)
    t = b['ns'][0]; K = d.namespaces[0]['K']
    se = o['ns'][0]['span_events']; se.retain_grad()
    loss, parts = L.native_total_loss(o, b, acc, event_eligible=[True] * len(acc), group1=True)
    loss.backward()
    g = se.grad                                                          # (N,T,K,E)
    wrong = torch.ones_like(g, dtype=torch.bool)
    tn = t['g1n']
    for i in range(tn.shape[0]):
        for p in range(tn.shape[1]):
            if 0 <= tn[i, p] < K:
                wrong[i, p, tn[i, p]] = False
    assert float(g[wrong].abs().max()) == 0.0                            # no gradient into other next-type rows
    assert 'ns0_next_span' in parts and 'ns0_span_events' in parts
    # positive-only namespace: span-event term exactly zero, next_span still trained
    o = m(b); loss2, parts2 = L.native_total_loss(o, b, acc, event_eligible=[False] * len(acc), group1=True)
    assert float(parts2['ns0_span_events']) == 0.0 and float(parts2['ns0_next_span']) > 0
    # group1=False reproduces the next-step loss exactly
    o = m(b)
    l_g1 = L.native_total_loss(o, b, acc, event_eligible=[True] * len(acc), group1=False)[0]
    pk1 = PackedData(pk); b1 = pk1.make_batch(pk1.ids('TRAIN')[:4], 'cpu')
    b1['weight'] = b['weight']                                           # same weights (w_pos of the G2 pack) for the comparison
    assert torch.allclose(l_g1, L.native_total_loss(o, b1, acc, event_eligible=[True] * len(acc))[0])


def test_marginal_event_probability_uses_no_true_type():
    out = dict(dest=torch.zeros(1, 2, 3), next_span=torch.randn(1, 2, 4), span_events=torch.randn(1, 2, 3, 2))
    p = L.native_group1_marginal_event_probs(out)
    ref = (torch.softmax(out['next_span'], -1)[..., :3, None] * torch.sigmoid(out['span_events'])).sum(-2)
    assert torch.allclose(p, ref) and p.shape == (1, 2, 2)


def _cfg(tmp, name, group1, cell='CT'):
    cfg = dict(arch='A', cell=cell, model=SMALL, group1=group1, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
               warmup_updates=2, cosine_floor=.1), effective_batch=8, micro_batch=4, passes=1, task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3)
    p = tmp / (name + '.json'); p.write_text(json.dumps(cfg)); return p


def test_runner_and_predict_g2(env):
    tmp, pk, ud, rp, dec, units, g2 = env
    with pytest.raises(ValueError, match='G2_CONFIG_REQUIRES_GROUP1_TARGETS_IN_PACK'):
        runner.main(['--config', str(_cfg(tmp, 'c0', True)), '--pack', str(pk), '--out', str(tmp / 'o0'), '--seed', '1', '--device', 'cpu',
                     '--max-seconds', '1000', '--purpose', 'test'])
    out = tmp / 'o1'
    rc = runner.main(['--config', str(_cfg(tmp, 'c1', True)), '--pack', str(g2), '--out', str(out), '--seed', '1', '--device', 'cpu',
                      '--max-seconds', '100000', '--purpose', 'test'])
    assert rc == 0 and (out / 'COMPLETE.json').exists()
    met = [json.loads(l) for l in open(out / 'metrics.jsonl')][-1]['dev']
    assert 'ns0_next_span' in met and 'ns0_span_events' in met
    predict.main(['--run', str(out), '--device', 'cpu'])
    M = json.loads((out / 'predictions_dev' / 'METRICS.json').read_text())
    g = next(iter(M['namespaces'].values()))['groups']['all']
    assert 'next_span' in g and g['next_span']['end_class'] == next(iter(json.loads((g2 / 'PACK_MANIFEST.json').read_text())['namespaces']))['K']
    # G1 config on the G2 pack: no G2 heads, no G2 term, denominator without G2-only positions
    rc = runner.main(['--config', str(_cfg(tmp, 'c2', False)), '--pack', str(g2), '--out', str(tmp / 'o2'), '--seed', '1', '--device', 'cpu',
                      '--max-seconds', '100000', '--purpose', 'test', '--max-updates', '1'])
    assert rc == runner.EXIT_RESUMABLE
    assert not any('next_span' in l for l in open(tmp / 'o2' / 'train_log.jsonl'))
