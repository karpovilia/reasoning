"""CPU checks before any GPU work. Run: python -m pytest -q tests (inside cpu_gate)."""
import json, shutil, sys, tempfile
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
from ptrain import model_a, losses as L, synthetic, runner  # noqa: E402
from ptrain.data import PackedData, make_windows  # noqa: E402
from ptrain.targets_group1 import derive  # noqa: E402

SMALL = dict(width=64, heads=4, layers=2, ff_width=128, dropout=0.1, max_segments=256)


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    d = tmp_path_factory.mktemp('pack')
    synthetic.build(d, n_train=24, n_dev=8, seed=1, max_len=300, emb_rows=500, native=False)
    return d


def test_windows_score_each_position_once(pack):
    data = PackedData(pack)
    seen = {}
    for ui, lo, hi, s in data.windows:
        assert hi - lo <= 256 and lo <= s < hi
        for p in range(s, hi):
            seen[(ui, p)] = seen.get((ui, p), 0) + 1
    assert all(v == 1 for v in seen.values())
    assert len(seen) == sum(u['n'] for u in data.units)


def test_right_context_invariance(pack):
    torch.manual_seed(0)
    data = PackedData(pack)
    m = model_a.build('CTES', data.nsolvers, SMALL).eval()
    ids = [i for i, w in enumerate(data.windows) if w[2] - w[1] > 40][:4]
    b = data.make_batch(ids, 'cpu')
    o1 = m(b)
    t = 20
    b2 = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in b.items() if k != 'valid'}; b2['valid'] = b['valid']
    for k in ('xin', 'text', 'prompt'):
        b2[k][:, t + 1:] = torch.randn_like(b2[k][:, t + 1:])
    b2['mode'][:, t + 1:] = 3; b2['solver'][:, t + 1:] = 1
    o2 = m(b2)
    for k in ('gate', 'dest', 'events', 'rem'):
        assert torch.allclose(o1[k][:, :t + 1], o2[k][:, :t + 1], atol=1e-5), k


def test_disabled_losses_have_no_gradient(pack):
    torch.manual_seed(0)
    data = PackedData(pack)
    m = model_a.build('CTES', data.nsolvers, SMALL).train()
    b = data.make_batch(data.ids('TRAIN')[:4], 'cpu')
    loss, parts, _ = L.total_loss(m(b), b, L.STAGE1_WEIGHTS)
    loss.backward()
    dis = runner.disabled_head_params(m)
    assert len(dis) == 6
    assert all(p.grad is None for _, p in dis)
    assert m.gate.weight.grad is not None and m.events.weight.grad is not None and m.dest.weight.grad is not None


def test_stage1_equals_v9_three_terms(pack):
    torch.manual_seed(0)
    data = PackedData(pack)
    m = model_a.build('CT', data.nsolvers, SMALL).eval()
    b = data.make_batch(data.ids('TRAIN')[:4], 'cpu')
    o = m(b)
    total, parts, _ = L.total_loss(o, b, L.STAGE1_WEIGHTS)
    full = L.v9_loss_parts(o, b, include_disabled=True)
    ref = full['joint'] + .5 * full['destination'] + .25 * full['event']
    assert torch.allclose(total, ref)
    assert set(full) == {'joint', 'destination', 'event', 'length', 'wait', 'remainder'}


def test_ctes_without_proven_es_equals_ct(pack):
    data = PackedData(pack)
    torch.manual_seed(3); a = model_a.build('CT', data.nsolvers, SMALL).eval()
    torch.manual_seed(3); c = model_a.build('CTES', data.nsolvers, SMALL).eval()
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), c.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)  # paired initialisation across cells
    b = data.make_batch(data.ids('DEV')[:3], 'cpu')
    b['es_ok'][:] = False
    assert torch.allclose(a(b)['gate'], c(b)['gate'], atol=1e-6)
    b['es_ok'][:] = True
    assert not torch.allclose(a(b)['gate'], c(b)['gate'], atol=1e-6)


def test_pack_refuses_test_role_and_overlap(tmp_path):
    synthetic.build(tmp_path / 'p', n_train=6, n_dev=3, seed=2, max_len=50, emb_rows=50, native=False)
    u = json.loads((tmp_path / 'p/units.json').read_text())
    u[0]['role'] = 'TEST'; (tmp_path / 'p/units.json').write_text(json.dumps(u))
    with pytest.raises(ValueError):
        PackedData(tmp_path / 'p', verify=False)
    u[0]['role'] = 'TRAIN'; u[-1]['question_component'] = u[0]['question_component']
    (tmp_path / 'p/units.json').write_text(json.dumps(u))
    with pytest.raises(ValueError, match='TRAIN_DEV_OVERLAP'):
        PackedData(tmp_path / 'p', verify=False)


# ---- group-1 target semantics ----
def test_g1_mode_change_and_events():
    mode = [0, 0, 1, 1, 2]
    ev = [[0] * 6, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0] * 6]
    kn = [[True] * 6] * 5
    ns, y, k, _ = derive(mode, ev, kn, 'natural')
    assert ns == [1, 1, 2, 2, 6]
    assert y[0] == [0, 1, 1, 0, 0, 0] and all(k[0])          # events of span [2,4): complete, all known
    assert y[2] == [0] * 6 and all(k[2])                      # next span [4,5) ends naturally: complete
    assert not any(k[4])                                      # END: no event target


def test_g1_internal_event_is_not_boundary_and_same_type_switch():
    mode = [1, 1, 1, 1]
    ev = [[0] * 6, [1, 0, 0, 0, 0, 0], [0] * 6, [0] * 6]; kn = [[True] * 6] * 4
    ns, _, _, _ = derive(mode, ev, kn, 'natural')
    assert ns == [6, 6, 6, 6]                                 # one span, event inside does not split
    ns, y, k, explicit = derive(mode, ev, kn, 'natural', span_start=[True, False, True, False])
    assert explicit and ns[0] == 1 and ns[1] == 1 and ns[2] == 6   # explicit same-type boundary = SWITCH to type 1


def test_g1_unknown_and_truncation_masks():
    mode = [0, 0, 7, 1]
    kn = [[True] * 6] * 4; ev = [[0] * 6] * 4
    ns, _, _, _ = derive(mode, ev, kn, 'natural')
    assert ns[0] == -1 and ns[1] == -1 and ns[2] == -1        # boundary undecidable through the unknown label
    mode = [0, 0, 1, 1]
    ev = [[0] * 6, [0] * 6, [0, 0, 1, 0, 0, 0], [0] * 6]
    ns, y, k, _ = derive(mode, ev, kn, 'confirmed_truncated')
    assert ns[:2] == [1, 1] and ns[2:] == [-1, -1]            # truncated: no END, current span end unknown
    assert y[0][2] == 1 and k[0][2] and not k[0][0]           # partial next span: positive known, absence unknown
    kn2 = [[True] * 6, [True] * 6, [True] * 6, [True, False, True, True, True, True]]
    ns, y, k, _ = derive([0, 0, 1, 1], [[0] * 6] * 4, kn2, 'natural')
    assert k[0][0] and not k[0][1]                            # absence known only if every segment knows it


def _run(pack, out, seed, max_updates, cell='CTES'):
    cfg = dict(arch='A', cell=cell, model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
               warmup_updates=2, cosine_floor=.1), effective_batch=8, micro_batch=4, passes=2, task_weights=L.STAGE1_WEIGHTS,
               checkpoint_every_updates=3, log_every=1)
    cp = Path(out).parent / (Path(out).name + '.cfg.json'); cp.write_text(json.dumps(cfg))
    args = ['--config', str(cp), '--pack', str(pack), '--out', str(out), '--seed', str(seed), '--device', 'cpu',
            '--max-seconds', '100000', '--purpose', 'test']
    if max_updates:
        args += ['--max-updates', str(max_updates)]
    return runner.main(args)


def test_resume_equals_continuous(pack, tmp_path):
    torch.use_deterministic_algorithms(True)
    assert _run(pack, tmp_path / 'cont', 42, None) == 0
    assert _run(pack, tmp_path / 'res', 42, 4) == runner.EXIT_RESUMABLE       # stop mid-pass
    assert _run(pack, tmp_path / 'res', 42, 9) == runner.EXIT_RESUMABLE       # stop in pass 2
    assert _run(pack, tmp_path / 'res', 42, None) == 0
    a = torch.load(tmp_path / 'cont/checkpoint.pt', weights_only=False); b = torch.load(tmp_path / 'res/checkpoint.pt', weights_only=False)
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k]), k
    ma = [json.loads(x) for x in (tmp_path / 'cont/metrics.jsonl').read_text().splitlines()]
    mb = [json.loads(x) for x in (tmp_path / 'res/metrics.jsonl').read_text().splitlines()]
    assert ma == mb and len(ma) == 2
    assert b['state']['resumes'] == 2
    ev = [json.loads(x) for x in (tmp_path / 'res/events.jsonl').read_text().splitlines()]
    chk = [e for e in ev if e['event'] == 'disabled_grad_check']
    assert len(chk) == 1 and chk[0]['nonzero'] == [] and chk[0]['none_count'] == 6


def test_binding_mismatch_refused(pack, tmp_path):
    assert _run(pack, tmp_path / 'x', 42, 2) == runner.EXIT_RESUMABLE
    with pytest.raises(ValueError, match='RUN_BINDING_MISMATCH'):
        _run(pack, tmp_path / 'x', 43, 2)


def test_synthetic_refused_for_production(pack, tmp_path):
    cp = tmp_path / 'c.json'; cp.write_text(json.dumps(dict(arch='A')))
    with pytest.raises(ValueError, match='SYNTHETIC'):
        runner.main(['--config', str(cp), '--pack', str(pack), '--out', str(tmp_path / 'o'), '--seed', '1', '--device', 'cpu',
                     '--max-seconds', '10', '--purpose', 'production'])


def test_lock_refuses_second_worker(pack, tmp_path):
    import fcntl
    out = tmp_path / 'l'; out.mkdir()
    f = open(out / 'LOCK', 'a+'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert _run(pack, out, 42, 1) == runner.EXIT_LOCKED
