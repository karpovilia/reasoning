"""CPU checks of architecture B with a tiny random Qwen3 (same code path as the 4B model)."""
import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
pytest.importorskip('peft')
from ptrain import synthetic, runner, losses as L  # noqa: E402
from ptrain.data import sha256_file  # noqa: E402
from ptrain.model_b import BData, token_windows, build  # noqa: E402

MODEL = dict(model_id='Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554', dtype='bfloat16',
             token_context=256, max_segments=64, question_max_tokens=32, segment_max_tokens=24, context_tokens=64,
             gradient_checkpointing=False, head_width=32,
             lora=dict(r=4, alpha=8, dropout=0.0, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))


def fake_tokens(pack, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    units = json.loads((Path(pack) / 'units.json').read_text())
    P = sum(u['n'] for u in units)
    lens = rng.integers(1, MODEL['segment_max_tokens'] + 1, P); seg_off = np.concatenate([[0], np.cumsum(lens)])
    ql = rng.integers(5, MODEL['question_max_tokens'] + 1, len(units)); q_off = np.concatenate([[0], np.cumsum(ql)])
    np.savez(Path(pack) / 'tokens_qwen3.npz', seg_tok=rng.integers(10, 190, seg_off[-1]).astype('i4'), seg_off=seg_off,
             q_tok=rng.integers(10, 190, q_off[-1]).astype('i4'), q_off=q_off, prefix_ids=np.array([1, 2], 'i4'), mid_ids=np.array([3, 4, 5], 'i4'))
    man = dict(model_id=MODEL['model_id'], revision=MODEL['revision'], question_max_tokens=MODEL['question_max_tokens'],
               segment_max_tokens=MODEL['segment_max_tokens'], sha256=sha256_file(Path(pack) / 'tokens_qwen3.npz'))
    (Path(pack) / 'TOKENS_MANIFEST.json').write_text(json.dumps(man))


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp('b')
    synthetic.build(d / 'pack', n_train=10, n_dev=4, seed=5, max_len=120, emb_rows=50, native=True)
    fake_tokens(d / 'pack')
    cfgq = transformers.Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(cfgq).save_pretrained(d / 'qwen')
    (d / 'qwen/PTRAIN_ASSET_BINDING.json').write_text(json.dumps(dict(model_id=MODEL['model_id'], revision=MODEL['revision'])))
    import os
    os.environ['PTRAIN_QWEN_DIR'] = str(d / 'qwen')
    cfg = dict(arch='B', cell='CTES', model=MODEL, optimizer=dict(adapter_lr=1e-3, heads_lr=1e-3, lr=1e-3, weight_decay=.01, betas=[.9, .999],
               eps=1e-8, clip_norm=1., warmup_updates=2, cosine_floor=.1), effective_batch=4, micro_batch=2, passes=2,
               task_weights=L.STAGE1_WEIGHTS, checkpoint_every_updates=3, log_every=1)
    return d, cfg


def test_token_windows_score_once_and_fit(env):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    seen = {}
    for ui, lo, hi, s in data.windows:
        u = data.units[ui]
        ntok = len(data.prefix_ids) + len(data.mid_ids) + int(data.q_off[ui + 1] - data.q_off[ui]) + int(data.seg_off[u['offset'] + hi] - data.seg_off[u['offset'] + lo])
        assert ntok <= MODEL['token_context'] and hi - lo <= MODEL['max_segments']
        for p in range(s, hi):
            seen[(ui, p)] = seen.get((ui, p), 0) + 1
    assert all(v == 1 for v in seen.values()) and len(seen) == sum(u['n'] for u in data.units)
    assert any(lo < s for _, lo, _, s in data.windows)          # left context is used
    assert all(s - lo <= 32 for _, lo, _, s in data.windows)    # SR-7: context capped at max_segments/2 (64/2)
    assert all(hi > s for _, lo, hi, s in data.windows)         # every window scores at least one new segment


def test_b_right_context_invariance_and_frozen_base(env):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    torch.manual_seed(1)
    m = build(cfg, data.nsolvers, 'cpu', namespaces=data.namespaces).eval()
    a = m.audit(); assert a['base_trainable_names'] == [] and a['trainable'] < a['total']
    with torch.no_grad():
        m.side.weight.normal_(0, .1)      # make side features matter for the test
    wi = max(range(len(data.windows)), key=lambda i: data.windows[i][2] - data.windows[i][1])
    b = data.make_batch([wi], 'cpu')
    o1 = m(b)
    t = 5
    cut = int(b['marker_pos'][0, t]) + 1
    b2 = dict(b); b2['input_ids'] = b['input_ids'].clone(); b2['input_ids'][0, cut:] = 7
    b2['xin'] = b['xin'].clone(); b2['xin'][0, t + 1:] = 3.0
    o2 = m(b2)
    for j in range(len(o1['ns'])):
        for k in ('gate', 'dest', 'ev'):
            assert torch.allclose(o1['ns'][j][k][0, :t + 1], o2['ns'][j][k][0, :t + 1], atol=1e-4), (j, k)
    assert not torch.allclose(o1['ns'][0]['gate'][0, t + 1:], o2['ns'][0]['gate'][0, t + 1:], atol=1e-4)
    assert [x['dest'].shape[-1] for x in o1['ns']] == [6, 7, 6] and [x['ev'].shape[-1] for x in o1['ns']] == [6, 4, 6]


def test_b_ctes_unavailable_equals_ct(env):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    torch.manual_seed(2); ct = build(dict(cfg, cell='CT'), data.nsolvers, 'cpu', namespaces=data.namespaces).eval()
    torch.manual_seed(2); ce = build(cfg, data.nsolvers, 'cpu', namespaces=data.namespaces).eval()
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(ct.named_parameters(), ce.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)
        g = torch.Generator().manual_seed(0)
        for p1, p2 in zip(ct.side.parameters(), ce.side.parameters()):
            r = torch.randn(p1.shape, generator=g) * .1; p1.copy_(r); p2.copy_(r)
    b = data.make_batch([0, 1], 'cpu'); b['es_ok'][:] = False
    assert torch.allclose(ct(b)['ns'][0]['gate'], ce(b)['ns'][0]['gate'], atol=1e-5)


def test_b_disabled_heads_and_resume(env, tmp_path):
    d, cfg = env
    torch.use_deterministic_algorithms(True)
    cp = tmp_path / 'b.json'; cp.write_text(json.dumps(cfg))
    def run(out, mu):
        args = ['--config', str(cp), '--pack', str(d / 'pack'), '--out', str(out), '--seed', '42', '--device', 'cpu', '--max-seconds', '100000', '--purpose', 'test']
        if mu:
            args += ['--max-updates', str(mu)]
        return runner.main(args)
    assert run(tmp_path / 'c', None) == 0
    assert run(tmp_path / 'r', 3) == runner.EXIT_RESUMABLE
    assert run(tmp_path / 'r', None) == 0
    a = torch.load(tmp_path / 'c/checkpoint.pt', weights_only=False); b = torch.load(tmp_path / 'r/checkpoint.pt', weights_only=False)
    assert set(a['model']) == set(b['model']) and all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert not any(k.startswith('lm.') and 'lora_' not in k for k in a['model'])     # base weights never saved/trained
    ev = [json.loads(x) for x in (tmp_path / 'r/events.jsonl').read_text().splitlines()]
    chk = [e for e in ev if e['event'] == 'disabled_grad_check'][0]
    assert chk['nonzero'] == [] and chk['checked'] == 6


def test_b_token_budget_groups_and_padding_equivalence(env):
    """Token-budget micro-batching covers every window once, respects the budget, and a padded group gives the same
    per-position outputs as each window alone (right padding + attention mask)."""
    d, cfg = env
    data = BData(d / 'pack', cfg)
    ids = list(range(len(data.windows)))
    groups = data.micro_batches(ids, 300)
    assert sorted(i for g in groups for i in g) == ids
    assert all(len(g) == 1 or max(data.win_tokens[i] for i in g) * len(g) <= 300 for g in groups)
    torch.manual_seed(3)
    m = build(cfg, data.nsolvers, 'cpu', namespaces=data.namespaces).eval()
    with torch.no_grad():
        m.side.weight.normal_(0, .1)
        g = max(groups, key=len)
        assert len(g) > 1
        ob = m(data.make_batch(g, 'cpu'))
        for k, wi in enumerate(g):
            o1 = m(data.make_batch([wi], 'cpu')); n = data.windows[wi][2] - data.windows[wi][1]
            for j in range(len(o1['ns'])):
                assert torch.allclose(ob['ns'][j]['gate'][k, :n], o1['ns'][j]['gate'][0, :n], atol=2e-4), (k, j)



def test_b_context_never_starves_window():
    """SR-8: context as large as the whole budget still leaves room for a scored segment."""
    import numpy as np
    units = [dict(role='TRAIN', n=40, offset=0, question_component='q')]
    seg = np.full(40, 50); q = np.array([30])
    w = token_windows(units, seg, q, ('TRAIN',), 300, 256, 10_000, 1000)
    assert sorted(p for _, lo, hi, s in w for p in range(s, hi)) == list(range(40))
    assert all(q[0] + seg[lo:hi].sum() <= 300 for _, lo, hi, s in w)
