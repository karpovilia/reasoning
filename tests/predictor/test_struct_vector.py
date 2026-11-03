"""Structure vector V (structural_embedding_into_lora_v1): frozen A encoder -> projection -> Qwen marker embedding."""
import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
torch = pytest.importorskip('torch')
pytest.importorskip('transformers'); pytest.importorskip('peft')
from ptrain import model_a, runner, losses as L  # noqa: E402
from ptrain.data import sha256_file  # noqa: E402
from ptrain.model_b import BData, build  # noqa: E402
from test_model_b import env, MODEL  # noqa: E402,F401

ENC_MODEL = dict(width=32, heads=4, layers=2, ff_width=64, dropout=0.1, max_segments=256)


def make_enc(d, data, seed=7):
    torch.manual_seed(seed)
    enc = model_a.build('CES', data.nsolvers, ENC_MODEL, namespaces=data.namespaces)
    p = d / f'enc_{seed}.pt'
    torch.save(dict(state={}, model=enc.state_dict(), weights_only=True), p)
    return p


def vcfg(cfg, d, data, cell='CTESV', wp=None):
    wp = wp or make_enc(d, data)
    return dict(cfg, cell=cell, struct_vector=dict(cell='CES', model=ENC_MODEL, weights=str(wp), weights_sha256=sha256_file(wp)))


def batch(data, n=3):
    return data.make_batch(data.ids('TRAIN')[:n], 'cpu')


def test_weights_sha_and_text_guard(env):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    c = vcfg(cfg, d, data); c['struct_vector']['weights_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='STRUCT_ENCODER_WEIGHTS_SHA'):
        build(c, data.nsolvers, 'cpu', namespaces=data.namespaces)
    c = vcfg(cfg, d, data); c['struct_vector']['cell'] = 'CTES'
    with pytest.raises(ValueError, match='STRUCT_ENCODER_MUST_NOT_READ_TEXT'):
        build(c, data.nsolvers, 'cpu', namespaces=data.namespaces)
    c = dict(cfg, cell='CTV')
    with pytest.raises(ValueError, match='V_CELL_REQUIRES_STRUCT_VECTOR_CONFIG'):
        build(c, data.nsolvers, 'cpu', namespaces=data.namespaces)


def test_vector_reaches_qwen_and_gradients(env):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    m = build(vcfg(cfg, d, data), data.nsolvers, 'cpu', namespaces=data.namespaces)
    m.train()
    assert not m.enc.training                                     # frozen source stays in eval inside model.train()
    assert all(not p.requires_grad for p in m.enc.parameters())
    assert not any(k.startswith('enc.') for k in m.trainable_state_dict())
    b = batch(data)
    seen = {}
    h = m.lm.base_model.model.model.register_forward_hook(lambda mod, i, o: seen.__setitem__('h', o.last_hidden_state.detach().clone()))
    m.eval()
    with torch.no_grad():
        o0 = m(b)['ns'][0]['gate'].clone(); h0 = seen['h']
        torch.nn.init.normal_(m.vproj.weight, std=0.5)
        o1 = m(b)['ns'][0]['gate'].clone(); h1 = seen['h']
        # replace the encoder output by zeros: the effect disappears (the vector, not the projection bias, carries it)
        real = m.struct_vector
        m.struct_vector = lambda bb: torch.zeros_like(real(bb))
        o2 = m(b)['ns'][0]['gate'].clone()
        m.struct_vector = real
    h.remove()
    assert not torch.allclose(h0, h1)                             # Qwen's own hidden states change: input-level entry
    assert not torch.allclose(o0, o1) and torch.allclose(o0, o2, atol=1e-5)
    m.train()
    out = m(b)
    acc = [ns['status'] == 'accepted' for ns in data.namespaces]
    loss, _ = L.native_total_loss(out, b, acc)
    loss.backward()
    assert m.vproj.weight.grad is not None and m.vproj.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for p in m.enc.parameters())


def test_causal_prefix_invariance(env):
    """Changing E/S inputs strictly after position t changes neither the vector nor the outputs at positions <= t."""
    d, cfg = env
    data = BData(d / 'pack', cfg)
    m = build(vcfg(cfg, d, data), data.nsolvers, 'cpu', namespaces=data.namespaces).eval()
    torch.nn.init.normal_(m.vproj.weight, std=0.5)
    b = batch(data, 2)
    n = int((~b['pad'][0]).sum()); t = n // 2
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2['ns'] = b['ns']; b2['valid'] = b['valid']
    g = torch.Generator().manual_seed(0)
    b2['xin'][0, t + 1:n, 3:] = torch.rand(n - t - 1, 38, generator=g)
    b2['mode'][0, t + 1:n] = (b2['mode'][0, t + 1:n] + 1) % 6
    with torch.no_grad():
        v1, v2 = m.struct_vector(b), m.struct_vector(b2)
        o1, o2 = m(b)['ns'][0]['gate'], m(b2)['ns'][0]['gate']
    assert torch.allclose(v1[0, :t + 1], v2[0, :t + 1], atol=1e-5) and not torch.allclose(v1[0, t + 1:n], v2[0, t + 1:n])
    assert torch.allclose(o1[0, :t + 1], o2[0, :t + 1], atol=1e-4)


def test_runner_production_v_needs_es_admission(env, tmp_path):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    c = vcfg(cfg, d, data, cell='CTV'); cp = tmp_path / 'c.json'; cp.write_text(json.dumps(c))
    import shutil
    pk = tmp_path / 'pack'; shutil.copytree(d / 'pack', pk)
    man = json.loads((pk / 'PACK_MANIFEST.json').read_text()); man.update(synthetic=False, production_admissible=True)
    (pk / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    pr = tmp_path / 'pass.json'
    pr.write_text(json.dumps(dict(verdict='PASS', code_hashes=runner.code_hashes(), pack_manifest_sha256=[sha256_file(pk / 'PACK_MANIFEST.json')],
                                  config_sha256=[sha256_file(cp)])))
    with pytest.raises(ValueError, match='CTES_PRODUCTION_REQUIRES_ACCEPTED_ES_ADMISSION'):
        runner.main(['--config', str(cp), '--pack', str(pk), '--out', str(tmp_path / 'o'), '--seed', '1', '--device', 'cpu',
                     '--max-seconds', '100', '--purpose', 'production', '--pass-record', str(pr)])


def test_runner_trains_ctv_briefly(env, tmp_path):
    d, cfg = env
    data = BData(d / 'pack', cfg)
    c = vcfg(cfg, d, data, cell='CTV'); cp = tmp_path / 'c.json'; cp.write_text(json.dumps(c))
    rc = runner.main(['--config', str(cp), '--pack', str(d / 'pack'), '--out', str(tmp_path / 'o'), '--seed', '1', '--device', 'cpu',
                      '--max-seconds', '100000', '--purpose', 'test', '--max-updates', '2'])
    assert rc == runner.EXIT_RESUMABLE
    ev = [json.loads(l) for l in open(tmp_path / 'o' / 'train_log.jsonl')]
    assert ev and all(e['grad_norm'] == e['grad_norm'] for e in ev)
