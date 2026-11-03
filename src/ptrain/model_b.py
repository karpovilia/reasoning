"""Architecture B: frozen pretrained Qwen3-4B-Instruct-2507 (BF16) + fresh LoRA + prediction heads.

Serialization of one window (identical text boundaries for CT and CTES):
  [chat prefix + question tokens (first `question_max_tokens`)] + for every segment j in [lo,hi):
  tokens(segment_j + "\n") truncated to its LAST `segment_max_tokens` tokens.
The prediction after segment j is read from the hidden state of the last token of segment j.
Side features (C, and for CTES the prefix-available E/S, masked exactly as in architecture A)
are projected to the model width and ADDED to the input embedding of that last token, so later
positions can attend to them; nothing to the right of segment j enters position j (causal LM).
Windows: at most `max_segments` segments and `token_context` tokens. Each target is scored once:
the first window starts at segment 0; every next window keeps up to `context_tokens` tokens and at most
`context_segments` (default max_segments/2 = 128) of the immediately preceding segments as unscored left context,
so each prediction sees between context_segments and max_segments segments of history (or the whole prefix). Counts of truncated segments and
questions are recorded in the pack token manifest.
Trainable: LoRA adapters (r, alpha, dropout, target modules from config), side projection,
mode/prev/solver embeddings and heads. Base weights frozen (requires_grad False, checked).
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import PackedData, make_windows  # noqa: E402
from ptrain.model_a import apply_cell_mask, native_outputs, native_group1_heads, native_group3_heads, g4_head  # noqa: E402

CHAT_PREFIX = '<|im_start|>user\n'
CHAT_MID = '<|im_end|>\n<|im_start|>assistant\n<think>\n'


def token_windows(units, seg_ntok, q_ntok, roles, token_context, max_segments, context_tokens, context_segments=None):
    """seg_ntok: per-position token counts (after truncation), q_ntok: per-unit question tokens.
    Returns (ui, lo, hi, score_lo). Every position of a unit scored exactly once."""
    out = []
    for ui, u in enumerate(units):
        if u['role'] not in roles:
            continue
        n = u['n']; off = u['offset']; budget = token_context - q_ntok[ui]
        if budget < 1:
            raise ValueError('QUESTION_EXCEEDS_CONTEXT')
        s = 0
        while s < n:
            # left context
            # SR-7: the left context is capped in segments too, so that a window keeps room for new scored segments
            cseg = (max_segments // 2) if context_segments is None else context_segments
            lo = s; ctx = 0
            # SR-8: context never takes the room of the first scored segment
            room = min(context_tokens, budget - int(seg_ntok[off + s]))
            while lo > 0 and ctx + seg_ntok[off + lo - 1] <= room and (s - lo) < cseg:
                lo -= 1; ctx += seg_ntok[off + lo]
            hi = s; used = ctx
            while hi < n and used + seg_ntok[off + hi] <= budget and hi - lo < max_segments:
                used += seg_ntok[off + hi]; hi += 1
            if hi == s:      # single segment longer than the budget after truncation cannot happen
                raise ValueError('SEGMENT_EXCEEDS_BUDGET')
            out.append((ui, lo, hi, s)); s = hi
    return out


class BData(PackedData):
    def __init__(self, root, cfg, verify=True):
        import numpy as np
        super().__init__(root, verify=verify, need_embeddings=False)
        tk = Path(root) / 'tokens_qwen3.npz'
        man = json.loads((Path(root) / 'TOKENS_MANIFEST.json').read_text())
        if verify:
            from ptrain.data import sha256_file
            if sha256_file(tk) != man['sha256']:
                raise ValueError('TOKENS_SHA')
        m = cfg['model']
        for k in ('model_id', 'revision', 'question_max_tokens', 'segment_max_tokens'):
            if man[k] != m[k]:
                raise ValueError('TOKENS_CONFIG_MISMATCH:' + k)
        z = np.load(tk, allow_pickle=False)
        self.seg_tok = z['seg_tok']; self.seg_off = z['seg_off']; self.q_tok = z['q_tok']; self.q_off = z['q_off']
        self.prefix_ids = z['prefix_ids'].tolist(); self.mid_ids = z['mid_ids'].tolist()
        seg_ntok = np.diff(self.seg_off); q_ntok = np.diff(self.q_off) + len(self.prefix_ids) + len(self.mid_ids)
        self.windows = token_windows(self.units, seg_ntok, q_ntok, ('TRAIN', 'DEV'), m['token_context'], m['max_segments'],
                                     m.get('context_tokens', 2048), m.get('context_segments'))
        self.token_manifest = man
        # padded token length of every window (prefix + question + segments), for token-budget micro-batching
        self.win_tokens = []
        for ui, lo, hi, s in self.windows:
            u = self.units[ui]
            self.win_tokens.append(len(self.prefix_ids) + len(self.mid_ids) + int(self.q_off[ui + 1] - self.q_off[ui])
                                   + int(self.seg_off[u['offset'] + hi] - self.seg_off[u['offset'] + lo]))

    def micro_batches(self, ids, token_budget):
        """Deterministic grouping of one effective batch: sort by token length, then pack greedily so that
        (max length in group) x (group size) <= token_budget. A single window longer than the budget forms its own group.
        Grouping does not change the update: the runner normalises with the common denominator of the whole step."""
        order = sorted(ids, key=lambda i: (self.win_tokens[i], i))
        groups = []; cur = []
        for i in order:
            if cur and self.win_tokens[i] * (len(cur) + 1) > token_budget:
                groups.append(cur); cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        return groups

    def make_batch(self, ids, device, T=None):
        import numpy as np
        import torch
        T = max(self.windows[i][2] - self.windows[i][1] for i in ids)
        b = super().make_batch(ids, 'cpu', T=T)
        seqs = []; marks = []
        for wi in ids:
            ui, lo, hi, s = self.windows[wi]; u = self.units[ui]
            ids_ = list(self.prefix_ids) + self.q_tok[self.q_off[ui]:self.q_off[ui + 1]].tolist() + list(self.mid_ids)
            mk = []
            for j in range(lo, hi):
                p = u['offset'] + j
                ids_ += self.seg_tok[self.seg_off[p]:self.seg_off[p + 1]].tolist(); mk.append(len(ids_) - 1)
            seqs.append(ids_); marks.append(mk)
        L = max(len(x) for x in seqs)
        inp = torch.zeros((len(ids), L), dtype=torch.long); att = torch.zeros((len(ids), L), dtype=torch.long)
        mpos = torch.zeros((len(ids), T), dtype=torch.long)
        for i, (x, mk) in enumerate(zip(seqs, marks)):
            inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1; mpos[i, :len(mk)] = torch.tensor(mk)
        b['input_ids'] = inp; b['attention_mask'] = att; b['marker_pos'] = mpos
        from ptrain.data import to_device
        return to_device(b, device)


def build_struct_encoder(cfg, nsolvers, namespaces):
    """Frozen structure encoder for cells with V: an architecture-A model trained earlier on the same pack split, loaded
    from its weights file (sha256 pinned in cfg['struct_vector']); all parameters frozen. Only its trunk is used."""
    import torch
    from ptrain import model_a
    from ptrain.data import sha256_file
    sv = cfg.get('struct_vector')
    if not sv:
        raise ValueError('V_CELL_REQUIRES_STRUCT_VECTOR_CONFIG')
    if any(x in sv['cell'] for x in ('T',)):
        raise ValueError('STRUCT_ENCODER_MUST_NOT_READ_TEXT')
    w = Path(sv['weights'])
    if sha256_file(w) != sv['weights_sha256']:
        raise ValueError('STRUCT_ENCODER_WEIGHTS_SHA')
    enc = model_a.build(sv['cell'], nsolvers, sv['model'], namespaces=namespaces)
    payload = torch.load(w, map_location='cpu', weights_only=False)
    enc.load_state_dict(payload['model'])
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.eval()


def build(cfg, nsolvers, device, namespaces=None):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = cfg['model']; cell = cfg['cell']
    if 'T' not in cell:
        raise ValueError('B_REQUIRES_TEXT_CELL')
    path = os.environ.get('PTRAIN_QWEN_DIR')
    if not path:
        raise ValueError('PTRAIN_QWEN_DIR_REQUIRED')
    rev_file = Path(path) / 'PTRAIN_ASSET_BINDING.json'
    binding = json.loads(rev_file.read_text())
    if binding['model_id'] != m['model_id'] or binding['revision'] != m['revision']:
        raise ValueError('QWEN_ASSET_REVISION_MISMATCH')
    if m['dtype'] != 'bfloat16':
        raise ValueError('ONLY_BF16')
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=m.get('attn', 'sdpa'))
    for p in base.parameters():
        p.requires_grad_(False)
    lc = m['lora']
    peft_model = get_peft_model(base, LoraConfig(r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], bias=lc['bias'],
                                                 target_modules=lc['target_modules'], task_type='CAUSAL_LM'))
    if m.get('gradient_checkpointing', True):
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    hid = base.config.hidden_size; hw = m['head_width']
    group1 = cfg.get('group1', False)
    enc = build_struct_encoder(cfg, nsolvers, namespaces) if 'V' in cell else None

    class BPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = cell; self.lm = peft_model
            self.mode = nn.Embedding(8, 24); self.prev = nn.Embedding(8, 16); self.solver = nn.Embedding(nsolvers, 16, padding_idx=0)
            self.side = nn.Linear(41 + 24 + 16 + 16 + 8, hid)
            nn.init.zeros_(self.side.weight); nn.init.zeros_(self.side.bias)   # start as the pure text model
            self.norm = nn.LayerNorm(hid); self.down = nn.Linear(hid, hw)
            self.native = bool(namespaces)
            if self.native:
                self.ns_heads = nn.ModuleList([nn.Linear(hw, 3 + ns['K'] + ns['E']) for ns in namespaces])
                self.ns_dims = [(ns['K'], ns['E']) for ns in namespaces]
                if group1:
                    self.ns_g1_heads = native_group1_heads(nn, hw, namespaces)
                if cfg.get('group3'):
                    self.ns_g3_heads = native_group3_heads(nn, hw, namespaces)
                if cfg.get('group4'):
                    self.g4 = g4_head(nn, hw)
            else:
                self.gate = nn.Linear(hw, 3); self.dest = nn.Linear(hw, 6); self.events = nn.Linear(hw, 6)
            self.length = nn.Linear(hw, 1); self.wait = nn.Linear(hw, 30); self.rem = nn.Linear(hw, 3)
            if group1 and not self.native:
                self.next_span = nn.Linear(hw, 7); self.span_events = nn.Linear(hw, 36)
            # structure vector V (structural_embedding_into_lora_v1): frozen causal A encoder -> trainable projection,
            # zero-init, added to the segment marker token embedding (same entry point as the E/S side features)
            self.enc = enc
            if enc is not None:
                self.vproj = nn.Linear(enc.norm.normalized_shape[0], hid)
                nn.init.zeros_(self.vproj.weight); nn.init.zeros_(self.vproj.bias)

        def train(self, mode=True):
            super().train(mode)
            if self.enc is not None:
                self.enc.eval()                      # frozen source: no dropout, no grad, in train and eval
            return self

        def struct_vector(self, b):
            with torch.no_grad():
                return self.enc.trunk(b)            # (N,T,width): causal mask inside the encoder, positions <= t only

        def forward(self, b):
            x, mode, prev, _, _ = apply_cell_mask(torch, self.cell, b)
            bench = x.new_zeros((*x.shape[:-1], 8))
            side = self.side(torch.cat((x, self.mode(mode), self.prev(prev), self.solver(b['solver']), bench), -1))  # (N,T,hid)
            if self.enc is not None:
                side = side + self.vproj(self.struct_vector(b).float())
            side = side * (~b['pad']).unsqueeze(-1).to(side.dtype)
            emb = self.lm.get_input_embeddings()(b['input_ids'])
            idx = b['marker_pos'].unsqueeze(-1).expand(-1, -1, emb.shape[-1])
            emb = emb.scatter_add(1, idx, side.to(emb.dtype))
            am = b['attention_mask']
            # no padding -> no explicit mask, so SDPA can use its causal flash kernel
            am = None if bool(am.all()) else am
            out = self.lm.base_model.model.model(inputs_embeds=emb, attention_mask=am, use_cache=False)
            h = out.last_hidden_state.gather(1, idx).float()
            h = F.gelu(self.down(self.norm(h)))
            p = self.wait(h).view(*h.shape[:-1], 3, 10)
            o = native_outputs(self, h) if self.native else dict(gate=self.gate(h), dest=self.dest(h), events=self.events(h))
            o.update(length=self.length(h).squeeze(-1),
                     wait_pi=p[..., 0], wait_logscale=p[..., 1].clamp(-3, 16), wait_shape=(F.softplus(p[..., 2]) + .2).clamp(max=5),
                     wait_mark=p[..., 3:], rem=self.rem(h))
            if group1 and not self.native:
                o['next_span'] = self.next_span(h); o['span_events'] = self.span_events(h).view(*h.shape[:-1], 6, 6)
            return o

        def trainable_state_dict(self):
            return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

        def load_trainable_state_dict(self, sd):
            names = {k for k, v in self.named_parameters() if v.requires_grad}
            if set(sd) != names:
                raise ValueError('TRAINABLE_KEYS_MISMATCH')
            params = dict(self.named_parameters())
            with torch.no_grad():
                for k, v in sd.items():
                    params[k].copy_(v)

        def param_groups(self, ocfg):
            lora = [p for n, p in self.named_parameters() if p.requires_grad and 'lora_' in n]
            other = [p for n, p in self.named_parameters() if p.requires_grad and 'lora_' not in n]
            return [{'params': lora, 'lr': ocfg['adapter_lr']}, {'params': other, 'lr': ocfg['heads_lr']}]

        def audit(self):
            tot = sum(p.numel() for p in self.parameters()); tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
            base_tr = [n for n, p in self.named_parameters() if p.requires_grad and n.startswith('lm.') and 'lora_' not in n]
            return dict(total=tot, trainable=tr, base_trainable_names=base_tr)

    model = BPredictor()
    for n, p in model.named_parameters():
        if not n.startswith('lm.'):
            p.data = p.data.float()
    model.to(device)
    a = model.audit()
    if a['base_trainable_names']:
        raise ValueError('BASE_WEIGHTS_TRAINABLE')
    return model


def tokenize_pack(root, tok_dir, model_cfg, texts_for_unit):
    """texts_for_unit(unit) -> (question_text, [segment_text...]). Writes tokens_qwen3.npz + manifest."""
    import numpy as np
    from transformers import AutoTokenizer
    from ptrain.data import sha256_file
    binding = json.loads((Path(tok_dir) / 'PTRAIN_ASSET_BINDING.json').read_text())
    tok = AutoTokenizer.from_pretrained(tok_dir)
    root = Path(root); units = json.loads((root / 'units.json').read_text())
    qmax = model_cfg['question_max_tokens']; smax = model_cfg['segment_max_tokens']
    seg_tok = []; seg_off = [0]; q_tok = []; q_off = [0]; trunc_q = 0; trunc_s = 0
    for u in units:
        q, segs = texts_for_unit(u)
        if len(segs) != u['n']:
            raise ValueError('SEGMENT_COUNT_MISMATCH:' + u['unit_id'])
        qi = tok(q, add_special_tokens=False)['input_ids']
        if len(qi) > qmax:
            qi = qi[:qmax]; trunc_q += 1
        q_tok.extend(qi); q_off.append(len(q_tok))
        enc = tok([s + '\n' for s in segs], add_special_tokens=False)['input_ids']
        for e in enc:
            if len(e) > smax:
                e = e[-smax:]; trunc_s += 1
            if not e:
                e = tok('\n', add_special_tokens=False)['input_ids']
            seg_tok.extend(e); seg_off.append(len(seg_tok))
    pre = tok(CHAT_PREFIX, add_special_tokens=False)['input_ids']; mid = tok(CHAT_MID, add_special_tokens=False)['input_ids']
    np.savez(root / 'tokens_qwen3.npz', seg_tok=np.array(seg_tok, np.int32), seg_off=np.array(seg_off, np.int64),
             q_tok=np.array(q_tok, np.int32), q_off=np.array(q_off, np.int64), prefix_ids=np.array(pre, np.int32), mid_ids=np.array(mid, np.int32))
    man = dict(model_id=binding['model_id'], revision=binding['revision'], question_max_tokens=qmax, segment_max_tokens=smax,
               truncated_questions=trunc_q, truncated_segments=trunc_s, segments=len(seg_off) - 1, units=len(units),
               segment_truncation='keep last segment_max_tokens tokens', question_truncation='keep first question_max_tokens tokens',
               sha256=sha256_file(root / 'tokens_qwen3.npz'))
    (root / 'TOKENS_MANIFEST.json').write_text(json.dumps(man, indent=1))
    return man
