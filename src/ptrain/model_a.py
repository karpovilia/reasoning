"""Architecture A: causal 12-layer Transformer over segments (port of V9 teacher_model_v2.build).

The core module, embeddings, projection 873->512 and the six V9 heads are kept exactly as in
teacher_model_v2.py
Changes, all documented in IMPLEMENTATION_HANDOFF.md:
  * per-position E/S availability mask `es_ok` (positions where E/S inputs are not proven
    prefix-available get the same UNKNOWN representation that a non-E/S cell uses);
  * optional group-1 heads (event -> next span -> events inside it), created only when enabled,
    so the first-stage model has exactly the V9 parameter set.
Torch is imported lazily so that inspection tools stay stdlib-only.
"""
CELLS = ('C', 'CE', 'CS', 'CES', 'CT', 'CTE', 'CTS', 'CTES')
N_TYPES = 6
N_EVENTS = 6
UNKNOWN_MODE = 7


def apply_cell_mask(torch, cell, b):
    """Return (x, mode, prev, text, prompt) after removing input groups absent in `cell`
    and E/S values whose availability at prediction time is not proven (b['es_ok']==False).
    xin layout (41): C3 | S2 (span lengths) | E36 (6 obs, 6 known, 24 counters)."""
    x = b['xin']; mode = b['mode']; prev = b['prev_mode']; text = b['text']; prompt = b['prompt']
    es_ok = b['es_ok'].unsqueeze(-1).to(x.dtype)            # (N,T,1)
    use_e = 'E' in cell
    use_s = 'S' in cell
    c = x[..., :3]
    s = x[..., 3:5] * es_ok if use_s else torch.zeros_like(x[..., 3:5])
    e = x[..., 5:] * es_ok if use_e else torch.zeros_like(x[..., 5:])
    x = torch.cat((c, s, e), dim=-1)
    if use_s:
        unk = torch.full_like(mode, UNKNOWN_MODE)
        ok = b['es_ok']
        mode = torch.where(ok, mode, unk); prev = torch.where(ok, prev, unk)
    else:
        mode = torch.full_like(mode, UNKNOWN_MODE); prev = torch.full_like(prev, UNKNOWN_MODE)
    if 'T' not in cell:
        text = torch.zeros_like(text); prompt = torch.zeros_like(prompt)
    return x, mode, prev, text, prompt


def native_outputs(m, h):
    outs = []
    for head, (K, E) in zip(m.ns_heads, m.ns_dims):
        z = head(h)
        outs.append(dict(gate=z[..., :3], dest=z[..., 3:3 + K], ev=z[..., 3 + K:3 + K + E]))
    for out, head, (K, E) in zip(outs, getattr(m, 'ns_g1_heads', None) or [], m.ns_dims):
        # G2: first next span (K types + natural END) and events inside it conditional on each next type
        z = head(h)
        out.update(next_span=z[..., :K + 1], span_events=z[..., K + 1:].reshape(*z.shape[:-1], K, E))
    res = {'ns': outs}
    if getattr(m, 'g4', None) is not None:
        z = m.g4(h); res.update(g4_changed=z[..., 0], g4_kind=z[..., 1:])
    for out, head, (K, E) in zip(outs, getattr(m, 'ns_g3_heads', None) or [], m.ns_dims):
        # G3: native label of the observed segment j at masked positions (K types, E events)
        z = head(h)
        out.update(g3_type=z[..., :K], g3_ev=z[..., K:K + E])
    return res


def native_group1_heads(nn, width, namespaces):
    return nn.ModuleList([nn.Linear(width, ns['K'] + 1 + ns['K'] * ns['E']) for ns in namespaces])


def g4_head(nn, width):
    """G4: one position-level head over the single input stream: changed (1 logit) + edit type (4 incl. clean)."""
    return nn.Linear(width, 5)


def native_group3_heads(nn, width, namespaces):
    return nn.ModuleList([nn.Linear(width, ns['K'] + ns['E']) for ns in namespaces])


def build(cell, nsolvers, cfg, group1=False, namespaces=None, group3=False, group4=False):
    import torch
    from torch import nn
    import torch.nn.functional as F
    if cell not in CELLS:
        raise ValueError('UNKNOWN_ARM')

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__(); w = cfg['width']; self.cell = cell
            self.mode = nn.Embedding(8, 24); self.prev = nn.Embedding(8, 16)
            self.solver = nn.Embedding(nsolvers, 16, padding_idx=0)
            self.proj = nn.Linear(873, w); self.position = nn.Embedding(cfg['max_segments'], w)
            layer = nn.TransformerEncoderLayer(w, cfg['heads'], cfg['ff_width'], dropout=cfg['dropout'],
                                               batch_first=True, norm_first=True, activation='gelu')
            self.core = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
            for lay in self.core.layers:
                for name, p in lay.named_parameters():
                    if p.ndim > 1:
                        nn.init.xavier_uniform_(p)
            self.norm = nn.LayerNorm(w)
            self.native = bool(namespaces)
            if self.native:
                # one small output head per native annotation namespace: 3 gate + K types + E events
                self.ns_heads = nn.ModuleList([nn.Linear(w, 3 + ns['K'] + ns['E']) for ns in namespaces])
                self.ns_dims = [(ns['K'], ns['E']) for ns in namespaces]
                if group1:
                    self.ns_g1_heads = native_group1_heads(nn, w, namespaces)
                if group3:
                    self.ns_g3_heads = native_group3_heads(nn, w, namespaces)
                if group4:
                    self.g4 = g4_head(nn, w)
            else:
                self.gate = nn.Linear(w, 3); self.dest = nn.Linear(w, 6); self.events = nn.Linear(w, 6)
            self.length = nn.Linear(w, 1); self.wait = nn.Linear(w, 30); self.rem = nn.Linear(w, 3)
            with torch.no_grad():
                self.wait.bias.view(3, 10)[:, 1] = 7.; self.wait.bias.view(3, 10)[:, 2] = 0.
            self.group1 = group1 and not self.native
            if self.group1:
                # next span after the current one: 6 types + natural END without a next span
                self.next_span = nn.Linear(w, N_TYPES + 1)
                # events inside the next span, conditional on each possible next-span type
                self.span_events = nn.Linear(w, N_TYPES * N_EVENTS)

        def trunk(self, b):
            # Whitelist only. No targets, benchmark, source/question/family IDs, full_n,
            # future durations, END flags or outcome fields are inputs.
            x, mode, prev, text, prompt = apply_cell_mask(torch, self.cell, b)
            bench = x.new_zeros((*x.shape[:-1], 8))
            h = self.proj(torch.cat((x, self.mode(mode), self.prev(prev), self.solver(b['solver']),
                                     bench, text, prompt), dim=-1))
            n = h.shape[1]; h = h + self.position(torch.arange(n, device=h.device))[None]
            mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=h.device), 1)
            return self.norm(self.core(h, mask=mask, src_key_padding_mask=b['pad']))

        def heads(self, h):
            p = self.wait(h).view(*h.shape[:-1], 3, 10)
            out = native_outputs(self, h) if self.native else dict(gate=self.gate(h), dest=self.dest(h), events=self.events(h))
            out.update(length=self.length(h).squeeze(-1), wait_pi=p[..., 0],
                       wait_logscale=p[..., 1].clamp(-3, 16),
                       wait_shape=(F.softplus(p[..., 2]) + .2).clamp(max=5), wait_mark=p[..., 3:],
                       rem=self.rem(h))
            if self.group1:
                out['next_span'] = self.next_span(h)
                out['span_events'] = self.span_events(h).view(*h.shape[:-1], N_TYPES, N_EVENTS)
            return out

        def forward(self, b):
            return self.heads(self.trunk(b))

    return Predictor()
