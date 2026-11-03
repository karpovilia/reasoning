"""Losses. `v9_loss_parts` is a verbatim port of loss_parts() from V9 teacher_model_v2.py.

First stage ("next structural step"):
    L = L_joint + 0.5 * L_destination + 0.25 * L_event
L_joint already contains the next-type cross-entropy on known SWITCH positions
(joint = gate_CE + dest_CE * known_switch); L_destination adds that same type CE again with
weight 0.5 on known switches; L_event is BCE of the next-segment events on known event cells.
These are three weighted terms of V9, not three independent categorical heads.
length / wait / remainder terms have weight 0 in the first stage: they are NOT added to the
graph at all, so their head parameters receive no gradient (checked in tests and at runtime).
"""
STAGE1_WEIGHTS = {'joint': 1.0, 'destination': 0.5, 'event': 0.25}
V9_FULL_WEIGHTS = {'joint': 1.0, 'destination': 0.5, 'event': 0.25, 'length': .05, 'wait': .2, 'remainder': .2}


def _avg(w, values, mask):
    weights = w.unsqueeze(-1) if values.ndim == 3 else w
    weights = weights * mask
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def v9_core_terms(o, b):
    """Per-position terms of the three first-stage parts (no reduction)."""
    import torch.nn.functional as F
    score = b['score']; valid = b['valid']; w = b['weight']
    gate = F.cross_entropy(o['gate'].reshape(-1, 3), b['gate'].clamp_min(0).reshape(-1), reduction='none').reshape_as(w)
    dest = F.cross_entropy(o['dest'].reshape(-1, 6), b['dest'].clamp(0, 5).reshape(-1), reduction='none').reshape_as(w)
    sw = score & valid['next_type']; gm = score & valid['step_outcome']; joint = gate + dest * sw
    events = F.binary_cross_entropy_with_logits(o['events'], b['event_y'], reduction='none')
    em = score[..., None] & valid['events']
    return dict(joint=(joint, gm), destination=(dest, sw), event=(events, em))


def v9_loss_parts(o, b, include_disabled=False):
    """Returns (parts dict of scalar tensors). Disabled parts only when include_disabled
    (used for monitoring under no_grad, never added to the first-stage total)."""
    import torch
    import torch.nn.functional as F
    w = b['weight']; score = b['score']; valid = b['valid']
    t = v9_core_terms(o, b)
    parts = {k: _avg(w, v, m) for k, (v, m) in t.items()}
    if include_disabled:
        length = F.smooth_l1_loss(o['length'], b['length_y'], reduction='none')
        ty = b['wait_y']; logt = ty.clamp_min(1e-5).log().unsqueeze(-1); scale = o['wait_logscale']; shape = o['wait_shape']
        power = (shape * (logt - scale)).clamp(max=20).exp(); lp = F.log_softmax(o['wait_pi'], -1)
        ld = shape.log() - scale + (shape - 1) * (logt - scale) - power; mark = b['next_span'].clamp(0, 6)
        lm = F.log_softmax(o['wait_mark'], -1).gather(-1, mark[..., None, None].expand(*mark.shape, 3, 1)).squeeze(-1)
        wait = -torch.where(b['wait_observed'], torch.logsumexp(lp + ld + lm, -1), torch.logsumexp(lp - power, -1))
        wm = score & (valid['wait_observed'] | valid['wait_censored'])
        rem = (o['rem'] - torch.log1p(b['rem'])).abs()
        parts.update(length=_avg(w, length, score & valid['length']), wait=_avg(w, wait, wm),
                     remainder=_avg(w, rem, score[..., None] & valid['remainder']))
    return parts


def group1_terms(o, b):
    """Event -> next span -> events inside it. Targets come only from the loss inputs.
    next_span target: 0..5 type of the first span after the current one, 6 = natural END without
    a next span, -1 unknown. span_ev_y/span_ev_known: events inside that next span; positives may be
    known while absences stay unknown for partially observed spans."""
    import torch
    import torch.nn.functional as F
    score = b['score']
    ns = b['g1_next_span']; nsm = score & (ns >= 0)
    ce = F.cross_entropy(o['next_span'].reshape(-1, 7), ns.clamp(0, 6).reshape(-1), reduction='none').reshape_as(ns).float()
    has_next = nsm & (ns < 6)
    logits = o['span_events']                                    # (N,T,6types,6events)
    idx = ns.clamp(0, 5)[..., None, None].expand(*ns.shape, 1, logits.shape[-1])
    cond = logits.gather(-2, idx).squeeze(-2)                    # conditional on the TRUE next type: loss only
    bce = F.binary_cross_entropy_with_logits(cond, b['g1_ev_y'], reduction='none')
    evm = has_next[..., None] & b['g1_ev_known']
    return dict(next_span=(ce, nsm), span_events=(bce, evm))


def group1_marginal_event_probs(o):
    """Inference: P(event a in next span) = sum_t P(next=t) * sigmoid(logit[t,a]); END has no events.
    No hard cascade and no true future type enters."""
    import torch
    p = torch.softmax(o['next_span'], -1)
    return (p[..., :6, None] * torch.sigmoid(o['span_events'])).sum(-2)


def total_loss(o, b, task_weights):
    """task_weights: dict of term -> weight, e.g. STAGE1_WEIGHTS or with 'next_span'/'span_events'."""
    w = b['weight']
    terms = v9_core_terms(o, b)
    if any(k in task_weights for k in ('next_span', 'span_events')):
        terms.update(group1_terms(o, b))
    parts = {k: _avg(w, v, m) for k, (v, m) in terms.items()}
    total = sum(task_weights[k] * parts[k] for k in task_weights)
    return total, parts, terms


# ---------------------------------------------------------------------------------------------
# Native annotation namespaces (COORDINATOR_NATIVE_TARGET_NAMESPACES_DECISION_V2)
NATIVE_COEF = {'joint': 1.0, 'destination': 0.5, 'event': 0.25}
# G2 (event -> first next span -> events inside it): weights mirror the next-step type (0.5) and events (0.25) terms
G1_COEF = {'next_span': 0.5, 'span_events': 0.25}


def native_group1_terms(out, t, score, event_ok):
    """G2 per namespace: CE over K types + natural END of the first next span; BCE of the events inside it, read from
    the logits conditional on the TRUE next type (loss only), mean over known event cells; END/unknown masked."""
    import torch.nn.functional as F
    K = out['dest'].shape[-1]
    g1n = t['g1n']
    ns_ce = F.cross_entropy(out['next_span'].reshape(-1, K + 1), g1n.clamp(0, K).reshape(-1), reduction='none').reshape(score.shape)
    Ins = score & (g1n >= 0)
    E = out['span_events'].shape[-1]
    idx = g1n.clamp(0, K - 1)[..., None, None].expand(*g1n.shape, 1, E)
    cond = out['span_events'].gather(-2, idx).squeeze(-2)
    bce = F.binary_cross_entropy_with_logits(cond, t['g1e'], reduction='none')
    kn = (t['g1k'] & (Ins & (g1n < K))[..., None]).float(); cnt = kn.sum(-1)
    Ise = (cnt > 0) & bool(event_ok)
    se = (bce * kn).sum(-1) / cnt.clamp_min(1)
    return ns_ce, Ins, se, Ise


G3_COEF = {'g3_type': 0.5, 'g3_events': 0.25}


def native_group3_terms(out, t, score, event_ok):
    """G3 per namespace at masked positions: CE of the native type of segment j, BCE of its events over known cells."""
    import torch.nn.functional as F
    K = out['g3_type'].shape[-1]
    m = t['g3m']
    ce = F.cross_entropy(out['g3_type'].reshape(-1, K), m.clamp(0, K - 1).reshape(-1), reduction='none').reshape(score.shape)
    It = score & (m >= 0)
    bce = F.binary_cross_entropy_with_logits(out['g3_ev'], t['g3e'], reduction='none')
    kn = (t['g3k'] & score[..., None]).float(); cnt = kn.sum(-1)
    Ie = (cnt > 0) & bool(event_ok)
    ev = (bce * kn).sum(-1) / cnt.clamp_min(1)
    return ce, It, ev, Ie


G4_COEF = {'g4_changed': 0.25, 'g4_kind': 0.25}


def g4_terms(o, b):
    """G4 per position (one input stream): BCE of 'this position's observed input label was changed' and CE of the edit
    type (0 clean, 1 type, 2 boundary, 3 event); scored only where the clean input label is known (g4b >= 0)."""
    import torch.nn.functional as F
    score = b['score']; y = b['g4b']; k = b['g4k']
    Ib = score & (y >= 0)
    bce = F.binary_cross_entropy_with_logits(o['g4_changed'], y.clamp_min(0).float(), reduction='none')
    ce = F.cross_entropy(o['g4_kind'].reshape(-1, 4), k.clamp(0, 3).reshape(-1), reduction='none').reshape(score.shape)
    return bce, ce, Ib


def comp_name(i, c):
    return 'g4' if '_g4' in c else f'ns{i}'


def native_group1_marginal_event_probs(out):
    """Inference: P(event a in the next span) = sum_t P(next=t) sigmoid(logit[t,a]); END has no events."""
    import torch
    K = out['dest'].shape[-1]
    p = torch.softmax(out['next_span'], -1)
    return (p[..., :K, None] * torch.sigmoid(out['span_events'])).sum(-2)


def native_position_terms(o, b, accepted, event_eligible=None, group1=True, group3=False, group4=False):
    """Returns per-namespace component tensors and the per-position combined loss.
    L[n,p] = I_joint*L_joint + 0.5*I_dest*L_dest + 0.25*I_ev*mean_known_axes(BCE)   (no division by the
    number of enabled components); L_p = mean over namespaces available at p (accepted namespaces with
    at least one true mask); eligible_p = any namespace available. Unknown terms contribute 0."""
    import torch
    import torch.nn.functional as F
    score = b['score']
    Ls = []; avails = []; comps = []
    for i, (t, out) in enumerate(zip(b['ns'], o['ns'])):
        K = out['dest'].shape[-1]
        gate = F.cross_entropy(out['gate'].reshape(-1, 3), t['gate'].clamp_min(0).reshape(-1), reduction='none').reshape(score.shape)
        dest = F.cross_entropy(out['dest'].reshape(-1, K), t['dest'].clamp(0, K - 1).reshape(-1), reduction='none').reshape(score.shape)
        bce = F.binary_cross_entropy_with_logits(out['ev'], t['ev'], reduction='none')
        Ij = score & t['vstep']; Id = score & t['vnt']
        kn = (t['vev'] & score[..., None]).float(); cnt = kn.sum(-1)
        # event term only for namespaces with certified absence (namespace_policy); exactly 0 otherwise, no empty mean
        Ie = (cnt > 0) & bool(True if event_eligible is None else event_eligible[i])
        evm = (bce * kn).sum(-1) / cnt.clamp_min(1)
        joint = gate + dest * Id
        L = Ij * joint + 0.5 * Id * dest + 0.25 * Ie * evm
        av = Ij | Id | Ie
        comp = dict(joint=(joint, Ij), destination=(dest, Id), event=(evm, Ie))
        if group1 and 'next_span' in out and 'g1n' in t:
            ns_ce, Ins, se, Ise = native_group1_terms(out, t, score, True if event_eligible is None else event_eligible[i])
            L = L + G1_COEF['next_span'] * Ins * ns_ce + G1_COEF['span_events'] * Ise * se
            av = av | Ins | Ise
            comp.update(next_span=(ns_ce, Ins), span_events=(se, Ise))
        if group3 and 'g3_type' in out and 'g3m' in t:
            ce3, It3, ev3, Ie3 = native_group3_terms(out, t, score, True if event_eligible is None else event_eligible[i])
            L = L + G3_COEF['g3_type'] * It3 * ce3 + G3_COEF['g3_events'] * Ie3 * ev3
            av = av | It3 | Ie3
            comp.update(g3_type=(ce3, It3), g3_events=(ev3, Ie3))
        av = av & bool(accepted[i])
        Ls.append(L * av); avails.append(av)
        comps.append(comp)
    if not Ls:
        raise ValueError('NO_NAMESPACES')
    A = torch.stack(avails).float(); Lsum = torch.stack(Ls).sum(0)
    navail = A.sum(0)
    Lp = Lsum / navail.clamp_min(1)
    elig = navail > 0
    if group4 and 'g4_changed' in o and 'g4b' in b:
        # G4 is position-level (one input stream), added once per position on top of the namespace mean
        bce4, ce4, Ib = g4_terms(o, b)
        Lp = Lp + Ib * (G4_COEF['g4_changed'] * bce4 + G4_COEF['g4_kind'] * ce4)
        elig = elig | Ib
        comps = comps + [dict(_g4=True, g4_changed=(bce4, Ib), g4_kind=(ce4, Ib))]
    return Lp, elig, comps


def native_eligible_weight(b, accepted, event_eligible=None, group1=True, group3=False, group4=False):
    """Sum of position weights that carry at least one accepted native target (computable from targets only,
    before the forward pass) - the common denominator of one effective batch."""
    import torch
    score = b['score']; e = None
    for i, t in enumerate(b['ns']):
        if not accepted[i]:
            continue
        ev_ok = True if event_eligible is None else bool(event_eligible[i])
        a = score & (t['vstep'] | t['vnt'] | (t['vev'].any(-1) if ev_ok else torch.zeros_like(t['vstep'])))
        if group1 and 'g1n' in t:
            a = a | (score & (t['g1n'] >= 0))        # next span known; its event cells are a subset of these positions
        if group3 and 'g3m' in t:
            a = a | (score & (t['g3m'] >= 0)) | (score & t['g3k'].any(-1) & ev_ok)
        e = a if e is None else (e | a)
    if group4 and 'g4b' in b and e is not None:
        e = e | (score & (b['g4b'] >= 0))
    return float((b['weight'] * e).sum()) if e is not None else 0.0


def native_total_loss(o, b, accepted, denominator=None, event_eligible=None, group1=True, group3=False, group4=False):
    """denominator=None: normalise inside this batch. Otherwise sum(w*L)/denominator, where the caller passes the
    eligible weight of the WHOLE effective batch, so micro-batch accumulation equals one big batch."""
    Lp, elig, comps = native_position_terms(o, b, accepted, event_eligible, group1, group3, group4)
    w = b['weight'] * elig
    den = w.sum().clamp_min(1e-12) if denominator is None else max(float(denominator), 1e-12)
    total = (Lp * w).sum() / den
    parts = {'native_total': total.detach()}
    for i, c in enumerate(comps):
        for k, vm in c.items():
            if k == '_g4':
                continue
            v, m = vm; mw = m.float() * b['weight']
            parts[f'{comp_name(i, c)}_{k}' if '_g4' not in c else k] = ((v * mw).sum() / mw.sum().clamp_min(1e-12)).detach()
    return total, parts
