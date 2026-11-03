"""Reproducible DEV predictions and metrics from a fit's best.pt (never reads TEST: the pack has no TEST).

Outputs in OUT/predictions_dev/:
  predictions.npz   per scored DEV position and namespace: unit index, position, gate probs (stay/switch/END),
                    type probs, event probs, targets and masks
  METRICS.json      per namespace: gate NLL/accuracy/Brier overall and by true class STAY/SWITCH/END,
                    next type NLL/accuracy on known switches, event BCE and per-axis AUC; the same by
                    solver, benchmark and position bucket with question/trace/position supports
"""
import argparse, collections, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain import runner  # noqa: E402
from ptrain import losses as L  # noqa: E402

BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 10 ** 9)]


def auc(y, p):
    import numpy as np
    y = np.asarray(y); p = np.asarray(p)
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p, kind='mergesort'); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarize(rows, K):
    import numpy as np
    out = {}
    if not rows['gate_y']:
        return out
    gy = np.array(rows['gate_y']); gp = np.array(rows['gate_p'])
    q = rows['q']; tr = rows['u']
    m = gy >= 0
    if m.any():
        p_true = gp[m, gy[m]]
        out['gate'] = dict(nll=float(-np.log(np.clip(p_true, 1e-12, 1)).mean()), acc=float((gp[m].argmax(1) == gy[m]).mean()),
                           brier=float(((gp[m] - np.eye(3)[gy[m]]) ** 2).sum(1).mean()), positions=int(m.sum()),
                           questions=len({q[i] for i in np.where(m)[0]}), traces=len({tr[i] for i in np.where(m)[0]}))
        for c, name in enumerate(('STAY', 'SWITCH', 'END')):
            mc = gy == c
            if mc.any():
                out['gate_' + name] = dict(nll=float(-np.log(np.clip(gp[mc, c], 1e-12, 1)).mean()), recall_argmax=float((gp[mc].argmax(1) == c).mean()),
                                           mean_prob=float(gp[mc, c].mean()), positions=int(mc.sum()))
    dy = np.array(rows['dest_y'])
    if len(dy) and (dy >= 0).any():
        dp = np.array(rows['dest_p']); md = dy >= 0
        out['next_type'] = dict(nll=float(-np.log(np.clip(dp[md, dy[md]], 1e-12, 1)).mean()), acc=float((dp[md].argmax(1) == dy[md]).mean()),
                                positions=int(md.sum()))
    ey = np.array(rows['ev_y']); ek = np.array(rows['ev_k']); ep = np.array(rows['ev_p'])
    if ek.size and ek.any():
        bce = -(ey * np.log(np.clip(ep, 1e-12, 1)) + (1 - ey) * np.log(np.clip(1 - ep, 1e-12, 1)))
        out['events'] = dict(bce=float(bce[ek].mean()), cells=int(ek.sum()),
                             auc_per_axis=[auc(ey[ek[:, a], a], ep[ek[:, a], a]) for a in range(ek.shape[1])],
                             positives_per_axis=[int(ey[ek[:, a], a].sum()) for a in range(ek.shape[1])])
    if rows.get('ns_y'):
        ny = np.array(rows['ns_y']); npr = np.array(rows['ns_p']); mn = ny >= 0
        if mn.any():
            out['next_span'] = dict(nll=float(-np.log(np.clip(npr[mn, ny[mn]], 1e-12, 1)).mean()), acc=float((npr[mn].argmax(1) == ny[mn]).mean()),
                                    positions=int(mn.sum()), end_class=K,
                                    by_true_class={str(c): dict(positions=int((ny == c).sum()), recall_argmax=float((npr[ny == c].argmax(1) == c).mean()))
                                                   for c in range(K + 1) if (ny == c).any()})
        sy = np.array(rows['se_y']); sk = np.array(rows['se_k']); sp = np.array(rows['se_p'])
        if sk.size and sk.any():
            bce = -(sy * np.log(np.clip(sp, 1e-12, 1)) + (1 - sy) * np.log(np.clip(1 - sp, 1e-12, 1)))
            out['span_events_marginal'] = dict(bce=float(bce[sk].mean()), cells=int(sk.sum()),
                                               auc_per_axis=[auc(sy[sk[:, a], a], sp[sk[:, a], a]) for a in range(sk.shape[1])],
                                               positives_per_axis=[int(sy[sk[:, a], a].sum()) for a in range(sk.shape[1])])
    return out


def average_precision(y, p):
    """PTV6-002: AP over distinct score thresholds (ties form one threshold), sum_k (R_k - R_{k-1}) * P_k; invariant to
    the order of rows with equal scores. None without positives."""
    import numpy as np
    y = np.asarray(y, float); p = np.asarray(p, float)
    if not len(y) or y.sum() == 0:
        return None
    thr, inv = np.unique(-p, return_inverse=True)               # ascending -p = descending score
    tp = np.bincount(inv, weights=y, minlength=len(thr)); cnt = np.bincount(inv, minlength=len(thr))
    TP = np.cumsum(tp); N = np.cumsum(cnt); P = TP / N; Rc = TP / y.sum()
    return float(np.sum(np.diff(np.concatenate(([0.0], Rc))) * P))


def summarize_g4(r):
    """G4 on the fixed corrupted DEV: ranking (AUROC, average precision) of 'changed', precision/recall per edit type of the
    4-class head, prevalence and support, false-positive rate (p >= 0.5) on positions of unchanged control traces."""
    import numpy as np
    y = np.array(r['y']); p = np.array(r['p']); k = np.array(r['k']); pk = np.array(r['pk']); ctrl = np.array(r['ctrl'])
    out = dict(positions=int(len(y)), changed=int(y.sum()), prevalence=float(y.mean()) if len(y) else None, traces=len(set(r['u'])))
    out['auroc'] = auc(y, p)
    out['average_precision'] = average_precision(y, p)
    pred = pk.argmax(1) if len(pk) else np.zeros(0, int)
    out['by_edit_type'] = {name: dict(support=int((k == c).sum()), recall=float((pred[k == c] == c).mean()) if (k == c).any() else None,
                                      precision=float((k[pred == c] == c).mean()) if (pred == c).any() else None)
                           for c, name in ((1, 'TYPE'), (2, 'BOUNDARY'), (3, 'EVENT'))}
    if ctrl.any():
        out['control_false_positive_rate_p05'] = float((p[ctrl] >= 0.5).mean()); out['control_positions'] = int(ctrl.sum())
    return out


def summarize_g3(rows, K, ev_ok):
    """G3 on the fixed diagnostic DEV mask: native type of the masked observed segment and its events."""
    import numpy as np
    from ptrain import namespace_policy
    out = dict(positions=len(rows['y']), questions=len(set(rows['q'])), traces=len(set(rows['u'])))
    y = np.array(rows['y']); p = np.array(rows['p']); m = y >= 0
    if m.any():
        pred = p[m].argmax(1); yy = y[m]
        rec = {str(c): float((pred[yy == c] == c).mean()) for c in range(K) if (yy == c).any()}
        out['type'] = dict(nll=float(-np.log(np.clip(p[m, yy], 1e-12, 1)).mean()), acc=float((pred == yy).mean()),
                           macro_recall=float(np.mean(list(rec.values()))), recall_by_class=rec, known=int(m.sum()))
    ey = np.array(rows['ey']); ek = np.array(rows['ek']); ep = np.array(rows['ep'])
    if not ev_ok:
        out['events'] = namespace_policy.NOT_TRAINED
    elif ek.size and ek.any():
        bce = -(ey * np.log(np.clip(ep, 1e-12, 1)) + (1 - ey) * np.log(np.clip(1 - ep, 1e-12, 1)))
        out['events'] = dict(bce=float(bce[ek].mean()), cells=int(ek.sum()), auc_per_axis=[auc(ey[ek[:, a], a], ep[ek[:, a], a]) for a in range(ek.shape[1])],
                             positives_per_axis=[int(ey[ek[:, a], a].sum()) for a in range(ek.shape[1])])
    return out


def main(argv=None):
    import numpy as np
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--which', default='best.pt')
    a = ap.parse_args(argv)
    run = Path(a.run); binding = json.loads((run / 'RUN_BINDING.json').read_text()); cfg = binding['config']
    # PTREV-006: predictions only from the exact code/pack/checkpoint the fit was trained and completed with
    if binding['code'] != runner.code_hashes():
        raise ValueError('PREDICT_CODE_DIFFERS_FROM_RUN_BINDING')
    if runner.sha_file(Path(binding['pack']) / 'PACK_MANIFEST.json') != binding['pack_manifest_sha256']:
        raise ValueError('PREDICT_PACK_DIFFERS_FROM_RUN_BINDING')
    done = json.loads((run / 'COMPLETE.json').read_text())
    if done.get('binding_sha256') != runner.sha_file(run / 'RUN_BINDING.json'):
        raise ValueError('PREDICT_COMPLETE_BINDING_MISMATCH')
    want = done.get('best_sha256') if a.which == 'best.pt' else done.get('final_sha256') if a.which == 'final.pt' else None
    if want is None or runner.sha_file(run / a.which) != want:
        raise ValueError('PREDICT_CHECKPOINT_NOT_BOUND_BY_COMPLETE')
    ad = runner.Adapter(cfg, binding['pack'], a.device)
    torch.manual_seed(binding['seed'])
    model, _ = ad.build(torch)
    payload = torch.load(run / a.which, map_location='cpu', weights_only=False)
    ad.load_trainable(model, payload['model']); model.eval()
    data = ad.data; nss = data.namespaces
    ids = ad.ids('DEV'); mb = cfg.get('micro_batch', 64)
    store = [collections.defaultdict(list) for _ in nss]
    groups = [collections.defaultdict(lambda: collections.defaultdict(list)) for _ in nss]
    ns_k = [ns['K'] for ns in nss]
    with torch.no_grad():
        for s in range(0, len(ids), mb):
            b = ad.batch(ids[s:s + mb]); o = model(b)
            for j, (t, out) in enumerate(zip(b['ns'], o['ns'])):
                gp = torch.softmax(out['gate'], -1).cpu().numpy(); dp = torch.softmax(out['dest'], -1).cpu().numpy()
                ep = torch.sigmoid(out['ev']).cpu().numpy()
                g2 = 'next_span' in out and 'g1n' in t
                if g2:   # marginal over the predicted next type: no true future type enters the published probability
                    nsp = torch.softmax(out['next_span'], -1).cpu().numpy(); sep = L.native_group1_marginal_event_probs(out).cpu().numpy()
                sc = b['score'].cpu().numpy()
                tt = {k: v.cpu().numpy() for k, v in t.items()}
                for i in range(sc.shape[0]):
                    ui = data.windows[ids[s + i]][0]; u = data.units[ui]; lo = b['_positions'][i][0]
                    for pos in np.where(sc[i])[0]:
                        if not (tt['vstep'][i, pos] or tt['vnt'][i, pos] or tt['vev'][i, pos].any() or (g2 and tt['g1n'][i, pos] >= 0)):
                            continue
                        absp = lo + int(pos)
                        rec = dict(u=ui, q=u['question_component'], pos=absp, gate_y=int(tt['gate'][i, pos]) if tt['vstep'][i, pos] else -1,
                                   gate_p=gp[i, pos], dest_y=int(tt['dest'][i, pos]) if tt['vnt'][i, pos] else -1, dest_p=dp[i, pos],
                                   ev_y=tt['ev'][i, pos], ev_k=tt['vev'][i, pos], ev_p=ep[i, pos])
                        if g2:
                            gn = int(tt['g1n'][i, pos])
                            rec.update(ns_y=gn, ns_p=nsp[i, pos], se_y=tt['g1e'][i, pos], se_k=tt['g1k'][i, pos] & (0 <= gn < ns_k[j]), se_p=sep[i, pos])
                        keys = ['all', 'solver=' + str(u.get('solver_ID')), 'benchmark=' + str(u.get('benchmark_audit_only'))]
                        keys.append('position=%d-%d' % next(bk for bk in BUCKETS if bk[0] <= absp < bk[1]))
                        for k, v in rec.items():
                            store[j][k].append(v)
                            for g in keys:
                                groups[j][g][k].append(v)
    outdir = run / ('predictions_dev' if a.which == 'best.pt' else 'predictions_dev_' + a.which.replace('.pt', ''))
    outdir.mkdir(exist_ok=True)
    arrays = {}
    metrics = dict(run=str(run), checkpoint=a.which, checkpoint_sha256=runner.sha_file(run / a.which),
                   run_binding_sha256=runner.sha_file(run / 'RUN_BINDING.json'), pack_manifest_sha256=binding['pack_manifest_sha256'],
                   code=binding['code'], config_sha256=binding['config_sha256'], seed=binding['seed'], namespaces={},
                   es_input_source=json.loads((Path(binding['pack']) / 'PACK_MANIFEST.json').read_text()).get('es_input_source'))
    for j, ns in enumerate(nss):
        st = store[j]
        if not st['u']:
            continue
        ev_ok_j = ns.get('event_supervision_eligible', True)
        for k in ('u', 'pos', 'gate_y', 'gate_p', 'dest_y', 'dest_p', 'ev_y', 'ev_k', 'ev_p', 'ns_y', 'ns_p', 'se_y', 'se_k', 'se_p'):
            if k not in st:
                continue
            if k in ('ev_p', 'se_p') and not ev_ok_j:
                continue                                          # do not publish untrained event outputs
            arrays[f'ns{j}_{k}'] = np.array(st[k])
        from ptrain import namespace_policy
        ev_ok = ns.get('event_supervision_eligible', namespace_policy.event_supervision_eligible(ns['id']))
        grp = {}
        for g, rows in sorted(groups[j].items()):
            sm = summarize(rows, ns['K'])
            if not ev_ok:
                sm['events'] = namespace_policy.NOT_TRAINED       # no event-quality claim for untrained heads
                if 'span_events_marginal' in sm:
                    sm['span_events_marginal'] = namespace_policy.NOT_TRAINED
            grp[g] = dict(sm, questions=len(set(rows['q'])), traces=len(set(rows['u'])))
        metrics['namespaces'][ns['id']] = dict(status=ns['status'], event_supervision_eligible=bool(ev_ok), groups=grp)
    if data.has_diag and any(ns.get('group3') for ns in nss) and 'g3_type' in (o['ns'][0] if nss else {}):
        # G3 metrics only on the fixed diagnostic DEV mask (masked inputs), never on natural DEV inputs
        g3 = [collections.defaultdict(lambda: collections.defaultdict(list)) for _ in nss]
        data.use_inputs('diag')
        try:
            with torch.no_grad():
                for s in range(0, len(ids), mb):
                    b = ad.batch(ids[s:s + mb]); o = model(b); sc = b['score'].cpu().numpy()
                    for j, (t, out) in enumerate(zip(b['ns'], o['ns'])):
                        tp = torch.softmax(out['g3_type'], -1).cpu().numpy(); ep3 = torch.sigmoid(out['g3_ev']).cpu().numpy()
                        tt = {k: t[k].cpu().numpy() for k in ('g3m', 'g3e', 'g3k')}
                        for i in range(sc.shape[0]):
                            ui = data.windows[ids[s + i]][0]; u = data.units[ui]; lo = b['_positions'][i][0]
                            for pos in np.where(sc[i] & ((tt['g3m'][i] >= 0) | tt['g3k'][i].any(-1)))[0]:
                                absp = lo + int(pos)
                                rec = dict(u=ui, q=u['question_component'], y=int(tt['g3m'][i, pos]), p=tp[i, pos], ey=tt['g3e'][i, pos], ek=tt['g3k'][i, pos], ep=ep3[i, pos])
                                keys = ['all', 'solver=' + str(u.get('solver_ID')), 'benchmark=' + str(u.get('benchmark_audit_only'))]
                                keys.append('position=%d-%d' % next(bk for bk in BUCKETS if bk[0] <= absp < bk[1]))
                                for g in keys:
                                    for k, v in rec.items():
                                        g3[j][g][k].append(v)
        finally:
            data.use_inputs(None)
        from ptrain import namespace_policy
        for j, ns in enumerate(nss):
            res = {}
            for g, rows in sorted(g3[j].items()):
                res[g] = summarize_g3(rows, ns['K'], ns.get('event_supervision_eligible', namespace_policy.event_supervision_eligible(ns['id'])))
            if res and ns['id'] in metrics['namespaces']:
                metrics['namespaces'][ns['id']]['g3_diag_dev'] = res
    if data.has_g4 and data.has_diag and 'g4_changed' in o:
        # G4 only on the fixed corrupted DEV (diag_* inputs); controls = DEV traces left unchanged by the plan
        aff = {}
        with open(Path(binding['pack']) / 'G4_EDIT_PLAN.jsonl') as f:
            for line in f:
                r = json.loads(line); aff[r['uid']] = bool(r.get('affected') and r.get('ops'))
        rows = collections.defaultdict(lambda: collections.defaultdict(list))
        data.use_inputs('diag')
        try:
            with torch.no_grad():
                for s in range(0, len(ids), mb):
                    b = ad.batch(ids[s:s + mb]); o4 = model(b); sc = b['score'].cpu().numpy()
                    pc = torch.sigmoid(o4['g4_changed']).cpu().numpy(); pk = torch.softmax(o4['g4_kind'], -1).cpu().numpy()
                    yb = b['g4b'].cpu().numpy(); yk = b['g4k'].cpu().numpy()
                    for i in range(sc.shape[0]):
                        ui = data.windows[ids[s + i]][0]; u = data.units[ui]
                        for pos in np.where(sc[i] & (yb[i] >= 0))[0]:
                            for g in ('all', 'solver=' + str(u.get('solver_ID')), 'benchmark=' + str(u.get('benchmark_audit_only'))):
                                rr = rows[g]; rr['y'].append(int(yb[i, pos])); rr['k'].append(int(yk[i, pos])); rr['p'].append(float(pc[i, pos]))
                                rr['pk'].append(pk[i, pos]); rr['ctrl'].append(not aff.get(u['physical_id'], False)); rr['u'].append(ui)
        finally:
            data.use_inputs(None)
        metrics['g4_diag_dev'] = {g: summarize_g4(r) for g, r in sorted(rows.items())}
    np.savez_compressed(outdir / 'predictions.npz', **arrays)
    (outdir / 'METRICS.json').write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print('PREDICT_DONE', outdir)


if __name__ == '__main__':
    main()
