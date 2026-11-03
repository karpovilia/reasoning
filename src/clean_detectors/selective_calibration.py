"""Deterministic joint two-route tuning; no test-set selection or label routing."""
from dataclasses import asdict
import numpy as np
from .selective import BinaryPolicy, any_event_target, binary_report, validation_role
from .dataset import Dataset


class RoleDataset(Dataset):
    def __init__(self, frozen, role):
        super().__init__(frozen, 'validation')
        self.ids = [sid for sid, q in self.db.execute(
            "SELECT sample_id,question_key FROM samples WHERE split='validation' ORDER BY sample_id")
            if validation_role(q, 'validation') == role]
        self.split = 'validation:' + role
        if not self.ids: raise ValueError('Empty validation role: ' + role)


def decisions(records, policy):
    policy = BinaryPolicy(**policy)
    return [policy.decide(r['absence_probability'], r['presence_probability']) for r in records]


def tune(records, cfg):
    """Exhaustive frozen quantile grid via 2-D histograms, including conflicts.

    Every candidate is evaluated on its actual non-conflicting routed subset.
    Unknown outcomes never enter known precision/support; coverage optimization
    counts known accepted cases. Disabling either or both routes is a candidate.
    """
    a = np.asarray([r['absence_probability'] for r in records])
    p = np.asarray([r['presence_probability'] for r in records])
    y = np.asarray([-1 if (t := any_event_target(r['event_targets'], r['event_mask'])) is None
                    else t for r in records])
    finite = np.isfinite(a) & np.isfinite(p) & (a >= 0) & (a <= 1) & (p >= 0) & (p <= 1)
    if not finite.any():
        return {'policy': asdict(BinaryPolicy()), 'score': -1e30, 'known_accepted': 0}
    qa = np.unique(np.quantile(a[finite], np.linspace(0, 1, cfg['grid_quantiles'])))
    qp = np.unique(np.quantile(p[finite], np.linspace(0, 1, cfg['grid_quantiles'])))
    ia = np.searchsorted(qa, a, side='left')
    ip = np.searchsorted(qp, p, side='right')
    def counts(mask):
        h = np.zeros((len(qa)+1, len(qp)+1), dtype=np.int64)
        take = mask & finite
        np.add.at(h, (ia[take], ip[take]), 1)
        prefix = np.pad(h.cumsum(0).cumsum(1), ((1,0), (1,0)))
        absent = prefix[:-1, 1:]
        present = h.sum() - prefix[:-1, -1, None] - prefix[-1, None, 1:] + absent
        return absent, present
    known_no, known_yes = counts(y >= 0)
    correct_no, _ = counts(y == 0)
    _, correct_yes = counts(y == 1)
    support_no = np.zeros_like(known_no); support_yes = np.zeros_like(known_yes)
    questions = np.asarray([r['question_key'] for r in records])
    for question in np.unique(questions):
        n, yes = counts((questions == question) & (y >= 0))
        support_no += n > 0; support_yes += yes > 0
    valid_no = ((known_no >= cfg['min_known']) & (support_no >= cfg['min_questions']) &
                (correct_no >= cfg['target_precision'] * known_no))
    valid_yes = ((known_yes >= cfg['min_known']) & (support_yes >= cfg['min_questions']) &
                 (correct_yes >= cfg['target_precision'] * known_yes))
    valid_no[0, :] = True; valid_yes[:, -1] = True
    coverage = np.where(valid_no & valid_yes, known_no + known_yes, -1)
    i, j = np.unravel_index(int(coverage.argmax()), coverage.shape)
    policy = BinaryPolicy(None if i == 0 else float(qa[i-1]), None if j == len(qp) else float(qp[j]))
    accepted = int(coverage[i,j])
    eligible = finite & (y >= 0)
    loss = 0.
    if eligible.any():
        for scores in (a[eligible], p[eligible]):
            scores = np.clip(scores, 1e-7, 1-1e-7); targets = y[eligible]
            loss += float(-(targets*np.log(scores)+(1-targets)*np.log1p(-scores)).mean()) / 2
    else: loss = 1e30
    return {'policy': asdict(policy), 'known_accepted': accepted,
            'score': accepted/max(1, int((y >= 0).sum())) if accepted else -loss,
            'diagnostic_binary_loss': loss, 'grid_shape': list(coverage.shape),
            'selection_scope': 'known binary outputs; not complete annotations'}


def report(records, policy, cfg):
    ds = decisions(records, policy)
    result = binary_report(ds, records, cfg['target_precision'], cfg['min_known'], cfg['min_questions'])
    # Resample questions, not correlated segments. These are descriptive intervals.
    questions = sorted({r['question_key'] for r in records})
    qi = {q:i for i,q in enumerate(questions)}
    counts = np.zeros((len(questions), 4), dtype=np.int64)
    for r,d in zip(records, ds, strict=True):
        y = any_event_target(r['event_targets'], r['event_mask'])
        if y is None or d not in ('absent','present'): continue
        offset = 0 if d == 'absent' else 2
        counts[qi[r['question_key']], offset] += 1
        counts[qi[r['question_key']], offset+1] += int(y == (d == 'present'))
    rng = np.random.default_rng(20261007)
    resampled = [[], []]
    if questions:
        for _ in range(cfg['bootstrap_replicates']):
            sampled = counts[rng.integers(0,len(questions),len(questions))].sum(0)
            for b in range(2):
                if sampled[2*b]: resampled[b].append(sampled[2*b+1]/sampled[2*b])
    for name, values in zip(('absent','present'),resampled):
        result[name]['question_bootstrap_95_interval'] = np.quantile(values,[.025,.975]).tolist() if values else None
        result[name]['zero_error_interval_is_not_a_guarantee'] = result[name]['errors'] == 0
    result['scope'] = 'empirical known-target quality; no distribution-free guarantee'
    return result


def audit_policy(records, tuned, cfg):
    metrics = report(records, tuned, cfg)
    required = [name for name,key in (('absent','absence_max'),('present','presence_min')) if tuned[key] is not None]
    passed = bool(required) and all(metrics[name]['empirical_and_support_pass'] for name in required)
    # Disabling just one conflicting route changes the other's population. Reject
    # the whole policy instead of claiming an unaudited new policy passed.
    return {'candidate': tuned, 'enabled': tuned if passed else asdict(BinaryPolicy()),
            'passed': passed, 'metrics': metrics, 'failure_action': 'all routes disabled; full LoRA fallback'}


def full_candidates(records, binary_policy):
    routes = decisions(records, binary_policy)
    result = []
    for r,route in zip(records,routes,strict=True):
        ep = np.asarray(r['event_probabilities']); mp = np.asarray(r['mode_probabilities'])
        events = np.zeros(6,dtype=bool) if route == 'absent' else ep >= .5
        valid = route in ('absent','present') and (route != 'present' or events.any())
        mode = int(mp.argmax())
        score = float(mp.max())
        # Confidence in the full event vector is its weakest marginal decision.
        if route == 'present': score = min(float(mp.max()), float(np.maximum(ep,1-ep).min()))
        known = bool(all(r['event_mask']) and r['mode_mask'])
        correct = known and np.array_equal(events,np.asarray(r['event_targets'],dtype=bool)) and mode == r['mode_target']
        result.append({'route':route,'valid':bool(valid),'score':score,'known':known,'correct':bool(correct),
                       'events':events.tolist(),'mode':mode,'question_key':r['question_key']})
    return result


def tune_full(records, binary_policy, cfg):
    data = full_candidates(records,binary_policy); result = {}
    for route in ('absent','present'):
        candidates = [r for r in data if r['route']==route and r['valid']]
        result[route] = None; best = 0
        if not candidates: continue
        for threshold in np.unique(np.quantile([r['score'] for r in candidates],np.linspace(0,1,cfg['grid_quantiles']))):
            accepted = [r for r in candidates if r['score'] >= threshold and r['known']]
            n = len(accepted); nq = len({r['question_key'] for r in accepted})
            if n >= cfg['min_known'] and nq >= cfg['min_questions'] and sum(r['correct'] for r in accepted) >= cfg['full_bypass_target']*n and n > best:
                result[route] = float(threshold);best = n
    return result


def full_report(records, binary_policy, thresholds, cfg):
    data = full_candidates(records,binary_policy); selected=[]; branches={}
    for route in ('absent','present'):
        threshold=thresholds[route]
        accepted=[r for r in data if threshold is not None and r['valid'] and r['route']==route and r['score']>=threshold]
        known=[r for r in accepted if r['known']]; correct=sum(r['correct'] for r in known)
        n=len(known); nq=len({r['question_key'] for r in known}); unknown=len(accepted)-n
        branches[route]={'accepted':len(accepted),'known':n,'unknown':unknown,'known_questions':nq,'correct':correct,
            'precision_known':correct/n if n else None,
            'precision_bounds_from_unknowns':[correct/len(accepted),(correct+unknown)/len(accepted)] if accepted else None,
            'pass':n>=cfg['min_known'] and nq>=cfg['min_questions'] and correct>=cfg['full_bypass_target']*n}
        selected.extend(accepted)
    return {'branches':branches,'full_bypass_count':len(selected),'total':len(records),
            'full_bypass_fraction':len(selected)/len(records) if records else None,
            'lora_call_fraction':1-len(selected)/len(records) if records else None,
            'scope':'routing replay; actual LoRA execution/cost and final chain quality require accepted LoRA checkpoint'}
