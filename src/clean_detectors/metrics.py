"""Mask-aware metrics, validation-only calibration, and question-paired bootstrap."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from .core import EVENTS, MODES, spans_from_modes


def divide(a,b):return float(a/b) if b else None


def confusion(y,p,k,weights=None):
    w=np.ones_like(y,dtype=float) if weights is None else np.broadcast_to(np.asarray(weights)[:,None],y.shape)
    return np.stack([((p & y & k)*w).sum(0),((p & ~y & k)*w).sum(0),
                     ((~p & y & k)*w).sum(0),((~p & ~y & k)*w).sum(0)],axis=-1)


def scores(c):
    tp,fp,fn,tn=map(float,c)
    return {'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),'f1':divide(2*tp,2*tp+fp+fn),
            'fpr':divide(fp,fp+tn),'fdr':divide(fp,tp+fp),'tp':tp,'fp':fp,'fn':fn,'tn':tn}


def thresholds(y,k,probs):
    """Call only on validation predictions. Descending unique score ties stay atomic."""
    best=[];recall70=[];support=[]
    for j in range(6):
        yy=y[k[:,j],j];pp=probs[k[:,j],j]
        support.append({'known':int(len(yy)),'positive':int(yy.sum())})
        if not len(yy) or not yy.sum():
            best.append(1.000001);recall70.append(None);continue
        order=np.argsort(-pp,kind='stable');pp=pp[order];yy=yy[order]
        ends=np.r_[np.flatnonzero(pp[1:]!=pp[:-1]),len(pp)-1]
        tp=np.cumsum(yy)[ends];pred=ends+1;positive=yy.sum()
        f1=2*tp/(pred+positive)
        best.append(float(pp[ends[np.argmax(f1)]]))
        eligible=np.flatnonzero(tp/positive>=.7)
        ii=eligible[np.argmax((tp/pred)[eligible])]
        recall70.append(float(pp[ends[ii]]))
    return {'f1':best,'recall70':recall70,'validation_support':support}


def evaluate(y,k,probs,mode_y,mode_k,mode_probs,threshold,weights=None):
    y=np.asarray(y,bool);k=np.asarray(k,bool);probs=np.asarray(probs,float)
    prediction=probs>=np.asarray(threshold)
    counts=confusion(y,prediction,k,weights)
    per={};aps=[]
    for j,kind in enumerate(EVENTS):
        known=k[:,j];yy=y[known,j];pp=probs[known,j]
        ap=float(average_precision_score(yy,pp)) if len(yy) and yy.any() and (~yy).any() else None
        s=scores(counts[j]);s.update(known=int(known.sum()),positive=int(yy.sum()),ap=ap,
          brier=float(np.mean((pp-yy)**2)) if len(yy) else None,
          calibration=[{'lo':a/10,'hi':(a+1)/10,'n':int(((pp>=a/10)&(pp<((a+1)/10) if a<9 else pp<=1)).sum()),
                        'mean_probability':float(pp[(pp>=a/10)&(pp<((a+1)/10) if a<9 else pp<=1)].mean()) if ((pp>=a/10)&(pp<((a+1)/10) if a<9 else pp<=1)).any() else None,
                        'positive_rate':float(yy[(pp>=a/10)&(pp<((a+1)/10) if a<9 else pp<=1)].mean()) if ((pp>=a/10)&(pp<((a+1)/10) if a<9 else pp<=1)).any() else None} for a in range(10)])
        per[kind]=s
        if ap is not None:aps.append(ap)
    macro={key:float(np.mean([s[key] for s in per.values() if s[key] is not None])) if any(s[key] is not None for s in per.values()) else None for key in ['precision','recall','f1']}
    mode_y=np.asarray(mode_y);mode_k=np.asarray(mode_k,bool);mode_probs=np.asarray(mode_probs)
    mp=mode_probs.argmax(-1)
    modes={'known':int(mode_k.sum()),'coverage':float(mode_k.mean()) if len(mode_k) else None,
           'accuracy':float((mp[mode_k]==mode_y[mode_k]).mean()) if mode_k.any() else None,
           'macro_f1':float(f1_score(mode_y[mode_k],mp[mode_k],labels=list(range(6)),average='macro',zero_division=0)) if mode_k.any() else None,
           'confusion_matrix':confusion_matrix(mode_y[mode_k],mp[mode_k],labels=list(range(6))).tolist(),
           'support':[int((mode_y[mode_k]==i).sum()) for i in range(6)]}
    return {'events':per,'micro':scores(counts.sum(0)),'macro':macro,
            'macro_ap':float(np.mean(aps)) if aps else None,'ap_classes':[name for name,s in per.items() if s['ap'] is not None],
            'modes':modes}


def span_metrics(records):
    """Exact mode spans and typed boundaries within contiguous KNOWN reference runs."""
    groups=defaultdict(list)
    for r in records:groups[r['trace_id']].append(r)
    tp=fp=fn=btp=bfp=bfn=0
    for rows in groups.values():
        rows.sort(key=lambda r:r['sid'])
        ids=[r['sid'] for r in rows]
        gold=[r['mode_target'] if r['mode_mask'] else None for r in rows]
        pred=[int(np.argmax(r['mode_probabilities'])) if r['mode_mask'] else None for r in rows]
        # Evaluate only complete spans internal to known runs, avoiding fake boundaries at gaps.
        g=spans_from_modes(ids,gold,True);p=spans_from_modes(ids,pred,True)
        known=set(sid for sid,m in zip(ids,gold) if m is not None)
        def eligible(span):return span['seg_start']-1 in known and span['seg_end'] in known
        gs={(s['seg_start'],s['seg_end'],s['mode']) for s in g if eligible(s)}
        ps={(s['seg_start'],s['seg_end'],s['mode']) for s in p if eligible(s)}
        tp+=len(gs&ps);fp+=len(ps-gs);fn+=len(gs-ps)
        gb={(s['seg_start'],s['mode']) for s in g if s['seg_start']-1 in known}
        pb={(s['seg_start'],s['mode']) for s in p if s['seg_start']-1 in known}
        btp+=len(gb&pb);bfp+=len(pb-gb);bfn+=len(gb-pb)
    return {'exact_internal_spans':scores([tp,fp,fn,0]),'typed_boundaries':scores([btp,bfp,bfn,0]),
            'scope':'Only boundaries and complete spans internal to contiguous known-reference runs; censored edges excluded.'}


def paired_bootstrap(records_a,records_b,threshold_a,threshold_b,seed=20261007,repeats=2000):
    b={r['sample_id']:r for r in records_b};groups=defaultdict(lambda:[np.zeros((6,4)),np.zeros((6,4))])
    if {r['sample_id'] for r in records_a}!=set(b):raise ValueError('paired sample panels differ')
    for a in records_a:
        z=b[a['sample_id']]
        if any(a[k]!=z[k] for k in ['event_targets','event_mask','question_key']):raise ValueError('paired targets differ')
        for side,r,t in [(0,a,threshold_a),(1,z,threshold_b)]:
            y=np.array([r['event_targets']],bool);k=np.array([r['event_mask']],bool)
            p=np.array([r['event_probabilities']])>=t
            groups[r['question_key']][side]+=confusion(y,p,k)
    values=np.array(list(groups.values()));rng=np.random.default_rng(seed);diff=[]
    for _ in range(repeats):
        draw=values[rng.integers(len(values),size=len(values))].sum(0)
        a=scores(draw[0].sum(0))['f1'];b=scores(draw[1].sum(0))['f1']
        if a is not None and b is not None:diff.append(b-a)
    total=values.sum(0)
    return {'direction':'model_b minus model_a','questions':len(groups),'repeats':repeats,
            'micro_f1_difference':scores(total[1].sum(0))['f1']-scores(total[0].sum(0))['f1'],
            'ci95':np.quantile(diff,[.025,.975]).tolist(),'seed':seed,
            'seed_variation':'reported separately; bootstrap resamples questions, not seeds'}
