"""Separate next-focus forecasting metrics from text-conditioned detection."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from .selective import any_event_target


def forecast_records(records):
    result=[]
    for row in records:
        prior=row['dynamics_prior']
        if prior['target_sid']!=row['sid'] or prior['as_of_sid']>=row['sid']:
            raise ValueError('Invalid forecast time')
        result.append({**row,'event_probabilities':prior['event_probabilities'],
            'mode_probabilities':prior['mode_probabilities']})
    return result


def binary_report(records):
    known=[]
    for row in records:
        target=any_event_target(row['event_targets'],row['event_mask'])
        if target is not None:known.append((target,row['dynamics_prior']['any_event_probability'],row['question_key']))
    if not known:return {'known':0,'unknown_excluded':len(records)}
    y=np.array([r[0] for r in known]);p=np.array([r[1] for r in known]);clipped=np.clip(p,1e-7,1-1e-7)
    return {'known':len(known),'unknown_excluded':len(records)-len(known),
        'questions':len({r[2] for r in known}),'positives':int(y.sum()),'prevalence':float(y.mean()),
        'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
        'roc_auc':float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else None,
        'brier':float(np.mean((p-y)**2)),'bce':float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
        'scope':'P(any event in next focus) BEFORE current text; direct targets; no 99.5% certification'}
