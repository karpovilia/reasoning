"""Prediction spans with explicit censoring at missing segments and open tails."""
from .core import spans_from_modes


def predicted_spans(segment_ids,modes,confirmed_end=False):
    spans=spans_from_modes(segment_ids,modes,confirmed_end)
    known={sid for sid,mode in zip(segment_ids,modes,strict=True) if mode is not None}
    observed_end=segment_ids[-1]+1 if segment_ids else None
    for span in spans:
        span['left_censored']=span['seg_start']!=0 and span['seg_start']-1 not in known
        transition=span['seg_end'] in known
        eos=bool(confirmed_end and span['seg_end']==observed_end)
        span['closed']=transition or eos
        span['right_censored']=not span['closed']
        span['right_boundary']='mode_transition' if transition else 'confirmed_end' if eos else 'unknown_or_unobserved'
    return spans
