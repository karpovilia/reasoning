"""Empirical features of the same fixed-length continuations for every model."""
import numpy as np


def features(modes, lengths, events, starts, natural, K, E, H=2048):
    # First continued span is left-censored unless it starts at the first future
    # segment. The last span is right-censored at H unless natural END is seen.
    ended=bool(natural and sum(lengths)<=H)
    duration=np.zeros(4);events_by_mode=np.zeros((K,E));transitions=np.zeros((K,K))
    pairs=np.zeros((E,E));spent=0;last=None;age=0;known_start=False
    for mode,length,event,start in zip(modes,lengths,events,starts):
        if spent>=H:break
        boundary=last is not None and (start or mode!=last)
        if last is None:
            known_start=bool(start);age=0
        elif boundary:
            if known_start:duration[np.searchsorted([64,256,1024],age,side='right')]+=1
            transitions[last,mode]+=1;known_start=True;age=0
        if spent+length<=H:
            e=np.asarray(event);events_by_mode[mode]+=e;pairs+=np.outer(e,e)
        age+=min(length,H-spent);spent+=length;last=mode
    if ended and known_start and last is not None:
        duration[np.searchsorted([64,256,1024],age,side='right')]+=1
    upper=pairs[np.triu_indices(E,1)]
    # Saturating count transform gives bounded, declared coordinates. All models
    # use precisely the same transformation; no fitted scaling on DEV.
    squash=lambda x: np.asarray(x)/(1+np.asarray(x))
    return dict(duration=squash(duration), mode_events=squash(np.r_[events_by_mode.ravel(),upper]),
                event_counts=squash(events_by_mode.ravel()), event_pairs=squash(upper),
                transitions=squash(transitions.ravel()))
