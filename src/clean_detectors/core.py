"""Shared, model-independent contracts. Never infer an unknown label."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import unicodedata

EVENTS = ['backtrack', 'verify', 'branch', 'subgoal_done', 'failed_attempt', 'commit']
MODES = ['SETUP', 'DERIVING', 'EXPLORING', 'VERIFYING', 'REVISING', 'CONCLUDING']
SALT = 'clean-flash-pro-detectors-v1|'


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(dumps(value) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def normalize_question(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def new_split(key):
    # Integer comparisons are exactly equivalent to digest / 2**256; no float rounding.
    number = int(sha(SALT + key), 16)
    total = 1 << 256
    return 'train' if number * 100 < 70 * total else 'validation' if number * 100 < 85 * total else 'test'


def serialize(question, segments, sid, history_start=None):
    if not 0 <= sid < len(segments):
        raise ValueError('focus is outside the observed segments')
    lo = max(0, sid - 16) if history_start is None else history_start
    if not max(0, sid - 16) <= lo <= sid:
        raise ValueError('invalid history start')
    # Explicit lengths prevent text resembling delimiters from changing the structure.
    text = f'QUESTION ({len(question)} characters)\n{question}\nHISTORY\n'
    for i in range(lo, sid):
        text += f'SEGMENT ({len(segments[i])} characters)\n{segments[i]}\n'
    text += f'FOCUS ({len(segments[sid])} characters)\n'
    start = len(text)
    text += segments[sid]
    return text, (start, len(text))


def encode_focus(tokenizer, text, offsets):
    out = tokenizer(text, add_special_tokens=True, truncation=False,
                    return_offsets_mapping=True, return_special_tokens_mask=True)
    lo, hi = offsets
    focus = []
    special_ids=set(tokenizer.all_special_ids)
    for tid,(a, b), special in zip(out['input_ids'],out['offset_mapping'], out['special_tokens_mask']):
        # No pooling over delimiter tokens; a boundary-crossing token is excluded.
        focus.append(int(not special and tid not in special_ids and lo <= a < b <= hi))
    if not any(focus):
        raise ValueError('focus has no complete nonspecial token')
    return {'input_ids': out['input_ids'], 'focus_mask': focus,
            'offset_mapping': out['offset_mapping']}


@lru_cache(maxsize=4096)
def field_length(tokenizer, text):
    # This is only a search hint. Boundary merges may change the sum; candidate
    # and predecessor are always encoded as complete strings before acceptance.
    return len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])


def shared_input(question, segments, sid, tokenizers, limit=2048):
    cache={}
    start=max(0,sid-16)
    prefix=f'QUESTION ({len(question)} characters)\n{question}\nHISTORY\n'
    focus=f'FOCUS ({len(segments[sid])} characters)\n{segments[sid]}'
    lengths={name:[field_length(tok,f'SEGMENT ({len(segments[i])} characters)\n{segments[i]}\n')
                    for i in range(start,sid)] for name,tok in tokenizers.items()}
    totals={name:field_length(tok,prefix)+field_length(tok,focus)+sum(lengths[name])+tok.num_special_tokens_to_add(False)
            for name,tok in tokenizers.items()}
    hint=start
    while hint<sid and any(n>limit for n in totals.values()):
        for name in totals:totals[name]-=lengths[name][hint-start]
        hint+=1
    def fits(lo):
        if lo not in cache:
            text,offsets=serialize(question,segments,sid,lo)
            cache[lo]=(text,offsets,{name:encode_focus(tok,text,offsets) for name,tok in tokenizers.items()})
        return all(len(e['input_ids'])<=limit for e in cache[lo][2].values())
    # Most foci now require one full encoding (or two when dropping history).
    # Approximate field lengths NEVER become model inputs or decide overflow.
    if fits(hint) and (hint==start or not fits(hint-1)):
        selected=hint
    elif not fits(sid):
        text,offsets,encodings=cache[sid]
        return {'status':'overflow','text_sha256':sha(text),'focus_offsets':list(offsets),
                'history_start':sid,'lengths':{k:len(v['input_ids']) for k,v in encodings.items()}}
    elif fits(start):
        selected=start
    else:
        # For these pinned NFC/byte-BPE tokenizers, complete newline-delimited
        # SEGMENT fields have monotone token lengths. Binary search finds the SAME
        # first fitting suffix as deleting one oldest field at a time. The contract
        # is checked against the sequential algorithm on real corpus inputs.
        left,right=start+1,sid
        while left<right:
            middle=(left+right)//2
            if fits(middle):right=middle
            else:left=middle+1
        selected=left
        if selected>start and fits(selected-1):raise ValueError('nonmonotone history budget')
        observed=sorted(cache)
        for a,b in zip(observed,observed[1:]):
            if fits(a) and not fits(b):raise ValueError('nonmonotone history budget')
    text,offsets,encodings=cache[selected]
    return {'text':text,'text_sha256':sha(text),'focus_offsets':list(offsets),
            'history_start':selected,'tokens':encodings,'status':'eligible'}


def majority(values):
    """Strict majority among KNOWN independent observations; ties stay unknown."""
    from collections import Counter
    c = Counter(v for v in values if v is not None)
    if not c:
        return None
    value, n = c.most_common(1)[0]
    return value if 2*n > sum(c.values()) else None


def spans_from_modes(segment_ids, modes, end_of_trace=False):
    """Half-open spans; gaps/unknowns break spans. Last span remains open online."""
    spans = []
    previous = None
    for sid, mode in zip(segment_ids, modes, strict=True):
        if mode is None:
            previous = None
            continue
        if previous == sid - 1 and spans[-1]['mode'] == mode:
            spans[-1]['seg_end'] = sid + 1
        else:
            spans.append({'seg_start': sid, 'seg_end': sid+1, 'mode': mode, 'closed': True})
        previous = sid
    if spans and segment_ids and spans[-1]['seg_end'] == segment_ids[-1] + 1:
        spans[-1]['closed'] = bool(end_of_trace)
    return spans
