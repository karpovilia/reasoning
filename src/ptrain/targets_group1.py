"""Group-1 targets (event -> next span -> events inside it), derived per unit from observed labels.

Inputs per segment j (all future values are used ONLY as targets):
  mode[j]        0..5 known span type, 7 unknown
  span_start[j]  True if an explicit span boundary starts at j (same-type boundaries allowed);
                 None -> boundaries are inferred from mode changes only (source limitation)
  ev_y[j][a], ev_known[j][a]  events of segment j
  end_status     'natural' | 'confirmed_truncated' | 'unknown'
Outputs per position j:
  next_span[j]   0..5 type of the first span AFTER the current one, 6 natural END without a next
                 span, -1 unknown (current type unknown, unknown label before the boundary, or the
                 trace stops without a confirmed natural end)
  ev_y_next[j][a], ev_known_next[j][a]  at least one event a inside that next span. A positive is
                 known as soon as it is observed; an absence is known only if the whole next span is
                 observed (it ends at a known boundary or at a natural end) and event a is known on
                 every segment of it. For END the event target is masked (no events).
An event inside a span never creates a boundary.
"""
UNKNOWN = 7
END = 6


def derive(mode, ev_y, ev_known, end_status, span_start=None):
    n = len(mode)
    if span_start is None:
        span_start = [j == 0 or (mode[j] != UNKNOWN and mode[j - 1] != UNKNOWN and mode[j] != mode[j - 1]) for j in range(n)]
        explicit = False
    else:
        explicit = True
    natural = end_status == 'natural'

    def boundary(k):
        """True if a new span starts at k (k>0), None if undecidable (unknown label)."""
        if mode[k] == UNKNOWN or mode[k - 1] == UNKNOWN:
            return None
        if explicit:
            return bool(span_start[k])
        return mode[k] != mode[k - 1]

    # next boundary index strictly after j, or n (trace end), or None (undecidable)
    nb = [None] * n
    # compute from the right: nb[j] = smallest k>j with boundary(k) True, None if an undecidable k comes first
    state = n  # value for j = n-1
    nb[n - 1] = n if n else None
    for j in range(n - 2, -1, -1):
        b = boundary(j + 1)
        if b is None:
            state = None
        elif b:
            state = j + 1
        nb[j] = state
    next_span = [-1] * n
    ev_out = [[0] * 6 for _ in range(n)]
    known_out = [[False] * 6 for _ in range(n)]
    span_cache = {}
    for j in range(n):
        if mode[j] == UNKNOWN:
            continue
        k = nb[j]
        if k is None:
            continue
        if k == n:
            if natural:
                next_span[j] = END
            continue
        next_span[j] = mode[k]
        if k not in span_cache:
            e = nb[k] if k < n else n           # end (exclusive) of the next span
            pos = [0] * 6; allknown = [True] * 6
            last = (e if e is not None else n)
            stop = last
            if e is None:
                # span cut by an undecidable label: scan while labels stay known and same span
                stop = k + 1
                while stop < n and mode[stop] != UNKNOWN and not (boundary(stop) or False):
                    stop += 1
            for s in range(k, stop):
                for a in range(6):
                    if ev_known[s][a] and ev_y[s][a]:
                        pos[a] = 1
                    if not ev_known[s][a]:
                        allknown[a] = False
            complete = e is not None and (e < n or natural)
            span_cache[k] = (pos, [bool(pos[a]) or (complete and allknown[a]) for a in range(6)])
        pos, kn = span_cache[k]
        ev_out[j] = list(pos); known_out[j] = list(kn)
    return next_span, ev_out, known_out, explicit
