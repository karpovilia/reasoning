"""First-stage targets of ONE native annotation namespace (COORDINATOR_NATIVE_TARGET_NAMESPACES_DECISION_V2).

Per segment j of a unit the namespace gives mode[j] (native index, -1 unknown), event[j][a],
event_known[j][a] and optionally span_start[j]. For the prediction after segment j:
  gate[j]  0 stay in the span, 1 switch to a new span, 2 natural END, -1 unknown
           * j < n-1: needs known native modes at j and j+1 of THIS namespace; switch = mode change, or an
             explicit span_start[j+1] when the source gives boundaries (same-type switch);
             an unknown label is never bridged
           * j == n-1: END only if end_status == 'natural' (accepted generation termination metadata);
             otherwise unknown. An annotation window edge is never an END.
  dest[j]  native type of segment j+1 on a known switch, else -1
  ev[j]    events of segment j+1 with their own known mask (all unknown at the last segment)
Valid masks: vstep (gate known), vnt (known switch with known next type), vev (per event axis).
"""


def derive(mode, event, event_known, end_status, span_start=None):
    n = len(mode)
    E = len(event[0]) if n else 0
    gate = [-1] * n; dest = [-1] * n
    ev = [[0] * E for _ in range(n)]; vev = [[False] * E for _ in range(n)]
    vstep = [False] * n; vnt = [False] * n
    for j in range(n):
        if j == n - 1:
            if end_status == 'natural' and mode[j] >= 0:
                gate[j] = 2; vstep[j] = True
            continue
        a, b = mode[j], mode[j + 1]
        if a >= 0 and b >= 0:
            sw = (bool(span_start[j + 1]) if span_start is not None else False) or a != b
            gate[j] = int(sw); vstep[j] = True
            if sw:
                dest[j] = b; vnt[j] = True
        ev[j] = [int(x) for x in event[j + 1]]
        vev[j] = [bool(k) for k in event_known[j + 1]]
    return dict(gate=gate, dest=dest, ev=ev, vstep=vstep, vnt=vnt, vev=vev)


def derive_group1(mode, event, event_known, end_status, K, span_start=None):
    """G2 targets per native namespace (event -> first next span -> events inside it); future labels only as targets.
    mode: 0..K-1 known, -1 unknown. Returns g1n (0..K-1 type of the first span after the current one, K = natural END
    without a next span, -1 unknown), g1e/g1k (E): at least one event inside that next span; a positive is known when
    observed, an absence only if the whole next span is observed (known boundary after it or natural end) and the event
    is known on every segment of it. END and unknown next span: events masked. Boundaries: explicit span_start when the
    source gives it, else mode changes; an unknown label on either side makes the boundary undecidable (no bridging).
    An event never creates a boundary (same rule as ptrain/targets_group1.py, generalised to K types, E events)."""
    n = len(mode); E = len(event[0]) if n else 0; natural = end_status == 'natural'

    def boundary(k):
        if mode[k] < 0 or mode[k - 1] < 0:
            return None
        return bool(span_start[k]) or mode[k] != mode[k - 1] if span_start is not None else mode[k] != mode[k - 1]
    nb = [None] * n
    state = n
    if n:
        nb[n - 1] = n
    for j in range(n - 2, -1, -1):
        b = boundary(j + 1)
        state = None if b is None else (j + 1 if b else state)
        nb[j] = state
    g1n = [-1] * n; g1e = [[0.0] * E for _ in range(n)]; g1k = [[False] * E for _ in range(n)]; cache = {}
    for j in range(n):
        if mode[j] < 0:
            continue
        k = nb[j]
        if k is None:
            continue
        if k == n:
            if natural:
                g1n[j] = K
            continue
        g1n[j] = mode[k]
        if k not in cache:
            e = nb[k]
            stop = e if e is not None else k + 1
            if e is None:
                while stop < n and mode[stop] >= 0 and not boundary(stop):
                    stop += 1
            pos = [0] * E; allk = [True] * E
            for s in range(k, stop):
                for a in range(E):
                    if event_known[s][a] and event[s][a]:
                        pos[a] = 1
                    if not event_known[s][a]:
                        allk[a] = False
            complete = e is not None and (e < n or natural)
            cache[k] = ([float(p) for p in pos], [bool(pos[a]) or (complete and allk[a]) for a in range(E)])
        g1e[j], g1k[j] = list(cache[k][0]), list(cache[k][1])
    return dict(g1n=g1n, g1e=g1e, g1k=g1k)
