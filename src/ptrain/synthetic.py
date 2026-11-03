"""Synthetic pack with the exact production array contract, for profiling and CPU tests only.
The manifest carries synthetic=true and production_admissible=false; the runner refuses it for
production. Lengths follow a heavy-tailed distribution similar to the corpus (median ~100
segments, some traces > 1000)."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import ARRAY_SPEC, write_pack, ns_spec, hierarchical_weights  # noqa: E402
from ptrain import targets_native  # noqa: E402
from ptrain.targets_group1 import derive  # noqa: E402


SYN_NAMESPACES = [dict(id='syn_flash', K=6, E=6, status='accepted', cover=.7), dict(id='syn_sysR', K=7, E=4, status='accepted', cover=.4),
                  dict(id='syn_M', K=6, E=6, status='candidate', cover=.3)]


def build(out, n_train=200, n_dev=40, seed=0, max_len=2000, nsolvers=16, emb_rows=5000, with_text=False, native=True):
    import numpy as np
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    units = []; cols = {k: [] for k in ARRAY_SPEC}
    texts = {}
    for i in range(n_train + n_dev):
        role = 'TRAIN' if i < n_train else 'DEV'
        n = max(2, min(max_len, int(rng.lognormvariate(4.6, 1.0))))
        q = f'q{i // 3}' if role == 'TRAIN' else f'dq{i // 3}'
        mode = []; m = rng.randrange(6)
        for j in range(n):
            if rng.random() < .08:
                m = rng.randrange(6)
            mode.append(7 if rng.random() < .05 else m)
        ev = [[int(rng.random() < .07) for _ in range(6)] for _ in range(n)]
        evk = [[rng.random() < .9 for _ in range(6)] for _ in range(n)]
        end = rng.choice(['natural', 'natural', 'confirmed_truncated', 'unknown'])
        g1, g1y, g1k, _ = derive(mode, ev, evk, end)
        xin = nrng.normal(0, .3, size=(n, 41)).astype('f4')
        prev = [7] + mode[:-1]
        gate = []; dest = []; vso = []; vnt = []
        for j in range(n):
            if j == n - 1:
                gate.append(2 if end == 'natural' else -1); dest.append(-1); vso.append(end == 'natural'); vnt.append(False); continue
            if mode[j] == 7 or mode[j + 1] == 7:
                gate.append(-1); dest.append(-1); vso.append(False); vnt.append(False); continue
            sw = mode[j] != mode[j + 1]
            gate.append(int(sw)); dest.append(mode[j + 1] if sw else -1); vso.append(True); vnt.append(sw)
        evy_next = ev[1:] + [[0] * 6]; evk_next = evk[1:] + [[False] * 6]
        sol = rng.randrange(1, nsolvers)
        col = dict(xin=xin, mode=np.array(mode), prev_mode=np.array(prev), solver=np.full(n, sol),
                   text_row=nrng.integers(0, emb_rows, n), prompt_row=np.full(n, rng.randrange(emb_rows)),
                   es_ok=np.ones(n, bool), gate=np.array(gate), dest=np.array(dest), event_y=np.array(evy_next, 'f4'),
                   length_y=nrng.random(n).astype('f4'), wait_y=nrng.random(n).astype('f4') * 100, wait_observed=np.zeros(n, bool),
                   next_span=np.full(n, -1), rem=nrng.random((n, 3)).astype('f4'),
                   v_step_outcome=np.array(vso), v_next_type=np.array(vnt), v_length=np.ones(n, bool),
                   v_events=np.array(evk_next, bool), v_wait_observed=np.zeros(n, bool), v_wait_censored=np.zeros(n, bool),
                   v_remainder=np.zeros((n, 3), bool), g1_next_span=np.array(g1), g1_ev_y=np.array(g1y, 'f4'), g1_ev_known=np.array(g1k, bool))
        if native:
            for ni, ns in enumerate(SYN_NAMESPACES):
                if rng.random() < ns['cover']:
                    mm = []; cur = rng.randrange(ns['K'])
                    for j in range(n):
                        if rng.random() < .1:
                            cur = rng.randrange(ns['K'])
                        mm.append(-1 if rng.random() < .05 else cur)
                    evn = [[int(rng.random() < .1) for _ in range(ns['E'])] for _ in range(n)]
                    evk = [[rng.random() < .85 for _ in range(ns['E'])] for _ in range(n)]
                    t = targets_native.derive(mm, evn, evk, end)
                else:
                    t = dict(gate=[-1] * n, dest=[-1] * n, ev=[[0] * ns['E']] * n, vstep=[False] * n, vnt=[False] * n, vev=[[False] * ns['E']] * n)
                for k in ns_spec(ns):
                    col[f'ns{ni}_{k}'] = np.array(t[k])
                    cols.setdefault(f'ns{ni}_{k}', [])
        for k in cols:
            cols[k].append(col[k])
        u = dict(unit_id=f'syn{i:05d}', physical_id=f'syn-phys-{i}', question_component=q, role=role, n=n,
                 solver_ID=f'solver{sol}', benchmark_audit_only='synthetic', es_class='synthetic', end_status=end)
        if with_text:
            u['question_text'] = 'Synthetic question number %d. ' % i * 8
            u['segment_texts'] = [('step %d of trace %d: ' % (j, i)) + 'x = y + 1. ' * rng.randrange(2, 30) for j in range(n)]
        units.append(u)
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    extra = dict(synthetic=True, production_admissible=False, seed=seed)
    if native:
        off = 0
        for u in units:
            u['offset'] = off; off += u['n']
        elig = np.zeros(off, bool)
        for i, ns in enumerate(SYN_NAMESPACES):
            if ns['status'] == 'accepted':
                elig |= arrays[f'ns{i}_vstep'] | arrays[f'ns{i}_vnt'] | arrays[f'ns{i}_vev'].any(-1)
        arrays['w_pos'] = hierarchical_weights(units, elig)
        extra['namespaces'] = [dict(id=ns['id'], K=ns['K'], E=ns['E'], status=ns['status']) for ns in SYN_NAMESPACES]
    emb = nrng.normal(0, .05, size=(emb_rows, 384)).astype('f2')
    vocab = ['__OOV__'] + [f'solver{s}' for s in range(1, nsolvers)]
    return write_pack(out, units, arrays, vocab, extra, embeddings=emb)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--train', type=int, default=200)
    ap.add_argument('--dev', type=int, default=40); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max-len', type=int, default=2000)
    ap.add_argument('--with-text', action='store_true'); ap.add_argument('--legacy-v9-targets', action='store_true')
    a = ap.parse_args()
    m = build(a.out, a.train, a.dev, a.seed, a.max_len, with_text=a.with_text, native=not a.legacy_v9_targets)
    print(json.dumps(m['role_counts']))
