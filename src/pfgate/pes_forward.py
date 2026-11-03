#!/usr/bin/env python3
"""The forecast heads of the process predictor P_ES for every completed segment of every prefix (PROCESS_FORECAST_GATE_20261013).
P_ES is the accepted corrected event-span Transformer of EVENT_SPAN_TRANSFORMER_ABLATION_20261013, cell ES (12 layers, width 512,
context 256): the owner's class Predictor is imported from the owner's file, whose SHA256 is checked first, and the checkpoint is
loaded as the owner's evaluation does. No weight is changed; inference only.

Inputs of P_ES for a prefix: the records of pfgate.pes_bridge (41 numbers, span type, previous span type) of its completed
segments 0..n-1, the solver label of the solver that wrote the trace, and a benchmark label. The whole prefix is one window
(n <= 256 is required; the positions are 0..n-1, as in training for traces that start a window at 0). The model is causal:
the output at position j reads the records 0..j only and is the forecast of what follows segment j.
Benchmark label. The gate must not receive the identity of the benchmark, and the bank has benchmarks the predictor never saw.
So the predictor is run once with every benchmark label it was trained with and the outputs are AVERAGED: probabilities are
averaged as probabilities, point forecasts in the scale they were trained in. The outputs for every single label are saved too.
Heads kept for a position, 29 numbers, all read from trained heads; nothing is invented. The type GROUNDING (index 6) was never
a target of the predictor and is not produced by the detector, so its probability is not kept as a feature (request of the owner
of the predictor): the remaining probabilities of dest and of the boundary type are kept as they are, not renormalised.
   0:3    gate probabilities: the next segment stays in the span, the span changes, the trace ends
   3:9    dest probabilities: the type of the next span, if the span changes (six types)
   9:15   event probabilities of the next segment
  15      forecast log1p(characters of the next segment) / 10
  16:23   type of the next boundary from the waiting-time head: the six span types and END (mixture marginal)
  23:26   probability that no boundary occurs within the next 250, 1000 and 4000 characters (survival of the Weibull mixture)
  26:29   point forecast log1p(remainder after this segment) / 10 in segments, span starts and characters. A point forecast:
          this head has no trained uncertainty, and the waiting-time head is not an uncertainty of the remainder.
There is no correctness head in this checkpoint.
  pes_forward.py ABLATION_PY SHA256 CORPUS_METADATA.json CHECKPOINT.pt SHA256 RECORDS.npz SOLVER_NAME OUT.npz [THREADS=1]"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

CONFIG = {'width': 512, 'context': 256, 'heads': 8, 'layers': 12}
HORIZONS = (250.0, 1000.0, 4000.0)
WIDTH = 29
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_owner_module(path, expected):
    if sha(path) != expected:
        raise SystemExit('the predictor code is not the pinned file')
    spec = importlib.util.spec_from_file_location('owner_ablation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def heads_of(output, torch):
    """[batch, n, 29] from the output dictionary of Predictor.forward."""
    F = torch.nn.functional
    pi = F.softmax(output['wait_pi'], -1)
    mark = (pi[..., None] * F.softmax(output['wait_mark'], -1)).sum(-2)
    scale, shape = output['wait_logscale'], output['wait_shape']
    survival = torch.stack([(pi * torch.exp(-torch.exp((shape * (float(np.log(t)) - scale)).clamp(max=20)))).sum(-1) for t in HORIZONS], -1)
    keep = [0, 1, 2, 3, 4, 5, 7]                               # the six span types and END; index 6 is GROUNDING
    return torch.cat([F.softmax(output['gate'], -1), F.softmax(output['dest'], -1)[..., :6], torch.sigmoid(output['events']), output['length'][..., None] / 10, mark[..., keep], survival, output['rem'] / 10], -1)


def forward_prefix(model, torch, xin, mode, prev_mode, solver, benches):
    """[len(benches), n, 29]: the heads of one prefix under every benchmark label."""
    n = len(xin)
    if n > CONFIG['context']:
        raise ValueError('a prefix of %d segments does not fit the context of the predictor' % n)
    k = len(benches)
    batch = {'xin': torch.as_tensor(xin, dtype=torch.float32)[None].expand(k, n, 41), 'mode': torch.as_tensor(mode, dtype=torch.long)[None].expand(k, n),
             'prev_mode': torch.as_tensor(prev_mode, dtype=torch.long)[None].expand(k, n), 'solver': torch.full((k, n), solver, dtype=torch.long),
             'bench': torch.as_tensor(benches, dtype=torch.long)[:, None].expand(k, n), 'pad': torch.zeros(k, n, dtype=torch.bool)}
    with torch.no_grad():
        return heads_of(model(batch), torch)


def main(argv):
    code, code_sha, meta_path, checkpoint, checkpoint_sha, records_path, solver_name, out = argv[:8]
    threads = int(argv[8]) if len(argv) > 8 else 1
    if Path(out).exists():
        raise SystemExit('the output exists')
    owner = load_owner_module(code, code_sha)
    import torch
    torch.set_num_threads(threads)
    if sha(checkpoint) != checkpoint_sha:
        raise SystemExit('the checkpoint is not the pinned file')
    meta = json.loads(Path(meta_path).read_text())
    if solver_name not in meta['solvers'] or meta['solvers'][0] != 'UNKNOWN' or meta['benchmarks'][0] != 'UNKNOWN' or len(meta['operators']) != 7:
        raise SystemExit('the dictionaries of the corpus are not the expected ones')
    solver, benches = meta['solvers'].index(solver_name), list(range(1, len(meta['benchmarks'])))        # every TRAINED benchmark label; 0 is the untrained UNKNOWN
    model = owner.Predictor('ES', meta, CONFIG)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    r = np.load(records_path)
    keys, offsets = [str(k) for k in r['keys']], r['offsets']
    by_bench = np.zeros((len(benches), len(r['xin']), WIDTH), np.float32)
    began = time.time()
    for key, (offset, length) in zip(keys, offsets):
        sl = slice(int(offset), int(offset + length))
        by_bench[:, sl] = forward_prefix(model, torch, r['xin'][sl], r['mode'][sl], r['prev_mode'][sl], solver, benches).numpy()
    seconds = time.time() - began
    if not np.isfinite(by_bench).all():
        raise SystemExit('a head value is not finite')
    np.savez(out, heads=by_bench.mean(0), heads_by_benchmark_label=by_bench, keys=np.array(keys), offsets=offsets)
    manifest = {'format': 'pfgate_pes_heads_v2', 'npz_sha256': sha(out), 'width': WIDTH, 'prefixes': len(keys), 'records': int(len(r['xin'])), 'predictor_code_sha256': code_sha, 'checkpoint_sha256': checkpoint_sha,
                'checkpoint_step': int(state.get('step', -1)), 'parameters': parameters, 'cell': 'ES', 'config': CONFIG, 'solver_label': [solver_name, solver], 'benchmark_labels_averaged': [meta['benchmarks'][b] for b in benches],
                'records_sha256': sha(records_path), 'horizons_characters': list(HORIZONS), 'seconds_of_the_forward': seconds, 'seconds_per_segment': seconds / len(r['xin']), 'threads': threads, 'torch': torch.__version__,
                'device': 'cpu', 'this_file_sha256': sha(__file__), 'columns': __doc__.split('Heads kept for a position')[1].split('There is no correctness head')[0]}
    Path(out + '.manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'columns'}))


if __name__ == '__main__':
    main(sys.argv[1:])
