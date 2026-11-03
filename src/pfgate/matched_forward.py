#!/usr/bin/env python3
"""The forecast heads of one weight of the matched pair, P_text (cell T) or P_ES_matched (cell TES), for every completed segment
of every prefix of the gate panel (PROCESS_FORECAST_GATE_20261013, rows C_text and C_matched). Inference only, no fit.
Everything of the model comes from the owner: the reference matched_reference.py (its SHA256 is checked first) loads the
accepted training code and the checkpoint (its SHA256 is checked by the owner's loader). The inputs of a prefix are the bridge
records already used for the row C (41 numbers, span type, previous span type: accepted, equal to the owner's reference), the
MiniLM vector of every completed segment and the vector of the question (pfgate.matched_text_vectors). The rules for what the
gate may not know are those of the row C and the same for both cells: the solver label given on the command line (index 3, the
named stand-in), and the mean over the four TRAINED benchmark labels 1..4. The 29 numbers kept for a position are those of the
row C, computed by the SAME function pfgate.pes_forward.heads_of from the output of the model.
The whole prefix is one window with the positions 0..n-1 (n <= 192 is required, so that the window of the owner's layout starts
at 0 for every position). Checks made in this run, on the first CHECKED prefixes, before the output is written:
  * the heads of every position equal those computed from the owner's forward_all with the same label, and the gate
    probabilities of the last position equal the owner's forward_last (difference below 1e-5);
  * cell T: zeroed structure (columns 3..40 of the records, span type 0, previous span type 3) gives the same heads, difference 0;
    cell TES: the same change DOES change the heads (the structure is read).
  matched_forward.py MATCHED_REFERENCE.py SHA256 SRC_DIR PROVIDER_DIR CHECKPOINT.pt SHA256 CELL RECORDS.npz VECTORS.npz SOLVER_NAME OUT.npz [THREADS=1]"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import matched_text_vectors, pes_forward  # noqa: E402

WIDTH, TEXT, LONGEST, CHECKED = 29, 384, 192, 12
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_prefix(model, torch, xin, mode, prev_mode, solver, benches, text, prompt):
    """[len(benches), n, 29]: the heads of one prefix under every benchmark label."""
    n, k = len(xin), len(benches)
    if n > LONGEST:
        raise ValueError('a prefix of %d segments is longer than the single window allows' % n)
    batch = {'xin': torch.as_tensor(np.ascontiguousarray(xin, np.float32))[None].expand(k, n, 41), 'mode': torch.as_tensor(np.asarray(mode, np.int64))[None].expand(k, n),
             'prev_mode': torch.as_tensor(np.asarray(prev_mode, np.int64))[None].expand(k, n), 'solver': torch.full((k, n), solver, dtype=torch.long),
             'bench': torch.as_tensor(benches, dtype=torch.long)[:, None].expand(k, n), 'pad': torch.zeros(k, n, dtype=torch.bool),
             'text': torch.as_tensor(np.asarray(text, np.float16)).float()[None].expand(k, n, TEXT), 'prompt': torch.as_tensor(np.asarray(prompt, np.float16)).float()[None, None].expand(k, n, TEXT)}
    with torch.no_grad():
        return pes_forward.heads_of(model(batch), torch)


def main(argv):
    reference_path, reference_sha, src, provider, checkpoint, checkpoint_sha, cell, records_path, vectors_path, solver_name, out = argv[:11]
    threads = int(argv[11]) if len(argv) > 11 else 1
    if Path(out).exists():
        raise SystemExit('the output exists')
    if cell not in ('T', 'TES'):
        raise SystemExit('the cell must be T or TES')
    M = matched_text_vectors.load_reference(reference_path, reference_sha)
    import torch
    torch.set_num_threads(threads)
    X, R = M.load_code(src, provider)
    if solver_name not in R.SOLVERS or R.SOLVERS[0] != 'UNKNOWN' or R.BENCHMARKS[0] != 'UNKNOWN' or len(R.OPERATORS) != 7:
        raise SystemExit('the dictionaries of the predictor are not the expected ones')
    solver, benches = R.SOLVERS.index(solver_name), list(range(1, len(R.BENCHMARKS)))
    model, step = M.load_model(X, R, checkpoint, cell, checkpoint_sha)
    parameters = sum(p.numel() for p in model.parameters())
    r, v = np.load(records_path), np.load(vectors_path)
    vectors_manifest = json.loads(Path(vectors_path + '.manifest.json').read_text())
    keys, offsets = [str(k) for k in r['keys']], r['offsets']
    if sha(vectors_path) != vectors_manifest['npz_sha256'] or vectors_manifest['records_sha256'] != sha(records_path) or [str(k) for k in v['keys']] != keys or not np.array_equal(v['offsets'], offsets) \
            or v['text'].shape != (len(r['xin']), TEXT) or v['prompt'].shape != (len(keys), TEXT):
        raise SystemExit('the text vectors are not those of these bridge records')
    by_bench = np.zeros((len(benches), len(r['xin']), WIDTH), np.float32)
    checks = {'prefixes_checked': 0, 'max_difference_from_the_owner_forward_all': 0.0, 'max_gate_difference_from_the_owner_forward_last': 0.0, 'max_change_when_the_structure_is_zeroed': 0.0}
    began = time.time()
    for number, (key, (offset, length)) in enumerate(zip(keys, offsets)):
        sl = slice(int(offset), int(offset + length))
        xin, mode, prev, text, prompt = r['xin'][sl], r['mode'][sl], r['prev_mode'][sl], v['text'][sl], v['prompt'][number]
        heads = forward_prefix(model, torch, xin, mode, prev, solver, benches, text, prompt)
        by_bench[:, sl] = heads.numpy()
        if number < CHECKED:
            for b, bench in enumerate(benches):
                with torch.no_grad():
                    owner = pes_forward.heads_of({k: val[None] for k, val in M.run(model, xin, mode, prev, solver, bench, text, prompt).items()}, torch)[0]
                checks['max_difference_from_the_owner_forward_all'] = max(checks['max_difference_from_the_owner_forward_all'], float((owner - heads[b]).abs().max()))
                last = M.forward_last(R, model, xin, mode, prev, solver, bench, text, prompt)
                checks['max_gate_difference_from_the_owner_forward_last'] = max(checks['max_gate_difference_from_the_owner_forward_last'], float(np.abs(np.asarray(last['gate'], np.float32) - heads[b, -1, 0:3].numpy()).max()))
            zeroed = np.array(xin, np.float32)
            zeroed[:, 3:41] = 0
            blind = forward_prefix(model, torch, zeroed, np.zeros_like(mode), np.full_like(prev, 3), solver, benches, text, prompt)
            checks['max_change_when_the_structure_is_zeroed'] = max(checks['max_change_when_the_structure_is_zeroed'], float((blind - heads).abs().max()))
            checks['prefixes_checked'] += 1
    seconds = time.time() - began
    reads_structure = checks['max_change_when_the_structure_is_zeroed'] > 0
    if checks['max_difference_from_the_owner_forward_all'] > 1e-5 or checks['max_gate_difference_from_the_owner_forward_last'] > 1e-5 or reads_structure != (cell == 'TES') or not np.isfinite(by_bench).all():
        raise SystemExit('a check of the forward failed, nothing was written: %s' % json.dumps(checks))
    np.savez(out, heads=by_bench.mean(0), heads_by_benchmark_label=by_bench, keys=np.array(keys), offsets=offsets)
    manifest = {'format': 'pfgate_matched_heads_v1', 'npz_sha256': sha(out), 'width': WIDTH, 'prefixes': len(keys), 'records': int(len(r['xin'])), 'cell': cell, 'checkpoint_sha256': checkpoint_sha, 'checkpoint_step': step,
                'parameters': parameters, 'reference_sha256': reference_sha, 'training_code_sha256': {'matched.py': sha(Path(src) / 'matched.py'), 'ablation.py': sha(Path(src) / 'ablation.py'), 'pes_reference.py': sha(Path(provider) / 'pes_reference.py')},
                'solver_label': [solver_name, solver], 'benchmark_labels_averaged': [R.BENCHMARKS[b] for b in benches], 'records_sha256': sha(records_path), 'text_vectors_sha256': vectors_manifest['npz_sha256'],
                'heads_function_sha256': sha(pes_forward.__file__), 'checks': checks, 'seconds_of_the_forward_with_its_checks': seconds, 'threads': threads, 'torch': torch.__version__, 'device': 'cpu', 'this_file_sha256': sha(__file__)}
    Path(out + '.manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main(sys.argv[1:])
