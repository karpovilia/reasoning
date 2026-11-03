#!/usr/bin/env python3
"""ONE per-segment input table for the gate rows A, B, C of PROCESS_FORECAST_GATE_20261013. numpy only, no model.
A record is one COMPLETED segment of a prefix; every source below is causal: record j was computed from the segments 0..j.
Columns (width 2676):
     0:8      lengths and the self-reported confidence: the 8 'C' columns of the accepted table of the GRU matrix (4 of them and the
              confidence are filled by the worker from the bank, as in the matrix)
     8:29     observations of the LoRA detector: the 12 event and 9 mode columns of the same accepted table (probabilities, running
              sums, boundary of the argmax mode, age and count of the runs)
    29:2589   the fixed text representation of the segment: the frozen base of the detector bundle WITHOUT the LoRA adapter and
              without any head, pooled (cache text_only of the accepted S1 pass)
  2589:2618   forecast heads of P_ES seed 17         2618:2647   of P_ES seed 29         2647:2676   of P_ES seed 43   (29 numbers each)
Rows of the experiment (a row is a mask of columns; the network and its width are the same):
  A  = 0:8 + 29:2589            T: text, lengths, position, budget, confidence
  B  = A + 8:29                 T + the observed history of events and spans
  C  = B + the heads of the P_ES seed that equals the seed of the gate member
Refused: sources that do not hold exactly the same prefixes with the same number of segments in the same order.
  build_gate_table.py MATRIX_FEATURES.npz ITS_MANIFEST TEXT_ONLY.npz ITS_MANIFEST HEADS_17.npz HEADS_29.npz HEADS_43.npz OUT.npz"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HEADS = 29
COLUMNS = {'lengths_and_confidence': [0, 8], 'detector_events_and_spans': [8, 29], 'text': [29, 2589], 'pes_heads_seed17': [2589, 2618], 'pes_heads_seed29': [2618, 2647], 'pes_heads_seed43': [2647, 2676]}
WIDTH = 2676
ROWS = {'A': [[0, 8], [29, 2589]], 'B': [[0, 29], [29, 2589]], 'C': {17: [[0, 2618]], 29: [[0, 2589], [2618, 2647]], 43: [[0, 2589], [2647, 2676]]}}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    matrix_path, matrix_manifest_path, text_path, text_manifest_path, h17, h29, h43, out = argv
    if Path(out).exists():
        raise SystemExit('the output exists')
    mm, tm = json.loads(Path(matrix_manifest_path).read_text()), json.loads(Path(text_manifest_path).read_text())
    if sha(matrix_path) != mm['npz_sha256'] or mm['width'] != 29 or sha(text_path) != tm['npz_sha256'] or tm['source'] != 'text_only' or tm['width'] != 2560:
        raise SystemExit('a source is not the file its manifest describes')
    matrix = np.load(matrix_path)
    keys, offsets = [str(k) for k in matrix['keys']], matrix['offsets']
    text = np.load(text_path)['values']
    if keys != tm['keys'] or any((int(o), int(n)) != (tm['records'][k]['offset'], tm['records'][k]['length']) for k, (o, n) in zip(keys, offsets)) or len(text) != len(matrix['x']):
        raise SystemExit('the text cache and the table of the matrix do not hold the same prefixes in the same order')
    x = np.zeros((len(text), WIDTH), np.float32)
    x[:, 0:29], x[:, 29:2589] = matrix['x'], text
    heads_sums = {}
    for seed, path in ((17, h17), (29, h29), (43, h43)):
        h, manifest = np.load(path), json.loads(Path(path + '.manifest.json').read_text())
        if sha(path) != manifest['npz_sha256'] or [str(k) for k in h['keys']] != keys or not np.array_equal(h['offsets'], offsets) or h['heads'].shape != (len(text), HEADS) or manifest['format'] != 'pfgate_pes_heads_v2':
            raise SystemExit('the heads of seed %d are not for the same prefixes' % seed)
        first = COLUMNS['pes_heads_seed%d' % seed][0]
        x[:, first:first + HEADS] = h['heads']
        heads_sums[seed] = {'npz_sha256': manifest['npz_sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'], 'records_sha256': manifest['records_sha256']}
    if not np.isfinite(x).all():
        raise SystemExit('a value of the table is not finite')
    np.savez(out, x=x, keys=np.array(keys), offsets=offsets)
    manifest = {'format': 'pfgate_gate_table_v2', 'npz_sha256': sha(out), 'width': WIDTH, 'columns': COLUMNS, 'rows': {'A': ROWS['A'], 'B': ROWS['B'], 'C': {str(k): v for k, v in ROWS['C'].items()}}, 'prefixes': len(keys), 'records': int(len(x)),
                'matrix_table_sha256': mm['npz_sha256'], 'matrix_table_columns': mm['columns'], 'decision_context_columns_filled_by_the_worker': mm['decision_context_columns_filled_by_the_worker'],
                'text_cache_sha256': tm['npz_sha256'], 'pes_heads': {str(k): v for k, v in heads_sums.items()}, 'this_file_sha256': sha(__file__), 'what_this_is': __doc__}
    Path(out + '.manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('what_this_is', 'matrix_table_columns')}))


if __name__ == '__main__':
    main(sys.argv[1:])
