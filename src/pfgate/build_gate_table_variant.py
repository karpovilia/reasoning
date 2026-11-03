#!/usr/bin/env python3
"""A VARIANT of the accepted gate table of PROCESS_FORECAST_GATE_20261013 for the rows C_text and C_matched: the same table,
byte for byte, except the three blocks of forecast heads (columns 2589:2676), which hold the heads of one cell of the matched
pair instead of the heads of P_ES: cell T gives the table of C_text, cell TES the table of C_matched. numpy only, no model.
Why a variant and not new columns: the width, the masks of the rows, and with them the network, its number of parameters, its
initial weights for a seed and the order of its batches stay those of the accepted fits. Row C of a variant is the new row
(B + the heads of the matched cell with the seed of the member); rows A and B of a variant read exactly the accepted columns,
so their members must reproduce the accepted members, which the evaluation checks.
Refused: a base table that is not the one of its manifest; heads that are not of the named cell, not for the same prefixes, or
made from other bridge records than the heads of the base table; three heads whose seeds are not checkpoints of 17, 29, 43 in
this order (the SHA256 of the three checkpoints are arguments).
  build_gate_table_variant.py BASE_TABLE.npz CELL HEADS_17.npz CKPT_SHA_17 HEADS_29.npz CKPT_SHA_29 HEADS_43.npz CKPT_SHA_43 OUT.npz"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HEADS, FIRST, LAST, WIDTH = 29, 2589, 2676, 2676
BLOCKS = {17: (2589, 2618), 29: (2618, 2647), 43: (2647, 2676)}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    base_path, cell, out = argv[0], argv[1], argv[8]
    given = {17: (argv[2], argv[3]), 29: (argv[4], argv[5]), 43: (argv[6], argv[7])}
    if Path(out).exists():
        raise SystemExit('the output exists')
    if cell not in ('T', 'TES'):
        raise SystemExit('the cell must be T or TES')
    base_manifest = json.loads(Path(base_path + '.manifest.json').read_text())
    if sha(base_path) != base_manifest['npz_sha256'] or base_manifest['width'] != WIDTH or base_manifest['format'] != 'pfgate_gate_table_v2' \
            or base_manifest['columns'] != {'lengths_and_confidence': [0, 8], 'detector_events_and_spans': [8, 29], 'text': [29, 2589], 'pes_heads_seed17': [2589, 2618], 'pes_heads_seed29': [2618, 2647], 'pes_heads_seed43': [2647, 2676]}:
        raise SystemExit('the base table is not the accepted gate table of its manifest')
    base = np.load(base_path)
    keys, offsets = [str(k) for k in base['keys']], base['offsets']
    x = np.array(base['x'], np.float32)
    records = {v['records_sha256'] for v in base_manifest['pes_heads'].values()}
    heads_sums = {}
    for seed, (path, checkpoint_sha) in given.items():
        h, manifest = np.load(path), json.loads(Path(path + '.manifest.json').read_text())
        if sha(path) != manifest['npz_sha256'] or manifest['format'] != 'pfgate_matched_heads_v1' or manifest['cell'] != cell or manifest['checkpoint_sha256'] != checkpoint_sha or manifest['width'] != HEADS:
            raise SystemExit('the heads given for seed %d are not the heads of cell %s from the named checkpoint' % (seed, cell))
        if [str(k) for k in h['keys']] != keys or not np.array_equal(h['offsets'], offsets) or h['heads'].shape != (len(x), HEADS) or {manifest['records_sha256']} != records:
            raise SystemExit('the heads of seed %d are not for the prefixes and bridge records of the base table' % seed)
        first, last = BLOCKS[seed]
        x[:, first:last] = h['heads']
        heads_sums[str(seed)] = {'npz_sha256': manifest['npz_sha256'], 'checkpoint_sha256': checkpoint_sha, 'checkpoint_step': manifest['checkpoint_step'], 'records_sha256': manifest['records_sha256'],
                                 'text_vectors_sha256': manifest['text_vectors_sha256'], 'checks': manifest['checks']}
    if len({v['checkpoint_sha256'] for v in heads_sums.values()}) != 3 or len({v['text_vectors_sha256'] for v in heads_sums.values()}) != 1:
        raise SystemExit('the three heads must come from three different checkpoints and one table of text vectors')
    if not np.isfinite(x).all() or not np.array_equal(x[:, :FIRST], base['x'][:, :FIRST]):
        raise SystemExit('the variant is not finite or changed a column outside the heads')
    np.savez(out, x=x, keys=np.array(keys), offsets=offsets)
    manifest = dict(base_manifest, format='pfgate_gate_table_v2_variant', npz_sha256=sha(out), pes_heads=heads_sums, heads_cell=cell, base_table_sha256=base_manifest['npz_sha256'],
                    columns_equal_to_the_base_table=[0, FIRST], this_file_sha256=sha(__file__), base_builder_sha256=base_manifest['this_file_sha256'],
                    what_this_is=__doc__, row_C_of_this_table='B + the forecast heads of cell %s of the matched pair with the seed of the member: the row %s of the experiment' % (cell, 'C_text' if cell == 'T' else 'C_matched'))
    Path(out + '.manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('what_this_is', 'matrix_table_columns')}))


if __name__ == '__main__':
    main(sys.argv[1:])
