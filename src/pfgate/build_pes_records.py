#!/usr/bin/env python3
"""The bridge records of every cached prefix: detector cache + segment texts of the panel -> the inputs of P_ES. numpy only.
The segment texts give the characters (len of the text, as in the corpus of the predictor); the detector cache gives the event and
mode probabilities of the immutable primary answer (its first 12 columns); the thresholds are those of the detector bundle.
Refused: a prefix whose number of segments differs between the panel and the cache, a key of the cache that is not in the panel.
  build_pes_records.py LORA_OUTPUTS.npz ITS_MANIFEST.json PANEL.json THRESHOLDS.json OUT.npz"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfgate import pes_bridge  # noqa: E402

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    cache_path, manifest_path, panel_path, thresholds_path, out = argv
    if Path(out).exists():
        raise SystemExit('the output exists')
    manifest = json.loads(Path(manifest_path).read_text())
    if sha(cache_path) != manifest['npz_sha256'] or manifest['source'] != 'lora_outputs' or manifest['width'] != 25:
        raise SystemExit('the detector cache is not the one its manifest describes')
    values = np.load(cache_path)['values']
    thresholds = json.loads(Path(thresholds_path).read_text())
    if thresholds.get('answer_version') != 'immutable primary +0' or thresholds.get('selection_split') != 'validation' or len(thresholds['f1']) != 6:
        raise SystemExit('the thresholds are not the validation F1 thresholds of the primary answer')
    panel = {p['key']: p for p in json.loads(Path(panel_path).read_text())['prefixes']}
    xin, mode, prev, offsets, at = [], [], [], [], 0
    events_on = np.zeros(6)
    for key in manifest['keys']:
        record = manifest['records'][key]
        if key not in panel or len(panel[key]['segments']) != record['length'] or panel[key]['prefix_hash'] != record['prefix_hash']:
            raise SystemExit('the prefix %s of the cache is not the prefix of the panel' % key)
        rows = values[record['offset']:record['offset'] + record['length']]
        chars = [len(text) for text in panel[key]['segments']]
        a, b, c = pes_bridge.bridge(chars, rows[:, 0:6], rows[:, 6:12], thresholds['f1'])
        xin.append(a), mode.append(b), prev.append(c), offsets.append((at, len(a)))
        events_on += a[:, 5:11].sum(0)
        at += len(a)
    xin, mode, prev = np.concatenate(xin), np.concatenate(mode), np.concatenate(prev)
    np.savez(out, xin=xin, mode=mode, prev_mode=prev, keys=np.array(manifest['keys']), offsets=np.array(offsets, np.int64))
    report = {'format': 'pfgate_pes_records_v1', 'npz_sha256': sha(out), 'prefixes': len(offsets), 'records': int(len(xin)), 'detector_cache_sha256': manifest['npz_sha256'], 'detector_cache_manifest_sha256': sha(manifest_path),
              'panel_sha256': sha(panel_path), 'thresholds_sha256': sha(thresholds_path), 'thresholds_f1': thresholds['f1'], 'bridge_sha256': sha(pes_bridge.__file__), 'this_file_sha256': sha(__file__),
              'share_of_segments_with_the_event': dict(zip(pes_bridge.EVENTS, (events_on / len(xin)).tolist())),
              'share_of_segments_by_span_type': {pes_bridge.MODES[m]: float((mode == m).mean()) for m in range(7)}, 'spans_per_prefix_mean': float(np.mean([1 + int((mode[a + 1:a + n] != mode[a:a + n - 1]).sum()) for a, n in offsets])),
              'segments_per_prefix_max': int(max(n for _, n in offsets))}
    Path(out + '.manifest.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main(sys.argv[1:])
