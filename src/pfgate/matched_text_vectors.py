#!/usr/bin/env python3
"""The text vectors of the matched pair P_text / P_ES_matched for every completed segment of every prefix of the gate panel, and
for its question (PROCESS_FORECAST_GATE_20261013, rows C_text and C_matched). Inference of the frozen encoder only; no fit.
The encoding is NOT written here: the function `encode` of the owner's reference (matched_text_pair_7c8cf777/interface/
matched_reference.py, its SHA256 is checked first) is called, so the procedure is the one of the corpus table: the frozen
MiniLM, truncation to 256 word pieces, mean of the last hidden states over the attention mask, L2 norm, float16.
  segment   its text as it is, the same text whose length in characters the bridge records carry: for every record the length
            of the text must equal the characters stored in the bridge records, or the run is refused;
  question  normalised as in the corpus of the predictor: ' '.join(str(question).lower().split()). An empty question is refused
            (in the corpus 130 traces had one; the bank must always give the real text).
The prefixes, their order and their offsets are those of the bridge records, so the vectors line up with every other table.
  matched_text_vectors.py MATCHED_REFERENCE.py SHA256 ENCODER_DIR ENCODER_SUMS.json PANEL.json RECORDS.npz OUT.npz [THREADS=1]
ENCODER_SUMS.json: {file name: SHA256} of every file of the encoder directory; the directory must hold exactly these files."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

TEXT = 384
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
normalise = lambda question: ' '.join(str(question).lower().split())


def load_reference(path, expected):
    if sha(path) != expected:
        raise SystemExit('the reference of the owner of the predictor is not the pinned file')
    spec = importlib.util.spec_from_file_location('matched_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def texts_of(panel_path, records_path):
    """(keys, offsets, segment texts in the order of the records, normalised questions in the order of the prefixes)."""
    panel = {p['key']: p for p in json.loads(Path(panel_path).read_text())['prefixes']}
    r = np.load(records_path)
    keys, offsets = [str(k) for k in r['keys']], r['offsets']
    chars = np.rint(np.expm1(r['xin'][:, 1].astype(np.float64) * 10)).astype(np.int64)      # column 1 of the bridge records: log1p(characters of the segment) / 10
    segments, questions = [], []
    for key, (offset, length) in zip(keys, offsets):
        prefix = panel.get(key)
        if prefix is None or len(prefix['segments']) != length:
            raise SystemExit('the prefix %s of the records is not a prefix of the panel with the same number of segments' % key)
        if [len(t) for t in prefix['segments']] != chars[offset:offset + length].tolist():
            raise SystemExit('the segment texts of %s are not the texts whose lengths the bridge records carry' % key)
        question = normalise(prefix['question'])
        if not question:
            raise SystemExit('the question of %s is empty' % key)
        segments += prefix['segments']
        questions.append(question)
    if len(segments) != len(r['xin']):
        raise SystemExit('the records and the texts do not have the same number of segments')
    return keys, offsets, segments, questions


def main(argv):
    reference_path, reference_sha, encoder, sums_path, panel_path, records_path, out = argv[:7]
    threads = int(argv[7]) if len(argv) > 7 else 1
    if Path(out).exists():
        raise SystemExit('the output exists')
    sums = json.loads(Path(sums_path).read_text())
    present = sorted(p.name for p in Path(encoder).iterdir() if p.is_file())
    if present != sorted(sums) or any(sha(Path(encoder) / name) != value for name, value in sums.items()):
        raise SystemExit('the encoder directory does not hold exactly the pinned files')
    keys, offsets, segments, questions = texts_of(panel_path, records_path)
    reference = load_reference(reference_path, reference_sha)
    import torch
    torch.set_num_threads(threads)
    began = time.time()
    text = reference.encode(segments, encoder)
    prompt = reference.encode(questions, encoder)
    seconds = time.time() - began
    if text.shape != (len(segments), TEXT) or prompt.shape != (len(keys), TEXT) or text.dtype != np.float16 or not np.isfinite(text.astype(np.float32)).all() or not np.isfinite(prompt.astype(np.float32)).all():
        raise SystemExit('the vectors are not finite float16 vectors of width 384')
    np.savez(out, text=text, prompt=prompt, keys=np.array(keys), offsets=offsets)
    norms = np.linalg.norm(text.astype(np.float32), axis=1)
    manifest = {'format': 'pfgate_matched_text_vectors_v1', 'npz_sha256': sha(out), 'width': TEXT, 'prefixes': len(keys), 'records': len(segments), 'distinct_questions': len(set(questions)),
                'reference_sha256': reference_sha, 'encoder_files_sha256': sums, 'panel_sha256': sha(panel_path), 'records_sha256': sha(records_path), 'question_rule': "' '.join(str(question).lower().split())",
                'segment_rule': 'the text of the segment as it is; its length equals the characters of the bridge record', 'longest_segment_characters': int(max(len(t) for t in segments)),
                'vector_norm_min_max': [float(norms.min()), float(norms.max())], 'seconds_of_the_encoding': seconds, 'threads': threads, 'torch': torch.__version__, 'device': 'cpu', 'this_file_sha256': sha(__file__)}
    Path(out + '.manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main(sys.argv[1:])
