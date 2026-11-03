"""Create Qwen token arrays for a predictor pack from a SHA-keyed text database."""
import argparse
import json
import sqlite3
from pathlib import Path
from ptrain.model_b import tokenize_pack
from ptrain.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--texts', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    con = sqlite3.connect(f'file:{a.texts}?mode=ro', uri=True)
    def get(h):
        row = con.execute('select text from texts where sha=?', (h,)).fetchone()
        if row is None:
            raise ValueError('TEXT_MISSING:' + h)
        import hashlib
        if hashlib.sha256(row[0].encode()).hexdigest() != h:
            raise ValueError('TEXT_SHA_MISMATCH:' + h)
        return row[0]
    cfg = json.loads(Path(a.config).read_text())['model']
    result = tokenize_pack(a.pack, a.tokenizer, cfg,
        lambda u: (get(u['prompt_sha256']), [get(h) for h in u['text_sha256']]))
    path = Path(a.pack) / 'PACK_MANIFEST.json'
    manifest = json.loads(path.read_text())
    for name in ('tokens_qwen3.npz', 'TOKENS_MANIFEST.json'):
        manifest['files'][name] = sha256_file(Path(a.pack) / name)
    from ptrain.runner import atomic_json
    atomic_json(path, manifest)
    con.close()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
