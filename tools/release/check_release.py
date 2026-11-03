#!/usr/bin/env python3
"""Validate the paper implementation without importing or running a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ('experiments', 'reports', 'docs/reports', 'docs/findings', 'docs/plans',
             'docs/assets', 'docs/release', 'papers', 'env', 'data/manifests',
             'data/releases', 'data/registry.tsv', 'data/fetch.sh')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', action='store_true')
    args = parser.parse_args()
    for name in FORBIDDEN:
        assert not (ROOT / name).exists(), f'Out-of-scope historical content: {name}'
    sources = list((ROOT / 'src').rglob('*.py'))
    for p in sources:
        ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
    for p in (ROOT / 'configs').rglob('*.json'):
        json.loads(p.read_text())
    origins = json.loads((ROOT / 'CODE_ORIGINS.json').read_text())
    assert origins['paper_pdf_sha256'] and origins['files']
    assert all(str(p.relative_to(ROOT)) in origins['files'] for p in sources), 'Incomplete code inventory'
    for relative, item in origins['files'].items():
        assert (ROOT / relative).is_file(), relative
        assert sha256(ROOT / relative) == item['release_sha256'], relative
    manifest = json.loads((ROOT / 'models/MANIFEST.json').read_text())
    pointers = 0
    for relative, item in manifest['artifacts'].items():
        p = ROOT / relative
        with p.open('rb') as f:
            prefix = f.read(200)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            assert not args.weights, f'Run git lfs pull: {relative}'
            assert ('oid sha256:' + item['sha256']).encode() in prefix, relative
            assert ('size ' + str(item['bytes'])).encode() in prefix, relative
            pointers += 1
        else:
            assert p.stat().st_size == item['bytes'] and sha256(p) == item['sha256'], relative
    for p in (ROOT / 'configs/gate').glob('*.json'):
        cfg = json.loads(p.read_text())
        assert cfg['weights'] in manifest['artifacts'], p.name
    print(json.dumps({'python_sources': len(sources), 'model_artifacts': len(manifest['artifacts']),
                      'lfs_pointers': pointers, 'historical_content': False}))


if __name__ == '__main__':
    main()
