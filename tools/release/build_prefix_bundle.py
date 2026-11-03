#!/usr/bin/env python3
"""Assemble a verified causal-detector bundle from locally supplied trained assets."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizers', type=Path, required=True,
                        help='directory containing encoder/ and lora4b/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True,
                        help='threshold files calibrated for this checkpoint')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new directory')
    for name in ('heads.pt', 'detector_config.json', 'backbone/adapter_model.safetensors',
                 'backbone/adapter_config.json', 'CHECKPOINT_COMPLETE.json'):
        if not (args.checkpoint / name).is_file():
            parser.error('checkpoint lacks ' + name)
    args.output.mkdir(parents=True)
    shutil.copytree(args.checkpoint, args.output / 'checkpoint')
    shutil.copytree(args.tokenizers, args.output / 'tokenizers')
    for name in ('API_CONTRACT.json', 'schema.json', 'LIMITATIONS.json'):
        shutil.copy2(ROOT / 'configs/detector' / name, args.output / name)
    (args.output / 'api').mkdir()
    for name in ('detector_api_v1.py', 'schema_check.py'):
        shutil.copy2(ROOT / 'src/prefix_api' / name, args.output / 'api' / name)
    runtime = args.output / 'code/clean_detectors'
    runtime.mkdir(parents=True)
    for name in ('__init__', 'core', 'model', 'dynamics', 'history_inference', 'history_rollout', 'spans'):
        shutil.copy2(ROOT / 'src/clean_detectors' / (name + '.py'), runtime)
    (args.output / 'calibration').mkdir()
    for name in ('thresholds.json', 'revision_thresholds.json', 'dynamics_thresholds.json'):
        shutil.copy2(args.calibration / name, args.output / 'calibration' / name)
    sys.dont_write_bytecode = True
    path = args.output / 'api/detector_api_v1.py'
    spec = importlib.util.spec_from_file_location('bundled_detector_api', path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    provenance = {
        'base_model': json.loads((ROOT / 'configs/detector/base_model.json').read_text()),
        'training_code_sha256': {'clean_detectors/' + p.name: api.file_sha(p) for p in runtime.glob('*.py')},
        'runtime_scope': 'Packaged inference subset of the published training implementation.',
        'checkpoint': {'heads_sha256': api.file_sha(args.output / 'checkpoint/heads.pt')},
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    table = api.file_table(args.output)
    manifest = {'api_version': api.API_VERSION, 'bundle_id': api.bundle_id_of(table),
                'status': 'assembled; static verification only', 'files': table}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    api.verify_bundle(args.output / 'manifest.json')
    print(json.dumps({'bundle_id': manifest['bundle_id'], 'verified': True}))


if __name__ == '__main__':
    main()
