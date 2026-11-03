"""Small offline demonstration of the same training/prediction implementation."""
import argparse
import json
from pathlib import Path
from ptrain import configs, synthetic, runner, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be a new directory')
    args.out.mkdir(parents=True)
    synthetic.build(args.out / 'pack', n_train=8, n_dev=4, max_len=16, emb_rows=64)
    cfg = configs.arm('A', 'CT')
    cfg.update(micro_batch=4, effective_batch=8,
               passes=1, fit_budget_seconds=600,
               model=dict(width=32, heads=4, layers=1, ff_width=64, dropout=0.0, max_segments=256),
               optimizer=dict(lr=.0003, weight_decay=.01, betas=[.9, .999], eps=1e-8,
                              clip_norm=1., warmup_updates=1, cosine_floor=.1),
               task_weights=dict(joint=1., destination=.5, event=.25),
               selection_weights=dict(joint=1., destination=.5, event=.25))
    config = args.out / 'config.json'
    config.write_text(json.dumps(cfg))
    rc = runner.main(['--config', str(config), '--pack', str(args.out / 'pack'),
                      '--out', str(args.out / 'fit'), '--seed', '42', '--device', 'cpu',
                      '--max-seconds', '600', '--purpose', 'test'])
    if rc:
        raise SystemExit(rc)
    predict.main(['--run', str(args.out / 'fit'), '--device', 'cpu'])


if __name__ == '__main__':
    main()
