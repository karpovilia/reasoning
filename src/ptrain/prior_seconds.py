"""Cumulative busy seconds of one fit from all saved mirror snapshots (PTREV-003).
Records are keyed by their unique attempt_id (file name), the largest elapsed seen per record is kept, and the total is
runner.ledger_total over the distinct records: max(sum elapsed, max(prior_seconds + elapsed))."""
import json, sys, tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.runner import ledger_total


def total(fit, root):
    best = {}
    for t in sorted(Path(root).glob('runs_*.tgz')):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.startswith(f'./{fit}/attempts/attempt_'):
                    r = json.load(tf.extractfile(m)); key = r.get('attempt_id') or m.name
                    if key not in best or float(r.get('elapsed_seconds', 0)) > float(best[key].get('elapsed_seconds', 0)):
                        best[key] = r
    return ledger_total(list(best.values()))


if __name__ == '__main__':
    print(round(total(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'outputs/predictor'), 1))
