"""Regression tests for review findings PTREV-001, 003, 004, 005, 006, 007 (REVIEW.md, verdict on REVIEW_REQUEST_V1)."""
import gzip, hashlib, io, json, os, subprocess, sys, tarfile, time
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
import numpy as np  # noqa: E402
from ptrain import runner, synthetic, losses as L, predict  # noqa: E402
from ptrain.data import sha256_file  # noqa: E402

SMALL = dict(width=32, heads=4, layers=1, ff_width=64, dropout=0.0, max_segments=256)


def cfg_file(tmp, budget=None, cell='CT'):
    c = dict(arch='A', cell=cell, model=SMALL, optimizer=dict(lr=1e-3, weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.,
             warmup_updates=1, cosine_floor=.1), effective_batch=8, micro_batch=8, passes=1, task_weights=L.STAGE1_WEIGHTS,
             checkpoint_every_updates=2, log_every=1)
    if budget:
        c['fit_budget_seconds'] = budget
    p = Path(tmp) / ('cfg_%s.json' % (budget or 0)); p.write_text(json.dumps(c)); return p


@pytest.fixture(scope='module')
def pk(tmp_path_factory):
    d = tmp_path_factory.mktemp('pk')
    synthetic.build(d, n_train=12, n_dev=4, seed=4, max_len=120, emb_rows=100, native=True)
    return d


def admissible_copy(pk, tmp, note=None):
    import shutil
    p2 = Path(tmp) / 'adm'; shutil.copytree(pk, p2)
    m = json.loads((p2 / 'PACK_MANIFEST.json').read_text()); m.update(synthetic=False, production_admissible=True)
    if note:
        m['coverage_note'] = note
    (p2 / 'PACK_MANIFEST.json').write_text(json.dumps(m)); return p2


def run(cfg, pack, out, *extra, purpose='test'):
    return runner.main(['--config', str(cfg), '--pack', str(pack), '--out', str(out), '--seed', '1', '--device', 'cpu',
                        '--max-seconds', '100000', '--purpose', purpose, *extra])


# ---------------- PTREV-001 ----------------
def test_production_requires_exact_pass_record(pk, tmp_path):
    p2 = admissible_copy(pk, tmp_path); cfg = cfg_file(tmp_path)
    with pytest.raises(ValueError, match='PRODUCTION_REQUIRES_PASS_RECORD'):
        run(cfg, p2, tmp_path / 'o1', purpose='production')
    good = dict(verdict='PASS', code_hashes=runner.code_hashes(), pack_manifest_sha256=[sha256_file(p2 / 'PACK_MANIFEST.json')],
                config_sha256=[sha256_file(cfg)])
    for key, val, err in (('verdict', 'CHANGES_REQUESTED', 'PASS_RECORD_NOT_PASS'), ('code_hashes', {'x.py': '0'}, 'PASS_RECORD_CODE_MISMATCH'),
                          ('pack_manifest_sha256', ['0' * 64], 'PASS_RECORD_PACK_MISMATCH'), ('config_sha256', ['0' * 64], 'PASS_RECORD_CONFIG_MISMATCH')):
        bad = dict(good); bad[key] = val; f = tmp_path / ('pr_%s.json' % key); f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=err):
            run(cfg, p2, tmp_path / ('o_' + key), '--pass-record', str(f), purpose='production')
    f = tmp_path / 'pr_ok.json'; f.write_text(json.dumps(good))
    assert run(cfg, p2, tmp_path / 'o_ok', '--pass-record', str(f), '--max-updates', '1', purpose='production') == runner.EXIT_RESUMABLE


def test_intermediate_pack_refused_for_production(pk, tmp_path):
    p2 = admissible_copy(pk, tmp_path, note='INTERMEDIATE_X'); cfg = cfg_file(tmp_path)
    f = tmp_path / 'pr.json'
    f.write_text(json.dumps(dict(verdict='PASS', code_hashes=runner.code_hashes(), pack_manifest_sha256=[sha256_file(p2 / 'PACK_MANIFEST.json')],
                                 config_sha256=[sha256_file(cfg)])))
    with pytest.raises(ValueError, match='INTERMEDIATE_PACK_NOT_ACCEPTED_BY_PASS'):
        run(cfg, p2, tmp_path / 'o', '--pass-record', str(f), purpose='production')


# ---------------- PTREV-003 ----------------
def test_ledger_total_survives_pod_loss_cycles():
    a1 = dict(attempt_id='1', prior_seconds=0, elapsed_seconds=500)
    a2 = dict(attempt_id='2', prior_seconds=500, elapsed_seconds=100)      # restored with mirror floor 500
    assert runner.ledger_total([a2]) == 600                               # reviewer case: must be 600, not 500
    assert runner.ledger_total([a1, a2]) == 600                           # no double count when both survive
    a3 = dict(attempt_id='3', prior_seconds=600, elapsed_seconds=50)      # second loss cycle
    assert runner.ledger_total([a3]) == 650 and runner.ledger_total([a2, a3]) == 650 and runner.ledger_total([a1, a2, a3]) == 650


def test_runner_restored_attempt_counts_and_budget(pk, tmp_path):
    cfg = cfg_file(tmp_path, budget=650)
    out = tmp_path / 'o'; (out / 'attempts').mkdir(parents=True)
    # pod lost: only the restored attempt record exists, it started from the mirrored floor 500 and ran 100 s
    (out / 'attempts' / 'attempt_x.json').write_text(json.dumps(dict(attempt_id='x', prior_seconds=500, elapsed_seconds=100)))
    assert run(cfg, pk, out) == runner.EXIT_BUDGET                        # 600 >= 650 - 60


def test_mirror_prior_seconds_unique_attempts(tmp_path):
    from ptrain import prior_seconds as ps   # noqa: E402
    def snap(name, recs):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tf:
            for fn, r in recs:
                data = json.dumps(r).encode(); ti = tarfile.TarInfo(f'./F/attempts/{fn}'); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        (tmp_path / name).write_bytes(buf.getvalue())
    snap('runs_1.tgz', [('attempt_100_aaa.json', dict(attempt_id='100_aaa', prior_seconds=0, elapsed_seconds=500))])
    snap('runs_2.tgz', [('attempt_200_bbb.json', dict(attempt_id='200_bbb', prior_seconds=500, elapsed_seconds=100))])  # after pod loss
    assert ps.total('F', tmp_path) == 600


# ---------------- PTREV-004 ----------------
def test_completed_fit_is_not_restarted(pk, tmp_path):
    cfg = cfg_file(tmp_path); out = tmp_path / 'done'
    assert run(cfg, pk, out) == 0
    (out / 'checkpoint.pt').unlink()                                      # exactly what launch_fit.sh does after COMPLETE
    before = (out / 'events.jsonl').read_text().count('"start"')
    assert run(cfg, pk, out) == 0
    ev = (out / 'events.jsonl').read_text()
    assert ev.count('"start"') == before and 'already_complete_no_work' in ev
    assert not (out / 'checkpoint.pt').exists()
    (out / 'final.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='COMPLETE_DIRECTORY_INCONSISTENT'):
        run(cfg, pk, out)


# ---------------- PTREV-005 ----------------
DL = str(Path(__file__).resolve().parents[2] / 'tools' / 'deadline_run.sh')


def test_deadline_terminates_hung_and_kills_term_ignoring(tmp_path):
    """PTREV-005: the process group is gone BY the deadline, also when it ignores TERM."""
    dl = int(time.time()) + 6
    r = subprocess.run(['bash', DL, str(dl), '4', '2', '--', 'sleep', '1000'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 124 and time.time() <= dl                       # TERM before the deadline ends a sleeping process
    script = tmp_path / 'stubborn.sh'; script.write_text("trap '' TERM; sleep 1000 & wait; while true; do sleep 1; done\n")
    dl = int(time.time()) + 6
    t = subprocess.Popen(['bash', DL, str(dl), '4', '2', '--', 'bash', str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = t.communicate(timeout=60); t_end = time.time()
    assert t.returncode == 137 and 'DEADLINE_KILL' in err
    assert t_end <= dl + 0.5, (t_end, dl)                                  # watchdog itself returned by the deadline
    left = subprocess.run(['pgrep', '-f', str(script)], capture_output=True, text=True).stdout.split()
    assert left == []                                                     # no member of the group survives
    r = subprocess.run(['bash', DL, str(int(time.time()) + 100), '10', '2', '--', 'bash', '-c', 'exit 7'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 7                                              # normal exit code passes through
    r = subprocess.run(['bash', DL, str(int(time.time()) + 100), '1', '2', '--', 'true'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2                                              # inconsistent lead/margin refused




# ---------------- PTREV-006 ----------------
def test_predict_refuses_changed_pack_and_code(pk, tmp_path, monkeypatch):
    import shutil
    p2 = tmp_path / 'p'; shutil.copytree(pk, p2); cfg = cfg_file(tmp_path); out = tmp_path / 'r'
    assert run(cfg, p2, out) == 0
    predict.main(['--run', str(out), '--device', 'cpu'])                 # bound run: works
    m = json.loads((out / 'predictions_dev' / 'METRICS.json').read_text())
    assert m['pack_manifest_sha256'] == json.loads((out / 'RUN_BINDING.json').read_text())['pack_manifest_sha256']
    man = json.loads((p2 / 'PACK_MANIFEST.json').read_text()); man['note'] = 'changed'; (p2 / 'PACK_MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(ValueError, match='PREDICT_PACK_DIFFERS_FROM_RUN_BINDING'):
        predict.main(['--run', str(out), '--device', 'cpu'])
    monkeypatch.setattr(runner, 'code_hashes', lambda: {'x.py': '0'})
    with pytest.raises(ValueError, match='PREDICT_CODE_DIFFERS_FROM_RUN_BINDING'):
        predict.main(['--run', str(out), '--device', 'cpu'])


# ---------------- PTREV-007 ----------------
def test_build_pack_keeps_producer_reasons(tmp_path):
    from ptrain import build_pack
    udir = tmp_path / 'u'; (udir / 'units').mkdir(parents=True)
    n = 4
    d = dict(question_component='q0', termination_receipt=dict(end_status='natural'),
             inputs=dict(xin=[[0.0] * 41] * n, mode=[0] * n, prev_mode=[7] * n, solver=[1] * n, text_sha256=['t'] * n, prompt_sha256=['p'] * n),
             targets=dict(gate=[0] * n, dest=[-1] * n, event_y=[[0] * 6] * n, length_y=[0.0] * n, wait_y=[0.0] * n, wait_observed=[False] * n,
                          next_span=[-1] * n, rem=[[0, 0, 0]] * n,
                          target_valid=dict(step_outcome=[True] * n, next_type=[False] * n, length=[True] * n, events=[[True] * 6] * n,
                                            wait_observed=[False] * n, wait_censored=[False] * n, remainder=[[False] * 3] * n)),
             native_labels={})
    raw = gzip.compress(json.dumps(d).encode()); (udir / 'units/u0.json.gz').write_bytes(raw)
    (udir / 'UNIT_MANIFEST_V1.json').write_text(json.dumps([dict(unit_id='u0', physical_id='a', path='units/u0.json.gz', sha256=hashlib.sha256(raw).hexdigest())]))
    rows = [dict(index_v2_id='a', physical_id='a', role='TRAIN', question_component='q0', solver='s', labelled=True),
            dict(index_v2_id='b', physical_id='b', role='TRAIN', question_component='q1', solver='s', labelled=True),
            dict(index_v2_id='c', physical_id='c', role='DEV', question_component='q2', solver='s', labelled=True)]
    rp = tmp_path / 'r.jsonl'; rp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    lp = tmp_path / 'l.jsonl'
    lp.write_text(json.dumps(dict(index_v2_id='b', disposition='CONVERSION_BLOCKED', reason='question text missing')) + '\n'
                  + json.dumps(dict(index_v2_id='c', disposition='LEDGERED_CANDIDATE_M_ONLY_NOT_MAIN_LABEL', reason='candidate M only')) + '\n')
    build_pack.main(['--roster', str(rp), '--roster-sha256', sha256_file(rp), '--units', str(udir), '--out', str(tmp_path / 'pk'),
                     '--producer-ledger', str(lp), '--producer-ledger-sha256', sha256_file(lp), '--coverage-note', 'INTERMEDIATE_T'])
    disp = {json.loads(x)['index_v2_id']: json.loads(x) for x in (tmp_path / 'pk/DISPOSITION.jsonl').read_text().splitlines()}
    assert disp['b']['reason'] == 'question text missing' and disp['b']['producer_disposition'] == 'CONVERSION_BLOCKED'
    assert disp['c']['reason'] == 'candidate M only'
    man = json.loads((tmp_path / 'pk/PACK_MANIFEST.json').read_text())
    assert man['production_admissible'] is False and man['producer_ledger']['sha256'] == sha256_file(lp)
    with pytest.raises(ValueError, match='INTERMEDIATE_PACK_CANNOT_BE_PRODUCTION_ADMISSIBLE'):
        build_pack.main(['--roster', str(rp), '--roster-sha256', sha256_file(rp), '--units', str(udir), '--out', str(tmp_path / 'pk2'),
                         '--coverage-note', 'INTERMEDIATE_T', '--production-admissible'])
