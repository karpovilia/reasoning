"""Durable finite trainer for one fit (architecture A or B, one input cell, one seed).

Guarantees
  * one worker per output directory (fcntl lock on OUT/LOCK, non-blocking, refuses a second worker)
  * RUN_BINDING.json written once; a resume must match it exactly (config, pack, code hashes, seed)
  * checkpoints at update boundaries with model (trainable part), optimizer, python/numpy/torch/cuda
    RNG, sampler order and cursor, cumulative counters; atomic replace + fsync
  * finite: total updates fixed from the pack (ceil(train_windows/effective_batch) * passes);
    wall deadline --max-seconds -> checkpoint and exit 75 (resumable); SIGTERM/SIGINT the same
  * DEV evaluation after each pass with the frozen selection score, stored in state before
    best.pt is written, so a crash between them replays nothing
  * disabled loss heads are checked on the first real update: their parameters must have no grad
Exit codes: 0 complete, 75 stopped resumable, 3 lock held, other = error.
"""
import argparse, fcntl, hashlib, json, math, os, random, signal, sys, time
import uuid as uuid_mod
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ptrain import losses as L  # noqa: E402

EXIT_RESUMABLE = 75
EXIT_LOCKED = 3
EXIT_BUDGET = 76   # cumulative fit budget exhausted: not resumable without a new controller decision


ES_INPUT_KINDS = ('retrospective_annotation', 'causal_detector')

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha_file(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def code_hashes():
    return {p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))}


def atomic_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); _fsync_dir(path.parent)


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())


def atomic_torch_save(torch, obj, path):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); _fsync_dir(path.parent)


def lr_factor(step, updates, warm, floor):
    if step < warm:
        return min(1., (step + 1) / max(1, warm))
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * (step - warm) / max(1, updates - warm)))


def rss_bytes():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class Adapter:
    """Architecture-specific pieces. A: PackedData + model_a. B: ptrain.model_b."""

    def __init__(self, cfg, pack, device, synthetic=False):
        self.cfg = cfg; self.device = device
        if cfg['arch'] == 'A':
            from ptrain.data import PackedData
            self.data = PackedData(pack, verify=True)
        elif cfg['arch'] == 'B':
            from ptrain.model_b import BData
            self.data = BData(pack, cfg, verify=True)
        else:
            raise ValueError('UNKNOWN_ARCH')

    def build(self, torch):
        cfg = self.cfg
        if cfg['arch'] == 'A':
            from ptrain import model_a
            model = model_a.build(cfg['cell'], self.data.nsolvers, cfg['model'], group1=cfg.get('group1', False),
                                  namespaces=self.data.namespaces or None, group3=cfg.get('group3', False), group4=cfg.get('group4', False))
            model.to(self.device)
            params = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': cfg['optimizer']['lr']}]
        else:
            from ptrain import model_b
            model = model_b.build(cfg, self.data.nsolvers, self.device, namespaces=self.data.namespaces or None)
            params = model.param_groups(cfg['optimizer'])
        return model, params

    def ids(self, role):
        return self.data.ids(role)

    def batch(self, ids):
        return self.data.make_batch(ids, self.device)

    def micro_batches(self, ids, train=True):
        """A: fixed-size chunks of micro_batch windows in sampler order. B: token-budget groups (micro_tokens for
        training, eval_micro_tokens for DEV) when configured, else chunks of micro_batch."""
        cfg = self.cfg
        key = 'micro_tokens' if train else 'eval_micro_tokens'
        if cfg['arch'] == 'B' and cfg['model'].get(key):
            return self.data.micro_batches(ids, int(cfg['model'][key]))
        mb = cfg.get('micro_batch', cfg['effective_batch'])
        return [ids[k:k + mb] for k in range(0, len(ids), mb)]

    def trainable_state(self, model):
        if self.cfg['arch'] == 'A':
            return model.state_dict()
        return model.trainable_state_dict()

    def load_trainable(self, model, sd):
        if self.cfg['arch'] == 'A':
            model.load_state_dict(sd)
        else:
            model.load_trainable_state_dict(sd)


def ledger_total(records):
    """Monotone cumulative busy seconds from attempt records (PTREV-003): each record carries the prior total it started
    from; total = max(sum of elapsed over distinct records, max(prior_seconds + elapsed)). Restored attempts after a pod
    loss keep the mirrored floor; distinct records in one directory are not double counted."""
    if not records:
        return 0.0
    s = sum(float(r.get('elapsed_seconds', 0)) for r in records)
    m = max(float(r.get('prior_seconds', 0)) + float(r.get('elapsed_seconds', 0)) for r in records)
    return max(s, m)


def disabled_head_params(model):
    names = ('length.', 'wait.', 'rem.')
    return [(n, p) for n, p in model.named_parameters() if any(n.startswith(x) or ('.' + x) in n for x in names)]


def dev_score(per_q, weights):
    """per_q: term -> {question: [sum, count]}. Macro over questions with support per term."""
    means = {}
    for term, rows in per_q.items():
        vals = [s / c for s, c in rows.values() if c > 0]
        means[term] = (sum(vals) / len(vals)) if vals else None
        means[term + '_questions'] = len(vals)
        means[term + '_cells'] = int(sum(c for s, c in rows.values()))
    if any(means[t] is None for t in weights):
        raise ValueError('DEV_HEAD_WITHOUT_SUPPORT')
    return sum(weights[t] * means[t] for t in weights), means


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--pack', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max-seconds', type=float, required=True)
    ap.add_argument('--max-updates', type=int, default=None, help='profile/smoke cap: stop resumable after this many total updates')
    ap.add_argument('--purpose', choices=['production', 'reproduce', 'profile', 'smoke', 'test'], required=True)
    ap.add_argument('--expect-gpu-uuid', default=None)
    ap.add_argument('--pass-record', default=None, help='production only: independent PASS record binding exact code/config/pack')
    ap.add_argument('--prior-seconds-floor', type=float, default=0.0,
                    help='cumulative busy seconds already recorded outside this directory; prior = max(files, floor)')
    a = ap.parse_args(argv)
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lockf = open(out / 'LOCK', 'a+')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('LOCK_HELD', file=sys.stderr); return EXIT_LOCKED
    lockf.seek(0); lockf.truncate(); lockf.write(f'{os.getpid()} {now()}\n'); lockf.flush()

    cfg = json.loads(Path(a.config).read_text())
    import numpy as np
    import torch
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA_NOT_AVAILABLE')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('EXACTLY_ONE_VISIBLE_GPU_REQUIRED')
        if a.expect_gpu_uuid:
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            if a.expect_gpu_uuid.replace('GPU-', '') not in uuid:
                raise RuntimeError('GPU_UUID_MISMATCH:' + uuid)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get('tf32', False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get('tf32', False))
    torch.set_num_threads(int(os.environ.get('PTRAIN_THREADS', '1')))

    pack_manifest_sha = sha_file(Path(a.pack) / 'PACK_MANIFEST.json')
    pack_manifest = json.loads((Path(a.pack) / 'PACK_MANIFEST.json').read_text())
    if a.purpose in ('production', 'reproduce') and pack_manifest.get('synthetic'):
        raise ValueError('SYNTHETIC_PACK_REFUSED_FOR_PRODUCTION')
    if a.purpose in ('production', 'reproduce') and pack_manifest.get('production_admissible') is not True:
        raise ValueError('PACK_NOT_PRODUCTION_ADMISSIBLE')
    if a.purpose == 'production':
        # PTREV-001: production needs an independent PASS bound to these exact code hashes, config and pack manifest
        if not a.pass_record:
            raise ValueError('PRODUCTION_REQUIRES_PASS_RECORD')
        pr = json.loads(Path(a.pass_record).read_text())
        if pr.get('verdict') != 'PASS':
            raise ValueError('PASS_RECORD_NOT_PASS')
        if pr.get('code_hashes') != code_hashes():
            raise ValueError('PASS_RECORD_CODE_MISMATCH')
        if pack_manifest_sha not in pr.get('pack_manifest_sha256', []):
            raise ValueError('PASS_RECORD_PACK_MISMATCH')
        if sha_file(a.config) not in pr.get('config_sha256', []):
            raise ValueError('PASS_RECORD_CONFIG_MISMATCH')
        if str(pack_manifest.get('coverage_note') or '').startswith('INTERMEDIATE') and not pr.get('accepts_intermediate_pack'):
            raise ValueError('INTERMEDIATE_PACK_NOT_ACCEPTED_BY_PASS')
    binding = dict(config=cfg, config_sha256=sha_file(a.config), pack=str(Path(a.pack).resolve()),
                   pack_manifest_sha256=pack_manifest_sha, seed=a.seed, code=code_hashes(), purpose=a.purpose)
    bpath = out / 'RUN_BINDING.json'
    if bpath.exists():
        old = json.loads(bpath.read_text())
        if old != binding:
            diff = sorted(k for k in set(old) | set(binding) if old.get(k) != binding.get(k))
            raise ValueError('RUN_BINDING_MISMATCH:' + ','.join(diff))
    else:
        atomic_json(bpath, binding)

    # PTREV-004: a completed fit is never restarted. Under the lock: verify COMPLETE against binding and files, then return.
    cpath = out / 'COMPLETE.json'
    if cpath.exists():
        done = json.loads(cpath.read_text())
        ok = done.get('binding_sha256') == sha_file(bpath)
        for name, key in (('final.pt', 'final_sha256'), ('best.pt', 'best_sha256')):
            ok = ok and (out / name).is_file() and sha_file(out / name) == done.get(key)
        if not ok:
            raise ValueError('COMPLETE_DIRECTORY_INCONSISTENT')
        append_jsonl(out / 'events.jsonl', dict(at=now(), event='already_complete_no_work'))
        print('ALREADY_COMPLETE', done.get('global_step')); return 0
    # cumulative busy-time ledger over ALL attempts (never reset): one heartbeat file per attempt
    budget = float(cfg.get('fit_budget_seconds', 0) or 0)
    att_dir = out / 'attempts'; att_dir.mkdir(exist_ok=True)
    prior = ledger_total([json.loads(f.read_text()) for f in sorted(att_dir.glob('attempt_*.json'))])
    if a.prior_seconds_floor > prior:
        append_jsonl(out / 'events.jsonl', dict(at=now(), event='prior_floor_applied', files_seconds=prior, floor=a.prior_seconds_floor))
        prior = a.prior_seconds_floor
    attempt_id = '%d_%s' % (int(t_start), uuid_mod.uuid4().hex[:12])      # unique across pod losses / restores
    att_file = att_dir / f'attempt_{attempt_id}.json'
    attempt_no = len(list(att_dir.glob('attempt_*.json'))) + 1
    def heartbeat(final=None):
        atomic_json(att_file, dict(attempt_id=attempt_id, pid=os.getpid(), started_at_epoch=t_start, elapsed_seconds=time.time() - t_start,
                                   prior_seconds=prior, budget_seconds=budget, final=final))
    if budget and prior >= budget - 60:
        print('FIT_BUDGET_EXHAUSTED', prior); return EXIT_BUDGET
    heartbeat()
    stop = [None]
    def on_signal(signum, frame):
        stop[0] = signum
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)

    t_data = time.time()
    ad = Adapter(cfg, a.pack, a.device)
    # E/S coverage of UNIQUE positions per role (no padding, no window repetition); recorded, not a gate
    es_cov = {}
    for role in ('TRAIN', 'DEV'):
        tot = cov = 0
        for u in ad.data.units:
            if u['role'] == role:
                tot += u['n']; cov += int(ad.data.a['es_ok'][u['offset']:u['offset'] + u['n']].sum())
        es_cov[role] = dict(positions=tot, es_known=cov)
    uses_es = any(x in cfg['cell'] for x in ('E', 'S', 'V'))          # V: structure vector computed from the E/S inputs
    if uses_es and a.purpose in ('production', 'reproduce'):
        kind = (pack_manifest.get('es_input_source') or {}).get('kind')
        if kind not in ES_INPUT_KINDS:
            raise ValueError('E/S input must declare retrospective_annotation or causal_detector')
    if uses_es and a.purpose == 'production':
        # PREDICTOR_CTES_UNAPPROVED_COVERAGE_GATE_001: no implementer threshold; production CTES needs an explicit
        # accepted admission of the causal E/S input (provenance, masks, per-role unique coverage) bound in the pack
        # Explicit input modes (RETRO_TES_EXECUTION_DECISION_V1): 'retrospective_annotation' (accepted producer labels of observed
        # segments, admitted by the coordinator decision) or 'causal_detector' (detector exports, admission of coverage).
        # The pack's es_input_source.kind and the admission kind must agree; a pack without a kind is refused.
        adm = pack_manifest.get('es_admission') or {}
        kind = (pack_manifest.get('es_input_source') or {}).get('kind')
        ok = kind in ES_INPUT_KINDS and adm.get('kind') == kind
        ok = ok and adm.get('accepted') is True and adm.get('acceptance_path') and adm.get('acceptance_sha256')
        if ok:
            ap = Path(adm['acceptance_path'])
            ok = ap.is_file() and sha_file(ap) == adm['acceptance_sha256']
        if not ok:
            raise ValueError('CTES_PRODUCTION_REQUIRES_ACCEPTED_ES_ADMISSION:kind=%s:coverage=%s' % (kind, json.dumps(es_cov)))
    train_ids = ad.ids('TRAIN'); dev_ids = ad.ids('DEV')
    eb = cfg['effective_batch']; mb = cfg.get('micro_batch', eb)
    if eb % mb:
        raise ValueError('MICRO_BATCH_MUST_DIVIDE')
    passes = cfg['passes']; per_pass = math.ceil(len(train_ids) / eb); updates = per_pass * passes
    data_seconds = time.time() - t_data

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model, groups = ad.build(torch)
    ocfg = cfg['optimizer']
    for g in groups:
        g.setdefault('weight_decay', ocfg['weight_decay']); g['initial_lr'] = g['lr']
    opt = torch.optim.AdamW(groups, betas=tuple(ocfg['betas']), eps=ocfg['eps'])
    weights = cfg['task_weights']
    native = bool(ad.data.namespaces)
    accepted = [ns.get('status') == 'accepted' for ns in ad.data.namespaces]
    from ptrain import namespace_policy
    event_ok = [bool(ns.get('event_supervision_eligible', namespace_policy.event_supervision_eligible(ns['id']))) for ns in ad.data.namespaces]
    if native:
        for ns, ok in zip(ad.data.namespaces, event_ok):
            if ok != namespace_policy.event_supervision_eligible(ns['id']):
                raise ValueError('PACK_EVENT_POLICY_DIFFERS_FROM_FROZEN_TABLE:' + ns['id'])
    if native and not any(accepted):
        raise ValueError('NO_ACCEPTED_NAMESPACE')
    g1 = bool(cfg.get('group1')) and native
    if g1 and not all(ns.get('group1') for ns in ad.data.namespaces):
        raise ValueError('G2_CONFIG_REQUIRES_GROUP1_TARGETS_IN_PACK')
    g3 = bool(cfg.get('group3')) and native
    if g3 and not (all(ns.get('group3') for ns in ad.data.namespaces) and ad.data.has_diag):
        raise ValueError('G3_CONFIG_REQUIRES_GROUP3_TARGETS_AND_DIAG_DEV_IN_PACK')
    if not g3 and any(ns.get('group3') for ns in ad.data.namespaces) and 'E' in cfg['cell']:
        raise ValueError('G3_PACK_HAS_MASKED_TRAIN_INPUTS_USE_G3_CONFIG')
    g4 = bool(cfg.get('group4')) and native
    if g4 and not (ad.data.has_g4 and ad.data.has_diag):
        raise ValueError('G4_CONFIG_REQUIRES_GROUP4_TARGETS_AND_DIAG_DEV_IN_PACK')
    if g4 and 'E' not in cfg['cell']:
        raise ValueError('G4_ONLY_WITH_RETRO_E_S_INPUTS')            # COORDINATOR_G3_G4_DESIGN_DECISION_V1: CT_G4 not scheduled
    if not g4 and ad.data.has_g4:
        raise ValueError('G4_PACK_HAS_CORRUPTED_TRAIN_INPUTS_USE_G4_CONFIG')
    sel_weights = cfg.get('selection_weights', L.STAGE1_WEIGHTS)

    ckpt = out / 'checkpoint.pt'
    state = None
    if ckpt.exists():
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        ad.load_trainable(model, payload['model']); opt.load_state_dict(payload['optimizer'])
        random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if payload.get('cuda_rng') is not None and a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        state = payload['state']
        if state['updates_total'] != updates:
            raise ValueError('UPDATE_COUNT_CHANGED_ON_RESUME')
        state['resumes'] += 1
        append_jsonl(out / 'events.jsonl', dict(at=now(), event='resume', global_step=state['global_step'], epoch=state['epoch'], cursor=state['cursor']))
    else:
        state = dict(epoch=0, cursor=0, global_step=0, order=None, best_dev=None, best_epoch=None, pending_dev=None,
                     updates_total=updates, first_started_at=now(), resumes=0, train_windows=len(train_ids), dev_windows=len(dev_ids),
                     disabled_grad_check=None, cumulative_train_seconds=0.0)
        append_jsonl(out / 'events.jsonl', dict(at=now(), event='start', updates_total=updates, train_windows=len(train_ids),
                                                dev_windows=len(dev_ids), data_seconds=data_seconds, es_coverage=es_cov,
                                                es_input_source=pack_manifest.get('es_input_source')))

    def save_weights(path):
        # model-only file for predictions (best/final); resume always uses checkpoint.pt
        atomic_torch_save(torch, dict(state=state, model=ad.trainable_state(model), weights_only=True), path)

    def save(path=ckpt):
        payload = dict(state=state, model=ad.trainable_state(model), optimizer=opt.state_dict(), python_rng=random.getstate(),
                       numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None)
        atomic_torch_save(torch, payload, path)

    def budget_left():
        return (budget - prior - (time.time() - t_start)) if budget else float('inf')

    def expired():
        heartbeat()
        if stop[0] is not None:
            return True
        if budget_left() < 120:
            return True
        if a.max_updates is not None and state['global_step'] >= a.max_updates:
            return True
        return time.time() - t_start >= a.max_seconds - 60

    def evaluate_native(ids):
        """V2 selection: L_p (namespace mean) -> mean over eligible positions of a trace -> mean over traces
        of a question -> equal mean over questions. Per-namespace head metrics keep own denominators."""
        model.eval(); per_trace = {}; heads = {}
        with torch.no_grad():
            for g in ad.micro_batches(ids, train=False):
                if expired():
                    return None
                b = ad.batch(g); o = model(b)
                # selection = next-step objective only (same as G1), so groups are compared on the same DEV criterion;
                # G2 heads are reported separately
                Lp, elig, comps = L.native_position_terms(o, b, accepted, event_ok, group1=False)
                if g1:
                    comps = L.native_position_terms(o, b, accepted, event_ok, group1=True)[2]
                for i, u in enumerate(b['_unit_ids']):
                    r = per_trace.setdefault(u, [b['_question_ids'][i], 0., 0])
                    r[1] += float((Lp[i] * elig[i]).sum()); r[2] += int(elig[i].sum())
                for j, c in enumerate(comps):
                    for k, (v, m) in c.items():
                        h = heads.setdefault(f'ns{j}_{k}', [0., 0]); h[0] += float((v * m).sum()); h[1] += int(m.sum())
        model.train()
        per_q = {}
        for q, ssum, cnt in per_trace.values():
            if cnt:
                per_q.setdefault(q, []).append(ssum / cnt)
        if not per_q:
            raise ValueError('DEV_WITHOUT_ELIGIBLE_POSITIONS')
        score = sum(sum(v) / len(v) for v in per_q.values()) / len(per_q)
        means = {k: (v[0] / v[1] if v[1] else None) for k, v in heads.items()}
        for j, ok in enumerate(event_ok):
            if not ok:
                means[f'ns{j}_event'] = namespace_policy.NOT_TRAINED
                if g1:
                    means[f'ns{j}_span_events'] = namespace_policy.NOT_TRAINED
        means.update({k + '_cells': v[1] for k, v in heads.items()})
        means.update(questions=len(per_q), traces=sum(len(v) for v in per_q.values()))
        if g3 or g4:
            # G3/G4 heads only on the fixed diagnostic DEV mask/corruption (diag_* inputs); never part of the selection score
            ad.data.use_inputs('diag'); model.eval(); h3 = {}
            try:
                with torch.no_grad():
                    for g in ad.micro_batches(ids, train=False):
                        if expired():
                            return None
                        b = ad.batch(g); o = model(b)
                        comps3 = L.native_position_terms(o, b, accepted, event_ok, group1=False, group3=g3, group4=g4)[2]
                        for j, c in enumerate(comps3):
                            for k in ('g3_type', 'g3_events', 'g4_changed', 'g4_kind'):
                                if k not in c:
                                    continue
                                v, m = c[k]; key = k if '_g4' in c else f'ns{j}_{k}'
                                hh = h3.setdefault(key, [0., 0]); hh[0] += float((v * m).sum()); hh[1] += int(m.sum())
            finally:
                ad.data.use_inputs(None); model.train()
            means.update({k: (v[0] / v[1] if v[1] else None) for k, v in h3.items()})
            means.update({k + '_cells': v[1] for k, v in h3.items()})
            for j, ok in enumerate(event_ok):
                if not ok and g3:
                    means[f'ns{j}_g3_events'] = namespace_policy.NOT_TRAINED
        return score, means

    def evaluate(ids):
        model.eval(); per_q = {}
        with torch.no_grad():
            for g in ad.micro_batches(ids, train=False):
                if expired():
                    return None
                b = ad.batch(g); o = model(b)
                terms = L.v9_core_terms(o, b)
                if 'next_span' in o:
                    terms.update(L.group1_terms(o, b))
                for t, (v, m) in terms.items():
                    if v.ndim == 3:
                        v = (v * m).sum(-1); m = m.sum(-1)
                    else:
                        v = v * m; m = m.long()
                    rows = per_q.setdefault(t, {})
                    for i, q in enumerate(b['_question_ids']):
                        r = rows.setdefault(q, [0., 0]); r[0] += float(v[i].sum()); r[1] += int(m[i].sum())
        model.train()
        return per_q

    log_every = cfg.get('log_every', 20)
    t_seg = time.time(); steps_seg = 0
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    while state['epoch'] < passes:
        if state['order'] is None:
            order = list(train_ids); random.Random(a.seed * 1000003 + state['epoch']).shuffle(order)
            state['order'] = order; state['cursor'] = 0
        model.train()
        while state['cursor'] < len(state['order']):
            if expired():
                save(); why = 'budget' if budget_left() < 120 else ('signal' if stop[0] is not None else 'deadline_or_cap')
                append_jsonl(out / 'events.jsonl', dict(at=now(), event='stop_checkpointed', global_step=state['global_step'], signal=stop[0], reason=why,
                                                        cumulative_seconds=prior + time.time() - t_start))
                heartbeat(final=why); print('STOP_CHECKPOINTED', state['global_step'], why)
                return EXIT_BUDGET if why == 'budget' else EXIT_RESUMABLE
            big = state['order'][state['cursor']:state['cursor'] + eb]
            opt.zero_grad(set_to_none=True)
            parts_acc = {}
            # normalise accumulated micro-batches by the effective batch so that the update equals
            # the loss over the whole effective batch window set (weighted means per micro-batch)
            groups = ad.micro_batches(big, train=True); nmb = len(groups)
            batches = [ad.batch(g) for g in groups]
            den = sum(L.native_eligible_weight(b, accepted, event_ok, group1=g1, group3=g3, group4=g4) for b in batches) if native else None
            for b in batches:
                o = model(b)
                if native:
                    # common denominator over the effective batch: accumulation == one batch of eb windows
                    loss, parts = L.native_total_loss(o, b, accepted, denominator=den, event_eligible=event_ok, group1=g1, group3=g3, group4=g4)
                    scale = 1.0
                else:
                    loss, parts, _ = L.total_loss(o, b, weights)
                    scale = 1.0 / nmb
                if not torch.isfinite(loss):
                    save(); raise ValueError('NONFINITE_LOSS_STOP_NO_RETRY')
                (loss * scale).backward()
                for kk, v in parts.items():
                    parts_acc[kk] = parts_acc.get(kk, 0.) + float(v.detach()) * (1.0 if (native and kk == 'native_total') else 1.0 / nmb)
            del batches
            if state['disabled_grad_check'] is None:
                bad = [n for n, p in disabled_head_params(model) if p.grad is not None and float(p.grad.abs().sum()) != 0.]
                none = [n for n, p in disabled_head_params(model) if p.grad is None]
                state['disabled_grad_check'] = dict(nonzero=bad, none_count=len(none), checked=len(disabled_head_params(model)))
                append_jsonl(out / 'events.jsonl', dict(at=now(), event='disabled_grad_check', **state['disabled_grad_check']))
                if bad and not any(k in weights for k in ('length', 'wait', 'remainder')):
                    raise ValueError('DISABLED_HEAD_HAS_GRADIENT:' + ','.join(bad))
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], ocfg['clip_norm'], error_if_nonfinite=True)
            f = lr_factor(state['global_step'], updates, ocfg['warmup_updates'], ocfg['cosine_floor'])
            for g in opt.param_groups:
                g['lr'] = g['initial_lr'] * f
            opt.step(); state['global_step'] += 1; state['cursor'] += len(big); steps_seg += 1
            if state['global_step'] % log_every == 0 or state['global_step'] <= 3:
                dt = time.time() - t_seg
                rec = dict(at=now(), step=state['global_step'], epoch=state['epoch'], lr_factor=f, grad_norm=float(gn),
                           sec_per_update=dt / max(1, steps_seg), host_rss=rss_bytes(), **{'loss_' + k: v for k, v in parts_acc.items()})
                if a.device.startswith('cuda'):
                    rec.update(gpu_max_alloc=torch.cuda.max_memory_allocated(), gpu_max_reserved=torch.cuda.max_memory_reserved())
                append_jsonl(out / 'train_log.jsonl', rec); t_seg = time.time(); steps_seg = 0
            if state['global_step'] % cfg['checkpoint_every_updates'] == 0:
                save()
        if state.get('pending_dev') is None:
            if native:
                res = evaluate_native(dev_ids)
            else:
                per_q = evaluate(dev_ids)
                res = None if per_q is None else dev_score(per_q, sel_weights)
            if res is None:
                save(); why = 'budget' if budget_left() < 120 else 'stopped_in_dev'
                heartbeat(final=why); append_jsonl(out / 'events.jsonl', dict(at=now(), event='stop_in_dev', reason=why, global_step=state['global_step']))
                print('STOP_CHECKPOINTED_BEFORE_DEV', why); return EXIT_BUDGET if why == 'budget' else EXIT_RESUMABLE
            score, means = res
            state['pending_dev'] = dict(epoch=state['epoch'], score=score, heads=means); save()
        pend = state['pending_dev']
        if pend['epoch'] != state['epoch']:
            raise ValueError('DEV_RESUME_EPOCH_MISMATCH')
        if state['best_dev'] is None or pend['score'] < state['best_dev'] or pend.get('selected_as_best'):
            state['best_dev'] = pend['score']; state['best_epoch'] = pend['epoch']; pend['selected_as_best'] = True
            save_weights(out / 'best.pt')
        metric = dict(epoch=state['epoch'] + 1, step=state['global_step'], selection=pend['score'], dev=pend['heads'], best_epoch=state['best_epoch'] + 1)
        mpath = out / 'metrics.jsonl'
        existing = [json.loads(x) for x in mpath.read_text().splitlines()] if mpath.exists() else []
        if not any(r['epoch'] == metric['epoch'] for r in existing):
            append_jsonl(mpath, metric)
        state['epoch'] += 1; state['order'] = None; state['cursor'] = 0; state['pending_dev'] = None; save()
    save_weights(out / 'final.pt')
    atomic_json(out / 'COMPLETE.json', dict(at=now(), binding_sha256=sha_file(bpath), global_step=state['global_step'],
                                            passes=state['epoch'], best_dev=state['best_dev'], best_epoch=state['best_epoch'] + 1,
                                            resumes=state['resumes'], cumulative_seconds=prior + time.time() - t_start, attempts=attempt_no, checkpoint_sha256=sha_file(ckpt), final_sha256=sha_file(out / 'final.pt'), best_sha256=sha_file(out / 'best.pt')))
    heartbeat(final='complete'); print('COMPLETE', state['global_step']); return 0


if __name__ == '__main__':
    sys.exit(main())
