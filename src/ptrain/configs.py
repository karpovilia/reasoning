"""Generate paired first-stage predictor configurations."""
import json, sys
from pathlib import Path

SEEDS = [42, 43, 44]
CELLS = ['CT', 'CTES']
OPT_A = dict(lr=3e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8, clip_norm=1.0, warmup_updates=100, cosine_floor=0.1)
OPT_B = dict(adapter_lr=1e-4, heads_lr=3e-4, lr=3e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8, clip_norm=1.0,
             warmup_updates=100, cosine_floor=0.1)
STAGE1 = {'joint': 1.0, 'destination': 0.5, 'event': 0.25}
COMMON = dict(effective_batch=64, task_weights=STAGE1, selection_weights=STAGE1, checkpoint_every_updates=200,
              es_input_policy='prefix_proven_only', log_every=20,
              selection_rule='after each completed pass on fixed natural DEV windows: native namespaces -> L[n,p] = I_joint*L_joint + 0.5*I_dest*L_dest '
                             '+ 0.25*I_ev*mean_known_axes(BCE) (no division by enabled components); mean over accepted namespaces available at p; '
                             'mean over eligible positions of a trace; mean over traces of a question; equal mean over questions. Lowest wins, ties '
                             'keep the earliest pass; TEST never read')


def arm(arch, cell):
    if arch == 'A':
        c = dict(arch='A', cell=cell, tf32=False, micro_batch=64, optimizer=OPT_A, passes=8, fit_budget_seconds=3600,
                 model=dict(width=512, heads=8, layers=12, ff_width=2048, dropout=0.1, max_segments=256),
                 text_encoder='sentence-transformers/all-MiniLM-L6-v2, fixed, 256 wordpieces, mean pooling, L2, float16 cache')
    else:
        c = dict(arch='B', cell=cell, tf32=False, micro_batch=1, optimizer=OPT_B, passes=2, fit_budget_seconds=21600,
                 model=dict(model_id='Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554',
                            dtype='bfloat16', token_context=8192, max_segments=256, question_max_tokens=1024,
                            segment_max_tokens=512, context_tokens=4096, context_segments=128, gradient_checkpointing=False, head_width=512,
                            micro_tokens=8192, eval_micro_tokens=32768,
                            lora=dict(r=32, alpha=64, dropout=0.05, bias='none',
                                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])))
    c.update(COMMON); c['name'] = f'{arch}_{cell}'
    return c


def main(outdir, matrix_path=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fits = []
    for arch in ('A', 'B'):
        for cell in CELLS:
            c = arm(arch, cell); p = out / f'{arch}_{cell}.json'
            p.write_text(json.dumps(c, indent=1))
            for s in SEEDS:
                fits.append(dict(fit=f'{arch}_{cell}_seed{s}', arch=arch, cell=cell, seed=s, config=str(p.resolve()), task='next_structural_step'))
    m = dict(schema='predictor_training_matrix_v1', first_stage=fits, seeds=SEEDS,
             pairing='CT and CTES of one architecture and seed share initial weights (torch.manual_seed(seed) before build, identical '
                     'parameter set) and the sampler order (random.Random(seed*1000003+epoch))',
             later_groups=['g1_event_next_span_events', 'g2_masked_reconstruction', 'g3_annotation_error', 'g4_verified_reasoning_error'])
    if matrix_path:
        Path(matrix_path).write_text(json.dumps(m, indent=1))
    return m


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
