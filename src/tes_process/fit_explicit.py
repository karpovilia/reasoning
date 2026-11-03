"""Full observed-prefix fitting, with identical splits for explicit and neural models."""
import collections
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT, atomic, read, digest, beat
from .models import Process, losses
from .event_memory import EventMemory

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
FIELDS = ['base','memory','event','known','text','text_ema','length','reset',
          'start','true_start','mode','next_mode','gate','gknown','cont',
          'next_event','next_known','next_length','next_text','weight']


def batch(data, items):
    T = max(m['n'] for m in items)
    b = {}
    for key in FIELDS:
        a = data[key]
        z = np.zeros((len(items), T)+a.shape[1:], a.dtype)
        for i,m in enumerate(items):
            z[i,:m['n']] = a[m['start']:m['start']+m['n']]
        b[key] = torch.from_numpy(z)
    b['score'] = torch.arange(T)[None,:] < torch.tensor([m['n'] for m in items])[:,None]
    return b


def model_loss(out, b):
    # Conditional observation factors make the event/length law consistent with
    # the sampled next mode. Unknown next modes are marginalized using the model.
    conditional = dict(out)
    if 'event_cond' in out:
        mode = b['next_mode'].long()
        index = mode.clamp_min(0)[...,None,None]
        selected = out['event_cond'].gather(-2,index.expand(*mode.shape,1,out['event'].shape[-1])).squeeze(-2)
        conditional['event'] = torch.where((mode>=0)[...,None],selected,out['event'])
        selected = out['length_cond'].gather(-2,index.expand(*mode.shape,1,out['length'].shape[-1])).squeeze(-2)
        conditional['length'] = torch.where((mode>=0)[...,None],selected,out['length'])
    return losses(conditional,b)[0]


def evaluate(net, data, items, batch_size=8):
    net.eval()
    values = collections.defaultdict(lambda:collections.defaultdict(list))
    with torch.no_grad():
        for k in range(0,len(items),batch_size):
            current=items[k:k+batch_size];b=batch(data,current);out=net(b)
            _,loss=losses(out,b)
            end_brier=(out['gate'].softmax(-1)[...,1]-(b['gate']==1).float()).square()
            metrics={'gate_nll':(loss['gate'],b['gknown']),
                     'event_bce':(loss['events'],loss['event_valid']),
                     'next_length_bin_nll':(loss['length'],b['cont']),
                     'END_brier':(end_brier,b['gknown'])}
            for name,(v,valid) in metrics.items():
                for i,m in enumerate(current):
                    mask=(valid[i]*b['score'][i]).float()
                    if mask.sum():
                        values[name][m['question']].append(float((v[i]*mask).sum()/mask.sum()))
    return {name:{q:float(np.mean(v)) for q,v in qs.items()} for name,qs in values.items()}


def main(ni,family,seed):
    ni,seed=int(ni),int(seed)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    task=f'ns{ni}_{family}_seed{seed}';out=ROOT/'results'/task;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    spec=read(ROOT/'prepared'/f'ns{ni}.json');data=dict(np.load(ROOT/'prepared'/f'ns{ni}.npz'))
    assert digest(ROOT/'prepared'/f'ns{ni}.npz')==spec['data_sha256']
    train,tune,dev=([m for m in spec['sequences'] if m['role']==r] for r in [0,1,2])
    if not train or not tune or not dev:
        atomic(out/'COMPLETE.json',dict(status='NOT_ESTIMABLE',reason='Missing fit, TRAIN tune or DEV sequences',
            namespace=spec['namespace'],family=family,seed=seed,
            fit_traces=len(train),tune_traces=len(tune),dev_traces=len(dev)));return
    net=EventMemory(family,spec['K'],spec['E']) if family.startswith('explicit_') else Process(family,spec['K'],spec['E'])
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.0001)
    cfg=read(ROOT/'RUN.json');rng=np.random.default_rng(seed)
    best=float('inf');stale=0;epoch0=0;history=[]
    checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        c=torch.load(checkpoint,weights_only=False)
        net.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        rng.bit_generator.state=c['rng'];torch.set_rng_state(c['torch_rng'])
        best,stale,epoch0,history=c['best'],c['stale'],c['epoch']+1,c['history']
    qn=len({m['question'] for m in train})
    atomic(out/'CONTRACT.json',dict(family=family,seed=seed,namespace=spec['namespace'],
        data_sha256=spec['data_sha256'],parameters=sum(p.numel() for p in net.parameters()),
        fit_traces=len(train),fit_questions=qn,tune_questions=len({m['question'] for m in tune}),
        dev_questions=len({m['question'] for m in dev}),
        code={p.name:digest(p) for p in sorted(ROOT.glob('*.py'))},
        history='Full observed traces, no arbitrary 32/64-position state resets; gradient covers entire trace',
        data_scope='All registered linked TRAIN/DEV observations at the new frozen cut; namespace-specific fits',
        observation='Retrospective machine annotation; segment grid in characters; no online detector or exact point-process claim',
        missing='Explicit model uses diagonal memory moments; gaps restart inference from an uncertain learned prior',
        objective='Question-weighted gate CE + event BCE + length CE + 0.1 content NLL; explicit event/length factors condition on next mode',
        text='32-dimensional signed lexical sketch and causal EMA; no semantic LM embedding'))
    for epoch in ([] if (out/'FIT_DONE.json').exists() else range(epoch0,cfg['epochs'])):
        epoch_started=time.monotonic()
        net.train();order=rng.permutation(len(train));total=den=0.
        # Length bucketing within random pools avoids excessive padding without
        # giving any trace more than one training appearance per epoch.
        order=np.concatenate([p[np.argsort([train[i]['n'] for i in p])]
                              for p in np.array_split(order,max(1,len(order)//128))])
        for k in range(0,len(order),cfg['batch_size']):
            current=[train[i] for i in order[k:k+cfg['batch_size']]]
            b=batch(data,current);prediction=net(b);loss=model_loss(prediction,b)
            w=b['weight']*b['score']
            value=(loss*w).sum()*len(train)/(len(current)*qn)
            if hasattr(net,'regularization'):value=value+net.regularization()
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite training objective')
            opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            total+=float((loss.detach()*w).sum());den+=float(w.sum())
            if k%(cfg['batch_size']*4)==0:beat('fit',task=task,epoch=epoch,traces=k,total_traces=len(train))
        scores=evaluate(net,data,tune,cfg['batch_size'])
        score=float(sum(np.mean(list(scores[m].values())) for m in ['gate_nll','event_bce','next_length_bin_nll'] if scores.get(m)))
        history.append(dict(epoch=epoch,fit_loss=total/den,TRAIN_tune_score=score,epoch_seconds=time.monotonic()-epoch_started))
        if score<best-1e-5:
            best,stale=score,0
            torch.save(net.state_dict(),out/'best.tmp.pt');(out/'best.tmp.pt').replace(out/'best.pt')
        else:stale+=1
        torch.save(dict(epoch=epoch,model=net.state_dict(),optimizer=opt.state_dict(),
                        rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                        best=best,stale=stale,history=history),out/'checkpoint.tmp.pt')
        (out/'checkpoint.tmp.pt').replace(checkpoint);atomic(out/'HISTORY.json',history)
        beat('epoch_complete',task=task,epoch=epoch,best=best,stale=stale,epoch_seconds=history[-1]['epoch_seconds'])
        if epoch+1>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    net.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    perq=evaluate(net,data,dev,cfg['batch_size']);atomic(out/'PER_QUESTION.json',perq)
    atomic(out/'METRICS.json',{k:dict(mean=float(np.mean(list(v.values()))),questions=len(v)) for k,v in perq.items()})
    if hasattr(net,'rules'):atomic(out/'LEARNED_RULES.json',net.rules(spec['mode_vocab'],spec['event_vocab']))
    atomic(out/'FIT_DONE.json',dict(status='FIT_AND_ONE_STEP_COMPLETE',epochs=len(history),model_sha256=digest(out/'best.pt')))
    from .rollout_process import run
    run(ni,family,seed,net,data,spec,out)
    atomic(out/'COMPLETE.json',dict(status='COMPLETE',model_sha256=digest(out/'best.pt'),
                                   rollout_sha256=digest(out/'ROLLOUT.json')))
    beat('condition_complete',task=task)

if __name__=='__main__':main(*sys.argv[1:])
