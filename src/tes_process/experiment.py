"""Matched readout refits on frozen states, and explicit-memory ablations."""
import collections
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import ROOT, read, atomic, digest, beat
from .models import Process
from .event_memory import EventMemory
from .fit_explicit import batch


def save_torch(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt'); torch.save(obj, tmp); tmp.replace(path)


class Encoder(nn.Module):
    def __init__(self, source, variant, seed, spec):
        super().__init__()
        self.variant, self.K, self.E = variant, spec['K'], spec['E']
        family = 'gru_text' if variant == 'gru' else 'explicit_erasure_text'
        self.net = (Process(family, self.K, self.E) if variant == 'gru'
                    else EventMemory(family, self.K, self.E))
        path = Path(source)/'results'/f'ns9_{family}_seed{seed}'/'best.pt'
        self.source_sha = digest(path)
        self.net.load_state_dict(torch.load(path, weights_only=True))
        if variant == 'explicit_reset_both':
            self.net.reset_persistent = True
            with torch.no_grad(): self.net.boundary_logit.fill_(-10000.)
        self.net.requires_grad_(False); self.net.eval()
        self.D = self.net.gate.in_features
        self.captured = []
        self.net.gate.register_forward_pre_hook(self.capture)

    def capture(self, module, args):
        self.captured.append(args[0].detach())

    def transform(self, x, b):
        if self.variant not in ('explicit_u_only', 'explicit_current_only'): return x
        x = x.clone(); B = self.K+16+5; E = self.E
        if self.variant == 'explicit_u_only':
            x[..., B+E:B+2*E] = 0
            x[..., B+3*E:B+4*E] = 0
        else:
            p = self.net.initial_event_logit.sigmoid()
            ev = torch.where(b['known'].bool(), b['event'], p)
            var = (~b['known'].bool())*p*(1-p)
            x[..., B:B+2*E] = torch.cat([torch.log1p(ev)]*2, -1)
            x[..., B+2*E:B+4*E] = torch.cat([torch.log1p(var)]*2, -1)
        return x

    def forward(self, b):
        self.captured = []
        with torch.no_grad(): self.net(b)
        if self.variant == 'gru': x = torch.stack(self.captured, 1)
        else: x = self.captured[0]
        self.captured = []
        return self.transform(x, b)

    def step(self, b, state=None):
        self.captured = []
        with torch.no_grad(): _, state = self.net.step(b, state)
        x = self.captured[0]; self.captured = []
        return self.transform(x, b), state


class Core(nn.Module):
    def __init__(self, D, K):
        super().__init__(); self.K=K
        self.gate=nn.Linear(D,K+2); self.lengths=nn.Linear(D,12)
        self.length_mode=nn.Parameter(torch.zeros(K,12)); self.future_text=nn.Linear(D,64)

    def forward(self,x,mode):
        gate=self.gate(x); gp=gate.softmax(-1)
        mode=torch.where(mode.sum(-1,keepdim=True)>0,mode,torch.full_like(mode,1/self.K))
        nxt=gp[...,2:]+gp[...,:1]*mode
        nxt=nxt/nxt.sum(-1,keepdim=True).clamp_min(1e-8)
        lc=self.lengths(x)[...,None,:]+self.length_mode
        lp=(nxt[...,None]*lc.softmax(-1)).sum(-2)
        z=self.future_text(x)
        return dict(gate=gate,next_mode=nxt,length_cond=lc,length=lp.clamp_min(1e-8).log(),
                    text_mean=z[...,:32],text_logsd=z[...,32:].clamp(-4,1))


class Events(nn.Module):
    def __init__(self,D,K,E,conditional):
        super().__init__(); self.K=K; self.conditional=conditional
        self.linear=nn.Linear(D,E)
        if conditional:self.offset=nn.Parameter(torch.zeros(K,E))

    def forward(self,x,next_mode):
        logits=self.linear(x)[...,None,:]
        if self.conditional: logits=logits+self.offset
        else: logits=logits.expand(*logits.shape[:-2],self.K,logits.shape[-1])
        p=(next_mode[...,None]*logits.sigmoid()).sum(-2).clamp(1e-6,1-1e-6)
        return dict(event_cond=logits,event=torch.log(p)-torch.log1p(-p))


class Generator(nn.Module):
    def __init__(self,encoder,core,events):
        super().__init__(); self.encoder=encoder; self.core=core; self.events=events
        self.with_text=True; self.K=encoder.K

    def decode(self,x,b):
        out=self.core(x,b['base'][...,:self.K]); out.update(self.events(x,out['next_mode']))
        return out

    def forward(self,b):return self.decode(self.encoder(b),b)

    def step(self,b,state=None):
        x,state=self.encoder.step(b,state)
        return self.decode(x,b),state


def cache(encoder,data,spec,path):
    if path.exists():return np.load(path)
    x=np.empty((len(data['base']),encoder.D),np.float32)
    seq=sorted(spec['sequences'],key=lambda m:m['n'])
    for k in range(0,len(seq),16):
        ms=seq[k:k+16]; b=batch(data,ms)
        with torch.no_grad(): z=encoder(b).numpy()
        for i,m in enumerate(ms):x[m['start']:m['start']+m['n']]=z[i,:m['n']]
        if k%128==0:beat('cache',variant=encoder.variant,traces=k,total=len(seq))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npy');np.save(tmp,x);tmp.replace(path)
    return x


def tensors(data, x, ids):
    keys=['base','gate','gknown','cont','next_mode','next_event','next_known','next_length','next_text','weight']
    b={k:torch.from_numpy(data[k][ids]) for k in keys}
    b['x']=torch.from_numpy(x[ids]);return b


def losses(out,b,stage,conditional=True,training=False):
    if stage=='core':
        g=F.cross_entropy(out['gate'],b['gate'].clamp_min(0).long(),reduction='none')*b['gknown']
        logits=out['length']
        if training:
            mode=b['next_mode'].long(); ix=mode.clamp_min(0)
            chosen=out['length_cond'][torch.arange(len(ix)),ix]
            logits=torch.where((mode>=0)[:,None],chosen,logits)
        l=F.cross_entropy(logits,b['next_length'].long(),reduction='none')*b['cont']
        if not training:return g,l
        z=out['text_logsd']; t=(.5*((b['next_text']-out['text_mean'])/z.exp()).square()+z).mean(-1)*b['cont']
        return g+l+.1*t
    logits=out['event']
    if training and conditional:
        mode=b['next_mode'].long(); chosen=out['event_cond'][torch.arange(len(mode)),mode.clamp_min(0)]
        logits=torch.where((mode>=0)[:,None],chosen,logits)
    mask=b['next_known'];den=mask.sum(-1)
    return (F.binary_cross_entropy_with_logits(logits,b['next_event'],reduction='none')*mask).sum(-1)/den.clamp_min(1)


def weighted_scores(values,data,spec,role):
    byq=collections.defaultdict(list)
    for m in spec['sequences']:
        if m['role']!=role:continue
        sl=slice(m['start'],m['start']+m['n']);v,mask=values[0][sl],values[1][sl]
        if mask.sum():byq[m['question']].append(float((v*mask).sum()/mask.sum()))
    return {q:float(np.mean(v)) for q,v in byq.items()}


def predict(core,events,data,x,K,role=None):
    acc={};active=np.arange(len(x)) if role is None else np.flatnonzero(data['role']==role)
    def put(name,ids,v):
        if name not in acc:acc[name]=np.zeros(len(x),np.float32)
        acc[name][ids]=v.numpy()
    with torch.no_grad():
        for k in range(0,len(active),8192):
            ids=active[k:k+8192];b=tensors(data,x,ids)
            o=core(b['x'],b['base'][:,:K]);g,l=losses(o,b,'core')
            put('gate_nll',ids,g);put('length_nll',ids,l)
            put('END_brier',ids,(o['gate'].softmax(-1)[:,1]-(b['gate']==1).float()).square())
            if events is not None:
                o.update(events(b['x'],o['next_mode']));put('event_bce',ids,losses(o,b,'events'))
    return acc


def fit(module,stage,core,data,x,spec,cfg,path,seed):
    receipt=path/'DONE.json'
    if receipt.exists():module.load_state_dict(torch.load(path/'best.pt',weights_only=True));return
    path.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(module.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    ids=np.flatnonzero(data['role']==0);wscale=float(data['weight'][ids].mean())
    best=float('inf');stale=0;epoch0=0;history=[];checkpoint=path/'checkpoint.pt'
    if checkpoint.exists():
        c=torch.load(checkpoint,weights_only=False);module.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        rng.bit_generator.state=c['rng'];best=c['best'];stale=c['stale'];epoch0=c['epoch']+1;history=c['history']
    for epoch in range(epoch0,cfg['epochs']):
        began=time.monotonic();order=rng.permutation(ids)
        for k in range(0,len(order),cfg['batch_size']):
            b=tensors(data,x,order[k:k+cfg['batch_size']])
            if stage=='core':out=module(b['x'],b['base'][:,:spec['K']])
            else:
                with torch.no_grad():out=core(b['x'],b['base'][:,:spec['K']])
                out.update(module(b['x'],out['next_mode']))
            val=losses(out,b,stage,getattr(module,'conditional',True),True)
            value=(val*b['weight']/wscale).mean()
            if not torch.isfinite(value):raise FloatingPointError('nonfinite head objective')
            opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),2.);opt.step()
        # Select only on the original internal TRAIN tuning questions.
        pp=predict(module if stage=='core' else core,None if stage=='core' else module,data,x,spec['K'],role=1)
        if stage=='core':
            score=sum(np.mean(list(weighted_scores((pp[k],data[mask]),data,spec,1).values()))
                      for k,mask in [('gate_nll','gknown'),('length_nll','cont')])
        else:score=np.mean(list(weighted_scores((pp['event_bce'],data['next_known'].sum(-1)>0),data,spec,1).values()))
        if score<best-1e-6:best=float(score);stale=0;save_torch(module.state_dict(),path/'best.pt')
        else:stale+=1
        history.append(dict(epoch=epoch,tune=float(score),seconds=time.monotonic()-began))
        save_torch(dict(epoch=epoch,model=module.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,
                        best=best,stale=stale,history=history),checkpoint)
        atomic(path/'HISTORY.json',history);beat('head_fit',head=str(path.relative_to(ROOT)),epoch=epoch,best=best)
        if epoch+1>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    module.load_state_dict(torch.load(path/'best.pt',weights_only=True))
    atomic(receipt,dict(status='COMPLETE',epochs=len(history),best_tune=best,sha256=digest(path/'best.pt')))


def check_environment(seed):
    import scipy
    actual = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__)
    path = ROOT / 'ENVIRONMENT.json'
    if path.exists() and read(path) != actual:
        raise RuntimeError('Numerical environment differs from the saved run')
    if not path.exists():
        atomic(path, actual)
    pins = ROOT / 'PINS.json'
    if pins.exists():
        for relative, value in read(pins).items():
            if digest(ROOT / relative) != value:
                raise ValueError('Input checksum mismatch: ' + relative)


def main(seed):
    seed=int(seed);cfg=read(ROOT/'RUN.json');check_environment(seed)
    spec=read(ROOT/'prepared/ns9.json');data=dict(np.load(ROOT/'prepared/ns9.npz'))
    source=str((ROOT/ cfg['source_root']).resolve());variants=['explicit_full','gru','explicit_u_only','explicit_reset_both','explicit_current_only']
    for variant in variants:
        conditional_values=[True,False] if variant in ('explicit_full','gru') else [True]
        if all((ROOT/'results'/f'{variant}_{"cond" if c else "marg"}_seed{seed}'/'COMPLETE.json').exists() for c in conditional_values):continue
        encoder=Encoder(source,variant,seed,spec)
        x=cache(encoder,data,spec,ROOT/'cache'/f'{variant}_seed{seed}.npy')
        torch.manual_seed(seed);core=Core(encoder.D,spec['K'])
        corepath=ROOT/'heads'/f'{variant}_seed{seed}'/'core'
        fit(core,'core',None,data,x,spec,cfg,corepath,seed)
        for conditional in conditional_values:
            name=f'{variant}_{"cond" if conditional else "marg"}_seed{seed}';out=ROOT/'results'/name
            if (out/'COMPLETE.json').exists():continue
            out.mkdir(parents=True,exist_ok=True)
            torch.manual_seed(seed+1000);events=Events(encoder.D,spec['K'],spec['E'],conditional)
            evpath=ROOT/'heads'/f'{variant}_seed{seed}'/('events_cond' if conditional else 'events_marg')
            fit(events,'events',core,data,x,spec,cfg,evpath,seed+1000)
            pp=predict(core,events,data,x,spec['K']);perq={}
            for k,mask in [('gate_nll',data['gknown']),('length_nll',data['cont']),('event_bce',data['next_known'].sum(-1)>0),('END_brier',data['gknown'])]:
                perq[k]=weighted_scores((pp[k],mask),data,spec,2)
            atomic(out/'PER_QUESTION.json',perq)
            atomic(out/'CONTRACT.json',dict(variant=variant,conditional=conditional,seed=seed,encoder_sha=encoder.source_sha,
                core_sha=digest(corepath/'best.pt'),event_sha=digest(evpath/'best.pt'),frozen_encoder=True,
                fit_questions=443,tune_questions=51,dev_questions=82,scope='Frozen-state readout refit; not end-to-end retraining'))
            generator=Generator(encoder,core,events);generator.eval()
            from .rollout_process import run
            run(9,name,seed,generator,data,spec,out)
            atomic(out/'COMPLETE.json',dict(status='COMPLETE',seed=seed,condition=name.rsplit('_seed',1)[0],
                core_sha=digest(corepath/'best.pt'),event_sha=digest(evpath/'best.pt'),finished=time.time()))
            beat('condition_complete',condition=name)
    atomic(ROOT/f'SEED_{seed}_COMPLETE.json',dict(status='COMPLETE',seed=seed,time=time.time()))


if __name__=='__main__':main(sys.argv[1])
