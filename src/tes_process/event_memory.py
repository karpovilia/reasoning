"""Explicit event-anchored memory on the TES observation grid.

This is a binned process, not an exact continuous-time point-process likelihood.
Each memory coordinate is a decaying count of ONE named event type. Events can
attenuate older event traces; only local memory attenuates at span boundaries.
There is no learned recurrent matrix, GRU cell, or hidden-state rotation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EventMemory(nn.Module):
    def __init__(self, family, K, E, L=12):
        super().__init__()
        self.family, self.K, self.E, self.L = family, K, E, L
        self.with_text = family.endswith('_text')
        self.erasure = family not in ('explicit_additive_text', 'explicit_additive')
        self.reset_persistent = family.startswith('explicit_reset')
        # Character half-lives; fixed unit event increments anchor state scale.
        h = torch.tensor([128., 2048.]).log()[None, :, None].repeat(K, 1, E)
        self.log_half_life = nn.Parameter(h)
        self.boundary_logit = nn.Parameter(torch.full((K, K, E), -1.))
        if self.erasure:
            self.erase_raw = nn.Parameter(torch.full((K, 2, E, E), -5.))
        self.initial_mean_raw = nn.Parameter(torch.full((2, E), -2.))
        self.initial_var_raw = nn.Parameter(torch.zeros(2, E))
        self.initial_event_logit = nn.Parameter(torch.full((E,), -2.))
        D = K + 16 + 5 + 4*E + (64 if self.with_text else 0)
        self.gate = nn.Linear(D, K + 2)
        self.events = nn.Linear(D, E)
        self.lengths = nn.Linear(D, L)
        # Events and lengths inside the next span depend on its actual mode.
        self.event_mode = nn.Parameter(torch.zeros(K, E))
        self.length_mode = nn.Parameter(torch.zeros(K, L))
        if self.with_text:
            self.future_text = nn.Linear(D, 64)

    def step(self, b, state=None):
        base = b['base']
        n = len(base)
        device = base.device
        observed_mode = base[:, :self.K]
        if state is None:
            m = torch.zeros(n, 2, self.E, device=device)
            var = torch.zeros_like(m)
            previous_mode = torch.full((n, self.K), 1/self.K, device=device)
            expected_mode = previous_mode
            event_prior = self.initial_event_logit.sigmoid().expand(n, self.E)
        else:
            m, var, previous_mode, expected_mode, event_prior = state
        mode = torch.where(observed_mode.sum(-1, keepdim=True) > 0,
                           observed_mode, torch.full_like(observed_mode,1/self.K))
        event_prior = self.initial_event_logit.sigmoid().expand(n,self.E)
        reset = b.get('reset', torch.zeros(n, dtype=torch.bool, device=device)).bool()
        true_start = b.get('true_start', reset).bool()
        # Gap/left-censoring starts an uncertain inference prior, NOT a physical
        # reset of the process. This version uses a diagonal moment approximation.
        prior_m = F.softplus(self.initial_mean_raw).expand(n, 2, self.E)
        prior_v = F.softplus(self.initial_var_raw).expand(n, 2, self.E)
        m = torch.where(reset[:, None, None], prior_m, m)
        var = torch.where(reset[:, None, None], prior_v, var)
        m = torch.where(true_start[:, None, None], torch.zeros_like(m), m)
        var = torch.where(true_start[:, None, None], torch.zeros_like(var), var)
        boundary = b.get('start', torch.zeros(n, dtype=torch.bool, device=device)).bool() & ~reset
        keep_local = torch.einsum('ni,ije,nj->ne', previous_mode,
                                  self.boundary_logit.sigmoid(), mode)
        keep = torch.stack([keep_local, torch.zeros_like(keep_local)
                            if self.reset_persistent else torch.ones_like(keep_local)], 1)
        keep = torch.where(boundary[:, None, None], keep, torch.ones_like(keep))
        half_life = torch.einsum('nk,kbe->nbe', mode,
                                 self.log_half_life.clamp(0, math.log(65536)).exp())
        decay = torch.exp(-math.log(2)*b['length'][:, None, None]/half_life)
        m, var = m*keep*decay, var*(keep*decay).square()
        known = b['known'].bool()
        event = torch.where(known, b['event'], event_prior)
        if self.erasure:
            strength = torch.einsum('nk,kbij->nbij', mode, F.softplus(self.erase_raw))
            retain = torch.exp(-torch.einsum('ni,nbij->nbj', event, strength))
            m, var = m*retain, var*retain.square()
        # Tied marks act as an unordered set. No fabricated within-bin order.
        m = m + event[:, None, :]
        var = var + ((~known)*event_prior*(1-event_prior))[:, None, :]
        out,next_mode,event_prob = self.decode(base,m,var,mode,b)
        return out, (m, var, mode, next_mode, event_prob)

    def decode(self,base,m,var,mode,b):
        pieces = [base, torch.log1p(m).flatten(-2), torch.log1p(var).flatten(-2)]
        if self.with_text:
            pieces += [b['text'], b['text_ema']]
        x = torch.cat(pieces, -1)
        gate = self.gate(x)
        # Conditional next-mode probabilities, given that the process continues.
        gp = gate.softmax(-1)
        next_mode = gp[..., 2:] + gp[..., :1]*mode
        next_mode = next_mode / next_mode.sum(-1, keepdim=True).clamp_min(1e-8)
        event_cond = self.events(x)[..., None, :] + self.event_mode
        length_cond = self.lengths(x)[..., None, :] + self.length_mode
        event_prob = (next_mode[...,None]*event_cond.sigmoid()).sum(-2)
        length_prob = (next_mode[...,None]*length_cond.softmax(-1)).sum(-2)
        out = dict(gate=gate, event=torch.logit(event_prob.clamp(1e-6, 1-1e-6)),
                   length=length_prob.clamp_min(1e-8).log(),
                   event_cond=event_cond, length_cond=length_cond)
        if self.with_text:
            z = self.future_text(x)
            out.update(text_mean=z[..., :32], text_logsd=z[..., 32:].clamp(-4, 1))
        return out,next_mode,event_prob

    def forward(self, b):
        # The explicit memory law is affine. Associative prefix composition gives
        # the SAME full-history recurrence in O(log T) tensor operations, rather
        # than dispatching a Python/neural recurrent cell at every position.
        base=b['base'];mode=base[...,:self.K]
        mode=torch.where(mode.sum(-1,keepdim=True)>0,mode,torch.full_like(mode,1/self.K))
        previous=torch.cat([torch.full_like(mode[:,:1],1/self.K),mode[:,:-1]],1)
        keep_local=torch.einsum('nti,ije,ntj->nte',previous,self.boundary_logit.sigmoid(),mode)
        keep=torch.stack([keep_local,torch.zeros_like(keep_local) if self.reset_persistent else torch.ones_like(keep_local)],-2)
        boundary=b['start'].bool()&~b['reset'].bool()
        keep=torch.where(boundary[...,None,None],keep,torch.ones_like(keep))
        half=torch.einsum('ntk,kbe->ntbe',mode,self.log_half_life.clamp(0,math.log(65536)).exp())
        a=keep*torch.exp(-math.log(2)*b['length'][...,None,None]/half)
        p=self.initial_event_logit.sigmoid()
        event=torch.where(b['known'].bool(),b['event'],p)
        if self.erasure:
            strength=torch.einsum('ntk,kbij->ntbij',mode,F.softplus(self.erase_raw))
            a=a*torch.exp(-torch.einsum('nti,ntbij->ntbj',event,strength))
        noise=((~b['known'].bool())*p*(1-p))[...,None,:].expand_as(a)
        delta=event[...,None,:].expand_as(a)
        uncertain_reset=b['reset'].bool()&~b['true_start'].bool()
        delta=delta+torch.where(uncertain_reset[...,None,None],a*F.softplus(self.initial_mean_raw),torch.zeros_like(a))
        noise=noise+torch.where(uncertain_reset[...,None,None],a.square()*F.softplus(self.initial_var_raw),torch.zeros_like(a))
        a=torch.where((b['reset'].bool()|b['true_start'].bool())[...,None,None],torch.zeros_like(a),a)
        def scan(mult,add):
            offset=1
            while offset<add.shape[1]:
                add=torch.cat([add[:,:offset],add[:,offset:]+mult[:,offset:]*add[:,:-offset]],1)
                mult=torch.cat([mult[:,:offset],mult[:,offset:]*mult[:,:-offset]],1)
                offset*=2
            return add
        m=scan(a,delta);var=scan(a.square(),noise)
        return self.decode(base,m,var,mode,b)[0]

    def regularization(self):
        # Sparsity penalty on erasing interactions; predictive losses remain primary.
        return .0001*F.softplus(self.erase_raw).mean() if self.erasure else self.gate.weight.sum()*0

    def rules(self, modes, events):
        with torch.no_grad():
            half = self.log_half_life.clamp(0, math.log(65536)).exp().cpu().tolist()
            keep = self.boundary_logit.sigmoid().cpu().tolist()
            erase = F.softplus(self.erase_raw).cpu().tolist() if self.erasure else None
        return dict(mode_names=modes, event_names=events,
                    memory_banks=['local', 'persistent'], half_lives_characters=half,
                    boundary_local_retention=keep, persistent_boundary_retention=0 if self.reset_persistent else 1,
                    event_erase_strength=erase,
                    erase_axes=['mode', 'bank', 'incoming_event', 'older_event_trace'],
                    event_increments='one for the corresponding event type in both banks',
                    measurement='binary event presence per segment; not exact repeated-event counts',
                    missing='diagonal moment approximation; learned event prior and uniform mode prior for missing labels; never zero-labelled')
