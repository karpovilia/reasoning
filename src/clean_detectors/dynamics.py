"""Train-only causal dynamics prior; no weights or labels from the oracle v2.

The last state forecasts the NEXT focus from completed past foci only. Detached
own probabilities and past text pools are inputs, direct labels are loss only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import masked_loss
from .selective import any_event_target


class DynamicsPrior(nn.Module):
    def __init__(self,text_width,cfg):
        super().__init__();self.context=cfg['context'];width=cfg['width']
        self.project=nn.Linear(text_width+14,width)
        self.bos=nn.Parameter(torch.zeros(1,1,width))
        self.position=nn.Embedding(self.context+1,width)
        layer=nn.TransformerEncoderLayer(width,cfg['heads'],4*width,cfg['dropout'],
            batch_first=True,norm_first=True,activation='gelu')
        self.core=nn.TransformerEncoder(layer,cfg['layers'],enable_nested_tensor=False)
        # Independent initialization for layers cloned by TransformerEncoder.
        for item in self.core.modules():
            if isinstance(item,nn.Linear):nn.init.xavier_uniform_(item.weight)
            if isinstance(item,nn.MultiheadAttention):nn.init.xavier_uniform_(item.in_proj_weight)
        self.norm=nn.LayerNorm(width);self.events=nn.Linear(width,6)
        self.modes=nn.Linear(width,6);self.any_event=nn.Linear(width,1)

    def forward(self,entries,target_sid,device):
        entries=entries[-self.context:]
        if any(e['row']['sid']>=target_sid or e['as_of_sid']>=target_sid for e in entries):
            raise ValueError('Dynamics prior contains current/future information')
        # Explicit field selection: labels, masks, full trace lengths and solver
        # annotations cannot enter the forecast, even if present in stored rows.
        h=self.bos
        if entries:
            text=torch.stack([e['pooled'].detach().float() for e in entries])
            scores=torch.stack([e['scores'].detach().float() for e in entries])
            time=torch.tensor([[(target_sid-e['row']['sid'])/self.context,
                math.log1p(e['row']['sid'])/10.] for e in entries],device=device,dtype=torch.float32)
            x=torch.cat((text,scores,time),-1)
            h=torch.cat((h,self.project(x)[None]),1)
        n=h.shape[1];h=h+self.position(torch.arange(n,device=device))[None]
        causal=torch.triu(torch.ones(n,n,device=device,dtype=torch.bool),diagonal=1)
        h=self.norm(self.core(h,mask=causal)[:,-1])
        return self.events(h),self.modes(h),self.any_event(h).squeeze(-1)


def auxiliary_loss(outputs,rows,den,binary_den):
    value=masked_loss(outputs[0],outputs[1],rows,den[:2])
    targets=[any_event_target(r['event_targets'],r['event_mask']) for r in rows]
    known=torch.tensor([t is not None for t in targets],device=outputs[2].device)
    binary=outputs[2][known].sum()*0
    if known.any() and binary_den>0:
        y=torch.tensor([t for t in targets if t is not None],device=outputs[2].device,dtype=torch.float32)
        weights=torch.tensor([r['weight'] for r,t in zip(rows,targets) if t is not None],device=y.device)
        binary=(F.binary_cross_entropy_with_logits(outputs[2].float()[known],y,reduction='none')*weights).sum()/binary_den
    return value+binary
