"""Identical focus pooling and heads for the full encoder and LoRA decoder."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .core import EVENTS, MODES, atomic_json


def loss_denominators(rows):
    """Compute ONCE over the effective batch, before microbatch accumulation."""
    return (sum(r['weight'] * sum(r['event_mask']) for r in rows),
            sum(r['weight'] * int(r['mode_mask']) for r in rows))


def masked_loss(event_logits, mode_logits, rows, denominators=None):
    device = event_logits.device
    weights = torch.tensor([r['weight'] for r in rows], device=device, dtype=torch.float32)
    mask = torch.tensor([r['event_mask'] for r in rows], device=device, dtype=torch.bool)
    mode_mask = torch.tensor([r['mode_mask'] for r in rows], device=device, dtype=torch.bool)
    den_e, den_m = denominators or loss_denominators(rows)
    # Index known targets BEFORE BCE/CE: unknown sentinels/NaNs cannot affect loss.
    ev = event_logits.sum() * 0
    mo = mode_logits.sum() * 0
    if mask.any() and den_e > 0:
        targets = torch.tensor([r['event_targets'] for r in rows], device=device, dtype=torch.float32)
        ev = (F.binary_cross_entropy_with_logits(event_logits.float()[mask], targets[mask], reduction='none')
              * weights[:, None].expand_as(mask)[mask]).sum() / den_e
    if mode_mask.any() and den_m > 0:
        targets = torch.tensor([r['mode_target'] for r in rows], device=device, dtype=torch.long)
        mo = (F.cross_entropy(mode_logits.float()[mode_mask], targets[mode_mask], reduction='none')
              * weights[mode_mask]).sum() / den_m
    return ev + mo


def collate(rows, branch, pad_id, device, padding_side='right'):
    n = max(len(r['tokens'][branch]['input_ids']) for r in rows)
    ids, masks, focus = [], [], []
    for r in rows:
        enc = r['tokens'][branch]
        length = len(enc['input_ids']); padding = n-length
        if n > 2048 or not any(enc['focus_mask']):
            raise ValueError('invalid frozen tokenizer input')
        if padding_side == 'right':
            ids.append(enc['input_ids'] + [pad_id]*padding)
            masks.append([1]*length + [0]*padding)
            focus.append(enc['focus_mask'] + [0]*padding)
        elif padding_side == 'left':
            ids.append([pad_id]*padding + enc['input_ids'])
            masks.append([0]*padding + [1]*length)
            focus.append([0]*padding + enc['focus_mask'])
        else:
            raise ValueError(padding_side)
    return {k: torch.tensor(v, device=device, dtype=torch.long)
            for k,v in [('input_ids',ids),('attention_mask',masks),('focus_mask',focus)]}


class Detector(nn.Module):
    def __init__(self, backbone, branch, config):
        super().__init__()
        self.backbone, self.branch, self.detector_config = backbone, branch, config
        d = backbone.config.hidden_size
        self.event_head = nn.Linear(d, len(EVENTS), dtype=torch.float32)
        self.mode_head = nn.Linear(d, len(MODES), dtype=torch.float32)
        nn.init.zeros_(self.event_head.bias); nn.init.zeros_(self.mode_head.bias)
        if config.get('selective995'):
            if branch != 'encoder': raise ValueError('selective995 requires a full encoder')
            self.absence_head = nn.Linear(d, 1, dtype=torch.float32)
            self.presence_head = nn.Linear(d, 1, dtype=torch.float32)
            nn.init.zeros_(self.absence_head.bias); nn.init.zeros_(self.presence_head.bias)

        if config.get('own_history'):
            h=config['own_history']['hidden_size']
            self.history_gru=nn.GRU(12,h,batch_first=True)
            self.history_projection=nn.Linear(h,d)
            nn.init.zeros_(self.history_projection.bias)
            self.revision_projection=nn.Sequential(nn.Linear(2*d+h+1,d),nn.GELU(),nn.LayerNorm(d))

        if config.get('dynamics_prior'):
            if not config.get('own_history'):raise ValueError('Dynamics requires chronological own-history')
            if config['dynamics_prior']['context']>config['own_history']['capacity']:
                raise ValueError('Dynamics history cannot exceed the frozen shared context')
            from .dynamics import DynamicsPrior
            self.dynamics=DynamicsPrior(d,config['dynamics_prior'])
            self.dynamics_fusion=nn.Linear(13,12,bias=False)
            nn.init.normal_(self.dynamics_fusion.weight,std=.01)

    def fuse_dynamics(self,primary,prior):
        # Learned combination, not a conditional-independence/product claim.
        features=torch.cat([prior[0],prior[1],prior[2][:,None]],-1)
        delta=self.dynamics_fusion(features)
        return (primary[0]+delta[:,:6],primary[1]+delta[:,6:])+primary[2:]

    def encode(self, input_ids, attention_mask, focus_mask):
        positions = (attention_mask.cumsum(-1)-1).clamp(min=0)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions)
        if self.branch == 'lora4b':
            kwargs['use_cache'] = False
        hidden = self.backbone(**kwargs).last_hidden_state
        mask = (focus_mask.bool() & attention_mask.bool()).unsqueeze(-1)
        if not torch.all(mask.sum(1) > 0):
            raise ValueError('empty focus pool')
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1)
        return pooled

    def heads(self, pooled):
        result = (self.event_head(pooled), self.mode_head(pooled))
        if self.detector_config.get('selective995'):
            result += (self.absence_head(pooled).squeeze(-1), self.presence_head(pooled).squeeze(-1))
        return result

    def forward(self, input_ids, attention_mask, focus_mask):
        if self.detector_config.get('own_history'):
            raise RuntimeError('History model requires ordered rollout, not independent forward')
        return self.heads(self.encode(input_ids,attention_mask,focus_mask))

    def history_context(self, scores, device):
        if not scores:
            return torch.zeros(1,self.detector_config['own_history']['hidden_size'],device=device)
        with torch.autocast(device_type=torch.device(device).type,enabled=False):
            _,hidden=self.history_gru(torch.stack(scores).float().unsqueeze(0))
        return hidden[-1]

    def primary_heads(self, pooled, history_context):
        return self.heads(pooled+self.history_projection(history_context))

    def revision_heads(self, old_pooled, current_pooled, history_context, lags):
        n=len(old_pooled)
        x=torch.cat([old_pooled,current_pooled.expand(n,-1),history_context.expand(n,-1),lags[:,None]],-1)
        return self.heads(self.revision_projection(x))

    @torch.no_grad()
    def predict(self, batch, pad_id, device, padding_side='right'):
        self.eval()
        inputs = collate(batch, self.branch, pad_id, device, padding_side)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
            outputs = self(**inputs)
            e, m = outputs[:2]
        result = {'event_logits': e.float().cpu(), 'mode_logits': m.float().cpu(),
                'sample_id': [r['sample_id'] for r in batch]}
        if len(outputs) == 4:
            result.update(absence_logits=outputs[2].float().cpu(), presence_logits=outputs[3].float().cpu())
        return result

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(path/'backbone', safe_serialization=True)
        heads = {'events':self.event_head.state_dict(), 'modes':self.mode_head.state_dict()}
        if self.detector_config.get('selective995'):
            heads.update(absence=self.absence_head.state_dict(), presence=self.presence_head.state_dict())
        if self.detector_config.get('own_history'):
            heads.update(history_gru=self.history_gru.state_dict(),history_projection=self.history_projection.state_dict(),
                         revision_projection=self.revision_projection.state_dict())
        if self.detector_config.get('dynamics_prior'):
            heads.update(dynamics=self.dynamics.state_dict(),dynamics_fusion=self.dynamics_fusion.state_dict())
        torch.save(heads,path/'heads.pt')
        atomic_json(path/'detector_config.json',self.detector_config)


def build_model(config, checkpoint=None, initialize_only=False):
    from transformers import AutoConfig, AutoModel
    from peft import LoraConfig, get_peft_model, PeftModel
    branch = config['branch']
    model = config['model']
    config_source=model['asset_directory'] if initialize_only and Path(model['asset_directory']).exists() else model['model_id']
    hf_config = AutoConfig.from_pretrained(config_source, revision=model['revision'])
    hf_config._attn_implementation = 'sdpa'
    if branch == 'encoder':
        hf_config.reference_compile = False
    kwargs = {'config':hf_config, 'torch_dtype':torch.float32 if branch=='encoder' else torch.bfloat16}
    if initialize_only:
        # Meta device: report module layout/counts without loading weights or doing inference.
        with torch.device('meta'):
            backbone = AutoModel.from_config(hf_config, **{k:v for k,v in kwargs.items() if k!='config'})
    else:
        source = str(Path(checkpoint)/'backbone') if checkpoint and branch=='encoder' else model['model_id']
        extra = {} if checkpoint and branch=='encoder' else {'revision':model['revision']}
        backbone = AutoModel.from_pretrained(source, **kwargs, **extra)
    if branch == 'lora4b':
        targets = config['lora']['target_modules']
        found = {n.rsplit('.',1)[-1] for n,m in backbone.named_modules() if isinstance(m,nn.Linear)}
        if not set(targets) <= found:
            raise ValueError(('LoRA modules absent',set(targets)-found))
        if checkpoint:
            backbone = PeftModel.from_pretrained(backbone,str(Path(checkpoint)/'backbone'),is_trainable=True)
        else:
            backbone = get_peft_model(backbone,LoraConfig(r=config['lora']['rank'],lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],target_modules=targets,bias='none'))
    detector = Detector(backbone,branch,config)
    if checkpoint:
        state = torch.load(Path(checkpoint)/'heads.pt',map_location='cpu',weights_only=True)
        detector.event_head.load_state_dict(state['events']);detector.mode_head.load_state_dict(state['modes'])
        if config.get('selective995'):
            detector.absence_head.load_state_dict(state['absence'])
            detector.presence_head.load_state_dict(state['presence'])
        if config.get('own_history'):
            for name in ('history_gru','history_projection','revision_projection'):
                getattr(detector,name).load_state_dict(state[name])
        if config.get('dynamics_prior'):
            detector.dynamics.load_state_dict(state['dynamics'])
            detector.dynamics_fusion.load_state_dict(state['dynamics_fusion'])
    return detector


def parameter_report(model):
    params = [{'name':n,'shape':list(p.shape),'numel':p.numel(),'trainable':p.requires_grad}
              for n,p in model.named_parameters()]
    return {'parameters':params,'total':sum(p['numel'] for p in params),
            'trainable':sum(p['numel'] for p in params if p['trainable'])}


def training_denominators(rows, selective=False):
    den = loss_denominators(rows)
    if selective:
        from .selective import binary_denominators
        den += binary_denominators(rows)
    return den


def training_loss(outputs, rows, denominators=None):
    den = denominators or training_denominators(rows, len(outputs) == 4)
    loss = masked_loss(*outputs[:2], rows, den[:2])
    if len(outputs) == 4:
        from .selective import binary_losses
        a, p = binary_losses(*outputs[2:], rows, den[2:])
        loss = loss + .5 * a + .5 * p
    return loss


def head_parameters(model):
    return [p for name, p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
