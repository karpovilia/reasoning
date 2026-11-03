"""Causal own predictions and bounded, simultaneous retrospective revisions.

Gold metadata is carried only for loss/evaluation. Only detached predicted event
and mode probabilities, cached embeddings, and relative age enter model features.
"""
from copy import deepcopy
import numpy as np
import torch
from .model import training_denominators, training_loss


class HistoryState:
    def __init__(self, capacity=16):
        self.capacity=capacity;self.trace=None;self.last_sid=None;self.entries=[]

    def prepare(self,row):
        if self.trace != row['trace_id'] or self.last_sid is None or row['sid'] != self.last_sid+1:
            self.entries=[]
        self.trace=row['trace_id']

    def snapshot(self):
        return {'schema':1,'capacity':self.capacity,'trace':self.trace,'last_sid':self.last_sid,
                'entries':[{'row':deepcopy(e['row']),'pooled':e['pooled'].detach().cpu().clone(),
                            'scores':e['scores'].detach().cpu().clone(), 'revision':e['revision'],
                            'as_of_sid':e['as_of_sid']} for e in self.entries]}

    @classmethod
    def restore(cls,saved,device):
        if saved['schema']!=1:raise ValueError('Wrong history schema')
        result=cls(saved['capacity']);result.trace=saved['trace'];result.last_sid=saved['last_sid']
        if len(saved['entries'])>result.capacity:raise ValueError('History budget exceeded')
        previous=None
        for entry in saved['entries']:
            e=deepcopy(entry);row=e['row']
            if row['trace_id']!=result.trace or (previous is not None and row['sid']!=previous+1):
                raise ValueError('Noncausal history order')
            if not row['sid']<=e['as_of_sid']<=result.last_sid:raise ValueError('History contains future revision')
            e['pooled']=e['pooled'].to(device);e['scores']=e['scores'].to(device)
            if e['scores'].shape!=(12,) or not torch.isfinite(e['scores']).all():raise ValueError('Invalid cached predictions')
            result.entries.append(e);previous=row['sid']
        if result.entries and previous!=result.last_sid:raise ValueError('History cursor mismatch')
        return result


def predicted_scores(outputs):
    return torch.cat([outputs[0].float().sigmoid(),outputs[1].float().softmax(-1)],-1).detach()


def rollout(model,rows,pooled,state):
    """Encode can be batched; prediction/update is strictly chronological.

    A frame holds its primary answer and all revisions to earlier targets. All
    revisions use the SAME pre-update history snapshot, never each other's
    intermediate edits. No backward gradient passes through cached predictions.
    """
    frames=[]
    for row,current in zip(rows,pooled,strict=True):
        state.prepare(row)
        history_used=[{'sid':e['row']['sid'],'revision_id':e['revision'],'as_of_sid':e['as_of_sid']} for e in state.entries]
        context=model.history_context([e['scores'] for e in state.entries],current.device)
        prior=model.dynamics(state.entries,row['sid'],current.device) if hasattr(model,'dynamics') else None
        primary=model.primary_heads(current[None,:],context)
        if prior is not None:primary=model.fuse_dynamics(primary,prior)
        past=list(state.entries)
        revised=None
        if past:
            old=torch.stack([e['pooled'] for e in past])
            lags=torch.tensor([(row['sid']-e['row']['sid'])/state.capacity for e in past],device=current.device)
            revised=model.revision_heads(old,current[None,:],context,lags)
            latest=predicted_scores(revised)
            for i,e in enumerate(past):
                e['scores']=latest[i];e['revision']+=1;e['as_of_sid']=row['sid']
        # Record immutable coordinates separately; later steps may revise entries again.
        frames.append({'dynamics_prior':prior,'dynamics_loss_weight':model.detector_config.get('dynamics_prior',{}).get('auxiliary_loss_weight',0.),'history_used':history_used,'row':row,'primary':primary,'targets':[e['row'] for e in past],
                       'revision_ids':[e['revision'] for e in past],'revised':revised})
        state.entries.append({'row':row,'pooled':current.detach(),'scores':predicted_scores(primary)[0],
                              'revision':0,'as_of_sid':row['sid']})
        state.entries=state.entries[-state.capacity:];state.last_sid=row['sid']
    return frames


def revision_targets(rows,state):
    """Plan revision loss denominators over the WHOLE effective batch first."""
    cached=[e['row'] for e in state.entries];trace=state.trace;last=state.last_sid;targets=[]
    for row in rows:
        if trace!=row['trace_id'] or last is None or row['sid']!=last+1:cached=[]
        targets.extend(cached)
        cached=(cached+[row])[-state.capacity:];trace=row['trace_id'];last=row['sid']
    return targets


def denominators(rows,state):
    from .selective import any_event_target
    binary=sum(r['weight'] for r in rows if any_event_target(r['event_targets'],r['event_mask']) is not None)
    return training_denominators(rows,True),training_denominators(revision_targets(rows,state),True),binary


def sequence_loss(frames,den,revision_weight=.5):
    primary=tuple(torch.cat([f['primary'][j] for f in frames]) for j in range(len(frames[0]['primary'])))
    loss=training_loss(primary,[f['row'] for f in frames],den[0])
    with_revisions=[f for f in frames if f['targets']]
    if with_revisions:
        revised=tuple(torch.cat([f['revised'][j] for f in with_revisions]) for j in range(len(frames[0]['primary'])))
        targets=[r for f in with_revisions for r in f['targets']]
        loss=loss+revision_weight*training_loss(revised,targets,den[1])
    if frames[0].get('dynamics_prior') is not None:
        from .dynamics import auxiliary_loss
        forecasts=tuple(torch.cat([f['dynamics_prior'][j] for f in frames]) for j in range(3))
        loss=loss+frames[0]['dynamics_loss_weight']*auxiliary_loss(forecasts,[f['row'] for f in frames],den[0],den[2])
    return loss


def chronological_indices(dataset,seed=None):
    indices={sid:i for i,sid in enumerate(dataset.ids)};groups={}
    for sid,trace in dataset.db.execute('SELECT sample_id,trace_id FROM samples ORDER BY trace_id,sid,sample_id'):
        if sid in indices:groups.setdefault(trace,[]).append(indices[sid])
    traces=sorted(groups)
    if seed is not None:
        traces=[traces[int(i)] for i in np.random.default_rng(seed).permutation(len(traces))]
    return np.asarray([i for trace in traces for i in groups[trace]],dtype=np.int64)


def probabilities(outputs):
    e,m,*aux=outputs
    e=e.float().sigmoid().cpu().tolist();m=m.float().softmax(-1).cpu().tolist()
    records=[{'event_probabilities':ev,'mode_probabilities':mo} for ev,mo in zip(e,m,strict=True)]
    if aux:
        if len(aux)!=2:raise ValueError('Unexpected auxiliary head count')
        a,p=[x.float().sigmoid().cpu().tolist() for x in aux]
        for record,absence,presence in zip(records,a,p,strict=True):
            record.update(absence_probability=absence,presence_probability=presence)
    return records


def version_records(frames):
    """Append primary forecast and detector answer; revisions retain own times."""
    for frame in frames:
        row=frame['row']
        common={'as_of_sid':row['sid'],'as_of_input_sha256':row.get('text_sha256'),
                'trace_id':row['trace_id'],'as_of_sample_id':row['sample_id']}
        prior_record={}
        if frame.get('dynamics_prior') is not None:
            prior=frame['dynamics_prior']
            prior_record={'dynamics_prior':{**probabilities(prior[:2])[0],
                'any_event_probability':float(prior[2].detach().float().sigmoid()[0]),
                'as_of_sid':row['sid']-1,'target_sid':row['sid']}}
        yield {**prior_record,**common,'target_sample_id':row['sample_id'],'target_sid':row['sid'],
               'revision_id':0,'lag':0,'history_used':frame['history_used'],**probabilities(frame['primary'])[0]}
        if frame['targets']:
            values=probabilities(frame['revised'])
            for target,revision,value in zip(frame['targets'],frame['revision_ids'],values,strict=True):
                yield {**common,'target_sample_id':target['sample_id'],'target_sid':target['sid'],
                       'revision_id':revision,'lag':row['sid']-target['sid'],**value}
