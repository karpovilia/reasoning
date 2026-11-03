"""Answer-only HotpotQA/MuSiQue EM and F1; never a supporting-fact/joint score.

Definitions checked against official primary sources saved in reference_sources:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py
"""
from __future__ import annotations
import argparse, collections, json, re, string
from pathlib import Path

def normalize_answer(text):
    text=''.join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
def pair_score(benchmark,prediction,answer):
    p=normalize_answer(prediction);g=normalize_answer(answer);em=float(p==g)
    pt=p.split();gt=g.split()
    if benchmark=='hotpotqa' and ((p in ['yes','no','noanswer'] or g in ['yes','no','noanswer']) and p!=g):return {'em':em,'f1':0.0}
    if benchmark=='musique' and (not pt or not gt):return {'em':em,'f1':float(pt==gt)}
    if benchmark not in ['musique','hotpotqa']:raise ValueError('unsupported benchmark')
    common=sum((collections.Counter(pt)&collections.Counter(gt)).values())
    f1=2*common/(len(pt)+len(gt)) if common else 0.0
    return {'em':em,'f1':f1}
def score_answer(benchmark,prediction,answer,aliases=()):
    targets=[str(answer)]+([str(a) for a in aliases] if benchmark=='musique' else [])
    scores=[pair_score(benchmark,str(prediction),x) for x in targets]
    return {'em':max(x['em'] for x in scores),'f1':max(x['f1'] for x in scores)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gold',type=Path,required=True);ap.add_argument('--task-manifest',type=Path,required=True);ap.add_argument('--responses',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--role',choices=['CAL','EVAL','ALL'],default='CAL');ap.add_argument('--policy-freeze',type=Path);args=ap.parse_args()
    if args.role!='CAL' and (not args.policy_freeze or not args.policy_freeze.is_file()):ap.error('EVAL grading requires existing --policy-freeze receipt')
    task_rows=[json.loads(x) for x in args.task_manifest.open() if x.strip()];tasks={x['task_id']:x for x in task_rows if args.role=='ALL' or x['role']==args.role}
    gold={x['case_id']:x for x in (json.loads(s) for s in args.gold.open() if s.strip())}
    responses={}
    if args.responses.is_dir():
        paths=sorted(args.responses.glob('*.json'))
        for p in paths:
            x=json.loads(p.read_text())
            if x.get('task_id') in tasks:
                if x['task_id'] in responses:raise ValueError('duplicate response task_id')
                responses[x['task_id']]=x
    else:
        for line in args.responses.open():
            x=json.loads(line)
            if x.get('task_id') in tasks:
                if x['task_id'] in responses:raise ValueError('duplicate response task_id')
                responses[x['task_id']]=x
    rows=[]
    for tid,t in tasks.items():
        x=responses.get(tid);g=gold[t['case_id']];row={'task_id':tid,'case_id':t['case_id'],'benchmark':g['benchmark'],'role':t['role'],'action':t['action']}
        if x is None:row.update(status='MISSING_RESPONSE',em=None,f1=None)
        elif x.get('status','').lower() in ['transport_failure','retryable','infrastructure_failure','incomplete']:row.update(status='INFRASTRUCTURE_UNRESOLVED',em=None,f1=None)
        else:
            answer=x.get('answer')
            if isinstance(answer,dict):answer=answer.get('concise_answer','')
            row.update(status=x.get('status','UNKNOWN'),**score_answer(g['benchmark'],str(answer or ''),g['answer'],g.get('answer_aliases',[])))
        rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'assigned':len(rows),'scored':sum(r['f1'] is not None for r in rows),'unresolved':sum(r['f1'] is None for r in rows),'role':args.role}))
if __name__=='__main__':main()
