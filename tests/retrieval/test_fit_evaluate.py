"""Finite mock: fit succeeds with no EVAL files, freeze stays immutable, missing task blocks."""
import hashlib,importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from retrieval_control import fit_evaluate as f

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n')
def test_calibration_freeze_and_incomplete_cost():
 with tempfile.TemporaryDirectory(prefix='web-heads-mock-') as tmp:
  p=Path(tmp);cases=[];tasks=[];gold=[]
  for i in range(115):
   role='CAL' if i<50 else 'EVAL';benchmark=['hotpotqa','musique'][i%2] if i<50 else 'hotpotqa';cid=benchmark+'::mock'+str(i);case={'case_id':cid,'role':role,'benchmark':benchmark,'prefix_sha256':'mock','selection_hash':f'{i:08d}'};cases.append(case);gold.append({'case_id':cid,'answer':'correct','answer_aliases':[]})
   key=hashlib.sha256(cid.encode()).hexdigest()[:24];write(p/'features'/(key+'.json'),{'case_id':cid,'prefix_sha256':'mock','status':'COMPLETE','C':{'values':[i%2,2.]},'T':[1.,1.],'ES':{'values':[.5,.5],'mask':[1.,1.]},'times':{'detector_wall_s':.1,'detector_process_cpu_s':.02,'encoder_process_cpu_s':.01}})
   for action in ['answer','continue']:
    tid=str(i)+'_'+action;tasks.append({'task_id':tid,'case_id':cid,'action':action,'role':role})
    if role=='CAL':write(p/'responses'/(tid+'.json'),{'task_id':tid,'status':'COMPLETE','answer':'correct','usage_totals':{'completion_tokens':100 if action=='answer' else 400,'prompt_tokens':50},'responses':[{'server_cost':{'generation_wall_s':.1 if action=='answer' else .4}}]})
  for name,rows in [('cases.jsonl',cases),('tasks.jsonl',tasks),('cal.jsonl',gold[:50])]:
   (p/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
  args=SimpleNamespace(cases=p/'cases.jsonl',tasks=p/'tasks.jsonl',features=p/'features',responses=p/'responses',gold=p/'cal.jsonl',gold_eval=p/'eval.jsonl',output=p/'policies')
  freeze=f.fit(args,cases,tasks);original=(args.output/'POLICY_FREEZE.json').read_bytes();assert not args.gold_eval.exists()
  try:f.evaluate(args,cases,tasks,freeze)
  except FileNotFoundError:pass
  else:raise AssertionError('missing EVAL must block')
  assert not (args.output/'COMPLETE.json').exists()
  args.gold_eval.write_text(''.join(json.dumps(x)+'\n' for x in gold[50:]))
  for t in tasks[100:]:write(p/'responses'/(t['task_id']+'.json'),{'task_id':t['task_id'],'status':'COMPLETE','answer':'correct' if t['action']=='continue' else 'wrong','usage_totals':{'completion_tokens':100 if t['action']=='answer' else 400,'prompt_tokens':50},'responses':[{'server_cost':{'generation_wall_s':.1 if t['action']=='answer' else .4}}]})
  result=f.evaluate(args,cases,tasks,freeze);assert (args.output/'POLICY_FREEZE.json').read_bytes()==original;assert result['status']=='COMPLETE';assert len(json.loads((args.output/'COMPLETE.json').read_text())['task_ids'])==230
  assert result['contrast']['cost_coverage_complete']
  assert result['contrast']['unconfirmed_llm_attempts_all_branches']==0
  # Simulate one EVAL retry whose first request has no recovered accounting.
  # Even an unselected branch remains an unknown total experimental cost.
  uncertain=p/'responses'/(tasks[100]['task_id']+'.json');record=json.loads(uncertain.read_text());record['unconfirmed_llm_attempts']=1;write(uncertain,record)
  censored=f.evaluate(args,cases,tasks,freeze)
  assert not censored['contrast']['cost_coverage_complete']
  assert censored['contrast']['unconfirmed_llm_attempts_all_branches']==1
  assert not censored['contrast']['primary_pass']
  assert censored['amortization']['versus_CT']['break_even_subsequent_requests'] is None
  assert censored['amortization']['versus_always_continue']['break_even_subsequent_requests'] is None
  assert (args.output/'POLICY_FREEZE.json').read_bytes()==original
  assert next(x for x in censored['primary_results'] if x['policy']=='always_answer')['selected_unconfirmed_llm_attempts']==1


def test_single_benchmark_mean_and_duplicate_membership():
 import pytest
 assert f.weighted_mean([.2, .6], [{'benchmark':'hotpotqa'}]*2)==pytest.approx(.4)
 with pytest.raises(ValueError, match='duplicate'):
  f.validate_membership([{'case_id':'one','benchmark':'hotpotqa','role':r} for r in ['CAL','EVAL']], [])
