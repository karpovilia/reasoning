"""Fit only on CAL; freeze policies before loading EVAL outcomes; finalize from disk."""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, re, time, unicodedata
from pathlib import Path
import numpy as np
from .grade import score_answer

ALPHAS=[1.,10.,100.]
LAMBDAS=[0.,.01,.02,.05,.1,.2]
ARMS=['C','CT','CTES']
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def load_lines(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def output_tokens(response):
    u=response.get('usage_totals',{})
    for k in ['completion_tokens','output_tokens']:
        if k in u and u[k] is not None:return float(u[k])
    raise ValueError('missing measured output-token cost')
def outcome_arrays(cases,tasks,response_dir,gold_file):
    # Called on CAL first. EVAL response bodies are not read until freeze exists.
    want={c['case_id'] for c in cases};gold={r['case_id']:r for r in load_lines(gold_file) if r['case_id'] in want}
    bytask={(t['case_id'],t['action']):t for t in tasks};f1=[];em=[];cost=[];pins={};telemetry={k:[] for k in ['solver_generation_wall_s','input_tokens','tool_calls','unconfirmed_llm_attempts']}
    for c in cases:
        fr=[];er=[];cr=[];measured={k:[] for k in telemetry}
        for action in ['answer','continue']:
            task=bytask[(c['case_id'],action)];p=Path(response_dir)/(task['task_id']+'.json')
            if not p.is_file():raise FileNotFoundError('missing response: '+str(p))
            r=json.loads(p.read_text());assert r['task_id']==task['task_id'];status=str(r.get('status','')).lower()
            if status in ['transport_failure','infrastructure_failure','retryable','incomplete']:raise ValueError('unresolved infrastructure task '+task['task_id'])
            if status not in ['complete','completed','success','ok','semantic_failure','invalid_answer','model_failure']:raise ValueError('unrecognized terminal status '+status)
            pred=r.get('answer') or ''
            if isinstance(pred,dict):pred=pred.get('concise_answer','')
            g=gold[c['case_id']];score=score_answer(c['benchmark'],str(pred),g['answer'],g.get('answer_aliases',[]))
            fr.append(score['f1']);er.append(score['em']);cr.append(output_tokens(r));pins[task['task_id']]=sha_file(p)
            responses=r.get('responses') or []
            if not responses or any('generation_wall_s' not in x.get('server_cost',{}) for x in responses):raise ValueError('missing measured solver generation wall time')
            measured['solver_generation_wall_s'].append(sum(float(x['server_cost']['generation_wall_s']) for x in responses))
            measured['input_tokens'].append(float(r['usage_totals']['prompt_tokens']))
            measured['tool_calls'].append(float(len(r.get('tool_calls',[]))))
            measured['unconfirmed_llm_attempts'].append(float(r.get('unconfirmed_llm_attempts',0)))
        f1.append(fr);em.append(er);cost.append(cr)
        for k in telemetry:telemetry[k].append(measured[k])
    return np.asarray(f1),np.asarray(em),np.asarray(cost),pins,{k:np.asarray(v) for k,v in telemetry.items()}
def feature_matrix(cases,feature_dir):
    arrays={a:[] for a in ARMS};obs=[];pins={}
    for c in cases:
        key=hashlib.sha256(c['case_id'].encode()).hexdigest()[:24];p=Path(feature_dir)/(key+'.json')
        r=json.loads(p.read_text());assert r['case_id']==c['case_id'] and r['prefix_sha256']==c['prefix_sha256'] and r['status']=='COMPLETE'
        C=r['C']['values'];T=r['T'];ES=r['ES']['values']+r['ES']['mask']
        for a,x in [('C',C),('CT',C+T),('CTES',C+T+ES)]:arrays[a].append(x)
        obs.append(r['times']);pins[c['case_id']]=sha_file(p)
    return {a:np.asarray(x,dtype=np.float64) for a,x in arrays.items()},obs,pins
def ridge_fit(X,y,alpha):
    xm=X.mean(0);xs=X.std(0);xs[xs<1e-10]=1.;ym=y.mean(0);ys=y.std(0);ys[ys<1e-10]=1.
    z=(X-xm)/xs;target=(y-ym)/ys
    dual=np.linalg.solve(z@z.T+alpha*np.eye(len(X)),target);W=z.T@dual
    return {'x_mean':xm,'x_scale':xs,'y_mean':ym,'y_scale':ys,'weights':W}
def predict(model,X):return (((X-model['x_mean'])/model['x_scale'])@model['weights'])*model['y_scale']+model['y_mean']
def folds(cases):
    f=np.zeros(len(cases),dtype=int)
    for b in ['hotpotqa','musique']:
        indices=sorted([i for i,c in enumerate(cases) if c['benchmark']==b],key=lambda i:cases[i]['selection_hash'])
        for j,i in enumerate(indices):f[i]=j%5
    return f
def benchmarks(cases):
    present={c['benchmark'] for c in cases}
    if not present or not present <= {'hotpotqa','musique'}:
        raise ValueError('expected a nonempty HotpotQA/MuSiQue membership')
    return sorted(present)
def validate_membership(cases,tasks):
    """Require explicit, disjoint membership and both actions for every question."""
    benchmarks(cases)
    ids=[c['case_id'] for c in cases]
    if len(set(ids))!=len(ids):raise ValueError('duplicate case_id across CAL/EVAL')
    if any(c['role'] not in {'CAL','EVAL'} for c in cases):raise ValueError('unknown role')
    cal=[c for c in cases if c['role']=='CAL']
    if any(sum(c['benchmark']==b for c in cal)!=25 for b in ['hotpotqa','musique']):
        raise ValueError('calibration requires 25 HotpotQA and 25 MuSiQue questions')
    if not any(c['role']=='EVAL' for c in cases):raise ValueError('empty evaluation membership')
    expected={(c['case_id'],a) for c in cases for a in ['answer','continue']}
    actual=[(t['case_id'],t['action']) for t in tasks]
    if len(actual)!=len(expected) or set(actual)!=expected or len({t['task_id'] for t in tasks})!=len(tasks):
        raise ValueError('tasks must contain exactly one answer/continue pair per case')
def weighted_mean(values,cases):return float(np.mean([np.mean([values[i] for i,c in enumerate(cases) if c['benchmark']==b]) for b in benchmarks(cases)]))
def fit(args,cases,tasks):
    cal=[c for c in cases if c['role']=='CAL'];assert len(cal)==50
    start=time.monotonic();X,obs,feature_pins=feature_matrix(cal,args.features);f1,em,cost,response_pins,telemetry=outcome_arrays(cal,tasks,args.responses,args.gold)
    y=np.stack([f1[:,1]-f1[:,0],(cost[:,1]-cost[:,0])/1000],axis=1);fold=folds(cal);selected={};weights={};fit_cpu={}
    for arm in ARMS:
        cpu_start=time.process_time()
        candidates=[]
        for alpha in ALPHAS:
            pred=np.zeros_like(y);loss=[]
            for k in range(5):
                train=fold!=k;test=~train;m=ridge_fit(X[arm][train],y[train],alpha);pred[test]=predict(m,X[arm][test]);loss.extend((((pred[test]-y[test])/m['y_scale'])**2).mean(1).tolist())
            candidates.append((float(np.mean(loss)),-alpha,pred))
        loss,negative_alpha,pred=min(candidates,key=lambda x:(x[0],x[1]));alpha=-negative_alpha
        options=[]
        for lam in LAMBDAS:
            actions=(pred[:,0]-lam*pred[:,1]>=0).astype(int)
            quality=weighted_mean(f1[np.arange(len(cal)),actions]-f1[:,1],cal)
            saving=weighted_mean(cost[:,1]-cost[np.arange(len(cal)),actions],cal)
            if quality>=-.02:options.append((saving,-lam,actions,quality))
        if options:
            saving,nlam,actions,quality=max(options,key=lambda x:(x[0],x[1]));lam=-nlam;fallback=False
        else:lam=0.;actions=np.ones(len(cal),dtype=int);quality=0.;saving=0.;fallback=True
        model=ridge_fit(X[arm],y,alpha)
        for k,v in model.items():weights[arm+'__'+k]=v
        selected[arm]={'alpha':alpha,'lambda':lam,'fallback_always_continue':fallback,'oof_standardized_target_mse':loss,'oof_delta_f1_vs_continue':quality,'oof_saved_output_tokens_vs_continue':saving,'oof_continue_fraction':float(actions.mean())}
        fit_cpu[arm]=time.process_time()-cpu_start
    args.output.mkdir(parents=True,exist_ok=True)
    temporary=args.output/'HEADS.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream,**weights);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,args.output/'HEADS.npz')
    freeze={'status':'FROZEN_CAL_ONLY','calibration_questions':50,'calibration_case_ids':[c['case_id'] for c in cal],'folds':fold.tolist(),'selected':selected,'heads_sha256':sha_file(args.output/'HEADS.npz'),'cases_sha256':sha_file(args.cases),'task_manifest_sha256':sha_file(args.tasks),'calibration_feature_pins':feature_pins,'calibration_response_pins':response_pins,'calibration_cost':{'paired_solver_output_tokens':float(cost.sum()),'paired_solver_generation_wall_s':float(telemetry['solver_generation_wall_s'].sum()),'always_continue_solver_generation_wall_s':float(telemetry['solver_generation_wall_s'][:,1].sum()),'unconfirmed_llm_attempts':int(telemetry['unconfirmed_llm_attempts'].sum()),'observer_wall_s':sum(o['detector_wall_s'] for o in obs),'observer_process_cpu_s':sum(o['detector_process_cpu_s'] for o in obs),'encoder_cpu_s':sum(o['encoder_process_cpu_s'] for o in obs),'fit_process_cpu_by_arm_s':fit_cpu,'fit_wall_s':time.monotonic()-start},'created_at':datetime.datetime.now().astimezone().isoformat(),'rule':'CAL only;5fold internal scaling; alpha1,10,100;lambda grid; admissible oof F1 loss<=.02 vs continue; utility ties continue; no EVAL outcomes read'}
    atom(args.output/'POLICY_FREEZE.json',freeze);return freeze
def evaluate(args,cases,tasks,freeze):
    assert freeze['status']=='FROZEN_CAL_ONLY';assert freeze['heads_sha256']==sha_file(args.output/'HEADS.npz');assert freeze['cases_sha256']==sha_file(args.cases)
    assert freeze['task_manifest_sha256']==sha_file(args.tasks)
    for tid,digest in freeze['calibration_response_pins'].items():
        assert sha_file(Path(args.responses)/(tid+'.json'))==digest
    evaluation=[c for c in cases if c['role']=='EVAL'];assert evaluation
    X,obs,pins=feature_matrix(evaluation,args.features);f1,em,cost,response_pins,telemetry=outcome_arrays(evaluation,tasks,args.responses,args.gold_eval)
    weight_file=np.load(args.output/'HEADS.npz');actions={'always_continue':np.ones(len(evaluation),dtype=int),'always_answer':np.zeros(len(evaluation),dtype=int)}
    inference_cpu={}
    for arm in ARMS:
        cpu_start=time.process_time()
        selected=freeze['selected'][arm];model={k:weight_file[arm+'__'+k] for k in ['x_mean','x_scale','y_mean','y_scale','weights']}
        y=predict(model,X[arm]);actions[arm]=np.ones(len(evaluation),dtype=int) if selected['fallback_always_continue'] else (y[:,0]-selected['lambda']*y[:,1]>=0).astype(int)
        inference_cpu[arm]=(time.process_time()-cpu_start)/len(evaluation)
    # Random allocation preserves the actual CTES action count within benchmark, without outcomes.
    random_actions=actions['CTES'].copy();rng=np.random.default_rng(261026)
    for b in benchmarks(evaluation):
        idx=np.array([i for i,c in enumerate(evaluation) if c['benchmark']==b]);random_actions[idx]=rng.permutation(random_actions[idx])
    actions['random_CTES_count']=random_actions
    values={};rows=[];wall_cost={};cpu_cost={};observer=np.asarray([o['detector_wall_s'] for o in obs]);observer_cpu=np.asarray([o['detector_process_cpu_s'] for o in obs]);encoder_cpu=np.asarray([o['encoder_process_cpu_s'] for o in obs])
    for arm,a in actions.items():
        q=f1[np.arange(len(a)),a];e=em[np.arange(len(a)),a];t=cost[np.arange(len(a)),a];values[arm]=(q,e,t)
        generation=telemetry['solver_generation_wall_s'][np.arange(len(a)),a]
        uses_observer=arm in ['CTES','random_CTES_count'];uses_encoder=arm in ['CT','CTES','random_CTES_count']
        wall_cost[arm]=generation+(observer if uses_observer else 0)
        cpu_cost[arm]=(observer_cpu if uses_observer else np.zeros(len(a)))+(encoder_cpu if uses_encoder else 0)+inference_cpu.get(arm,0)
        unknown=telemetry['unconfirmed_llm_attempts'][np.arange(len(a)),a]
        rows.append({'policy':arm,'questions':len(evaluation),'f1':weighted_mean(q,evaluation),'em':weighted_mean(e,evaluation),'output_tokens_per_question':weighted_mean(t,evaluation),'continue_fraction':float(a.mean()),'solver_generation_wall_s_per_question':weighted_mean(generation,evaluation),'solver_plus_observer_wall_s_per_question':weighted_mean(wall_cost[arm],evaluation),'observer_encoder_head_cpu_s_per_question':weighted_mean(cpu_cost[arm],evaluation),'input_tokens_per_question':weighted_mean(telemetry['input_tokens'][np.arange(len(a)),a],evaluation),'tool_calls_per_question':weighted_mean(telemetry['tool_calls'][np.arange(len(a)),a],evaluation),'selected_unconfirmed_llm_attempts':int(unknown.sum()),'selected_questions_with_unknown_cost':int((unknown>0).sum())})
    q=values['CTES'][0]-values['CT'][0];c=values['CT'][2]-values['CTES'][2];base=values['CTES'][0]-values['always_continue'][0]
    rng=np.random.default_rng(261027);groups=[];cluster_log={}
    for b in benchmarks(evaluation):
        by={}
        for i,x in enumerate(evaluation):
            if x['benchmark']!=b:continue
            normalized=re.sub(r'\W+',' ',unicodedata.normalize('NFKC',x.get('question',x['case_id'])).casefold()).strip();by.setdefault(normalized,[]).append(i)
        groups.append([np.asarray(v,dtype=int) for v in by.values()]);cluster_log[b]={'questions':sum(map(len,by.values())),'normalized_question_clusters':len(by),'repeated_groups':[[evaluation[i]['case_id'] for i in v] for v in by.values() if len(v)>1]}
    draws=[];wall_saved=wall_cost['CT']-wall_cost['CTES']
    tools_saved=telemetry['tool_calls'][np.arange(len(evaluation)),actions['CT']]-telemetry['tool_calls'][np.arange(len(evaluation)),actions['CTES']]
    for _ in range(10000):
        ds=[np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),size=len(clusters))]) for clusters in groups]
        draws.append([float(np.mean([v[d].mean() for d in ds])) for v in [q,c,base,wall_saved,tools_saved]])
    draws=np.asarray(draws);ci=np.quantile(draws,[.025,.975],axis=0);point=[weighted_mean(v,evaluation) for v in [q,c,base]]
    contrast={'delta_f1_CTES_minus_CT':{'mean':point[0],'ci95':ci[:,0].tolist()},'saved_output_tokens_CT_minus_CTES':{'mean':point[1],'ci95':ci[:,1].tolist()},'delta_f1_CTES_minus_always_continue':{'mean':point[2],'ci95':ci[:,2].tolist()},'primary_pass':bool(ci[0,0]>-.02 and ci[0,1]>0),'primary_definition':'F1 CTES−CT lower one-sided97.5%>−.02 and CT−CTES output-token saving lower one-sided97.5%>0','quality_vs_continue_tolerance_pass':bool(ci[0,2]>-.02),'bootstrap':'10000 stratified paired question resamples; no repeated fits'}
    contrast['saved_solver_plus_observer_wall_s_CT_minus_CTES']={'mean':weighted_mean(wall_saved,evaluation),'ci95':ci[:,3].tolist()};contrast['bootstrap_clusters']=cluster_log
    contrast['saved_tool_calls_CT_minus_CTES']={'mean':weighted_mean(tools_saved,evaluation),'ci95':ci[:,4].tolist()}
    unknown_total=freeze['calibration_cost']['unconfirmed_llm_attempts']+int(telemetry['unconfirmed_llm_attempts'].sum())
    contrast['known_accepted_cost_numerical_criterion']=contrast['primary_pass'];contrast['cost_coverage_complete']=unknown_total==0;contrast['primary_pass']=contrast['primary_pass'] and unknown_total==0;contrast['unconfirmed_llm_attempts_all_branches']=unknown_total
    cal=freeze['calibration_cost'];extra_adaptation=cal['observer_wall_s'];saving=weighted_mean(wall_saved,evaluation);full_adaptation=cal['paired_solver_generation_wall_s']+cal['observer_wall_s'];saving_vs_continue=weighted_mean(wall_cost['always_continue']-wall_cost['CTES'],evaluation)
    amortization={'unit':'sum measured serial solver generation wall seconds plus detector wall seconds on the same GPU; not active GPU seconds or service latency','cost_coverage_complete':unknown_total==0,'versus_CT':{'extra_calibration_wall_s':extra_adaptation,'saved_wall_s_per_eval_question':saving,'break_even_subsequent_requests':math.ceil(max(0.,extra_adaptation)/saving) if saving>0 and unknown_total==0 else None},'versus_always_continue':{'calibration_wall_s':full_adaptation,'saved_wall_s_per_eval_question':saving_vs_continue,'break_even_subsequent_requests':math.ceil(full_adaptation/saving_vs_continue) if saving_vs_continue>0 and unknown_total==0 else None},'nominal_total_questions_including50cal':[500,900],'serial_wall_totals_including_calibration':{str(n):{'subsequent_questions':n-50,'CT':cal['paired_solver_generation_wall_s']+(n-50)*weighted_mean(wall_cost['CT'],evaluation),'CTES':full_adaptation+(n-50)*weighted_mean(wall_cost['CTES'],evaluation),'always_continue':cal['always_continue_solver_generation_wall_s']+(n-50)*weighted_mean(wall_cost['always_continue'],evaluation)} for n in [500,900]},'cpu_reported_separately':True,'scope':'Known accepted costs only if any unknown attempts. 900 is a same-distribution extrapolation, not900observedquestions. Research cost of generating unselected EVAL branches, annotation/pruning, server loads and queueing is excluded from selected-policy execution cost and must not be called serving latency.'}
    result={'status':'COMPLETE','primary_results':rows,'contrast':contrast,'observer_cost':{'detector_wall_s_total':sum(o['detector_wall_s'] for o in obs),'encoder_process_cpu_s_total':sum(o['encoder_process_cpu_s'] for o in obs),'head_batch_cpu_s_per_question':inference_cpu,'note':'Measured observer service times on allocated hardware; not measured complete-service latency or accelerator active time. Solver token savings do not by themselves prove service-cost savings.'},'calibration_cost':freeze['calibration_cost'],'amortization':amortization,'policy_freeze_sha256':sha_file(args.output/'POLICY_FREEZE.json'),'eval_feature_pins':pins,'eval_response_pins':response_pins,'completed_at':datetime.datetime.now().astimezone().isoformat()};atom(args.output/'RESULT.json',result)
    atom(args.output/'COMPLETE.json',{'status':'COMPLETE','calibration_questions':50,'evaluation_questions':len(evaluation),'paired_branches':len(tasks),'tasks':len(tasks),'task_ids':sorted(t['task_id'] for t in tasks),'result_sha256':sha_file(args.output/'RESULT.json')})
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--tasks',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--responses',type=Path,required=True);ap.add_argument('--gold',type=Path,required=True,help='CAL-only gold file');ap.add_argument('--gold-eval',type=Path,required=True,help='EVAL-only file, opened only after POLICY_FREEZE');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mode',choices=['fit','evaluate','all'],default='all');args=ap.parse_args()
    cases=load_lines(args.cases);tasks=load_lines(args.tasks);validate_membership(cases,tasks)
    p=args.output/'POLICY_FREEZE.json'
    if args.mode in ['fit','all'] and not p.exists():fit(args,cases,tasks)
    if args.mode in ['evaluate','all']:
        if not p.is_file():raise ValueError('evaluation requires frozen policies')
        result=evaluate(args,cases,tasks,json.loads(p.read_text()));print(json.dumps({'status':result['status'],'primary':result['contrast']}))
if __name__=='__main__':main()
