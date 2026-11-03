"""Question grouping, reconciliation and a single atomic two-tokenizer freeze."""
from __future__ import annotations

from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zlib
import time

from .core import EVENTS, MODES, atomic_json, dumps, file_sha, majority, new_split, sha, shared_input
ROOT = Path(os.environ.get('REASONING_DETECTOR_DATA', 'data/local/detector'))


class Union:
    def __init__(self):self.parents={}
    def root(self,x):
        self.parents.setdefault(x,x)
        if self.parents[x]!=x:self.parents[x]=self.root(self.parents[x])
        return self.parents[x]
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        self.parents[max(a,b)]=min(a,b)


def reconcile(annotations, n):
    layers=defaultdict(list)
    for a in annotations:
        a=dict(a);a['data']=json.loads(a['data'])
        for sid in range(a['lo'],a['hi']):layers[sid].append(a)
    for sid,records in sorted(layers.items()):
        # Newer accepted protocol replaces the whole observation, including UNKNOWN.
        version=max(r['version'] for r in records)
        records=[r for r in records if r['version']==version]
        values=[];masks=[];origins=[]
        for j in range(7):
            by_teacher=defaultdict(list)
            by_source=defaultdict(list)
            for r in records:
                i=sid-r['lo'];d=r['data']
                known=d['event_mask'][i][j] if j<6 else d['mode_mask'][i]
                v=d['event_targets'][i][j] if j<6 else d['mode_target'][i]
                if known:
                    by_teacher[r['teacher']].append(v);by_source[r['teacher']].append(r['response_id'])
            teacher='Pro' if by_teacher.get('Pro') else 'Flash'
            value=majority(by_teacher[teacher])
            values.append(value if value is not None else (0 if j<6 else -1))
            masks.append(value is not None)
            origins.append({'teacher':teacher if by_teacher[teacher] else None,
                            'responses':by_source[teacher],
                            'flash_pro_disagreement':bool(by_teacher.get('Pro') and by_teacher.get('Flash') and
                               majority(by_teacher['Pro'])!=majority(by_teacher['Flash']))})
        yield {'sid':sid,'event_targets':values[:6],'event_mask':masks[:6],
               'mode_target':values[6],'mode_mask':masks[6],'origins':origins,
               'protocols':sorted({r['protocol'] for r in records})}


class Dataset:
    def __init__(self, frozen, split):
        self.path=Path(frozen).resolve()
        self.db=sqlite3.connect((self.path/'samples.sqlite').as_uri()+'?mode=ro',uri=True)
        self.ids=[r[0] for r in self.db.execute('SELECT sample_id FROM samples WHERE split=? ORDER BY sample_id',(split,))]
        self.split=split
    def __len__(self):return len(self.ids)
    def get(self, sample_id):
        row=self.db.execute('SELECT weight,data FROM samples WHERE sample_id=?',(sample_id,)).fetchone()
        r=json.loads(zlib.decompress(row[1]));r['weight']=row[0]
        return r
    def __getitem__(self,i):return self.get(self.ids[i])


def verify_manifest(path):
    path=Path(path)
    m=json.loads((path/'manifest.json').read_text())
    if m['status']!='frozen':raise ValueError('not a frozen dataset')
    for name,digest in m['files'].items():
        if file_sha(path/name)!=digest:raise ValueError('Frozen dataset changed: '+name)
    return m


def availability(trace,sid):
    if trace['source_coordinate']=='original_reasoning':
        bounds=trace['segment_offsets']
        closure=bounds[sid+1][0]+1 if sid+1<len(bounds) else bounds[sid][1]
        return {'available_at_char':closure,'focus_end_char':bounds[sid][1],
                'boundary_contract':'canonical lookahead closure; next character confirms boundary but is never a model feature',
                'record_format':2}
    return {'available_at_char':None,'focus_end_char':trace['segment_offsets'][sid][1],
            'boundary_contract':'payload segment coordinates; live caller must supply confirmed segment closure',
            'record_format':2}


def main():
    from transformers import AutoTokenizer
    source=ROOT/'preparation/source_snapshot'
    complete=json.loads((source/'COMPLETE.json').read_text())
    if file_sha(source/'sources.sqlite')!=complete['sources_sha256']:
        raise ValueError('source snapshot changed')
    frozen=ROOT/'common/v1';stage=ROOT/'common/.v1.staging'
    with (ROOT/'common/build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if frozen.exists():
            verify_manifest(frozen);print('Frozen v1 already exists.');return
        stage.mkdir(exist_ok=True)
        models=json.loads((ROOT/'preparation/models.json').read_text())
        tokenizers={k:AutoTokenizer.from_pretrained(ROOT/'preparation/models'/k,local_files_only=True,use_fast=True) for k in models}
        context_audit=json.loads((ROOT/'preparation/context_equivalence.json').read_text())
        if context_audit['status']!='pass' or context_audit['core_sha256']!=file_sha(Path(__file__).with_name('core.py')):
            raise ValueError('Context selection implementation requires current reference audit')
        src=sqlite3.connect((source/'sources.sqlite').as_uri()+'?mode=ro',uri=True);src.row_factory=sqlite3.Row
        uf=Union();traces={};roles=defaultdict(set);history=defaultdict(list)
        trace_identities={};trace_aliases=defaultdict(list)
        extra_history=json.loads((ROOT/'preparation/history.json').read_text())
        historical_raw_ids={r['alias'] for r in extra_history['records'] if r['alias'].startswith('rawqid:')}
        for row in src.execute('SELECT * FROM traces ORDER BY trace_id'):
            r=json.loads(row['data'])
            uf.join(r['question_alias'],r['question_text_key'])
            if 'rawqid:'+r['question_id'] in historical_raw_ids:
                uf.join(r['question_alias'],'rawqid:'+r['question_id'])
            # Same solver/seed/effort and exact segment text = copied trace, even when
            # exported under another tag/arm/cap. Keep every original response as provenance.
            identity=sha(dumps([r['question_text_key'],r['solver'],r['condition'].get('seed'),
                                r['condition'].get('effort_level'),r['segments']]))
            canonical=trace_identities.setdefault(identity,r['trace_id'])
            traces.setdefault(canonical,r);trace_aliases[canonical].append(r['trace_id'])
        for row in src.execute('SELECT * FROM histories'):
            group=uf.root(row['alias']);roles[group].add(row['role'])
            history[group].append(dict(row))
        overlaps=defaultdict(dict)
        for row in extra_history['records']:
            group=uf.root(row['alias']);roles[group].add(row['role']);history[group].append(row)
            prior=overlaps[group].get(row['baseline'],False)
            overlaps[group][row['baseline']]=bool(prior or row['training_overlap'])
        split={};split_records={}
        for tid,r in traces.items():
            key=uf.root(r['question_alias']);prior=roles[key]
            part='test' if 'test' in prior else 'validation' if 'validation' in prior else 'train' if 'train' in prior else new_split(key)
            split[tid]=(key,part)
            split_records[key]={'split':part,'historical_roles':sorted(prior),'history':history[key],
              'historical_holdout':bool(prior and part!='train'),
              'new_holdout':bool(not prior and part!='train')}
        atomic_json(stage/'split.json',split_records)
        db=sqlite3.connect(stage/'samples.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,trace_id TEXT,sid INTEGER,question_key TEXT,split TEXT,weight REAL,data BLOB);
          CREATE INDEX IF NOT EXISTS sample_split ON samples(split);
          CREATE INDEX IF NOT EXISTS sample_question ON samples(question_key);
          CREATE TABLE IF NOT EXISTS omitted(sample_id TEXT PRIMARY KEY,status TEXT,detail TEXT);
          CREATE TABLE IF NOT EXISTS completed_traces(trace_id TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS traces(trace_id TEXT PRIMARY KEY,data BLOB);
        ''')
        done={r[0] for r in db.execute('SELECT trace_id FROM completed_traces')}
        # Resumable preparation format migration, before any final freeze/weights.
        if not (stage/'record_format.json').exists():
            for sample_id,blob in db.execute('SELECT sample_id,data FROM samples').fetchall():
                rec=json.loads(zlib.decompress(blob));rec.update(availability(traces[rec['trace_id']],rec['sid']))
                for name,enc in rec['tokens'].items():
                    special=set(tokenizers[name].all_special_ids)
                    enc['focus_mask']=[int(m and token not in special) for token,m in zip(enc['input_ids'],enc['focus_mask'])]
                if all(any(enc['focus_mask']) for enc in rec['tokens'].values()):
                    db.execute('UPDATE samples SET data=? WHERE sample_id=?',(zlib.compress(dumps(rec).encode(),3),sample_id))
                else:
                    db.execute('DELETE FROM samples WHERE sample_id=?',(sample_id,))
                    db.execute('INSERT OR REPLACE INTO omitted VALUES(?,?,?)',(sample_id,'token_alignment_error','"special-only focus"'))
            db.commit();atomic_json(stage/'record_format.json',{'version':2})
        for number,(tid,tr) in enumerate(traces.items()):
            if tid in done:continue
            anns=[];seen=set()
            for original in trace_aliases[tid]:
                for ann in src.execute('SELECT * FROM annotations WHERE trace_id=? ORDER BY response_id',(original,)):
                    provenance=json.loads(ann['provenance'])
                    fingerprint=(ann['teacher'],ann['protocol'],ann['lo'],ann['hi'],provenance['response_fingerprint'])
                    if fingerprint not in seen:anns.append(ann);seen.add(fingerprint)
            group,part=split[tid]
            for target in reconcile(anns,len(tr['segments'])):
                sid=target.pop('sid');sample_id=sha(dumps([tid,sid]))
                if not any(target['event_mask']) and not target['mode_mask']:
                    db.execute('INSERT OR REPLACE INTO omitted VALUES(?,?,?)',(sample_id,'all_unknown',dumps({'trace_id':tid,'sid':sid})));continue
                try:
                    inp=shared_input(tr['question'],tr['segments'],sid,tokenizers)
                except ValueError as exc:
                    db.execute('INSERT OR REPLACE INTO omitted VALUES(?,?,?)',(sample_id,'token_alignment_error',dumps({'trace_id':tid,'sid':sid,'error':str(exc)})));continue
                if inp['status']!='eligible':
                    db.execute('INSERT OR REPLACE INTO omitted VALUES(?,?,?)',(sample_id,inp['status'],dumps(inp)));continue
                rec={'sample_id':sample_id,'trace_id':tid,'sid':sid,'question_key':group,'split':part,
                     'benchmark':tr['benchmark'],'solver':tr['solver'],'condition':tr['condition'],
                     'source_offsets':tr['segment_offsets'][sid],
                     **availability(tr,sid),
                     'trace_segment_count_evaluation_only':len(tr['segments']),
                     'finish_reason_evaluation_only':tr['original_trace'].get('finish_reason'),
                     'baseline_train_overlap':{name:overlaps[group].get(name,'unknown') for name in ['lexical_gate','detector_left','lora_ck22000']},
                     **target,**inp}
                db.execute('INSERT OR REPLACE INTO samples VALUES(?,?,?,?,?,?,?)',
                  (sample_id,tid,sid,group,part,1.,zlib.compress(dumps(rec).encode(),level=3)))
            db.execute('INSERT OR REPLACE INTO traces VALUES(?,?)',(tid,zlib.compress(dumps(tr).encode(),3)))
            db.execute('INSERT OR REPLACE INTO completed_traces VALUES(?)',(tid,));db.commit()
            atomic_json(ROOT/'preparation/dataset_progress.json',{'at':time.time(),'trace_index':number,
              'total_traces':len(traces),'trace_id':tid,'status':'building'})
            if number%50==0:print('TOKENIZED TRACES',number,len(traces),flush=True)
        for key,n in db.execute("SELECT question_key,count(*) FROM samples WHERE split='train' GROUP BY question_key").fetchall():
            db.execute("UPDATE samples SET weight=? WHERE question_key=? AND split='train'",(min(1.,1000/n),key))
        db.commit()
        stats={};strata=defaultdict(lambda:{'samples':0,'known':[0]*6,'positive':[0]*6,'modes':[0]*6})
        for part in ['train','validation','test']:
            stats[part]={'samples':db.execute('SELECT count(*) FROM samples WHERE split=?',(part,)).fetchone()[0],
                         'questions':db.execute('SELECT count(DISTINCT question_key) FROM samples WHERE split=?',(part,)).fetchone()[0],
                         'traces':db.execute('SELECT count(DISTINCT trace_id) FROM samples WHERE split=?',(part,)).fetchone()[0]}
        token_hist={name:Counter() for name in models};longest=[]
        for (blob,) in db.execute('SELECT data FROM samples'):
            r=json.loads(zlib.decompress(blob))
            if r['split']=='train':
                for name in models:token_hist[name][len(r['tokens'][name]['input_ids'])]+=1
                if r['mode_mask'] and any(r['event_mask']):
                    longest.append((max(len(r['tokens'][name]['input_ids']) for name in models),r['sample_id']))
                    longest=sorted(longest,reverse=True)[:8]
            for name in [r['split'],r['split']+'|'+r['benchmark'],r['split']+'|'+','.join(r['protocols'])]:
                s=strata[name];s['samples']+=1
                for j in range(6):
                    s['known'][j]+=r['event_mask'][j];s['positive'][j]+=r['event_mask'][j]*r['event_targets'][j]
                if r['mode_mask']:s['modes'][r['mode_target']]+=1
        stats['strata']=dict(strata);stats['omitted']=dict(db.execute('SELECT status,count(*) FROM omitted GROUP BY status'))
        stats['trace_deduplication']={'raw_traces':sum(len(x) for x in trace_aliases.values()),'unique_traces':len(traces)}
        stats['train_token_length_histograms']={name:dict(hist) for name,hist in token_hist.items()}
        stats['gpu_smoke_sample_ids']=[sample for _,sample in longest]
        # Language audit uses original question/trace characters, never alters targets.
        language=Counter()
        for r in traces.values():
            text=r['question']+' '+''.join(r['segments'])
            letters=[c for c in text if c.isalpha()]
            fraction=sum('a'<=c.lower()<='z' for c in letters)/max(1,len(letters))
            language['latin_dominant' if fraction>=.9 else 'requires_language_review']+=1
        stats['language_script_audit']=dict(language)
        if db.execute('SELECT question_key FROM samples GROUP BY question_key HAVING count(DISTINCT split)>1').fetchone():
            raise ValueError('question split leakage')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();src.close()
        schema={'events':EVENTS,'modes':MODES,'grounding':'not included; no accepted domain-R supervision',
          'production_contract_difference':'6 modes, not the 7-channel production interface',
          'span_contract':'half-open contiguous predicted modes; unknown/reference gaps split spans; online last span is open',
          'loss':'known-pair weighted BCE + known-mode weighted CE; separate effective-batch denominators, ratio 1:1',
          'checkpoint_selection':'validation macro average_precision_score; classes with both positives and negatives fixed before training',
          'threshold_ties':'highest threshold among equal best F1; recall>=0.70 point maximizes precision, then threshold',
          'context':{'history_segments':16,'max_tokens_each':2048,'future_segments':0,'truncation':False},
          'models':models,'source_cutoff':complete['cutoff_utc']}
        atomic_json(stage/'schema.json',schema);atomic_json(stage/'coverage.json',stats)
        atomic_json(stage/'baseline_history.json',extra_history)
        atomic_json(stage/'context_equivalence.json',context_audit)
        atomic_json(stage/'trace_aliases.json',dict(trace_aliases))
        shutil.copyfile(source/'sources.sqlite',stage/'provenance.sqlite')
        for k,tok in tokenizers.items():tok.save_pretrained(stage/'tokenizers'/k)
        files={str(p.relative_to(stage)):file_sha(p) for p in stage.rglob('*') if p.is_file()}
        atomic_json(stage/'manifest.json',{'status':'frozen','version':1,'files':files,
          'source_snapshot_sha256':complete['sources_sha256'],'schema_sha256':files['schema.json'],
          'stats':stats,'baseline_comparison_status':'overlap masks recorded; unknown memberships excluded from baseline-independent panels',
          'api_cost_usd':0,'code_sha256':file_sha(__file__),
          'preparation_code_sha256':{p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')},
          'protocol_files':{p.name:file_sha(p) for p in (ROOT/'preparation/protocols').glob('*.txt')}})
        os.rename(stage,frozen)
        print('FROZEN',json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
