import hashlib,json,os,time
from pathlib import Path
ROOT=Path(os.environ.get('TES_COMPARISON_ROOT','outputs/process'))
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def beat(stage,**kw):
 d=dict(time=time.time(),stage=stage,**kw)
 task=kw.get('task') or os.environ.get('TES_TASK_ID')
 if task:atomic(ROOT/'heartbeats'/(task+'.json'),d)
 else:atomic(ROOT/'HEARTBEAT.json',d)
def qsplit(q):return int(hashlib.sha256(('tes-process-20261026:'+q).encode()).hexdigest()[:8],16)%10
