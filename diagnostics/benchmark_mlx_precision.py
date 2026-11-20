"""Isolated MLX GPU float32/float16 precision benchmark using cached public queries."""
import json,sys,time,subprocess,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'diagnostics'

def run(dtype, dimensions=512):
 import mlx.core as mx
 mx.set_default_device(mx.gpu)
 a=mx.load(str(ROOT/'data/search/vectors.npy'))
 if dimensions != 512:
  a=a[:,:dimensions]
  a=a/mx.linalg.norm(a,axis=1,keepdims=True)
 a=a.astype(getattr(mx,dtype))
 q=np.load(D/'msmarco-precision-query-vectors.npy')[:,:dimensions].copy()
 q/=np.linalg.norm(q,axis=1,keepdims=True)
 b=mx.array(q,dtype=getattr(mx,dtype));mx.eval(a,b);mx.synchronize();mx.clear_cache();mx.reset_peak_memory()
 def search(i):
  scores=a@b[i]
  ids=mx.argpartition(-scores,9)[:10]
  values=scores[ids];mx.eval(ids,values);mx.synchronize()
  ix=np.array(ids);vs=np.array(values)
  return ix[np.lexsort((ix,-vs))].tolist()
 # Select ten to measure Hit@10 too; use five separately for requested latency.
 def search5(i):
  scores=a@b[i];ids=mx.argpartition(-scores,4)[:5];values=scores[ids]
  mx.eval(ids,values);mx.synchronize()
  ix=np.array(ids);vs=np.array(values)
  return ix[np.lexsort((ix,-vs))].tolist()
 for _ in range(3):search5(0)
 rows=[];ts=[]
 for i in range(100):
  t=time.perf_counter();rows.append(search5(i));ts.append((time.perf_counter()-t)*1000)
 peak=mx.get_peak_memory();active=mx.get_active_memory()
 top10=[search(i) for i in range(100)]
 cases=json.loads((D/'msmarco-precision-queries.json').read_text())['queries']
 hit5=sum(bool(set(map(str,row))&set(c['relevant_pids'])) for row,c in zip(rows,cases))
 hit10=sum(bool(set(map(str,row))&set(c['relevant_pids'])) for row,c in zip(top10,cases))
 return dict(dimensions=dimensions,dtype=dtype,matrix_bytes=a.nbytes,mean_ms=float(np.mean(ts)),median_ms=float(np.median(ts)),p95_ms=float(np.percentile(ts,95)),mlx_peak_allocated_bytes=peak,mlx_active_allocated_bytes=active,hit_at_5=hit5,hit_at_10=hit10,ids=rows,per_query_ms=ts)

if __name__=='__main__':
 if len(sys.argv)>1:print(json.dumps(run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 512)))
 else:
  variants=[]
  for dtype in ('float32','float16'):
   p=subprocess.run([sys.executable,__file__,dtype],cwd=ROOT,capture_output=True,text=True)
   if p.returncode:raise RuntimeError(p.stderr)
   variants.append(json.loads(p.stdout.strip().splitlines()[-1]))
  base=variants[0]['ids']
  for v in variants:
   v['exact_top5_match_count']=sum(a==b for a,b in zip(base,v['ids']))
   v['mean_top5_set_overlap']=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(base,v['ids'])]))
  report=dict(queries=100,k=5,dimensions=512,warmups=3,variants=variants,method='Separate processes; matrix and query use the same dtype, native MLX GPU matvec and partial selection. Three warmups; synchronized execution and result transfer timed. Hit@10 computed in separate untimed scans. Float16 values are rounded from the normalized float32 inputs; no per-query matrix renormalization. MLX allocation excludes Python/process overhead; no API calls.')
  (D/'mlx-precision-results.json').write_text(json.dumps(report,indent=2))
  print(json.dumps([{k:x for k,x in v.items() if k not in ('ids','per_query_ms')} for v in variants],indent=2))
