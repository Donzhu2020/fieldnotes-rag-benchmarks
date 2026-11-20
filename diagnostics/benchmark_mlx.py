"""100 cached queries, exact 512d float32 search, K=5; synchronized MLX GPU timing."""
import json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
root=Path(__file__).resolve().parents[1]
m=np.load(root/'data/search/vectors.npy',allow_pickle=False)
q=np.load(root/'diagnostics/msmarco-precision-query-vectors.npy',allow_pickle=False)[:,:512].copy()
q/=np.linalg.norm(q,axis=1,keepdims=True)
assert m.shape==(1000000,512) and m.dtype==np.float32
results={}; timings={}
def numpy_search(v,partial):
 s=m@v
 if partial:
  ids=np.argpartition(-s,4)[:5]
  return ids[np.lexsort((ids,-s[ids]))].tolist()
 return np.argsort(-s,kind='stable')[:5].tolist()
for name,partial in [('numpy_full_sort',False),('numpy_partial',True)]:
 for _ in range(3):numpy_search(q[0],partial)
 rows=[];ts=[]
 for v in q:
  t=time.perf_counter();rows.append(numpy_search(v,partial));ts.append((time.perf_counter()-t)*1000)
 results[name]=rows;timings[name]=ts
 print(name,round(float(np.mean(ts)),2),flush=True)
mx.set_default_device(mx.gpu)
a=mx.array(m);b=mx.array(q);mx.eval(a,b);mx.synchronize()
def mlx_search(v):
 s=a@v
 ids=mx.argpartition(-s,4)[:5]
 values=s[ids]
 mx.eval(ids,values);mx.synchronize()
 ix=np.array(ids);scores=np.array(values)
 return ix[np.lexsort((ix,-scores))].tolist()
for _ in range(3):mlx_search(b[0])
rows=[];ts=[]
for i in range(100):
 t=time.perf_counter();rows.append(mlx_search(b[i]));ts.append((time.perf_counter()-t)*1000)
results['mlx_gpu_partial']=rows;timings['mlx_gpu_partial']=ts
report={'shape':list(m.shape),'k':5,'queries':100,'warmups':3,'embedding_time_included':False,'mlx_peak_allocated_bytes':mx.get_peak_memory(),'variants':{n:{'mean_ms':float(np.mean(t)),'median_ms':float(np.median(t)),'p95_ms':float(np.percentile(t,95)),'exact_top5_match_count':sum(x==y for x,y in zip(results['numpy_full_sort'],results[n])),'per_query_ms':t,'ids':results[n]} for n,t in timings.items()},'limits':'Single run per variant. Partial selection does not guarantee the stable full-sort boundary choice on exactly tied scores. GPU and NumPy arrays coexist in this benchmark; MLX allocated memory is not total process RSS.'}
(root/'diagnostics/mlx-benchmark-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps({n:{k:v for k,v in x.items() if k not in ('ids','per_query_ms')} for n,x in report['variants'].items()},indent=2))
