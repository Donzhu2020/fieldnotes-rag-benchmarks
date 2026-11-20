"""Compare native MLX float16 dimensions in separate GPU processes."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
rows=[]
for dimensions in (512,256):
 p=subprocess.run([sys.executable,str(D/'benchmark_mlx_precision.py'),'float16',str(dimensions)],capture_output=True,text=True,check=True)
 rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
base=rows[0]['ids']
for row in rows:
 row['exact_top5_match_count_vs_512']=sum(a==b for a,b in zip(base,row['ids']))
 row['top5_set_overlap_vs_512']=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(base,row['ids'])]))
r=dict(variants=rows,queries=100,k=5,method='Same fixed 100 qrels queries, native float16 MLX GPU, separate processes, 3 warmups, synchronized K=5 latency including result transfer. Hit@10 measured separately. Reduced vectors truncated and normalized before rounding to float16. No API calls. One run per configuration.')
(D/'mlx-dimensions-results.json').write_text(json.dumps(r,indent=2))
print(json.dumps([{k:v for k,v in row.items() if k not in ('ids','per_query_ms')} for row in rows],indent=2))
