"""Locate qrels positives in exact full-corpus rankings without new API calls."""
from pathlib import Path
import json,time
import numpy as np
D=Path(__file__).resolve().parent
root=D.parent
cases=json.loads((D/'msmarco-precision-queries.json').read_text())['queries']
q=np.load(D/'msmarco-precision-query-vectors.npy')[:,:512].copy();q/=np.linalg.norm(q,axis=1,keepdims=True)
a=np.load(root/'data/search/vectors.npy');assert a.shape==(1000000,512)
rows=[]
for case,v in zip(cases,q):
 scores=a@v
 ranks=[]
 for pid in case['relevant_pids']:
  i=int(pid);score=scores[i]
  rank=1+int(np.count_nonzero(scores>score))+int(np.count_nonzero(scores[:i]==score))
  ranks.append(rank)
 top=np.argpartition(-scores,4)[:5];top=top[np.lexsort((top,-scores[top]))]
 rows.append(dict(qid=case['qid'],query=case['query'],positive_pids=case['relevant_pids'],best_positive_rank=min(ranks),top5_ids=[str(i) for i in top]))
r=dict(method='Exact NumPy float32 full scan; qrels-positive rank counts all greater scores and preceding row ties; same selected 100 queries; no threshold.',hit_at_k={str(k):sum(row['best_positive_rank']<=k for row in rows) for k in (1,5,10,20,50,100,1000)},cases=rows)
(D/'retrieval-diagnosis.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r['hit_at_k']))
# Inspect nearest misses with known answers in top20, without changing judgments.
selected=[x for x in rows if 5<x['best_positive_rank']<=20][:4]
want={pid for row in selected for pid in row['top5_ids'][:2]+row['positive_pids']}
texts={}
with (root/'data/msmarco/collection.first1m.tsv').open() as f:
 for line in f:
  pid,text=line.rstrip('\n').split('\t',1)
  if pid in want:texts[pid]=text
for row in selected:
 print(json.dumps({'query':row['query'],'gold_rank':row['best_positive_rank'],'gold':[texts[p][:260] for p in row['positive_pids']],'top2':[texts[p][:260] for p in row['top5_ids'][:2]]}))
