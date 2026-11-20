"""Exact MLX top-100 retrieval followed by a local MS MARCO cross-encoder."""
import gc,json,time,resource,os
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import mlx.core as mx
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'diagnostics';CACHE=ROOT/'data/reranker'
CACHE.mkdir(parents=True,exist_ok=True)
MODEL='cross-encoder/ms-marco-MiniLM-L6-v2'
cases=json.loads((D/'msmarco-precision-queries.json').read_text())['queries']
candidate_path=CACHE/'mlx-512-f16-top100.json'
if candidate_path.exists():
 candidate_data=json.loads(candidate_path.read_text());candidates=candidate_data['ids'];retrieval_ms=candidate_data['retrieval_ms']
else:
 mx.set_default_device(mx.gpu)
 a=mx.load(str(ROOT/'data/search/vectors.npy')).astype(mx.float16)
 q=np.load(D/'msmarco-precision-query-vectors.npy')[:,:512].copy();q/=np.linalg.norm(q,axis=1,keepdims=True)
 b=mx.array(q,dtype=mx.float16);mx.eval(a,b);mx.synchronize()
 def retrieve(i):
  scores=a@b[i];ids=mx.argpartition(-scores,99)[:100];values=scores[ids]
  mx.eval(ids,values);mx.synchronize();ix=np.array(ids);vs=np.array(values)
  return ix[np.lexsort((ix,-vs))].tolist()
 for _ in range(3):retrieve(0)
 candidates=[];retrieval_ms=[]
 for i in range(100):
  t=time.perf_counter();candidates.append(retrieve(i));retrieval_ms.append((time.perf_counter()-t)*1000)
 candidate_path.write_text(json.dumps(dict(ids=candidates,retrieval_ms=retrieval_ms)))
 del a,b;gc.collect();mx.clear_cache()
needed={str(i) for row in candidates for i in row};texts={}
with (ROOT/'data/msmarco/collection.first1m.tsv').open() as f:
 for line in f:
  pid,text=line.rstrip('\n').split('\t',1)
  if pid in needed:texts[pid]=text
assert len(texts)==len(needed)
print('Loading public cross-encoder model',flush=True)
tokenizer=AutoTokenizer.from_pretrained(MODEL,cache_dir=CACHE/'model')
model=AutoModelForSequenceClassification.from_pretrained(MODEL,cache_dir=CACHE/'model',use_safetensors=True)
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=model.to(device).eval()
def sync():
 if device=='mps':torch.mps.synchronize()
def rerank(index):
 case=cases[index];ids=candidates[index];scores=[];truncated=0
 for start in range(0,100,16):
  batch=ids[start:start+16]
  features=tokenizer([case['query']]*len(batch),[texts[str(i)] for i in batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
  with torch.inference_mode():logits=model(**features).logits.flatten()
  scores.extend(logits.float().cpu().tolist())
 order=np.lexsort((np.array(ids),-np.array(scores)))
 return [ids[i] for i in order]
rerank(0);sync()
rows=[];rerank_ms=[]
for i in range(100):
 sync();start=time.perf_counter();ranked=rerank(i);sync();rerank_ms.append((time.perf_counter()-start)*1000);rows.append(ranked)
 if (i+1)%10==0:print(f'{i+1}/100 reranked',flush=True)
def metrics(rankings):
 hits={str(k):sum(bool(set(map(str,row[:k]))&set(c['relevant_pids'])) for row,c in zip(rankings,cases)) for k in (1,5,10,100)}
 mrr=np.mean([next((1/(j+1) for j,pid in enumerate(row[:10]) if str(pid) in set(c['relevant_pids'])),0) for row,c in zip(rankings,cases)])
 return dict(hit_at_k=hits,mrr_at_10=float(mrr))
def latency(values):return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
report=dict(model=MODEL,model_revision=getattr(model.config,'_commit_hash',None),device=device,queries=100,candidate_count=100,max_length=512,batch_size=16,retriever='MLX GPU 512d float16 exact scan, top100, no cutoff',baseline=metrics(candidates),reranked=metrics(rows),retrieval_latency=latency(retrieval_ms),rerank_latency=latency(rerank_ms),combined_latency=latency(np.array(retrieval_ms)+np.array(rerank_ms)),protocol='Same cached dev queries; no training or qrels given to reranker; stable ID tie-break within candidate scores. Retrieval and reranking stages measured separately; combined values are sums, not a timed deployed endpoint. One warmup query for reranker. Tokenization, model inference and synchronization included; loading excluded. No paid API calls.',limitations=['Selected reused 100-query sample, not held out for project tuning.','Model pretrained on MS MARCO training data; split overlap with this expanded dev sample has not been independently audited.','Sequences longer than 512 tokens are truncated.','Candidate retrieval runs with MLX and cross-encoder with PyTorch MPS; production memory and concurrency not measured.'],cases=[dict(qid=c['qid'],baseline_ids=[str(x) for x in before],reranked_ids=[str(x) for x in after]) for c,before,after in zip(cases,candidates,rows)])
(D/'reranker-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
