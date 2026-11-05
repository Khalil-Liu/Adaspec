from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import binomtest
import argparse,os
parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);config=json.loads(Path(args.config).read_text(encoding='utf-8'));OUT=ROOT/config['output_dir']/'existing_records'
if (OUT/'paired_statistics.json').exists():raise FileExistsError(OUT/'paired_statistics.json')
OUT.mkdir(parents=True,exist_ok=True)
B=10000; rng=np.random.default_rng(42)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ci(x):return [float(v) for v in np.quantile(x,[.025,.975])]
def holm(ps):
 order=np.argsort(ps);out=[0.]*len(ps);last=0
 for j,i in enumerate(order):last=max(last,min(1.,(len(ps)-j)*ps[i]));out[i]=last
 return out
reports={}
for label,base,ada,status in [
 ('historical_candidate','exp_results/truthfulqa/sea_llama2_7b/result1.json','exp_results/truthfulqa/probe_router_abstain_040_060_L2/result.json','AdaSpec matches manuscript; saved SEA baseline differs from manuscript. Exploratory only; do not insert these deltas as manuscript-table comparisons.'),
 ('t438_matched','exp_results/p0_cost_t438/svd_l2_result.json','exp_results/p0_cost_t438/adaspec_l2_result.json','Saved t438 pair, distinct from original manuscript table. Statistical analysis is exploratory; complete protocol provenance remains separate.')]:
 a=load(base);b=load(ada);assert a['question']==b['question'];n=len(a['model_scores']); assert n==817
 aa=np.array([[x[k] for k in ['MC1','MC2','MC3']] for x in a['model_scores']]);bb=np.array([[x[k] for k in ['MC1','MC2','MC3']] for x in b['model_scores']]);ds=bb-aa
 abs_a=[];abs_b=[];deltas=[]
 for _ in range(B//500):
  idx=rng.integers(0,n,size=(500,n));abs_a.append(aa[idx].mean(axis=1));abs_b.append(bb[idx].mean(axis=1));deltas.append(ds[idx].mean(axis=1))
 abs_a=np.concatenate(abs_a);abs_b=np.concatenate(abs_b);deltas=np.concatenate(deltas);metrics={};ps=[]
 for j,k in enumerate(['MC1','MC2','MC3']):
  obs=abs(ds[:,j].mean())
  if j==0:
   n01=int(((aa[:,0]==0)&(bb[:,0]==1)).sum());n10=int(((aa[:,0]==1)&(bb[:,0]==0)).sum());p=binomtest(min(n01,n10),n01+n10,.5).pvalue if n01+n10 else 1.;test='exact two-sided McNemar'
  else:
   exceeds=0
   for _ in range(B//500):
    signs=rng.choice([-1,1],size=(500,n));exceeds+=int((np.abs((signs*ds[:,j]).mean(axis=1))>=obs-1e-15).sum())
   p=(exceeds+1)/(B+1);test='two-sided paired sign-flip Monte Carlo, 10000 draws, plus-one p'
  ps.append(p);metrics[k]={'SEA_pct':float(aa[:,j].mean()*100),'AdaSpec_pct':float(bb[:,j].mean()*100),'SEA_95CI_pct':ci(abs_a[:,j]*100),'AdaSpec_95CI_pct':ci(abs_b[:,j]*100),'delta_pp':float(ds[:,j].mean()*100),'paired_delta_95CI_pp':ci(deltas[:,j]*100),'p_raw':float(p),'test':test}
 for k,p in zip(metrics,holm(ps)):metrics[k]['p_Holm_within_three_metrics']=p
 reports[label]={'baseline':base,'comparison':ada,'n':n,'status':status,'MC1_paired_counts':{'AdaSpec_only':n01,'SEA_only':n10},'metrics':metrics}
rawp=Path('data/BBQ/disambig/All-test-disambig.jsonl');raw=[json.loads(l) for l in rawp.read_text(encoding='utf-8').splitlines()]
basep='exp_results/bbq/sea_llama2_7b/bias_metrics.json';adap='exp_results/bbq/probe_router_abstain_040_060/result.json';base=load(basep)['results'];ada=load(adap)['results']
assert len(base)==len(ada)==len(raw)==5246;assert {x['context_condition'] for x in raw}=={'disambig'}
for x,a,b in zip(raw,base,ada):
 assert all(x[k]==a[k]==b[k] for k in ['category','example_id','question','context'])
 assert x['label']==a['correct_idx']==b['correct_idx'];assert a['unknown_idx']==b['unknown_idx']
 assert b['candidate_answers']=={f'ans{i}':x[f'ans{i}'] for i in range(3)}
 assert a['pred_answer']==x[f"ans{a['pred_idx']}"]
clusters=sorted({(x['category'],str(x['question_index'])) for x in raw}); index={k:i for i,k in enumerate(clusters)};totals=[]
for records in [base,ada]:
 group=np.zeros((len(clusters),6))
 for x,r,canonical in zip(raw,records,ada):
  pred=r['pred_idx'];unk=pred==canonical['unknown_idx'];biased=pred==canonical['biased_idx'];anti=bool(canonical['gold_is_antistereotypical'])
  group[index[(x['category'],str(x['question_index']))]] += [1,pred==x['label'],unk,biased and not unk,anti and biased,anti]
 totals.append(group)
def bbqmetrics(s):
 n=s[...,0];return np.stack([s[...,1]/n,s[...,2]/n,2*s[...,3]/(n-s[...,2])-1,s[...,4]/s[...,5]],axis=-1)*100
bbq_a=[];bbq_b=[]
for _ in range(B//500):
 idx=rng.integers(0,len(clusters),size=(500,len(clusters)));bbq_a.append(bbqmetrics(totals[0][idx].sum(axis=1)));bbq_b.append(bbqmetrics(totals[1][idx].sum(axis=1)))
ba=np.concatenate(bbq_a);bb=np.concatenate(bbq_b);pa=bbqmetrics(totals[0].sum(axis=0));pb=bbqmetrics(totals[1].sum(axis=0));bm={}
for j,k in enumerate(['Acc','Unk','signed_BS','SR']):bm[k]={'SEA_pct':float(pa[j]),'AdaSpec_pct':float(pb[j]),'SEA_95CI_pct':ci(ba[:,j]),'AdaSpec_95CI_pct':ci(bb[:,j]),'delta_pp':float(pb[j]-pa[j]),'paired_cluster_delta_95CI_pp':ci(bb[:,j]-ba[:,j])}
reports['bbq_disambiguated']={'n':5246,'clusters':len(clusters),'cluster_key':'category,question_index','baseline':basep,'comparison':adap,'canonical_bias_metadata':'AdaSpec result official metadata; aligned raw labels, option order, contexts, question IDs checked for both predictions','metrics':bm,'ambiguous':'No ambiguous evaluation predictions found in current local result inventory; do not infer from disambiguated data.'}
sources=[v[k] for v in reports.values() for k in ['baseline','comparison']]+[str(rawp)]
reports['protocol']={'bootstrap_replicates':B,'seed':42,'TruthfulQA_unit':'question','BBQ_unit':'paired category/question_index cluster; percentile CI','scope':'existing records only, no inference/training, does not resolve R3-5','source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}
(OUT/'paired_statistics.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 第二部分：已有逐题记录分析','', '本轮只运行CPU统计，未重跑模型；历史候选TruthfulQA基线与主表略有不同，不能直接替代稿件中SEA对比。','']
for label in ['historical_candidate','t438_matched','bbq_disambiguated']:
 x=reports[label];lines += ['## '+label,'',x.get('status','BBQ disambiguated，统一元数据重算，327个题型cluster配对bootstrap。'),'', '|指标|SEA|AdaSpec|差值(pp)|配对95% CI(pp)|','|---|---:|---:|---:|---|']
 for k,m in x['metrics'].items():
  c=m.get('paired_delta_95CI_pp',m.get('paired_cluster_delta_95CI_pp'));lines.append(f"|{k}|{m['SEA_pct']:.2f}|{m['AdaSpec_pct']:.2f}|{m['delta_pp']:+.2f}|[{c[0]:+.2f}, {c[1]:+.2f}]|")
 lines.append('')
(OUT/'results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
