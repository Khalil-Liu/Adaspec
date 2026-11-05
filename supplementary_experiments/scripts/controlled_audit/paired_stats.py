"""CPU-only paired inference on saved new-protocol outputs; no model loading."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from runtime import ROOT

def holm(ps):
    order=np.argsort(ps);result=[0.]*len(ps);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,(len(ps)-rank)*ps[i]));result[i]=last
    return result

def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975]).tolist() if len(x) else None

def bbq_metric(s,subset):
    n,acc,unk,biased,nonunk,srn,srd,valid,valid_correct=np.moveaxis(s,-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        bs=2*biased/nonunk-1
        if subset=='ambig':bs=(1-valid_correct/valid)*bs
        return np.stack([acc/n,unk/n,bs,srn/srd],axis=-1)

def analyze(path,B=10000,seed=42):
    source=json.loads(Path(path).read_text(encoding='utf-8'))
    if source.get('smoke_limit'):raise ValueError('Smoke outputs are excluded from formal statistics')
    records=source['rows'];n=len(records);names=list(records[0]['actions']);rng=np.random.default_rng(seed)
    isbbq=Path(path).stem.startswith('bbq_');subset='ambig' if Path(path).stem=='bbq_ambig' else 'disambig'
    keys=['MC1','Unk','Biased','NonUnknown','SR_numerator','SR_denominator','BiasValid','BiasCorrect'] if isbbq else ['MC1','MC2','MC3']
    values={name:np.array([[r['scores'][r['actions'][name]][k] for k in keys] for r in records]) for name in names}
    if isbbq:
        clusters=[tuple(r['cluster']) for r in records]
        if any(None in v for v in clusters):raise ValueError('Missing BBQ cluster identifiers')
        labels=sorted(set(clusters),key=str);lookup={k:i for i,k in enumerate(labels)}
        values={name:np.column_stack([np.ones(n),a]) for name,a in values.items()}
        groups={}
        for name,a in values.items():
            g=np.zeros((len(labels),a.shape[1]));np.add.at(g,[lookup[k] for k in clusters],a);groups[name]=g
        unit=len(labels);metric=lambda a:bbq_metric(a,subset);metric_names=['Acc','Unk','signed_BS','SR']
        point={name:metric(a.sum(0)) for name,a in groups.items()}
    else:
        groups=values;unit=n;metric=lambda a:a;metric_names=keys
        point={name:a.mean(0) for name,a in groups.items()}
    if not unit:raise ValueError('No samples')
    draws={name:[] for name in names}
    for start in range(0,B,100):
        size=min(100,B-start);idx=rng.integers(0,unit,(size,unit))
        for name,a in groups.items():draws[name].append(metric(a[idx].sum(1)) if isbbq else a[idx].mean(1))
    draws={name:np.concatenate(a) for name,a in draws.items()}
    comparisons={};baseline='best_fixed_validation'
    for name in names:
        comparisons[name]={}
        for j,k in enumerate(metric_names):
            a=point[name][j];b=point[baseline][j]
            comparisons[name][k]={'value_pct':float(a*100) if np.isfinite(a) else None,'95CI_pct':ci(draws[name][:,j]*100),'baseline_pct':float(b*100) if np.isfinite(b) else None,'delta_pp':float((a-b)*100) if np.isfinite(a-b) else None,'paired_delta_95CI_pp':ci((draws[name][:,j]-draws[baseline][:,j])*100)}
    # Predeclared primary comparison: AdaSpec versus validation-chosen best fixed editor.
    # Other policies receive intervals, without adding exploratory p values to this family.
    if not isbbq:
        delta=values['adaspec']-values[baseline];ps=[]
        for j,k in enumerate(metric_names):
            if j==0:
                w=int((delta[:,j]>0).sum());l=int((delta[:,j]<0).sum());pv=float(binomtest(min(w,l),w+l,.5).pvalue) if w+l else 1.;method='exact two-sided McNemar'
            else:
                exceed=0;obs=abs(delta[:,j].mean())
                for start in range(0,B,100):
                    signs=rng.choice([-1,1],size=(min(100,B-start),n));exceed+=int((abs((signs*delta[:,j]).mean(1))>=obs-1e-15).sum())
                pv=(exceed+1)/(B+1);method='paired two-sided sign-flip Monte Carlo, plus-one correction'
            comparisons['adaspec'][k].update(p_raw=pv,test=method);ps.append(pv)
        for k,pv in zip(metric_names,holm(ps)):comparisons['adaspec'][k]['p_Holm_three_metrics']=pv
    return {'source':str(path),'n':n,'resampling_units':unit,'unit':'category/question_index cluster' if isbbq else 'question','replicates':B,'seed':seed,'baseline':baseline,'primary_comparison':'adaspec vs best_fixed_validation','other_comparisons':'descriptive intervals','policies':comparisons}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).with_name('config.json')));p.add_argument('--replicates',type=int,default=10000);a=p.parse_args()
    if a.replicates<100:raise ValueError('At least 100 bootstrap replicates required')
    c=json.loads(Path(a.config).read_text(encoding='utf-8'));out=ROOT/c['output_dir'];reports={}
    target=out/'paired_statistics.json'
    if target.exists():raise FileExistsError(target)
    for name in ['truthfulqa','bbq_disambig','bbq_ambig']:
        source=out/(name+'.json')
        if source.exists():reports[name]=analyze(source,a.replicates,c['seed'])
    if not reports:raise FileNotFoundError('No complete evaluation records')
    target.write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print('Saved',target)
if __name__=='__main__':main()
