"""Small synthetic checks only; never loads a language model or editor weights."""
import ast,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from runtime import Runtime,ROOT,rows
from validate import probs,policies,preference_metrics,train_linear,split_pool
from paired_stats import analyze,holm,bbq_metric

def main():
    checked=[];c=json.loads(Path(__file__).with_name('config.json').read_text())
    # Check that calibration bins cover all probability mass, including p=1.
    m=preference_metrics(np.array([0,1,1]),np.array([0.,.75,1.]),1)
    assert sum(b['n'] for b in m['calibration_bins'])==3 and m['accuracy']==1.;checked.append('calibration boundaries')
    # The prompt must reuse the repository BBQ context/question format exactly.
    rt=Runtime.__new__(Runtime);rt.c=c
    prompt=rt.prompt('Which person?','A and B were present.')
    assert 'Q: A and B were present. Which person?\nA:' in prompt;checked.append('BBQ prompt/context formatting')
    assert rows(ROOT/c['truthfulqa'])[82]['Question'].lower().find('sword')>=0
    assert 'vote' in rows(ROOT/c['truthfulqa'])[120]['Question'].lower();checked.append('two actual case IDs')
    # Explicitly do not enable the split gate just to make collection run.
    try:split_pool(dict(c,confirm_projector_split=False))
    except ValueError:pass
    else:raise AssertionError('Unconfirmed split was accepted')
    checked.append('projection provenance gate')
    toy={'mean':[0,0],'std':[1,1],'weight':[[0,0],[0,0]],'bias':[0,0]}
    tr={'binary':toy,'threeway':dict(toy,weight=[[0,0],[0,0],[0,0]],bias=[1,0,0]),'temperature':1,'thresholds':[.4,.6],'best_fixed':1}
    acts,p=policies(c,np.zeros(2),tr);assert acts['adaspec']==0 and acts['router_no_bypass']==1 and acts['direct_threeway']==0;checked.append('bypass/no-bypass/three-way actions')
    x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
    fitted=train_linear(x,np.array([0,0,1,1]),2,dict(c,epochs=50));assert (probs(fitted,x).argmax(1)==np.array([0,0,1,1])).all();checked.append('CPU linear router fitting')
    assert holm([.01,.03,.04])==[.03,.06,.06];checked.append('Holm adjustment')
    # Hand-calculated paired rows verify positive differences and CI units.
    with tempfile.TemporaryDirectory(prefix='adaspec_validation_') as tmp:
        file=Path(tmp)/'truthfulqa.json';rs=[]
        for i in range(16):
            rs.append({'scores':[{'MC1':0,'MC2':.2,'MC3':0},{'MC1':1,'MC2':.8,'MC3':1}], 'actions':{'best_fixed_validation':0,'adaspec':1}})
        file.write_text(json.dumps({'rows':rs}),encoding='utf-8');r=analyze(file,200,42)
        assert r['policies']['adaspec']['MC1']['delta_pp']==100
        assert r['policies']['adaspec']['MC1']['paired_delta_95CI_pp']==[100,100]
        assert r['policies']['adaspec']['MC1']['p_raw']<.001
        checked.append('paired bootstrap and exact McNemar')
        # Same synthetic predictions, grouped into four paired BBQ clusters.
        file=Path(tmp)/'bbq_ambig.json';rs=[]
        for i in range(16):
            scores=[dict(MC1=0,Unk=0,Biased=1,NonUnknown=1,SR_numerator=1,SR_denominator=1,BiasValid=1,BiasCorrect=0),dict(MC1=1,Unk=1,Biased=0,NonUnknown=0,SR_numerator=0,SR_denominator=1,BiasValid=1,BiasCorrect=1)]
            rs.append({'scores':scores,'actions':{'best_fixed_validation':0,'adaspec':1},'cluster':['test',i//4]})
        file.write_text(json.dumps({'rows':rs}),encoding='utf-8');r=analyze(file,200,42)
        assert r['resampling_units']==4 and r['policies']['adaspec']['Acc']['delta_pp']==100
        assert r['policies']['adaspec']['signed_BS']['value_pct'] is None;checked.append('paired BBQ clusters and undefined denominators')
    report={'checks_passed':checked,'n_passed':len(checked),'language_model_inference':False,'scope':'synthetic code checks; not manuscript experimental evidence'}
    print(json.dumps(report,indent=2));Path(__file__).with_name('selfcheck_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
