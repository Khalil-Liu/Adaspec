"""Unified new-protocol runtime. Saved projectors are reused without retraining.
Token-wise scaling follows the current routed implementation. This is not a
bit-for-bit reproduction of the old static adapter's feature-wise normalization.
"""
import ast, csv, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
ACTIONS=('bypass','svd','gevd')
def rows(path):
    p=Path(path)
    if p.suffix=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    text=p.read_text(encoding='utf-8-sig')
    try:
        d=json.loads(text)
        if isinstance(d,list):return d
    except ValueError:pass
    return [json.loads(l) for l in text.splitlines() if l.strip()]

class Runtime:
    def __init__(self,c):
        if c['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if c['device']=='cuda' and torch.cuda.get_device_properties(0).total_memory < 15*1024**3:
            raise RuntimeError('This unquantized 7B protocol requires a larger GPU (recommend >=20 GiB). Do not quantize silently.')
        self.c=c;self.action=0;self.capture=False;self.feature=None
        dt=getattr(torch,c['dtype'])
        self.tokenizer=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(c['model_path'],torch_dtype=dt,local_files_only=True).to(c['device']).eval()
        layers=self.model.model.layers;self.layer_ids=list(range(len(layers)-c['edit_layers'],len(layers)))
        if not 0<=c['probe_layer']<len(layers):raise ValueError('probe_layer outside model')
        self.proj={}
        for name in ['svd_positive','svd_negative','gevd_positive','gevd_negative']:
            tensor=torch.load(c[name],map_location='cpu',weights_only=True,mmap=True)
            if not isinstance(tensor,torch.Tensor) or tensor.ndim!=3:raise ValueError('Expected [layers,d,d] tensor: '+name)
            self.proj[name]={i:tensor[i].to(device=c['device'],dtype=dt).clone() for i in self.layer_ids}
            del tensor
        self.handles=[]
        self.handles.append(layers[c['probe_layer']].mlp.register_forward_hook(self._feature))
        for i in self.layer_ids:self.handles.append(layers[i].mlp.register_forward_hook(self._hook(i)))
    def _feature(self,module,inputs,output):
        if self.capture:self.feature=output.detach().float().mean(1)[0].cpu()
    def _hook(self,i):
        def hook(module,inputs,x):
            if self.action==0:return x
            name=ACTIONS[self.action];pos=self.proj[name+'_positive'][i];neg=self.proj[name+'_negative'][i]
            z=x@pos.T+x@neg.T
            y=z.float()*x.float().norm(dim=-1,keepdim=True)/(z.float().norm(dim=-1,keepdim=True)+1e-8)
            return y.to(x.dtype)
        return hook
    def prompt(self,question,context=None):
        if context is None:
            # Reuse the repository's TruthfulQA demonstrations and Llama chat prompt.
            from extract_probe_router_features import build_prompt,resolve_prompt_template
            return build_prompt(question,is_chat=True,prompt_template=resolve_prompt_template(self.c['model_path']))
        # Reuse just the repository prompt/demo definitions, without importing its evaluator.
        source=(ROOT/'src/benchmark_evaluation/bbq_bias_metrics_eval.py').read_text(encoding='utf-8-sig')
        tree=ast.parse(source);ns={}
        defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_demo_text']
        exec(compile(ast.Module(body=defs,type_ignores=[]),'<repo BBQ demo>','exec'),ns)
        from extract_probe_router_features import resolve_prompt_template
        tpl=resolve_prompt_template(self.c['model_path'])
        instruction=ns['create_demo_text']()+'Q: '+context+' '+question+'\nA:'
        return tpl['prompt'].format(instruction=instruction)
    @torch.no_grad()
    def features(self,prompt):
        self.action=0;self.capture=True;self.feature=None
        try:self.model(**self.tokenizer(prompt,return_tensors='pt').to(self.c['device']),use_cache=False)
        finally:self.capture=False
        if self.feature is None:raise RuntimeError('Feature hook did not run')
        return self.feature.clone()
    @torch.no_grad()
    def score(self,prompt,answer,action,average=False):
        self.action=int(action)
        prefix=self.tokenizer(prompt,return_tensors='pt').input_ids
        full=self.tokenizer(prompt+' '+answer.strip(),return_tensors='pt').input_ids
        n=prefix.shape[1]
        if not torch.equal(prefix,full[:,:n]):raise ValueError('Prompt/answer token boundary changed; inspect tokenizer before proceeding')
        if full.shape[1]<=n:raise ValueError('Empty candidate')
        ids=full.to(self.c['device']);logp=self.model(ids,use_cache=False).logits.float().log_softmax(-1)
        values=logp[0,n-1:-1].gather(1,ids[0,n:,None]).squeeze(1)
        return float(values.mean() if average else values.sum())
    @torch.no_grad()
    def generate(self,prompt,action):
        self.action=int(action);ids=self.tokenizer(prompt,return_tensors='pt').to(self.c['device'])
        out=self.model.generate(**ids,do_sample=False,max_new_tokens=self.c['max_new_tokens'],pad_token_id=self.tokenizer.eos_token_id)
        new=out[0,ids.input_ids.shape[1]:];eos=self.tokenizer.eos_token_id
        return {'raw_text':self.tokenizer.decode(new,skip_special_tokens=True),'new_tokens':len(new),'hit_length_limit':len(new)>=self.c['max_new_tokens'],'ended_with_eos':bool(len(new) and int(new[-1])==eos)}
