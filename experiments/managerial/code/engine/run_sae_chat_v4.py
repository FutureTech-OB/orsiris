"""Chat answer-boundary SAE search. Fixed prompts, no thinking, no target at test.

Candidate discovery uses human aggregate hints. Calibration uses rotations 0/1;
rotations >=2 are format holdouts on the SAME studies, not new human validation.
"""
from pathlib import Path
import argparse, collections, gc, itertools, json, math, re, time
import numpy as np
import torch
import transformers
from run_experiment import ModelService, make_prompt, save, sha, ROOT

W = Path(__file__).resolve().parent
LAYERS = [24, 36, 48]

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def avg(xs): return sum(xs) / len(xs)
def emit(f, r): f.write(json.dumps(r) + '\n'); f.flush()

def effects_of(rows, cases, effects):
    idx = {(r['case_id'], r['rotation']): r for r in rows}
    result = []
    for e in effects:
        relevant = [c for c in cases if c['study_id'] == e['study_id'] and c['item'] in e['items'] and c['group'] in e['groups1'] + e['groups2']]
        rotations = sorted(set(r['rotation'] for r in rows if r['study_id'] == e['study_id']))
        for rot in rotations:
            means = []
            for side in [1, 2]:
                gm, gn = [], []
                for group in e[f'groups{side}']:
                    cc = [c for c in relevant if c['group'] == group]
                    gm.append(avg([idx[(c['id'], rot)]['mean'] for c in cc])); gn.append(cc[0]['group_n'])
                means.append(sum(m*n for m,n in zip(gm,gn))/sum(gn))
            delta = means[0]-means[1]
            result.append(dict(effect_id=e['effect_id'], study_id=e['study_id'], rotation=rot,
                human_delta=e['human_delta'], model_delta=delta, gap=abs(delta-e['human_delta'])/abs(e['human_delta'])))
    return result

def score(rows, cases, effects):
    ee = effects_of(rows, cases, effects)
    grouped = collections.defaultdict(list)
    for r in ee: grouped[r['effect_id']].append(r['gap'])
    return avg([avg(x) for x in grouped.values()])

class Engine:
    def __init__(self, design):
        self.model = ModelService(design['model'], variant='chat', dtype='bf16', device_map='cuda', enable_thinking=False)
        self.labels = []
        for i in range(11):
            ids = [self.model.tokenizer.encode(s, add_special_tokens=False) for s in [chr(65+i), ' '+chr(65+i)]]
            assert len(ids[0]) == 1
            self.labels.append(sorted({t[0] for t in ids if len(t)==1}))
        self.count = 0

    def inputs(self, c, rot, mode='base'):
        prompt, values = make_prompt(c, mode, rot)
        formatted = self.model.format_prompt(prompt)
        ids = self.model.tokenizer(formatted, return_tensors='pt')['input_ids'].to(self.model.device)
        return ids, values, prompt

    def read(self, c, rot, mode='base', delta=None, capture=False):
        ids, values, prompt = self.inputs(c, rot, mode)
        captured, perturbations, handles = {}, {}, []
        def hook_for(layer):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                old = h[0,-1].detach().float()
                if capture: captured[layer] = old.cpu()
                if not delta or layer not in delta: return output
                d = delta[layer].to(device=old.device)
                edited = h.clone(); edited[0,-1] = (old+d).to(h.dtype)
                perturbations[str(layer)] = dict(requested_norm=float(d.norm()), actual_norm=float((edited[0,-1].float()-old).norm()), residual_norm=float(old.norm()))
                return (edited,)+output[1:] if isinstance(output,tuple) else edited
            return hook
        try:
            for l in LAYERS if capture else sorted(delta or {}):
                handles.append(self.model.get_decoder_layers()[l].register_forward_hook(hook_for(l)))
            with torch.inference_mode():
                logits = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[0,-1].float()
            p = logits.softmax(-1); weights = [float(p[t].sum()) for t in self.labels[:len(values)]]
            mass = sum(weights); assert mass > 0 and math.isfinite(mass)
            probs = [v/mass for v in weights]; top = int(logits.argmax()); self.count += 1
            return dict(case_id=c['id'], study_id=c['study_id'], group=c['group'], item=c['item'], rotation=rot, mode=mode,
                values=values, probabilities=probs, mean=sum(v*p for v,p in zip(values,probs)), label_mass=mass,
                top_token=self.model.tokenizer.decode([top],skip_special_tokens=False), top_legal=top in sum(self.labels[:len(values)],[]),
                n_tokens=int(ids.shape[1]), prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
                h_norms={str(l):float(h.norm()) for l,h in captured.items()}, perturbations=perturbations), captured
        finally:
            for h in handles: h.remove()

    def generate(self, c, rot, delta=None):
        ids, values, prompt = self.inputs(c, rot); handles=[]
        def hook_for(layer):
            def hook(module, inputs, output):
                h=output[0] if isinstance(output,tuple) else output
                # The intervention is only at the initial answer boundary.
                if h.shape[1] != ids.shape[1]: return output
                edited=h.clone(); edited[0,-1]=(h[0,-1].float()+delta[layer].to(h.device)).to(h.dtype)
                return (edited,)+output[1:] if isinstance(output,tuple) else edited
            return hook
        try:
            for l in sorted(delta or {}): handles.append(self.model.get_decoder_layers()[l].register_forward_hook(hook_for(l)))
            with torch.inference_mode():
                result=self.model.model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,do_sample=False,pad_token_id=self.model.tokenizer.pad_token_id)
            text=self.model.tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
            return dict(case_id=c['id'],rotation=rot,response=text,valid=bool(re.fullmatch('([A-'+chr(64+len(values))+'])[.\\s]*',text)),tokens=int(result.shape[1]-ids.shape[1]))
        finally:
            for h in handles: h.remove()
