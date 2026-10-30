"""Calibrate one shared SAE controller on ten organizational contrasts."""
from pathlib import Path
import hashlib,json,math,time,os,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from run_sae_chat_v4 import Engine,load,save,sha,emit,W
from sae_domain_v12_metrics import score,compare

class DomainEngine(Engine):
    def inputs(self,c,rot,mode='base'):
        assert mode=='base'
        vals=c['values'];n=len(vals)
        if rot==1:vals=list(reversed(vals))
        elif rot==2:vals=vals[n//2:]+vals[:n//2]
        else:assert rot==0
        return super().inputs(dict(c,values=vals),0,'base')
    def state(self,c,order,gradient=False):
        ids,values,prompt=self.inputs(c,order);formatted=self.model.format_prompt(prompt)
        prefix=prompt.split('\n\nCurrent survey item:')[0];assert prefix!=prompt
        end=formatted.index(prefix)+len(prefix)
        tok=self.model.tokenizer(formatted,return_offsets_mapping=True);assert tok['input_ids']==ids[0].tolist()
        positions=[i for i,(a,b) in enumerate(tok['offset_mapping']) if b>0 and b<=end][-16:]
        assert len(positions)==16 and max(positions)<ids.shape[1]-10
        captured={};leaf=torch.zeros(5120,device=self.model.device,dtype=torch.float32,requires_grad=gradient)
        def hook(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            captured['answer']=h[0,-1].detach().float().cpu();captured['context']=h[0,positions].detach().float().cpu()
            if not gradient:return output
            edited=h.clone();edited[0,-1]=(h[0,-1].float()+leaf).to(h.dtype)
            return (edited,)+output[1:] if isinstance(output,tuple) else edited
        handle=self.model.get_decoder_layers()[48].register_forward_hook(hook)
        try:
            with torch.enable_grad() if gradient else torch.inference_mode():
                logits=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[0,-1].float()
                lw=torch.stack([torch.logsumexp(logits[t],dim=0) for t in self.labels[:len(values)]])
                mu=(lw.softmax(0)*torch.tensor(values,device=logits.device)).sum()
                grad=torch.autograd.grad(mu,leaf)[0].detach().cpu() if gradient else None
            p=logits.detach().softmax(-1);weights=[float(p[t].sum()) for t in self.labels[:len(values)]];mass=sum(weights);probs=[w/mass for w in weights];top=int(logits.detach().argmax())
            row=dict(case_id=c['id'],study_id=c['study_id'],group=c['group'],item=c['item'],rotation=order,mode='base',values=values,
                probabilities=probs,mean=sum(v*p for v,p in zip(values,probs)),label_mass=mass,
                top_token=self.model.tokenizer.decode([top],skip_special_tokens=False),top_legal=top in sum(self.labels[:len(values)],[]),
                n_tokens=int(ids.shape[1]),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),h_norms={'48':float(captured['answer'].norm())},
                context_sha256=hashlib.sha256(json.dumps(ids[0,:max(positions)+1].tolist()).encode()).hexdigest(),perturbations={})
            assert abs(row['mean']-float(mu.detach()))<1e-4
            if gradient:assert torch.isfinite(grad).all() and grad.norm()>0
            self.count+=1;return row,captured,grad
        finally:handle.remove()

    def prefix_context(self,c,order):
        ids,_,prompt=self.inputs(c,order);formatted=self.model.format_prompt(prompt)
        prefix=prompt.split('\n\nCurrent survey item:')[0];end=formatted.index(prefix)+len(prefix)
        tok=self.model.tokenizer(formatted,return_offsets_mapping=True)
        positions=[i for i,(a,b) in enumerate(tok['offset_mapping']) if b>0 and b<=end][-16:]
        short=ids[:,:max(positions)+1];captured={}
        def hook(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            captured['context']=h[0,positions].detach().float().cpu()
        handle=self.model.get_decoder_layers()[48].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model.model(input_ids=short,attention_mask=torch.ones_like(short),use_cache=False,logits_to_keep=1)
            return captured['context']
        finally:handle.remove()


WORKER=None
def initialize_worker(queue,design):
    global WORKER
    gpu=queue.get();os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    torch.set_num_threads(2);torch.manual_seed(design['seed']);torch.backends.cuda.matmul.allow_tf32=False
    WORKER=DomainEngine(design)
    first=next(c for c in design['cases'] if c['id']==design['fit_case_ids'][0])
    probe,_=WORKER.read(first,0)
    save(W/'sae_domain_v12'/f'worker_{gpu}_probe.json',dict(gpu=gpu,pid=os.getpid(),row=probe))

def worker_read(c,order,delta):
    row,_=WORKER.read(c,order,delta=delta)
    row['worker_gpu']=os.environ['CUDA_VISIBLE_DEVICES']
    return row


def encode(h,blob):
    out=[]
    for batch in h.split(32):
        pre=batch@blob['W_enc'].T+blob['b_enc'];v,i=pre.topk(100,dim=-1)
        out.append(torch.zeros_like(pre).scatter_(-1,i,v))
    return torch.cat(out)

def main():
    start=time.monotonic();out=W/'sae_domain_v12';out.mkdir(exist_ok=False)
    d=load(W/'sae_domain_v12_protocol.json');cases=d['cases'];byid={c['id']:c for c in cases}
    fitcases=[byid[cid] for cid in d['fit_case_ids']];fiteffects=[e for e in d['effects'] if e['split']=='calibration']
    assert sha(W/'run_sae_chat_v4.py')==d['source_engine_sha256']
    save(out/'frozen_protocol.json',d)
    save(out/'metadata.json',dict(script_sha256=sha(Path(__file__)),protocol_sha256=sha(W/'sae_domain_v12_protocol.json'),
        metrics_sha256=sha(W/'sae_domain_v12_metrics.py'),source_engine_sha256=sha(W/'run_sae_chat_v4.py'),
        model_service_sha256=sha(W/'src/model_service.py')))
    torch.set_num_threads(4);torch.manual_seed(d['seed']);torch.backends.cuda.matmul.allow_tf32=False
    engine=DomainEngine(d)
    for p in engine.model.model.parameters():p.requires_grad_(False)
    print(json.dumps(dict(stage='model_ready',seconds=round(time.monotonic()-start,1))),flush=True)
    blob=torch.load(Path(d['sae_root'])/'layer48.sae.pt',weights_only=True,map_location='cpu',mmap=True)
    baseline={};states={};context_codes={};context_hidden={};context_audit=[]
    with (out/'baseline.jsonl').open('w') as bf:
        def collect(cc,orders,gradient=False):
            for order in orders:
                for c in cc:
                    key=(c['id'],order)
                    if key in baseline:continue
                    row,h,g=engine.state(c,order,gradient);baseline[key]=row;emit(bf,row)
                    ph=row['context_sha256'];drift=0.
                    if ph not in context_codes:
                        context_hidden[ph]=engine.prefix_context(c,order)
                        context_codes[ph]=encode(context_hidden[ph],blob).mean(0)
                    drift=float((h['context']-context_hidden[ph]).abs().max())
                    rel=float((h['context']-context_hidden[ph]).norm()/context_hidden[ph].norm().clamp_min(1e-12))
                    context_audit.append(dict(case_id=c['id'],order=order,context_sha256=ph,max_full_vs_canonical_difference=drift,relative_full_vs_canonical_l2=rel))
                    states[key]=dict(answer=h['answer'],gradient=g,context_code=context_codes[ph])
                print(json.dumps(dict(stage='collect',gradient=gradient,order=order,baseline_rows=len(baseline),seconds=round(time.monotonic()-start,1))),flush=True)
        collect(fitcases,d['fit_orders'],True)
        keys=[(c['id'],o) for o in d['fit_orders'] for c in fitcases];nv=len(d['fit_orders']);nc=len(fitcases);ne=len(fiteffects);nf=d['feature_count']
        h=torch.stack([states[k]['answer'] for k in keys]);g=torch.stack([states[k]['gradient'] for k in keys])*h.norm(dim=1,keepdim=True)
        z=torch.stack([states[k]['context_code'] for k in keys]);center=z.mean(0);scale=z.std(0,unbiased=False).clamp_min(1e-6);gates=((z-center)/scale).clamp(-2,2)
        dec=blob['W_dec'];norms=dec.norm(dim=0).clamp_min(1e-8);raw=(g@dec)/norms
        contrast=torch.tensor([[e['coefficients'].get(c['id'],0.) for c in fitcases] for e in fiteffects])
        assert int(torch.linalg.matrix_rank(contrast))==8
        base=torch.tensor([baseline[k]['mean'] for k in keys]).reshape(nv,nc)
        targets=torch.tensor([e['human_delta'] for e in fiteffects]);spans=torch.tensor([max(c['values'])-min(c['values']) for c in fitcases])
        pools={};details=[]
        for mode in ['context','static']:
            deriv=(raw*gates if mode=='context' else raw).reshape(nv,nc,-1)
            cd=torch.einsum('ec,vcf->evf',contrast,deriv)
            noise=((deriv-deriv.mean(0,keepdim=True))/spans[None,:,None]).square().mean((0,1)).sqrt()
            scores=((cd.mean(1).abs()-.5*cd.std(1,unbiased=False))/targets.abs()[:,None]).mean(0)-.1*noise
            if mode=='context':
                active=z.reshape(nv,nc,-1).ne(0).any(1).all(0)&(scale>1.000001e-6);scores[~active]=-torch.inf
            chosen=[];vecs=[]
            for fid in scores.argsort(descending=True).tolist():
                if not torch.isfinite(scores[fid]):continue
                v=dec[:,fid]/norms[fid]
                if any(abs(float(v@q))>.9 for q in vecs):continue
                chosen.append(fid);vecs.append(v)
                if len(chosen)==nf:break
            assert len(chosen)==nf
            kind='sae_'+mode;pools[kind]=dict(mode=mode,fids=chosen,vectors=torch.stack(vecs),derivative=deriv[:,:,chosen])
            details.append(dict(kind=kind,fids=chosen,ranking_scores=scores[chosen].tolist()))
        for repeat in [1,2]:
            gen=torch.Generator().manual_seed(d['seed']+repeat);vectors=torch.nn.functional.normalize(torch.randn(nf,5120,generator=gen),dim=1);fids=pools['sae_context']['fids']
            pools[f'random_context_{repeat}']=dict(mode='context',fids=fids,vectors=vectors,derivative=((g@vectors.T)*gates[:,fids]).reshape(nv,nc,nf))
            pools[f'random_static_{repeat}']=dict(mode='static',fids=[],vectors=vectors,derivative=(g@vectors.T).reshape(nv,nc,nf))
        configs=[];controllers={};fitdetails=[]
        for kind,pool in pools.items():
            dd=pool['derivative'];den=targets.abs()[:,None]*math.sqrt(nv*ne)
            x=(torch.einsum('ec,vcf->evf',contrast,dd)/den[:,:,None]).reshape(-1,nf)
            y=((targets[:,None]-contrast@base.T)/den).reshape(-1)
            xn=((dd-dd.mean(0,keepdim=True))/spans[None,:,None]).reshape(-1,nf)/math.sqrt(nv*nc)
            xx=torch.cat([x,math.sqrt(d['fit_penalty'])*xn,d['ridge']*torch.eye(nf)]).double();yy=torch.cat([y,torch.zeros(nv*nc+nf)]).double()
            weights=torch.linalg.lstsq(xx,yy).solution.float();vectors=pool['vectors'];fids=pool['fids']
            perturb=weights@vectors if pool['mode']=='static' else (gates[:,fids]*weights)@vectors
            rms=float(perturb.norm()) if pool['mode']=='static' else float(perturb.square().sum(1).mean().sqrt());assert rms>1e-8
            controllers[kind]=dict(mode=pool['mode'],fids=fids,vectors=vectors,weights=weights,rms=rms)
            fitdetails.append(dict(kind=kind,mode=pool['mode'],fids=fids,weights=weights.tolist(),rms=rms))
            for dose in d['doses']:configs.append(dict(id=f'{kind}_d{dose}',kind=kind,dose=dose))
        save(out/'feature_details.json',details);save(out/'fit_details.json',fitdetails);save(out/'candidates.json',configs)
        torch.save(dict(controllers=controllers,center=center,scale=scale),out/'controllers.pt')
        torch.save(dict(keys=keys,h=h,scaled_gradients=g,context_features=z),out/'fit_states.pt')
        np.savez_compressed(out/'controllers.npz',**{k+'__vectors':v['vectors'].numpy() for k,v in controllers.items()},**{k+'__weights':v['weights'].numpy() for k,v in controllers.items()})
        def delta_for(c,o,cfg):
            ctrl=controllers[cfg['kind']];key=(c['id'],o)
            if ctrl['mode']=='static':direction=ctrl['weights']@ctrl['vectors']/ctrl['rms']
            else:
                fids=ctrl['fids'];gate=((states[key]['context_code'][fids]-center[fids])/scale[fids]).clamp(-2,2)
                direction=(gate*ctrl['weights'])@ctrl['vectors']/ctrl['rms']
            direction*=min(1.,2./max(float(direction.norm()),1e-12))
            return {48:direction*(cfg['dose']*baseline[key]['h_norms']['48'])}
        ctx=multiprocessing.get_context('spawn');gpu_queue=ctx.Queue()
        for gpu in d['worker_gpus']:gpu_queue.put(gpu)
        executor=ProcessPoolExecutor(max_workers=len(d['worker_gpus']),mp_context=ctx,initializer=initialize_worker,initargs=(gpu_queue,d))
        cache={};tuning=[];winners={}
        def reference(cc,orders):return [baseline[(c['id'],o)] for o in orders for c in cc]
        with (out/'trials.jsonl').open('w') as tf:
            def evaluate(cfg,cc,orders,phase):
                jobs={}
                for o in orders:
                    for c in cc:
                        key=(cfg['id'],c['id'],o)
                        if key not in cache:jobs[executor.submit(worker_read,c,o,delta_for(c,o,cfg))]=key
                for future in as_completed(jobs):
                    key=jobs[future];r=future.result();r.update(configuration=cfg,phase=phase);cache[key]=r;emit(tf,r);engine.count+=1
                return [cache[(cfg['id'],c['id'],o)] for o in orders for c in cc]
            for i,cfg in enumerate(configs):
                rr=evaluate(cfg,fitcases,d['fit_orders'],'fit_screen');met=score(rr,fiteffects,d['fit_orders'],d['probability_quality'])
                tuning.append(dict(configuration=cfg,fit=met));save(out/'tuning.json',tuning)
                print(json.dumps(dict(stage='fit_screen',done=i+1,total=len(configs),configuration=cfg['id'],gap=met['mean_gap'],quality=met['quality'],seconds=round(time.monotonic()-start,1))),flush=True)
            collect(fitcases,d['selection_orders'])
            for kind in d['families']:
                pool=sorted([e for e in tuning if e['configuration']['kind']==kind],key=lambda e:(not e['fit']['quality'],e['fit']['mean_gap']))[:d['shortlist_per_family']]
                for entry in pool:
                    rr=evaluate(entry['configuration'],fitcases,d['selection_orders'],'calibration_selection');entry['selection']=score(rr,fiteffects,d['selection_orders'],d['probability_quality'])
                eligible=[e for e in pool if e['fit']['quality'] and e['selection']['quality']]
                if eligible:winners[kind]=min(eligible,key=lambda e:e['selection']['mean_gap'])
                print(json.dumps(dict(stage='family_selected',kind=kind,configuration=winners[kind]['configuration']['id'] if kind in winners else None,seconds=round(time.monotonic()-start,1))),flush=True)
            save(out/'tuning.json',tuning);save(out/'frozen_winners.json',winners)
            base_selection=score(reference(fitcases,d['selection_orders']),fiteffects,d['selection_orders'],d['probability_quality'])
            sae=[e for k,e in winners.items() if k.startswith('sae_')]
            primary=min(sae,key=lambda e:e['selection']['mean_gap']) if sae else None
            if primary and primary['selection']['mean_gap']>=base_selection['mean_gap']:primary=None
            save(out/'primary_frozen.json',dict(configuration=primary['configuration'] if primary else None,base_selection_gap=base_selection['mean_gap'],
                selected_calibration_gap=primary['selection']['mean_gap'] if primary else base_selection['mean_gap'],selection_effect_ids=d['fit_effect_ids'],
                noncalibrated_effects_scored=False,criterion=d['selection']))
            frozen=sha(out/'primary_frozen.json');winners_sha=sha(out/'frozen_winners.json')
            print(json.dumps(dict(stage='frozen',primary=primary['configuration'] if primary else 'base',seconds=round(time.monotonic()-start,1))),flush=True)
            collect(cases,d['all_orders'])
            allrows={'base':reference(cases,d['all_orders'])}
            for kind,entry in winners.items():
                allrows[kind]=evaluate(entry['configuration'],cases,d['all_orders'],'whole_domain')
                print(json.dumps(dict(stage='domain_done',kind=kind,seconds=round(time.monotonic()-start,1))),flush=True)
            summary={}
            for split in ['all','calibration','not_calibrated']:
                ee=d['effects'] if split=='all' else [e for e in d['effects'] if e['split']==split]
                summary[split]={kind:compare(rr,allrows['base'],ee,d['all_orders'],d['probability_quality']) for kind,rr in allrows.items()}
            save(out/'analysis.json',dict(primary=load(out/'primary_frozen.json'),summary=summary))
            format_cases=[]
            for sid in sorted({c['study_id'] for c in cases}):
                cc=[c for c in cases if c['study_id']==sid];format_cases.extend([cc[0],cc[-1]])
            with (out/'format.jsonl').open('w') as ff:
                for kind in allrows:
                    cfg=winners[kind]['configuration'] if kind!='base' else None
                    for c in format_cases:
                        r=engine.generate(c,2,delta_for(c,2,cfg) if cfg else None);r['kind']=kind;emit(ff,r)
            assert frozen==sha(out/'primary_frozen.json') and winners_sha==sha(out/'frozen_winners.json')
    executor.shutdown(wait=True)
    save(out/'context_audit.json',context_audit)
    save(out/'complete.json',dict(status='complete',forward_reads=engine.count,gradient_reads=len(keys),baseline_reads=len(baseline),intervention_reads=len(cache),
        prefix_context_reads=len(context_codes),worker_gpus=d['worker_gpus'],seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),primary_sha256=frozen,protocol_sha256=sha(W/'sae_domain_v12_protocol.json')))
    print(json.dumps(dict(stage='complete',summary={s:{k:{f:v[f] for f in ['mean_gap','relative_improvement','quality']} for k,v in mm.items()} for s,mm in summary.items()},seconds=round(time.monotonic()-start,1))),flush=True)

if __name__=='__main__':main()
