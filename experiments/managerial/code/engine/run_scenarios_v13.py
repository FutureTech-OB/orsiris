"""One independently calibrated study per GPU pair; no model or SAE training."""
from pathlib import Path
import argparse, hashlib, json, math, multiprocessing, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from domain_engine import DomainEngine, encode
from run_sae_chat_v4 import load, save, sha, emit
from sae_domain_v12_metrics import score, compare

ROOT=Path(__file__).resolve().parent
WORKER=None
def init_worker(d,out):
    global WORKER
    os.environ['CUDA_VISIBLE_DEVICES']=str(d['worker_gpu'])
    torch.set_num_threads(2);torch.manual_seed(d['seed']);torch.backends.cuda.matmul.allow_tf32=False
    WORKER=DomainEngine(d)
    row,_=WORKER.read(d['cases'][0],0)
    save(Path(out)/'worker_probe.json',dict(gpu=d['worker_gpu'],pid=os.getpid(),row=row))
def worker_read(c,order,delta):
    row,_=WORKER.read(c,order,delta=delta)
    row['worker_gpu']=int(os.environ['CUDA_VISIBLE_DEVICES'])
    return row

def run(d):
    start=time.monotonic();out=ROOT/'results'/d['study_id'];out.mkdir(parents=True,exist_ok=False)
    save(out/'frozen_protocol.json',d)
    save(out/'metadata.json',dict(script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'protocol.json'),engine_sha256=sha(ROOT/'domain_engine.py'),model_service_sha256=sha(ROOT/'src/model_service.py')))
    os.environ['CUDA_VISIBLE_DEVICES']=str(d['main_gpu'])
    torch.set_num_threads(2);torch.manual_seed(d['seed']);torch.backends.cuda.matmul.allow_tf32=False
    def log(stage,**kw):print(json.dumps(dict(study=d['study_id'],stage=stage,seconds=round(time.monotonic()-start,1),**kw)),flush=True)
    ctx=multiprocessing.get_context('spawn')
    executor=ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=init_worker,initargs=(d,str(out)))
    # Eagerly load the companion model while the main model loads.
    startup=executor.submit(worker_read,d['cases'][0],0,None)
    engine=DomainEngine(d)
    for p in engine.model.model.parameters():p.requires_grad_(False)
    log('model_ready')
    blob=torch.load(Path(d['sae_root'])/'layer48.sae.pt',weights_only=True,map_location='cpu',mmap=True)
    cases=d['cases'];effects=d['effects'];baseline={};states={};prefix_codes={};prefix_hidden={};audit=[]
    try:
        with (out/'baseline.jsonl').open('w') as bf:
            for order in d['all_orders']:
                for c in cases:
                    grad=order in d['fit_orders'];r,h,g=engine.state(c,order,grad);key=(c['id'],order)
                    baseline[key]=r;emit(bf,r);ph=r['context_sha256']
                    if ph not in prefix_codes:
                        prefix_hidden[ph]=engine.prefix_context(c,order)
                        prefix_codes[ph]=encode(prefix_hidden[ph],blob).mean(0)
                    audit.append(dict(case_id=c['id'],order=order,prefix_sha256=ph,relative_full_vs_prefix=float((h['context']-prefix_hidden[ph]).norm()/prefix_hidden[ph].norm().clamp_min(1e-12))))
                    states[key]=dict(answer=h['answer'],gradient=g,context=prefix_codes[ph])
                log('baseline',order=order,rows=len(baseline))
        probe=startup.result();b=baseline[(cases[0]['id'],0)]
        assert probe['prompt_sha256']==b['prompt_sha256']
        assert max(abs(x-y) for x,y in zip(probe['probabilities'],b['probabilities']))<1e-5
        keys=[(c['id'],o) for o in d['fit_orders'] for c in cases]
        nv=len(d['fit_orders']);nc=len(cases);ne=len(effects);nf=d['feature_count']
        h=torch.stack([states[k]['answer'] for k in keys]);g=torch.stack([states[k]['gradient'] for k in keys])*h.norm(dim=1,keepdim=True)
        z=torch.stack([states[k]['context'] for k in keys]);center=z.mean(0);scale=z.std(0,unbiased=False).clamp_min(1e-6);gates=((z-center)/scale).clamp(-2,2)
        dec=blob['W_dec'];norms=dec.norm(dim=0).clamp_min(1e-8);raw=(g@dec)/norms
        contrast=torch.tensor([[e['coefficients'].get(c['id'],0.) for c in cases] for e in effects])
        assert int(torch.linalg.matrix_rank(contrast))==d['contrast_rank']
        base=torch.tensor([baseline[k]['mean'] for k in keys]).reshape(nv,nc)
        targets=torch.tensor([e['human_delta'] for e in effects]);spans=torch.tensor([max(c['values'])-min(c['values']) for c in cases])
        pools={};features=[]
        for mode in ['static','context']:
            deriv=(raw if mode=='static' else raw*gates).reshape(nv,nc,-1)
            cd=torch.einsum('ec,vcf->evf',contrast,deriv)
            noise=((deriv-deriv.mean(0,keepdim=True))/spans[None,:,None]).square().mean((0,1)).sqrt()
            ranking=((cd.mean(1).abs()-.5*cd.std(1,unbiased=False))/targets.abs()[:,None]).mean(0)-.1*noise
            if mode=='context':
                active=z.reshape(nv,nc,-1).ne(0).any(1).all(0)&(scale>1.000001e-6);ranking[~active]=-torch.inf
            chosen=[];vectors=[]
            for fid in ranking.argsort(descending=True).tolist():
                if not torch.isfinite(ranking[fid]):continue
                v=dec[:,fid]/norms[fid]
                if any(abs(float(v@q))>.9 for q in vectors):continue
                chosen.append(fid);vectors.append(v)
                if len(chosen)==nf:break
            assert len(chosen)==nf
            kind='sae_'+mode;pools[kind]=dict(mode=mode,fids=chosen,vectors=torch.stack(vectors),derivative=deriv[:,:,chosen])
            features.append(dict(kind=kind,fids=chosen,scores=ranking[chosen].tolist()))
        controllers={};configs=[];fitdetails=[]
        for kind,pool in pools.items():
            dd=pool['derivative'];den=targets.abs()[:,None]*math.sqrt(nv*ne)
            x=(torch.einsum('ec,vcf->evf',contrast,dd)/den[:,:,None]).reshape(-1,nf)
            y=((targets[:,None]-contrast@base.T)/den).reshape(-1)
            xn=((dd-dd.mean(0,keepdim=True))/spans[None,:,None]).reshape(-1,nf)/math.sqrt(nv*nc)
            for penalty in d['fit_penalties']:
                xx=torch.cat([x,math.sqrt(penalty)*xn,d['ridge']*torch.eye(nf)]).double();yy=torch.cat([y,torch.zeros(nv*nc+nf)]).double()
                w=torch.linalg.lstsq(xx,yy).solution.float();vs=pool['vectors'];fids=pool['fids']
                perturb=w@vs if pool['mode']=='static' else (gates[:,fids]*w)@vs
                rms=float(perturb.norm()) if pool['mode']=='static' else float(perturb.square().sum(1).mean().sqrt());assert rms>1e-8
                key=f'{kind}_p{penalty}';controllers[key]=dict(mode=pool['mode'],fids=fids,vectors=vs,weights=w,rms=rms)
                fitdetails.append(dict(key=key,kind=kind,mode=pool['mode'],fids=fids,weights=w.tolist(),rms=rms,penalty=penalty))
                for dose in d['doses']:configs.append(dict(id=f'{key}_d{dose}',kind=kind,controller=key,dose=dose))
        save(out/'features.json',features);save(out/'fit_details.json',fitdetails);save(out/'candidates.json',configs)
        torch.save(dict(controllers=controllers,center=center,scale=scale),out/'controllers.pt')
        torch.save(dict(keys=keys,answer=h,scaled_gradients=g,context_features=z),out/'fit_states.pt')
        np.savez_compressed(out/'controllers.npz',**{k+'__vectors':v['vectors'].numpy() for k,v in controllers.items()},**{k+'__weights':v['weights'].numpy() for k,v in controllers.items()})
        def delta(c,o,cfg):
            ctrl=controllers[cfg['controller']];key=(c['id'],o)
            if ctrl['mode']=='static':direction=ctrl['weights']@ctrl['vectors']/ctrl['rms']
            else:
                fids=ctrl['fids'];gate=((states[key]['context'][fids]-center[fids])/scale[fids]).clamp(-2,2)
                direction=(gate*ctrl['weights'])@ctrl['vectors']/ctrl['rms']
            direction*=min(1.,2./max(float(direction.norm()),1e-12))
            return {48:direction*(cfg['dose']*baseline[key]['h_norms']['48'])}
        cache={};tuning=[];winners={}
        def ref(orders):return [baseline[(c['id'],o)] for o in orders for c in cases]
        with (out/'trials.jsonl').open('w') as tf:
            def evaluate(cfg,orders,phase):
                todo=[(c,o) for o in orders for c in cases if (cfg['id'],c['id'],o) not in cache]
                futures={executor.submit(worker_read,c,o,delta(c,o,cfg)):(c,o) for c,o in todo[::2]}
                def accept(r,c,o):
                    b=baseline[(c['id'],o)];assert r['prompt_sha256']==b['prompt_sha256'] and r['values']==b['values']
                    r.update(configuration=cfg,phase=phase);cache[(cfg['id'],c['id'],o)]=r;emit(tf,r)
                for c,o in todo[1::2]:
                    r,_=engine.read(c,o,delta=delta(c,o,cfg));r['worker_gpu']=d['main_gpu'];accept(r,c,o)
                for f in as_completed(futures):
                    c,o=futures[f];accept(f.result(),c,o)
                return [cache[(cfg['id'],c['id'],o)] for o in orders for c in cases]
            for i,cfg in enumerate(configs):
                rr=evaluate(cfg,d['fit_orders'],'fit');met=score(rr,effects,d['fit_orders'],d['probability_quality'])
                tuning.append(dict(configuration=cfg,fit=met));save(out/'tuning.json',tuning)
                log('fit_screen',done=i+1,total=len(configs),config=cfg['id'],gap=met['mean_gap'],quality=met['quality'])
            for kind in d['families']:
                shortlist=sorted([e for e in tuning if e['configuration']['kind']==kind],key=lambda e:(not e['fit']['quality'],e['fit']['mean_gap']))[:d['shortlist_per_family']]
                for entry in shortlist:
                    rr=evaluate(entry['configuration'],d['selection_orders'],'selection');entry['selection']=score(rr,effects,d['selection_orders'],d['probability_quality'])
                eligible=[e for e in shortlist if e['fit']['quality'] and e['selection']['quality']]
                if eligible:winners[kind]=min(eligible,key=lambda e:e['selection']['mean_gap'])
            save(out/'tuning.json',tuning);save(out/'frozen_winners.json',winners)
            bs=score(ref(d['selection_orders']),effects,d['selection_orders'],d['probability_quality'])
            selected=min(winners.values(),key=lambda e:e['selection']['mean_gap']) if winners else None
            primary=selected if selected and selected['selection']['mean_gap']<bs['mean_gap'] else None
            save(out/'primary_frozen.json',dict(configuration=primary['configuration'] if primary else None,base_selection_gap=bs['mean_gap'],selected_gap=primary['selection']['mean_gap'] if primary else bs['mean_gap'],selection_effects=d['fit_effect_ids'],all_human_effects_used_for_calibration=True,random_controls=False))
            frozen=sha(out/'primary_frozen.json');whash=sha(out/'frozen_winners.json')
            rows={'base':ref(d['all_orders'])}
            for kind,entry in winners.items():rows[kind]=evaluate(entry['configuration'],d['all_orders'],'calibration_summary')
            summary={name:{kind:compare(rr,rows['base'],effects,orders,d['probability_quality']) for kind,rr in rows.items()} for name,orders in [('all_orders',d['all_orders']),('original',[0]),('selection',d['selection_orders'])]}
            save(out/'analysis.json',dict(primary=load(out/'primary_frozen.json'),summary=summary))
            with (out/'format.jsonl').open('w') as ff:
                for kind in rows:
                    cfg=winners[kind]['configuration'] if kind!='base' else None
                    for c in [cases[0],cases[-1]]:
                        r=engine.generate(c,0,delta(c,0,cfg) if cfg else None);r['kind']=kind;emit(ff,r)
            assert frozen==sha(out/'primary_frozen.json') and whash==sha(out/'frozen_winners.json')
        save(out/'context_audit.json',audit)
        save(out/'complete.json',dict(status='complete',seconds=time.monotonic()-start,baseline_reads=len(baseline),gradient_reads=len(keys),intervention_reads=len(cache),prefix_reads=len(prefix_codes),primary_sha256=frozen,gpus=[d['main_gpu'],d['worker_gpu']],gpu=torch.cuda.get_device_name(),torch=torch.__version__))
        log('complete',primary=primary['configuration'] if primary else 'base',scores={k:dict(gap=v['mean_gap'],improvement=v['relative_improvement']) for k,v in summary['all_orders'].items()})
    finally:executor.shutdown(wait=True,cancel_futures=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);args=parser.parse_args()
    run(next(d for d in load(ROOT/'protocol.json')['studies'] if d['study_id']==args.study))
