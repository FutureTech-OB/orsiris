"""Passive, fixed-input J-Lens readout. No questions, steering, or generation."""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText
from safetensors import safe_open

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    for k in ['protocol','model','size','output']: p.add_argument('--'+k,required=True)
    p.add_argument('--lens');p.add_argument('--directions');p.add_argument('--extract-only',action='store_true')
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.protocol).read_text(encoding='utf-8')); spec=cfg['models'][a.size]
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if a.extract_only:
        assert sha(a.lens)==spec['lens_sha256']
        accepted={};excluded={}
        for w in cfg['emotion_words']+cfg['appraisal_words']:
            variants=[(s,tok.encode(s,add_special_tokens=False)) for s in [w,' '+w]]
            good=[(s,ids) for s,ids in variants if len(ids)==1]
            if good:accepted[w]={'surface':good[0][0],'id':good[0][1][0]}
            else:excluded[w]=variants
        index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map']
        with safe_open(Path(a.model)/index['lm_head.weight'],framework='pt',device='cpu') as f:
            head=f.get_slice('lm_head.weight')
            rows=torch.stack([head[v['id']:v['id']+1].squeeze(0).to(torch.bfloat16).float() for v in accepted.values()])
        lens=torch.load(a.lens,map_location='cpu',weights_only=True)
        vectors={l:rows@lens['J'][l].float() for l in spec['layers']}
        meta={'words':list(accepted),'tokens':accepted,'excluded':excluded,'lens_sha256':sha(a.lens),
              'model_config_sha256':sha(Path(a.model)/'config.json'),'protocol_sha256':sha(a.protocol),
              'lens_model_id':spec['lens_model_id'],'formula':'float32(BF16(W_U[token])) @ float32(J[layer])',
              'layer_index':'zero-based post-block residual; J[layer], matching existing interventions'}
        torch.save({'vectors':vectors,'metadata':meta},out/'directions.pt')
        dump(out/'directions.json',{**meta,'directions_sha256':sha(out/'directions.pt')})
        print('extracted',len(accepted),'excluded',list(excluded),flush=True);return
    asset=torch.load(a.directions,map_location='cpu',weights_only=True)
    assert asset['metadata']['protocol_sha256']==sha(a.protocol)
    assert asset['metadata']['model_config_sha256']==sha(Path(a.model)/'config.json')
    vectors=asset['vectors'];words=asset['metadata']['words']
    print('loading',a.size,flush=True)
    model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda',
              attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval()
    blocks=model.model.language_model.layers
    assert len(blocks)==spec['num_layers']
    records=[];activations={}
    for name,case in cfg['scenarios'].items():
        prompt=case['prompt']
        text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                    add_generation_prompt=True,enable_thinking=False)
        enc=tok(text,return_tensors='pt',return_offsets_mapping=True)
        offsets=enc.pop('offset_mapping')[0].tolist()
        positions={'decision_end':enc['input_ids'].shape[1]-1}
        inputs={k:v.to(model.device) for k,v in enc.items()};captured={};handles=[]
        def hook(l):
            def f(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                captured[l]=h[0].detach().float().cpu().clone()
            return f
        for l in spec['layers']:handles.append(blocks[l].register_forward_hook(hook(l)))
        with torch.inference_mode(): logits=model(**inputs,use_cache=False).logits[0,-1].float().cpu()
        for h in handles:h.remove()
        with torch.inference_mode(): plain=model(**inputs,use_cache=False).logits[0,-1].float().cpu()
        delta=float((logits-plain).abs().max());assert delta==0,delta
        lp=logits.log_softmax(-1)
        row={'model':a.size,'scenario':name,'prompt':prompt,'formatted_prompt':text,
             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'token_ids':enc['input_ids'][0].tolist(),
             'positions':positions,'position_tokens':{k:tok.decode([enc['input_ids'][0,i].item()]) for k,i in positions.items()},
             'passive_max_logit_difference':delta,'readouts':{}}
        for l,h in captured.items():
            v=vectors[l].float();nv=torch.nn.functional.normalize(v,dim=-1)
            row['readouts'][str(l)]={}
            for pos,i in positions.items():
                z=h[i];projection=nv@z;cos=projection/z.norm()
                row['readouts'][str(l)][pos]={w:{'cosine':float(cos[j]),'unit_direction_projection':float(projection[j]),
                                                              'direction_norm':float(v[j].norm()),'activation_norm':float(z.norm())} for j,w in enumerate(words)}
        row['first_token_logp']={choice:{'token_ids':tok.encode(choice,add_special_tokens=False),
          'logp':float(lp[tok.encode(choice,add_special_tokens=False)[0]])} for choice in case['choices']}
        records.append(row);activations[name]=captured
        dump(out/'readouts.json',records)
        print('read',name,'passive logit difference',delta,flush=True)
    torch.save(activations,out/'activations.pt')
    dump(out/'metadata.json',{'created_unix':time.time(),'model':a.size,'python':platform.python_version(),
      'torch':str(torch.__version__),'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
      'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'dtype':'bfloat16','attention':'sdpa','thinking':False,
      'protocol_sha256':sha(a.protocol),'script_sha256':sha(__file__),'directions_sha256':sha(a.directions),
      'readouts_sha256':sha(out/'readouts.json'),'activations_sha256':sha(out/'activations.pt'),
      'directions_metadata':asset['metadata']})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
