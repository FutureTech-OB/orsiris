"""Extract only the layer/concept pairs required by a model's four recipes."""
from pathlib import Path
import argparse,hashlib,json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['config','model','lens','output']:p.add_argument('--'+name,required=True)
    args=p.parse_args()
    protocol=json.loads(Path(args.config).read_text(encoding='utf-8'))
    configs=[s['backend_config'] for s in protocol['studies'].values()]
    base=configs[0];model=Path(args.model);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    assert sha(args.lens)==base['lens_sha256']
    config=json.loads((model/'config.json').read_text())['text_config']
    assert config['hidden_size']==base['expected_hidden_size'] and config['num_hidden_layers']==base['expected_layers']
    pairs={}
    for c in configs:
        for op in c['operations'].values():
            for layer in c['layers']:
                pairs.setdefault(layer,set()).update([op['word']])
                if op.get('target_word'):pairs[layer].add(op['target_word'])
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    words=sorted(set().union(*pairs.values()))
    ids={word:next(v[0] for v in [tokenizer.encode(s,add_special_tokens=False) for s in [word,' '+word]] if len(v)==1) for word in words}
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model/index['lm_head.weight'],framework='pt',device='cpu') as f:
        head=f.get_slice('lm_head.weight')
        rows={w:head[i:i+1].squeeze(0).to(torch.bfloat16).contiguous() for w,i in ids.items()}
    lens=torch.load(args.lens,map_location='cpu',weights_only=True)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vectors={layer:{w:(rows[w].to(device).float()@lens['J'][layer].to(device=device,dtype=torch.float32)).cpu() for w in sorted(requested)} for layer,requested in pairs.items()}
    meta={'model_id':base['model_id'],'lens_model_id':base['lens_model_id'],'lens_sha256':sha(args.lens),
          'model_config_sha256':sha(model/'config.json'),'expected_hidden_size':base['expected_hidden_size'],
          'expected_layers':base['expected_layers'],'token_ids':ids,
          'unembedding_row_sha256':{w:hashlib.sha256(r.view(torch.uint8).numpy().tobytes()).hexdigest() for w,r in rows.items()},
          'extraction_device':device,'torch':str(torch.__version__),'formula':'float32(W_U[token]) @ float32(J[layer]); TF32 disabled'}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'vectors':vectors,'metadata':meta},output)
    output.with_suffix('.json').write_text(json.dumps({**meta,'directions_sha256':sha(output)},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
