"""Rank decision-time J-Lens readouts for two models and two fixed scenarios."""
from pathlib import Path
import argparse,csv,json,math
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.data);cfg=json.loads(Path(a.protocol).read_text(encoding='utf-8'))
rows=[];scenarios={}
for size in ['9b','27b']:
    records=json.loads((root/size/'readouts.json').read_text(encoding='utf-8'))
    assert len(records)==2 and {r['scenario'] for r in records}=={'ultimatum','reporting'}
    for r in records:
        pos=r['positions']['decision_end']
        assert pos==len(r['token_ids'])-1 and r['passive_max_logit_difference']==0
        scores={}
        for word in cfg['emotion_words']+cfg['appraisal_words']:
            vals=[r['readouts'][str(layer)]['decision_end'][word]['cosine'] for layer in cfg['models'][size]['primary_layers']]
            assert all(math.isfinite(v) for v in vals)
            scores[word]=sum(vals)/len(vals)
        ranks={}
        for cat in ['emotion','appraisal']:
            ranks[cat]=sorted(cfg[cat+'_words'],key=lambda w:(-scores[w],w))
            for rank,w in enumerate(ranks[cat],1):rows.append(dict(model=size,scenario=r['scenario'],category=cat,rank=rank,word=w,mean_cosine=scores[w]))
        scenarios[size+'_'+r['scenario']]={'scores':scores,'ranks':ranks,'position':pos}
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps({'position':'decision_end','metric':'mean signed cosine across three specified layers','scenarios':scenarios},ensure_ascii=False,indent=2),encoding='utf-8')
with (out/'rankings.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'model_scenario_records':4,'ranked_entries':len(rows)}))
