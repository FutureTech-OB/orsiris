"""Reconstruct executive/manager treatment effects from saved probabilities."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics

STUDIES=['HRM_1_1','AMJ_1_3','JAP_4_1_new']


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(reference,methods):
    sid=reference['study_id'];orders=reference['orders']
    aggregation=reference.get('aggregation','pooled_orders')
    if aggregation not in {'per_order','pooled_orders'}:raise ValueError('Unknown aggregation')
    cases={c.get('source_case_id',c['id']):c for c in reference['cases']}
    indices={};metrics={};details=[]
    for method,rows in methods.items():
        index={}
        for r in rows:
            key=(r.get('source_case_id',r['case_id']),r['rotation'])
            if key in index:raise ValueError(f'Duplicate row: {sid}/{method}/{key}')
            assert r['study_id']==sid
            probs=r['probabilities'];values=r['values']
            assert len(probs)==len(values) and all(math.isfinite(x) and 0<=x<=1 for x in probs)
            assert abs(sum(probs)-1)<1e-6
            expected=sum(v*p for v,p in zip(values,probs))
            assert abs(expected-r['mean'])<1e-8
            index[key]={**r,'reconstructed_mean':expected}
        needed={(cid,order) for cid in cases for order in orders}
        if set(index)!=needed:raise ValueError(f'Missing/extra rows: {sid}/{method}')
        indices[method]=index
    for key,r in indices['sae'].items():
        b=indices['prompt'][key]
        assert r['prompt_sha256']==b['prompt_sha256'] and r['values']==b['values']
    for method,index in indices.items():
        effect_details=[]
        for effect in reference['effects']:
            by_order=[]
            for order in orders:
                if 'coefficients' not in effect:
                    sides=[]
                    for side in [1,2]:
                        means=[];weights=[]
                        for group in effect['groups'+str(side)]:
                            group_cases=[c for c in cases.values() if c['group']==group and c['item'] in effect['items']]
                            values=[c.get('score_offset',0)+c.get('score_multiplier',1)*index[c['id'],order]['reconstructed_mean'] for c in group_cases]
                            means.append(statistics.mean(values));weights.append(group_cases[0]['group_n'])
                        sides.append(sum(v*w for v,w in zip(means,weights))/sum(weights))
                    delta=sides[0]-sides[1]
                else:
                    delta=sum(weight*index[cid,order]['reconstructed_mean'] for cid,weight in effect['coefficients'].items())
                by_order.append(delta)
            human=effect['human_delta'];span=effect['span']
            if aggregation=='per_order':
                raw_error=statistics.mean(abs(v-human) for v in by_order)
            else:
                raw_error=abs(statistics.mean(by_order)-human)
            row={'study_id':sid,'method':method,'effect_id':effect['effect_id'],
                 'human_delta':human,'span':span,'order_deltas':by_order,
                 'mean_model_delta':statistics.mean(by_order),'raw_error':raw_error,
                 'scale_error':raw_error/span,
                 'per_order_raw_error':statistics.mean(abs(v-human) for v in by_order)}
            effect_details.append(row);details.append(row)
        quality_rows=list(index.values())
        metrics[method]={'rows':len(index),'effects':len(effect_details),
                         'raw_effect_mae':statistics.mean(r['raw_error'] for r in effect_details),
                         'scale_normalized_effect_mae':statistics.mean(r['scale_error'] for r in effect_details),
                         'mean_per_order_scale_error':statistics.mean(r['per_order_raw_error']/r['span'] for r in effect_details),
                         'quality_pass':sum(r['top_legal'] and r['label_mass']>=reference['probability_quality'] for r in quality_rows),
                         'min_label_mass':min(r['label_mass'] for r in quality_rows)}
    base=metrics['prompt']['scale_normalized_effect_mae'];sae=metrics['sae']['scale_normalized_effect_mae']
    return {'study_id':sid,'title':reference['title'],'evidence':reference['evidence'],'kind':reference['kind'],'aggregation':aggregation,
            'cases':len(cases),'orders':orders,'methods':metrics,'relative_error_reduction':1-sae/base if base>0 else None},details


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--references',required=True)
    p.add_argument('--output',required=True);p.add_argument('--check-original')
    p.add_argument('--studies',nargs='+',help='Study IDs; default includes CEO source when its reference is present')
    args=p.parse_args();studies=[];details=[];sources=[]
    selected=args.studies or (STUDIES+(['JAP_4_2b'] if (Path(args.references)/'JAP_4_2b.json').is_file() else []))
    for sid in selected:
        ref_path=Path(args.references)/f'{sid}.json';ref=load(ref_path);methods={}
        for method in ['prompt','sae']:
            file=Path(args.data)/sid/f'{method}.jsonl'
            methods[method]=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
            sources.append({'path':f'{sid}/{method}.jsonl','sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        result,rows=summarize(ref,methods);studies.append(result);details.extend(rows)
    if args.check_original:
        saved=load(args.check_original)
        for s in studies:
            for method in ['prompt','sae']:
                if s['kind']=='calibration':
                    expected=next(r for r in saved['calibration']['studies'] if r['study_id']==s['study_id'])['methods']['base' if method=='prompt' else 'sae_context']['scale_error']
                else:
                    expected=saved['transfer']['studies'][s['study_id']]['summary'][method]['mean_scale_normalized_error']
                assert abs(s['methods'][method]['scale_normalized_effect_mae']-expected)<1e-10
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    summary={'studies':studies,'source_records':sources,
             'interpretation':'Source calibration for pay, voice and CEO trust; frozen transfer for restaurant trust. Reconstructed from probabilities, not new model calls or new human samples. All four studies use pooled-order effects before absolute error; per-order sensitivity is retained as a diagnostic. Study reductions are reported separately.'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'effects.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        rows=[{'study_id':s['study_id'],'title':s['title'],'evidence_type':s['kind'],
               'prompt_error':s['methods']['prompt']['scale_normalized_effect_mae'],
               'sae_error':s['methods']['sae']['scale_normalized_effect_mae'],
               'relative_error_reduction':s['relative_error_reduction'],'effects':s['methods']['prompt']['effects']} for s in studies]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(studies,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
