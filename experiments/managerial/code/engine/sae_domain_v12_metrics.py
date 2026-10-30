"""Dependency-free effect scoring for the personnel-judgment domain pilot."""
import collections,statistics

def score(rows,effects,orders,quality_threshold=.9):
    idx={(r['case_id'],r['rotation']):r for r in rows}
    details=[]
    for e in effects:
        for order in orders:
            delta=sum(w*idx[(cid,order)]['mean'] for cid,w in e['coefficients'].items())
            err=abs(delta-e['human_delta'])
            details.append(dict(effect_id=e['effect_id'],study_id=e['study_id'],split=e['split'],order=order,
                human_delta=e['human_delta'],model_delta=delta,gap=err/abs(e['human_delta']),raw_error=err,scale_error=err/e['span']))
    bystudy=collections.defaultdict(list)
    for r in details:bystudy[r['study_id']].append(r['gap'])
    needed={cid for e in effects for cid in e['coefficients']}
    selected=[r for r in rows if r['case_id'] in needed and r['rotation'] in orders]
    expected=len(needed)*len(orders);assert len(selected)==expected
    qual=sum(r['top_legal'] and r['label_mass']>=quality_threshold for r in selected)
    noise=statistics.mean(statistics.pstdev(idx[(cid,o)]['mean'] for o in orders) for cid in needed) if len(orders)>1 else None
    return dict(mean_gap=statistics.mean(r['gap'] for r in details),scale_mae=statistics.mean(r['scale_error'] for r in details),
        study_balanced_gap=statistics.mean(statistics.mean(v) for v in bystudy.values()),
        study_gaps={k:statistics.mean(v) for k,v in bystudy.items()},order_sd=noise,
        quality=qual==expected,quality_pass=qual,rows=expected,min_label_mass=min(r['label_mass'] for r in selected),details=details)

def compare(rows,base,effects,orders,threshold=.9):
    a=score(rows,effects,orders,threshold);b=score(base,effects,orders,threshold)
    a.update(baseline_mean_gap=b['mean_gap'],relative_improvement=1-a['mean_gap']/b['mean_gap'],
        baseline_scale_mae=b['scale_mae'],scale_mae_improvement=1-a['scale_mae']/b['scale_mae'],
        baseline_order_sd=b['order_sd'],order_sd_change=a['order_sd']-b['order_sd'] if a['order_sd'] is not None else None)
    return a
