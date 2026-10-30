"""CPU-only reanalysis of the selected four interventions; standard library."""
from pathlib import Path
from collections import Counter
from decimal import Decimal,localcontext
from fractions import Fraction
import argparse,csv,hashlib,json,math


def wilson(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def paired_probability(positive,negative):
    n=positive+negative;k=min(positive,negative)
    if not n or k==n/2:return Fraction(1)
    coefficient=1;total=0
    for i in range(k+1):
        total+=coefficient;coefficient=coefficient*(n-i)//(i+1)
    return min(Fraction(1),Fraction(2*total,2**n))


def scientific(p):
    with localcontext() as context:
        context.prec=14
        return f'{Decimal(p.numerator)/Decimal(p.denominator):.8E}'


def summarize(protocol,records):
    output={}
    expected=set()
    indexed={}
    for study,c in protocol['studies'].items():
        aliases={**{arm:arm for arm in c['conditions']},
                 **{source:arm for arm,source in c['source_conditions'].items()}}
        for r in records:
            if r['study']!=study:continue
            if r['condition'] not in aliases:raise ValueError(f'Unexpected condition: {r["condition"]}')
            key=(study,aliases[r['condition']],r['sample_id'])
            if key in indexed:raise ValueError(f'Duplicate sample: {key}')
            if r['response'] not in c['backend_config']['choices']+['Other']:raise ValueError('Invalid response label')
            indexed[key]=r
        n=protocol['n_per_arm'];target=c['target_choice']
        expected.update((study,arm,i) for arm in c['conditions'] for i in range(n))
        if any((study,arm,i) not in indexed for arm in c['conditions'] for i in range(n)):
            raise ValueError(f'Missing planned samples: {study}')
        baseline={i:indexed[study,'baseline',i] for i in range(n)}
        arms={}
        for arm in c['conditions']:
            samples={i:indexed[study,arm,i] for i in range(n)}
            counts=Counter(r['response'] for r in samples.values());k=counts[target]
            row={'n':n,'counts':dict(counts),'target':target,'target_rate':k/n,
                 'target_rate_wilson_95':wilson(k,n),'other_rate':counts['Other']/n,
                 'token_limit_count':sum(r['reached_token_limit'] for r in samples.values())}
            if arm!='baseline':
                differences=[int(samples[i]['response']==target)-int(baseline[i]['response']==target) for i in range(n)]
                positive=differences.count(1);negative=differences.count(-1);delta=(positive-negative)/n
                se=math.sqrt(sum((x-delta)**2 for x in differences)/(n-1)/n)
                ci=[max(-1,delta-1.959963984540054*se),min(1,delta+1.959963984540054*se)]
                if positive+negative==0:
                    upper=1-math.pow(.05,1/n);ci=[-upper,upper]
                exact=paired_probability(positive,negative)
                factor=8*c['original_search_family_sizes'][arm]
                row.update(difference=delta,difference_paired_95=ci,
                           paired_increase=positive,paired_decrease=negative,
                           exact_p_scientific=scientific(exact),
                           conservative_correction_factor=factor,
                           conservative_p_scientific=scientific(min(Fraction(1),factor*exact)),
                           operation=c['backend_config']['operations'][arm])
            arms[arm]=row
        output[study]=arms
    if set(indexed)!=expected or len(indexed)!=len(records):raise ValueError('Unexpected study/sample IDs')
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--configs',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result={'models':{},'sources':[],'interpretation':'Selected exploratory contrasts, not a new independent confirmation. Intervals quantify generation sampling conditional on fixed prompts/models. 9B and 27B anger doses differ; no isolated model-size or equivalence claim. Bonferroni factors are eight displayed contrasts times each original search-family size; not the original Holm p values.'}
    for model in ['9b','27b']:
        config=Path(args.configs)/f'{model}.json'
        protocol=json.loads(config.read_text(encoding='utf-8'));records=[]
        for study in protocol['studies']:
            merged=Path(args.data)/model/f'{study}.jsonl'
            files=[merged] if merged.is_file() else sorted((Path(args.data)/model).glob(f'{study}_*/records.jsonl'))
            if not files:raise FileNotFoundError(f'No complete data for {model}/{study}')
            for file in files:
                if file!=merged:
                    state=json.loads(file.with_name('status.json').read_text(encoding='utf-8'))
                    if state['state']!='complete' or state['smoke']:raise ValueError(f'Incomplete/smoke run: {file}')
                records.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip())
                result['sources'].append({'path':file.relative_to(Path(args.data)).as_posix(),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        result['models'][model]=summarize(protocol,records)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table=[]
    for study in ['ultimatum','reporting']:
        for arm in ['unfair','fair'] if study=='ultimatum' else ['calm','anger']:
            row={'study':study,'intervention':arm}
            for model in ['9b','27b']:
                baseline=result['models'][model][study]['baseline'];r=result['models'][model][study][arm]
                row.update({f'{model}_baseline':baseline['target_rate'],f'{model}_rate':r['target_rate'],
                            f'{model}_difference':r['difference'],f'{model}_ci95':r['difference_paired_95'],
                            f'{model}_other_rate':r['other_rate'],f'{model}_strength':r['operation']['strength'],
                            f'{model}_corrected_p':r['conservative_p_scientific']})
            table.append(row)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
