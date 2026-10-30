"""Collect final primary conditions from new runs without changing raw lines."""
from pathlib import Path
import argparse,json,shutil


def main():
    p=argparse.ArgumentParser()
    for name in ['bundle','pay-run','voice-run','trust-run','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--ceo-run',help='Optional new CEO source calibration run; does not reuse historical source readouts')
    a=p.parse_args();bundle=Path(a.bundle);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'references').mkdir();(out/'data').mkdir()
    sources=[('HRM_1_1',Path(a.pay_run)),('AMJ_1_3',Path(a.voice_run))]
    if a.ceo_run:sources.append(('JAP_4_2b',Path(a.ceo_run)))
    for sid,run in sources:
        directory=run/'results'/sid
        if json.loads((directory/'complete.json').read_text())['status']!='complete':raise ValueError('Incomplete calibration run')
        primary=json.loads((directory/'primary_frozen.json').read_text())['configuration']
        if primary is None:raise ValueError('No improved primary controller was selected in this new calibration')
        target=out/'data'/sid;target.mkdir()
        shutil.copy2(directory/'baseline.jsonl',target/'prompt.jsonl')
        lines=[line for line in (directory/'trials.jsonl').read_bytes().splitlines(keepends=True) if json.loads(line)['configuration']['id']==primary['id']]
        (target/'sae.jsonl').write_bytes(b''.join(lines))
        ref=json.loads((bundle/f'references/{sid}.json').read_text(encoding='utf-8'));ref['configuration']=primary
        (out/f'references/{sid}.json').write_text(json.dumps(ref,ensure_ascii=False,indent=2),encoding='utf-8')
    sid='JAP_4_1_new';target=out/'data'/sid;target.mkdir()
    methods={'prompt':[],'sae':[]}
    for file in sorted((Path(a.trust_run)/'results').glob('*/readouts.jsonl')):
        done=json.loads(file.with_name('complete.json').read_text())
        if done['status']!='complete':raise ValueError('Incomplete transfer shard')
        for line in file.read_bytes().splitlines(keepends=True):
            r=json.loads(line)
            if r['study_id']==sid and r['persona_id']=='none':methods[r['method']].append(line)
    if not all(methods.values()):raise ValueError('No transfer records')
    for method,lines in methods.items():(target/f'{method}.jsonl').write_bytes(b''.join(lines))
    shutil.copy2(bundle/f'references/{sid}.json',out/f'references/{sid}.json')
    print(str(out.resolve()))


if __name__=='__main__':main()
