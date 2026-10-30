"""Portable preparation/launch of the two original sealed research workers."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(bundle,action,study,model,sae_root,gpus,output,controller=None):
    bundle=Path(bundle).resolve();model=Path(model).resolve();sae_root=Path(sae_root).resolve()
    if not (model/'config.json').is_file() or not (sae_root/'layer48.sae.pt').is_file():
        raise FileNotFoundError('Provide a local Qwen3.5-27B checkpoint and SAE directory containing layer48.sae.pt')
    cfg=load(model/'config.json')['text_config']
    if cfg['hidden_size']!=5120 or cfg['num_hidden_layers']!=64:raise ValueError('Expected Qwen3.5-27B dimensions')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for p in (bundle/'code/engine').rglob('*.py'):
        target=output/p.relative_to(bundle/'code/engine');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    if action=='calibrate':
        if len(gpus)!=2:raise ValueError('Original calibration worker needs exactly two distinct GPUs')
        d=copy.deepcopy(next(s for s in load(bundle/'protocols/calibration.json')['studies'] if s['study_id']==study))
        d.update(model=str(model),sae_root=str(sae_root),main_gpu=gpus[0],worker_gpu=gpus[1])
        protocol={'studies':[d]}
        commands=[[sys.executable,str(output/'run_scenarios_v13.py'),'--study',study]]
    else:
        d=copy.deepcopy(load(bundle/'protocols/transfer.json'));sid='JAP_4_1_new'
        s=copy.deepcopy(next(s for s in d['studies'] if s['study_id']==sid))
        s['controller_directory']=str(Path(controller).resolve() if controller else bundle/'assets/JAP_4_2b')
        prior=Path(s['controller_directory'])
        if not (prior/'controllers.pt').is_file() or not (prior/'primary_frozen.json').is_file():raise FileNotFoundError('Missing frozen source controller')
        d.update(model=str(model),sae_root=str(sae_root),gpus=gpus,studies=[s])
        d['cases']=[c for c in d['cases'] if c['study_id']==sid and c['persona_id']=='none']
        for i,c in enumerate(d['cases']):c['shard']=i%len(gpus)
        d['expected_reads_by_shard']=[sum(c['shard']==i for c in d['cases'])*len(s['orders'])*2 for i in range(len(gpus))]
        protocol=d
        commands=[[sys.executable,str(output/'run_transfer.py'),'--shard',str(i)] for i in range(len(gpus))]
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'portable_launch.json').write_text(json.dumps({'action':action,'commands':commands,
        'scope':'Original worker source retained; only asset paths, GPU assignment and selected study/cases relocated.'},indent=2),encoding='utf-8')
    return output,commands


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['plan','calibrate','transfer'])
    p.add_argument('--bundle',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--study',choices=['HRM_1_1','AMJ_1_3','JAP_4_2b'],default='HRM_1_1')
    p.add_argument('--model');p.add_argument('--sae-root');p.add_argument('--output')
    p.add_argument('--gpus',default='0,1');p.add_argument('--controller-dir');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();bundle=Path(a.bundle).resolve()
    if a.action=='plan':
        calibration=load(bundle/'protocols/calibration.json')
        print(json.dumps({'calibration_studies':[s['study_id'] for s in calibration['studies']],
            'primary_transfer':'JAP_4_1_new','source_controller':'JAP_4_2b',
            'required_assets':['Qwen3.5-27B BF16, hidden=5120, layers=64','layer48.sae.pt, width=81920, TopK=100'],
            'calibration_gpus':2,'transfer_gpus':'one or more; cases evenly sharded',
            'note':'Plan only; no model is loaded and no new experiment is run.'},ensure_ascii=False,indent=2));return
    if not all([a.model,a.sae_root,a.output]):p.error('Supply --model, --sae-root and --output')
    gpus=[int(x) for x in a.gpus.split(',')]
    if len(set(gpus))!=len(gpus):p.error('GPU indices must be distinct')
    output,commands=prepare(bundle,a.action,a.study,a.model,a.sae_root,gpus,a.output,a.controller_dir)
    if a.prepare_only:print(json.dumps(commands,indent=2));return
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    jobs=[];logs=[]
    for i,command in enumerate(commands):
        log=(output/f'worker_{i}.log').open('x');logs.append(log)
        jobs.append(subprocess.Popen(command,cwd=output,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
    codes=[job.wait() for job in jobs]
    (output/'exit_codes.json').write_text(json.dumps(codes))
    for log in logs:log.close()
    if any(codes):raise SystemExit(1)


if __name__=='__main__':main()
