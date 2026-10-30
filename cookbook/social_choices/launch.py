"""Launch independent GPU shards; never resume or overwrite earlier runs."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for key in ['model','directions','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-size',type=int,default=4)
    p.add_argument('--protocol',required=True)
    p.add_argument('--study-gpus',help='Optional study groups, e.g. 3,4:5,6,7')
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    gpus=a.gpus.split(',')
    if a.study_gpus:
        groups=[g.split(',') for g in a.study_gpus.split(':')]
        if len(groups)!=2 or any(not g or '' in g for g in groups):raise ValueError('Supply one nonempty GPU group per study')
        if len(set(sum(groups,[])))!=sum(map(len,groups)):raise ValueError('GPU groups must be disjoint')
    else:
        if len(gpus)%2 or len(gpus)<2:raise ValueError('Use an even number of GPUs, half per study')
        groups=[gpus[:len(gpus)//2],gpus[len(gpus)//2:]]
    jobs=[]
    files=[]
    for si,study in enumerate(['ultimatum','reporting']):
        shards=len(groups[si])
        for shard,gpu in enumerate(groups[si]):
            label=f'{study}_{shard}'
            command=[sys.executable,str(root/'run.py'),'--protocol',a.protocol,
                '--study',study,'--model',a.model,'--directions',a.directions,'--output',str(out/label),
                '--shard',str(shard),'--shards',str(shards)]
            if a.smoke:command.extend(['--smoke','--smoke-size',str(a.smoke_size)])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log=(out/(label+'.log')).open('x');files.append(log)
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((label,process))
            time.sleep(.25)
    (out/'pids.json').write_text(json.dumps({label:proc.pid for label,proc in jobs},indent=2))
    results={label:proc.wait() for label,proc in jobs}
    (out/'exit_codes.json').write_text(json.dumps(results,indent=2))
    for f in files:f.close()
    if any(results.values()):sys.exit(1)


if __name__=='__main__':main()
