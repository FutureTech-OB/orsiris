from pathlib import Path
import hashlib,json
from src.model_service import ModelService
ROOT=Path(__file__).resolve().parent

def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def make_prompt(c,mode,rotation):
    values=c['values'][rotation:]+c['values'][:rotation]
    text=('Simulate one human participant in the following organizational experiment. '
          'Report the response this participant would give, not the response of an ideal decision maker.\n\n'+c['scenario'])
    if mode=='human':text+='\n\n'+c['human_hint']
    if mode=='synthetic':text+='\n\n'+c['synthetic_hint']
    text+='\n\nCurrent survey item: '+c['question']+'\n'
    text+='\n'.join(f'{chr(65+i)}: response {v}' for i,v in enumerate(values))
    text+='\nChoose exactly one response. Output only its letter, without explanation.\nAnswer:'
    return text,values
