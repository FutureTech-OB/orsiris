"""Autoregressive sampling of four fixed social-choice interventions (9B/27B).

Uniforms are keyed by study/replicate/token, shared across arms. Inverse-CDF
sampling is performed before forcing the sampled token through HF generate.
This preserves per-replicate random numbers across batches and GPU shards.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from cookbook.social_choices.numerics import NumericIntervention
from repsteer import InterventionSession

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_response(text, choices):
    import re
    # Do not count an explanation or text mentioning both labels as a choice.
    cleaned = text.strip().strip('"\u201c\u201d\u2018\u2019\'').strip()
    match = re.fullmatch('(' + '|'.join(re.escape(x) for x in choices) + r')[.!]?', cleaned, re.I)
    if match:
        return next(c for c in choices if c.lower() == match.group(1).lower())
    return 'Other'


def uniforms(seed, study, ids, steps):
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed, study, int(i)])).random(steps)
                     for i in ids])


class RecordedSampler:
    def __init__(self, draws, prompt_length, temperature, top_k, top_p):
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        self.draws = torch.as_tensor(draws, dtype=torch.float64)
        self.prompt_length = prompt_length
        self.warpers = [TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k), TopPLogitsWarper(top_p)]
        self.trace = []

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        raw = scores.float()
        for warp in self.warpers:
            raw = warp(input_ids, raw)
        p = raw.softmax(-1)
        cdf = p.double().cumsum(-1)
        cdf[:, -1] = 1.0
        u = self.draws[:, step].to(scores.device).contiguous()
        selected = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1)).squeeze(-1)
        probability = p.gather(1, selected[:, None]).squeeze(-1)
        self.trace.append({'tokens': selected.cpu().tolist(), 'probabilities': probability.cpu().tolist()})
        sampled_scores = torch.full_like(scores, -float('inf'))
        return sampled_scores.scatter_(1, selected[:, None], 0.0)


class SamplingBackend:
    def __init__(self, config, model_path, directions_path):
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
            local_files_only=True, trust_remote_code=False).eval()
        self.layers = self.model.model.language_model.layers
        self.device = self.model.get_input_embeddings().weight.device
        head = self.model.get_output_embeddings().weight.detach()
        asset = torch.load(directions_path, map_location='cpu', weights_only=True)
        meta = asset['metadata']
        assert len(self.layers) == config['expected_layers'] == meta['expected_layers']
        assert head.shape[1] == config['expected_hidden_size'] == meta['expected_hidden_size']
        assert sha(Path(model_path) / 'config.json') == meta['model_config_sha256']
        for word, tid in meta['token_ids'].items():
            row = head[tid].cpu().contiguous()
            assert hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest() == meta['unembedding_row_sha256'][word]
            variants = [self.tokenizer.encode(s, add_special_tokens=False) for s in [word, ' ' + word]]
            ids = next(v for v in variants if len(v) == 1)
            assert ids == [tid], (word, ids, tid)
        self.vectors = asset['vectors']
        self.formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': config['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.inputs = self.tokenizer(self.formatted, return_tensors='pt').to(self.device)
        self.metadata = {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': importlib.metadata.version('transformers'), 'numpy': np.__version__,
            'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'directions_sha256': sha(directions_path), 'directions_metadata': meta,
            'prompt_sha256': hashlib.sha256(config['prompt'].encode()).hexdigest(),
            'formatted_prompt': self.formatted, 'prompt_token_ids': self.inputs['input_ids'][0].tolist(),
            'original_generation_defaults': self.model.generation_config.to_dict(),
            'attention_implementation': 'sdpa', 'tf32': False,
            'sampling': 'autoregessive inverse CDF on temperature/top_k/top_p distribution, common uniforms across arms',
            'note': 'HF greedy selection passes through the already sampled one-hot score; this is token-level stochastic generation, not greedy decoding or resampling stored outcomes.'}
        self.metadata['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p)
            for folder in [ROOT/'src/repsteer', ROOT/'cookbook/social_choices']
            for p in sorted(folder.glob('*.py'))}
        self.metadata['runner_note'] = 'Consolidated runner; original archived runs used earlier implementations. Arithmetic parity checked; metadata source hashes refer to the code actually executed.'

    def interventions(self, condition):
        if condition == 'baseline':
            return []
        c = self.config
        spec = c['operations'][condition]
        operations=[]
        for layer in c['layers']:
            direction=self.vectors[layer][spec['word']]
            if spec['mode']=='add':
                direction=direction/direction.norm()
            target=self.vectors[layer][spec['target_word']] if spec['mode']=='swap' else None
            operations.append(NumericIntervention(layer,spec['mode'],direction,
                strength=spec['strength'],target=target,position=spec['position'],name=condition))
        return operations

    def batch(self, ids, condition, draws, params, scope):
        from transformers import GenerationConfig, LogitsProcessorList
        sampler = RecordedSampler(draws, self.inputs['input_ids'].shape[1],
                                  params['temperature'], params['top_k'], params['top_p'])
        config = GenerationConfig(do_sample=False, max_new_tokens=params['max_new_tokens'],
            eos_token_id=self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True, repetition_penalty=1.0)
        inputs = {k: v.repeat(len(ids), 1) for k, v in self.inputs.items()}
        with torch.inference_mode(), InterventionSession(self.layers, self.interventions(condition), scope=scope) as session:
            outputs = self.model.generate(**inputs, generation_config=config,
                                          logits_processor=LogitsProcessorList([sampler]))
        completions = outputs[:, inputs['input_ids'].shape[1]:].cpu().tolist()
        rows = []
        eos = config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        for j, (sample_id, tokens) in enumerate(zip(ids, completions)):
            end = next((i + 1 for i, t in enumerate(tokens) if t in stops), len(tokens))
            tokens = tokens[:end]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            rows.append({'condition': condition, 'sample_id': int(sample_id), 'completion': text,
                         'response': parse_response(text, self.config['choices']),
                         'token_ids': tokens, 'sampled_token_probabilities': [t['probabilities'][j] for t in sampler.trace[:end]],
                         'ended_with_eos': bool(tokens and tokens[-1] in stops),
                         'reached_token_limit': end == params['max_new_tokens'] and tokens[-1] not in stops})
        return rows, session.records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True)
    p.add_argument('--study', choices=['ultimatum', 'reporting'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--directions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-size', type=int, default=4)
    a = p.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    study = protocol['studies'][a.study]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {'state': 'running', 'study': a.study, 'shard': a.shard, 'shards': a.shards, 'smoke': a.smoke}
    def save_status():
        (out / 'status.json').write_text(json.dumps(status, indent=2))
    save_status()
    try:
        backend = SamplingBackend(study['backend_config'], a.model, a.directions)
        (out / 'metadata.json').write_text(json.dumps({**backend.metadata, 'protocol_sha256': sha(a.protocol)}, indent=2))
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        ids = list(range(protocol['n_per_arm']))[a.shard::a.shards]
        if a.smoke:
            ids = ids[:a.smoke_size]
        completed = 0
        with (out / 'records.jsonl').open('x') as f, (out / 'interventions.jsonl').open('x') as g:
            for condition in study['conditions']:
                for offset in range(0, len(ids), protocol['batch_size']):
                    batch_ids = ids[offset:offset + protocol['batch_size']]
                    draws = uniforms(protocol['seed'], study['study_code'], batch_ids, protocol['sampling']['max_new_tokens'])
                    rows, interventions = backend.batch(batch_ids, condition, draws, protocol['sampling'], study['scope'])
                    for row in rows:
                        row.update(study=a.study, shard=a.shard)
                        f.write(json.dumps(row) + '\n')
                    f.flush()
                    g.write(json.dumps({'study': a.study, 'condition': condition, 'sample_ids': batch_ids,
                                        'records': interventions}) + '\n')
                    g.flush()
                    completed += len(rows)
                    status.update(rows=completed, condition=condition, elapsed_seconds=time.time()-started)
                    save_status()
                    print(json.dumps(status), flush=True)
        status.update(state='complete', elapsed_seconds=time.time()-started)
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_status()


if __name__ == '__main__':
    main()
