"""Pinned HarmBench behavior classifier; only used after final conditions freeze."""
from __future__ import annotations
import ast
import hashlib
import json
import time
import urllib.request

import torch

from .common import append_unique, atomic_json, digest, rows_by_id
from .pipeline import check_frozen


def parse_official_template(source):
    # Extract a literal string only. Never import or execute downloaded code.
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LLAMA2_CLS_PROMPT' for t in node.targets):
            value=ast.literal_eval(node.value)
            template=value['prompt']
            if not isinstance(template,str) or '{behavior}' not in template or '{generation}' not in template:
                raise ValueError('Invalid official classifier template')
            return template
    raise ValueError('Official classifier template was not found')


def template_for_run(c,run):
    path=run/'harmbench_classifier_template.json'
    url=f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/{c["data"]["harmbench_revision"]}/eval_utils.py'
    if path.exists():
        value=json.loads(path.read_text())
        if value['source']!=url or value['template_hash']!=digest(value['template']):
            raise ValueError('Classifier template provenance changed')
        return value['template']
    with urllib.request.urlopen(url,timeout=60) as response:
        raw=response.read()
    template=parse_official_template(raw.decode())
    atomic_json(path,dict(source=url,source_sha256=hashlib.sha256(raw).hexdigest(),
                         template=template,template_hash=digest(template)))
    return template


class OfficialJudge:
    def __init__(self,c,run):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.template=template_for_run(c,run)
        spec=c['official_judge']
        self.tokenizer=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'])
        # Unquantized FP16 weights, CPU offload if necessary on a 24GB card.
        # Quantization would be a separate evaluator protocol, not implicit.
        self.model=AutoModelForCausalLM.from_pretrained(spec['name'],revision=spec['revision'],
            dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True,
            max_memory={0:f'{spec["gpu_memory_gib"]}GiB','cpu':f'{spec["cpu_memory_gib"]}GiB'}).eval()

    def score(self,behavior,response):
        text=self.template.format(behavior=behavior,generation=response)
        inputs=self.tokenizer(text,return_tensors='pt')
        if inputs.input_ids.shape[-1]>4096:
            raise ValueError('Official classifier input exceeds 4096; do not silently truncate')
        device=self.model.get_input_embeddings().weight.device
        inputs=inputs.to(device)
        start=time.monotonic()
        with torch.inference_mode():
            generated=self.model.generate(**inputs,do_sample=False,max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        label=self.tokenizer.decode(generated[0,inputs.input_ids.shape[-1]:],skip_special_tokens=True).strip().lower()
        if label not in ('yes','no'):
            raise ValueError(f'Malformed official behavior label: {label!r}')
        return dict(behavior_success=label=='yes',raw=label,seconds=time.monotonic()-start,
                    input_tokens=int(inputs.input_ids.shape[-1]))

    def close(self):
        import gc
        del self.model,self.tokenizer
        gc.collect();torch.cuda.empty_cache()


def judge_harmbench(c,run,budget):
    check_frozen(run)
    if not c.get('official_judge',{}).get('enabled'):
        return
    generations=rows_by_id(run/'test_generations.jsonl')
    wanted=[r for r in generations.values() if r['corpus'].startswith('harmbench_')]
    path=run/'harmbench_judgements.jsonl'
    known=rows_by_id(path)
    if all(r['id'] in known for r in wanted):
        return
    judge=OfficialJudge(c,run)
    identity=digest(dict(config=c['official_judge'],template=judge.template))
    try:
        for r in wanted:
            if r['id'] in known:
                if known[r['id']]['judge_identity']!=identity:
                    raise ValueError('Official classifier identity changed')
                continue
            budget.check()
            append_unique(path,dict(id=r['id'],judge_identity=identity,
                **judge.score(r['behavior'],r['response'])),known)
    finally:
        judge.close()
