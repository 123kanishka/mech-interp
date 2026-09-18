#!/usr/bin/env python3
"""Detached two-host campaign with an auditable saved-pilot reuse option.

The coordinator owns a task-specific SSH key. The worker accepts only a small
stage queue; model work continues even when the laptop or an SSH session ends.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jlens_safety.common import atomic_json, append_unique, file_hash, rows_by_id
from jlens_safety.protocol_v3 import fitting_conditions
import yaml


def checked_merge(destination, source, stage):
    shared = ['manifest.json', 'prompts.jsonl', 'runtime_environment.json',
              'directions.npz', 'directions.json', 'layer_selection.json']
    for name in shared:
        if file_hash(destination/name) != file_hash(source/name):
            raise ValueError('Worker identity mismatch: '+name)
    if stage == 'fit-shard':
        c = json.loads((destination/'manifest.json').read_text())['config']
        layer = json.loads((destination/'layer_selection.json').read_text())['layer']
        expected = fitting_conditions(c, layer)
        results = []
        for index, folder in enumerate((destination, source)):
            items = json.loads((folder/f'learned_coefficients_shard_{index}.json').read_text())
            wanted = expected[index::2]
            if len(items) != len(wanted) or any(any(r.get(k) != v for k,v in e.items()) for r,e in zip(items,wanted)):
                raise ValueError('Coefficient shard condition coverage differs')
            results.extend(items)
        by_name = {r['name']:r for r in results}
        atomic_json(destination/'learned_coefficients.json', [by_name[r['name']] for r in expected])
    elif stage == 'validate-shard':
        if file_hash(destination/'learned_coefficients.json') != file_hash(source/'learned_coefficients.json'):
            raise ValueError('Worker coefficients differ')
        for name in ('validation_generations.jsonl', 'validation_judgements.jsonl'):
            known = rows_by_id(destination/name)
            for row in rows_by_id(source/name).values():
                append_unique(destination/name,row,known)
    else:
        raise ValueError('Unknown merge stage')


def wait_setup():
    deadline = time.monotonic()+3600
    while not (ROOT/'SETUP_COMPLETE').exists():
        status = subprocess.run(['supervisorctl','status','jlens_v3_setup'],capture_output=True,text=True).stdout
        if any(word in status for word in ('FATAL','EXITED','STOPPED','BACKOFF')):
            raise RuntimeError('Environment setup did not pass: '+status.strip())
        if time.monotonic()>deadline:
            raise TimeoutError('Environment setup exceeded one hour')
        time.sleep(60)


def prefetch(config):
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    for key in ('model','judge','official_judge'):
        c=config[key]
        if c.get('enabled',True):
            print('Downloading pinned artifacts:',c['name'],flush=True)
            files=HfApi().list_repo_files(c['name'],revision=c['revision'])
            weights='*.safetensors' if any(p.endswith('.safetensors') for p in files) else 'pytorch_model*.bin'
            snapshot_download(c['name'],revision=c['revision'],max_workers=4,
                allow_patterns=[weights,'*.json','*.model','*.txt','*.jinja'])
    c=config['jlens']
    hf_hub_download(c['repo'],c['filename'],revision=c['revision'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=['coordinator','worker','prefetch'],required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--worker-host')
    parser.add_argument('--worker-port',type=int)
    parser.add_argument('--worker-key',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/jlens_safety_learned.yaml')
    parser.add_argument('--pilot-source',type=Path)
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):
        parser.error('Run name must be a single directory name')
    config=yaml.safe_load(args.config.read_text())
    if args.role=='prefetch':
        prefetch(config);return
    wait_setup()
    run=ROOT/'results/safety_runs_v3'/args.run_name
    control=ROOT/'results/campaigns'/args.run_name
    control.mkdir(parents=True,exist_ok=True)
    deadline=time.time()+config['budget']['hard_hours']*3600

    def execute(stage,index=None):
        command=[sys.executable,str(ROOT/'experiments/run_jlens_safety_learned.py'),
                 '--config',str(args.config),'--run-dir',str(run),'--stage',stage,'--shard-count','2']
        if stage=='inherit-pilot': command+=['--pilot-source',str(args.pilot_source)]
        if index is not None: command+=['--shard-index',str(index)]
        subprocess.run(command,cwd=ROOT,check=True,timeout=max(1,deadline-time.time()))

    if args.role=='worker':
        previous=None
        while time.time()<deadline:
            task_file=control/'task.json'
            task=json.loads(task_file.read_text()) if task_file.exists() else None
            if not task or task['id']==previous:
                time.sleep(30);continue
            previous=task['id']
            deadline=min(deadline,task['deadline'])
            stage=task['stage']
            atomic_json(control/'worker_status.json',dict(id=previous,stage=stage,status='RUNNING'))
            try:
                if stage=='done':
                    atomic_json(control/'worker_status.json',dict(id=previous,stage=stage,status='COMPLETE'))
                    return
                if stage=='prefetch':
                    subprocess.run([sys.executable,__file__,'--role','prefetch','--run-name',args.run_name,
                                    '--config',str(args.config)],
                                   check=True,timeout=max(1,deadline-time.time()))
                elif stage in ('fit-shard','validate-shard','test-shard'):
                    execute(stage,1)
                else: raise ValueError('Unrecognized worker stage')
                atomic_json(control/'worker_status.json',dict(id=previous,stage=stage,status='COMPLETE'))
            except Exception as exc:
                atomic_json(control/'worker_status.json',dict(id=previous,stage=stage,status='FAILED',error=str(exc)))
                raise
        raise TimeoutError('Worker campaign wall-clock budget reached')

    if not all((args.worker_host,args.worker_port,args.worker_key)):
        parser.error('Coordinator requires worker host, port and task key')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',
         '-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
         '-o','UserKnownHostsFile='+str(args.worker_key)+'.known_hosts',
         '-i',str(args.worker_key),'-p',str(args.worker_port)]
    endpoint='root@'+args.worker_host
    def remote(command):
        return subprocess.run(ssh+[endpoint,command],check=True,capture_output=True,text=True,timeout=120).stdout
    def sync(source,destination,*extra):
        subprocess.run(['rsync','-a','--partial','--timeout=120','-e',shlex.join(ssh),
                        *extra,source,destination],check=True,timeout=max(1,deadline-time.time()))
    def enqueue(stage):
        task=dict(id=f'{stage}-{time.time_ns()}',stage=stage,deadline=deadline)
        atomic_json(control/'task.json',task)
        remote('mkdir -p '+shlex.quote(str(control)))
        sync(str(control/'task.json'),f'{endpoint}:{control}/task.next.json')
        remote(f'mv {shlex.quote(str(control/"task.next.json"))} {shlex.quote(str(control/"task.json"))}')
        return task['id']
    def wait_worker(task_id):
        while time.time()<deadline:
            text=remote('test ! -f '+shlex.quote(str(control/'worker_status.json'))+' || cat '+shlex.quote(str(control/'worker_status.json')))
            status=json.loads(text) if text.strip() else {}
            if status.get('id')==task_id:
                if status['status']=='COMPLETE': return
                if status['status']=='FAILED': raise RuntimeError('Worker failed: '+str(status))
            time.sleep(60)
        raise TimeoutError('Campaign wall-clock budget reached')
    def snapshot():
        remote('mkdir -p '+shlex.quote(str(run)))
        sync(str(run)+'/',f'{endpoint}:{run}/','--exclude=.lock','--exclude=coefficient_checkpoints','--exclude=activations')
    def status(stage,state='RUNNING',**extra):
        print(stage,state,flush=True)
        atomic_json(control/'campaign_status.json',dict(stage=stage,status=state,deadline=deadline,**extra))
    mirror=ROOT/'results/worker_shards'/args.run_name
    mirror.mkdir(parents=True,exist_ok=True)
    active='prepare'
    try:
        warm=enqueue('prefetch')
        status(active);execute(active)
        active='inherit-pilot' if args.pilot_source else 'preflight'
        status(active);execute(active)
        wait_worker(warm)
        active='screen';status(active);execute(active)
        for active in ('fit-shard','validate-shard','test-shard'):
            snapshot()
            status(active)
            task_id=enqueue(active)
            execute(active,0)
            wait_worker(task_id)
            sync(f'{endpoint}:{run}/',str(mirror)+'/','--exclude=.lock','--exclude=activations')
            if active=='test-shard':
                subprocess.run([sys.executable,str(ROOT/'scripts/merge_safety_shards.py'),
                    '--coordinator',str(run),'--worker',str(mirror)],check=True)
            else:
                checked_merge(run,mirror,active)
                if active=='validate-shard': execute('freeze')
        active='report';status(active);execute(active)
        snapshot()
        status(active,'COMPLETE',results_mirrored=True,external_backup_verified=False)
    except Exception as exc:
        status(active,'FAILED',error=str(exc))
        raise
    finally:
        try: enqueue('done')
        except Exception: pass


if __name__=='__main__': main()
