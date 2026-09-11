"""Pinned benchmark inputs; text is data, never executable instructions."""
from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import urllib.request

from .common import atomic_json, digest, file_hash
from sae_jlens.run_io import append_jsonl


def fetch_csv(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))), hashlib.sha256(raw).hexdigest()


def normalized(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def grouped_splits(rows, counts, seed):
    """Assign entire focus groups with exact class counts and category balance.

    Binary integer allocation uses labels/categories only, NEVER model outputs.
    Absolute category-count deviations are minimized. An unused partition
    supports smaller test fixtures without splitting groups. Infeasibility is
    explicit: never fall back to a leaky prompt-level split.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    groups = sorted({r['group'] for r in rows})
    partitions = ['train', 'validation', 'test', 'unused']
    total = [sum(r['harmful'] == y for r in rows) for y in (0, 1)]
    wanted = {s: [counts[s], counts[s]] if isinstance(counts[s], int)
              else [counts[s]['safe'], counts[s]['unsafe']] for s in partitions[:3]}
    wanted['unused'] = [total[y] - sum(wanted[s][y] for s in partitions[:3]) for y in (0, 1)]
    if any(n < 0 for value in wanted.values() for n in value):
        raise ValueError('Requested more prompts than available')
    categories = sorted({(r['harmful'], r.get('category', 'unknown')) for r in rows})
    gindex = {g: i for i, g in enumerate(groups)}
    by_label = np.zeros((2, len(groups)))
    by_category = np.zeros((len(categories), len(groups)))
    cindex = {c: i for i, c in enumerate(categories)}
    for r in rows:
        by_label[r['harmful'], gindex[r['group']]] += 1
        by_category[cindex[(r['harmful'], r.get('category', 'unknown'))], gindex[r['group']]] += 1
    nbin = len(groups)*4
    ndev = len(categories)*4
    nvars = nbin + ndev
    nconstraints = len(groups) + 8 + 2*ndev
    A = lil_matrix((nconstraints, nvars))
    lower = np.full(nconstraints, -np.inf)
    upper = np.full(nconstraints, np.inf)
    k = 0
    for g in range(len(groups)):
        A[k, g*4:g*4+4] = 1
        lower[k] = upper[k] = 1; k += 1
    for p, split in enumerate(partitions):
        columns = np.arange(len(groups))*4+p
        for y in (0, 1):
            A[k, columns] = by_label[y]
            lower[k] = upper[k] = wanted[split][y]; k += 1
        for ci, (y, category) in enumerate(categories):
            desired = by_category[ci].sum()*wanted[split][y]/max(1, total[y])
            deviation = nbin+p*len(categories)+ci
            A[k, columns] = by_category[ci]; A[k, deviation] = -1
            upper[k] = desired; k += 1
            A[k, columns] = -by_category[ci]; A[k, deviation] = -1
            upper[k] = -desired; k += 1
    objective = np.r_[np.random.default_rng(seed).uniform(0, 1e-6, nbin), np.ones(ndev)]
    result = milp(objective, integrality=np.r_[np.ones(nbin), np.zeros(ndev)],
        bounds=Bounds(np.zeros(nvars), np.r_[np.ones(nbin), np.full(ndev, np.inf)]),
        constraints=LinearConstraint(A.tocsr(), lower, upper), options={'time_limit': 30})
    if result.x is None:
        raise ValueError('No feasible focus-disjoint allocation; revise counts, not grouping')
    assignment = np.rint(result.x[:nbin]).reshape(-1, 4)
    if not np.all(assignment.sum(1) == 1):
        raise ValueError('Nonintegral allocation returned')
    for p, split in enumerate(partitions):
        if not np.allclose(by_label @ assignment[:, p], wanted[split]):
            raise ValueError('Allocation violates requested class counts')
    output = []
    for r in rows:
        split = partitions[int(assignment[gindex[r['group']]].argmax())]
        if split != 'unused':
            output.append({**r, 'split': split})
    return output


def stratified_subset(rows, count, seed):
    """Deterministic category round-robin; no outcomes consulted."""
    if count > len(rows):
        raise ValueError('Subset exceeds available prompts')
    buckets = {}
    for r in rows:
        buckets.setdefault(r['category'], []).append(r)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    categories = sorted(buckets)
    rng.shuffle(categories)
    selected = []
    while len(selected) < count:
        for category in categories:
            if buckets[category] and len(selected) < count:
                selected.append(buckets[category].pop())
    return selected


def scheduled_subsets(rows, config):
    result = {}
    for kind, split, key in [('screening', 'validation', 'screening_counts'),
                              ('diagnostic', 'test', 'diagnostic_counts')]:
        selected = []
        for i, (role, count) in enumerate(config['data'][key].items()):
            candidates = [r for r in rows if r['split'] == split and (
                (role == 'xstest_safe' and r['corpus'] == 'xstest' and r['harmful'] == 0) or
                (role == 'xstest_unsafe' and r['corpus'] == 'xstest' and r['harmful'] == 1) or
                (role in ('harmbench', 'gsm8k') and r['corpus'] == role))]
            selected.extend(stratified_subset(candidates, count, config['experiment']['seed']+i))
        result[kind] = [r['id'] for r in selected]
    return result


def validate_rows(rows):
    ids, texts, groups = set(), {}, {}
    for r in rows:
        if r['id'] in ids:
            raise ValueError('Duplicate benchmark ID')
        ids.add(r['id'])
        for key, mapping in ((normalized(r['prompt']), texts), (r['group'], groups)):
            if key in mapping and mapping[key] != r['split']:
                raise ValueError('Cross-split prompt/group leakage')
            mapping[key] = r['split']
    return rows


def prepare(config, run):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    d = config['data']
    urls = {
        'xstest': f"https://raw.githubusercontent.com/paul-rottger/xstest/{d['xstest_revision']}/xstest_prompts.csv",
        'harmbench': f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{d['harmbench_revision']}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    }
    xs, xs_hash = fetch_csv(urls['xstest'])
    hb, hb_hash = fetch_csv(urls['harmbench'])
    candidates = []
    for r in xs:
        if r['label'] not in ('safe', 'unsafe'):
            raise ValueError(f"Unexpected XSTest label: {r['label']}")
        candidates.append(dict(id='xs-' + r['id'], prompt=r['prompt'],
            harmful=int(r['label'] == 'unsafe'), group='xs-focus-' + normalized(r['focus']),
            corpus='xstest', category=r['type'], answer=None))
    rows = grouped_splits(candidates, d['xstest_counts'], config['experiment']['seed'])
    rng = random.Random(config['experiment']['seed'])
    # Only direct, standard text behaviours: no contextual/copyright special judges.
    eligible = [r for r in hb if r['FunctionalCategory'] == 'standard']
    rng.shuffle(eligible)
    used_text = {normalized(r['prompt']) for r in rows}
    eligible = [r for r in eligible if normalized(r['Behavior']) not in used_text]
    if len(eligible) < d['harmbench_test']:
        raise ValueError('Not enough standard HarmBench behaviours')
    rows.extend(dict(id='hb-' + r['BehaviorID'], prompt=r['Behavior'], harmful=1,
        group='hb-' + r['BehaviorID'], split='test', corpus='harmbench',
        category=r['SemanticCategory'], answer=None) for r in eligible[:d['harmbench_test']])
    parquet_hashes = {}
    for split, original in [('validation', 'train'), ('test', 'test')]:
        filename = f'main/{original}-00000-of-00001.parquet'
        path = hf_hub_download('openai/gsm8k', filename, repo_type='dataset', revision=d['gsm8k_revision'])
        parquet_hashes[filename] = file_hash(path)
        # Direct pinned parquet read avoids version-sensitive dataset glob resolution.
        ds = pd.read_parquet(path).to_dict('records')
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for i in indices[:d['gsm8k_' + split]]:
            item = ds[i]
            rows.append(dict(id=f'gsm-{original}-{i}', group=f'gsm-{original}-{i}',
                prompt=item['question'] + '\nGive your final numeric answer after ####.',
                harmful=0, split=split, corpus='gsm8k', category='math',
                answer=item['answer'].split('####')[-1].strip().replace(',', '')))
    validate_rows(rows)
    atomic_json(run / 'data_manifest.json', dict(urls=urls, sha256={'xstest': xs_hash,
        'harmbench': hb_hash}, gsm8k_revision=d['gsm8k_revision'], gsm8k_sha256=parquet_hashes, rows_hash=digest(rows),
        scheduled_subsets=scheduled_subsets(rows, config),
        notes='XSTest focus-disjoint split; standard direct HarmBench only; not full benchmark ASR.'))
    # All-or-nothing file avoids duplicated preparation on resume.
    tmp = run / 'prompts.tmp'
    with tmp.open('w', encoding='utf-8') as handle:
        import json
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    tmp.replace(run / 'prompts.jsonl')
    return rows
