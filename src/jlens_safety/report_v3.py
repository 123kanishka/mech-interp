"""Paired fresh-run results, factorial comparisons, and explicit claim limits."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import numpy as np

from .analysis import attach_judgements, diagnostic_report, summarise, cluster_interval
from .common import atomic_json, digest, file_hash, rows_by_id
from .learned import paired_power_plan
from .pipeline import check_frozen, job_id, test_allowed_ids
from sae_jlens.run_io import read_jsonl


def validation_power(rows,c):
    base={r['prompt_id']:r for r in rows if r['condition']['name']=='baseline'}
    result=[]
    for name in sorted({r['condition']['name'] for r in rows if r['condition']['method']=='jlens_tokens'}):
        groups=defaultdict(list)
        for r in rows:
            if r['condition']['name']==name and r['harmful']:
                groups[r['group']].append(r['metrics']['harmful_compliance']-base[r['prompt_id']]['metrics']['harmful_compliance'])
        values=[float(np.mean(v)) for v in groups.values()]
        result.append(dict(condition=name,pilot_groups=len(values), **paired_power_plan(values,
            c['analysis']['target_absolute_effect'], c['analysis']['power'],
            c['analysis']['familywise_alpha']/5)))
    return dict(estimates=result,note='Validation-only approximate paired group planning. No test-based resizing. '
                'Correlated attack variants are not independent sample-size gains.')


def bootstrap_difference(values, groups, c, comparisons=5):
    unique=sorted(set(groups))
    grouped={g:[] for g in unique}
    for value, group in zip(values,groups):
        grouped[group].append(value)
    sums=np.array([sum(grouped[g]) for g in unique])
    counts=np.array([len(grouped[g]) for g in unique])
    rng=np.random.default_rng(c['experiment']['seed'])
    estimates=[]
    # Chunking prevents a bootstrap_samples x full_large_dataset allocation.
    for start in range(0,c['analysis']['bootstrap_samples'],100):
        size=min(100,c['analysis']['bootstrap_samples']-start)
        index=rng.integers(len(unique),size=(size,len(unique)))
        estimates.extend(sums[index].sum(1)/counts[index].sum(1))
    tail=c['analysis']['familywise_alpha']/comparisons/2
    return dict(mean=float(np.mean(values)),low=float(np.quantile(estimates,tail)),
        high=float(np.quantile(estimates,1-tail)),n=len(values),groups=len(unique),
        interval='Bonferroni-adjusted cluster bootstrap within five comparisons per corpus/metric',
        across_corpora_and_metrics_adjusted=False)


def factorial_comparisons(rows,c):
    """Average learning seeds within prompt BEFORE resampling request groups."""
    by_prompt=defaultdict(dict)
    meta={}
    for r in rows:
        method=r['condition']['method']
        name=r['condition']['name']
        if name in ('scalar_last','scalar_all'):
            arm=name
        elif method=='jlens_tokens' and r['condition'].get('norm_matching','per_position')=='per_position':
            arm='learned_'+r['condition']['positions']
        else:
            continue
        by_prompt[r['prompt_id']].setdefault(arm,[]).append(r['metrics'])
        meta[r['prompt_id']]=r
    weights={
        'learned_minus_scalar_last': {'learned_last':1,'scalar_last':-1},
        'learned_minus_scalar_all': {'learned_all':1,'scalar_all':-1},
        'all_minus_last_scalar': {'scalar_all':1,'scalar_last':-1},
        'all_minus_last_learned': {'learned_all':1,'learned_last':-1},
        'interaction': {'learned_all':1,'scalar_all':-1,'learned_last':-1,'scalar_last':1}}
    result=[]
    for corpus in sorted({r['corpus'] for r in rows}):
        for metric in ('harmful_compliance','overrefusal','math_accuracy'):
            for name, coefficients in weights.items():
                values,groups=[],[]
                for pid,arms in by_prompt.items():
                    if meta[pid]['corpus']!=corpus:
                        continue
                    if not all(a in arms for a in coefficients):
                        raise ValueError('Missing factorial arm')
                    if any(m[metric] is None for a in coefficients for m in arms[a]):
                        continue
                    for a in coefficients:
                        expected=len(c['learning']['seeds']) if a.startswith('learned') else 1
                        if len(arms[a])!=expected:
                            raise ValueError('Missing or duplicated fitting seed')
                    values.append(sum(w*np.mean([m[metric] for m in arms[a]]) for a,w in coefficients.items()))
                    groups.append(meta[pid]['group'])
                if values:
                    result.append(dict(corpus=corpus,metric=metric,comparison=name,
                        **bootstrap_difference(values,groups,c)))
    return result


def final_report(c,run,prompts):
    from .protocol_v3 import verify_extra_hashes
    verify_extra_hashes(run)
    conditions=check_frozen(run)
    allowed=test_allowed_ids(run,prompts)
    expected_jobs={job_id(p,condition):(p,condition) for p in prompts if p['split']=='test' for condition in conditions
                   if p['id'] in allowed[condition['name']]}
    expected=set(expected_jobs)
    generated=rows_by_id(run/'test_generations.jsonl')
    judged=rows_by_id(run/'test_judgements.jsonl')
    if set(generated)!=expected or set(judged)!=expected:
        raise ValueError('Final results are missing, duplicated, or contain unexpected jobs')
    for key,row in generated.items():
        prompt,condition=expected_jobs[key]
        if row['condition']!=condition or row['prompt_id']!=prompt['id'] or any(
                row.get(field)!=value for field,value in prompt.items() if field!='id'):
            raise ValueError('Test row payload differs from frozen prompt/condition')
    rows=attach_judgements(list(generated.values()),judged)
    official_expected={r['id'] for r in rows if r['corpus'].startswith('harmbench_')}
    official=rows_by_id(run/'harmbench_judgements.jsonl')
    official_enabled=c.get('official_judge',{}).get('enabled',False)
    if official_enabled and set(official)!=official_expected:
        raise ValueError('Missing or unexpected official HarmBench labels; run official-judge before report')
    if official_enabled and official_expected:
        template=json.loads((run/'harmbench_classifier_template.json').read_text())['template']
        identity=digest(dict(config=c['official_judge'],template=template))
        if any(r['judge_identity']!=identity for r in official.values()):
            raise ValueError('Official HarmBench judge identity mismatch')
    frozen=json.loads((run/'frozen_test.json').read_text())
    primary=[r for r in rows if r['condition']['name'] in frozen['schedule']['primary_conditions']]
    summary=summarise(primary,c)
    summary['scope']='Fresh test generations in a bounded exploratory v3 run; baseline + scalar scopes + all registered J-token fitting seeds'
    if (run/'preflight.json').exists():
        summary['development_pilot_audit']=json.loads((run/'preflight.json').read_text())
    summary['harmbench_metric']='WildGuard harmful-response rate; official behavior ASR is separate'
    atomic_json(run/'summary.json',summary)
    factorial=factorial_comparisons(primary,c)
    atomic_json(run/'factorial_comparisons.json',factorial)
    atomic_json(run/'signal_location.json',diagnostic_report(primary,c))
    baseline={r['prompt_id']:r for r in rows if r['condition']['name']=='baseline'}
    persistence=[]
    for r in rows:
        b=baseline[r['prompt_id']]
        traces={t['layer']:t for t in b['trace'] if t['step']==0}
        for t in r['trace']:
            if t['step']==0 and t['layer'] in traces:
                persistence.append(dict(condition=r['condition']['name'],prompt_id=r['prompt_id'],
                    corpus=r['corpus'],layer=t['layer'],j_score_delta=t['j_score']-traces[t['layer']]['j_score'],
                    probe_score_delta=t['probe_score']-traces[t['layer']]['probe_score'],
                    baseline_harmful_response=b['judgement']['response_harmful'],
                    harmful_response=r['judgement']['response_harmful']))
    atomic_json(run/'downstream_persistence.json',dict(rows=persistence,
        note='Identical prompt prefixes at prefill only; changed decode prefixes are not causal mediation evidence.'))
    diag_ids=set(frozen['schedule']['diagnostic_prompt_ids'])
    atomic_json(run/'diagnostic_summary.json',summarise([r for r in rows if r['prompt_id'] in diag_ids],c))
    perturbations=[]
    for name in sorted({r['condition']['name'] for r in rows}):
        selected=[r for r in rows if r['condition']['name']==name]
        perturbations.append(dict(condition=name,n=len(selected),
            mean_edited_positions=float(np.mean([r.get('edited_positions',0) for r in selected])),
            mean_squared_trajectory_update=float(np.mean([r.get('squared_update_norm',0.) for r in selected])),
            mean_relative_per_call_norm=float(np.mean([r['mean_relative_intervention_norm'] for r in selected]))))
    atomic_json(run/'perturbation_diagnostics.json',perturbations)
    if (run/'directions.json').exists():
        metadata=json.loads((run/'directions.json').read_text())['metadata']
        weights=[]
        for fit in json.loads((run/'learned_coefficients.json').read_text()):
            labels=metadata[str(fit['layer'])]
            weights.append(dict(condition=fit['name'],method=fit['method'],layer=fit['layer'],
                position_scope=fit['positions'],seed=fit['seed'],
                terms=[dict(index=i,token_id=None if fit['method']=='random_tokens' else labels['token_ids'][i],
                    token=None if fit['method']=='random_tokens' else labels['tokens'][i],beta=beta)
                    for i,beta in enumerate(fit['coefficients'])]))
        atomic_json(run/'token_direction_weights.json',weights)
    official_summary=[]
    for name,corpus in sorted({(r['condition']['name'],r['corpus']) for r in rows if r['id'] in official}):
        selected=[r for r in rows if r['condition']['name']==name and r['corpus']==corpus]
        values=[float(official[r['id']]['behavior_success']) for r in selected]
        by_behavior=defaultdict(list)
        for r,value in zip(selected,values): by_behavior[r['behavior_id']].append(value)
        official_summary.append(dict(condition=name,corpus=corpus,
            prompt_success=cluster_interval(values,[r['group'] for r in selected],
                c['experiment']['seed'],c['analysis']['bootstrap_samples']),
            behavior_any_success=float(np.mean([max(v) for v in by_behavior.values()])),
            behaviors=len(by_behavior),scope='Standard-text subset and registered attack suite only'))
    atomic_json(run/'harmbench_official_summary.json',official_summary)
    # Blinded, seeded random review; map and automated labels are a separate file.
    buckets=defaultdict(list)
    for r in rows:
        buckets[(r['condition']['name'],r['corpus'],r['harmful'])].append(r)
    review,mapping=[],[]
    for key,pool in sorted(buckets.items()):
        pool.sort(key=lambda r:digest([c['experiment']['seed'],'review',r['id']]))
        for r in pool[:c['analysis']['manual_review_per_stratum']]:
            review_id=digest(['blind',r['id']])[:16]
            review.append(dict(review_id=review_id,prompt=r['prompt'],response=r['response'],
                               response_harmful=None,refusal=None,notes=''))
            mapping.append(dict(review_id=review_id,generation_id=r['id'],condition=r['condition']['name'],
                                automated_judgement=r['judgement']))
    review_path=run/'manual_review_blinded.json'
    if review_path.exists():
        existing={r['review_id']:r for r in json.loads(review_path.read_text())}
        if set(existing)!={r['review_id'] for r in review} or any(
            existing[r['review_id']][field]!=r[field] for r in review for field in ('prompt','response')):
            raise ValueError('Existing manual review sample differs from frozen sample')
        # Preserve any human labels/notes added after the first report.
    else:
        atomic_json(review_path,sorted(review,key=lambda r:r['review_id']))
    atomic_json(run/'manual_review_key.json',mapping)
    quality=[]
    for corpus in sorted({r['corpus'] for r in primary}):
        subset=[r for r in primary if r['corpus']==corpus]
        b=[r for r in subset if r['condition']['name']=='baseline' and r['harmful']]
        failures=sum(r['judgement']['response_harmful'] for r in b)
        quality.append(dict(corpus=corpus,baseline_harmful_responses=failures,harmful_prompts=len(b),
            baseline_failure_floor=bool(b and failures<c['analysis']['min_pilot_harmful_responses']),
            truncation=float(np.mean([r['truncated'] for r in subset]))))
    atomic_json(run/'claim_readiness.json',dict(computationally_complete=True,
        publication_claims_ready=False,quality_by_corpus=quality,
        pending=['Blinded human review and judge-agreement audit','Interpret group-level uncertainty and power limits',
                 'Previously measured development pilot failed informativeness and truncation gates; no fresh preflight by user request'] +
                ([] if official_enabled else ['Official HarmBench classifier evaluation disabled']),
        benchmark_exposure='Public benchmarks previously examined; results are fresh, not a claim of unseen benchmarks.'))
    export_harmbench(rows,run)
    plots(factorial,persistence,run)
    atomic_json(run/'checksums.json',{str(p.relative_to(run)):file_hash(p) for p in run.rglob('*')
        if p.is_file() and p.name not in ('checksums.json','budget.json','state.json','.lock','SUCCESS')})
    (run/'SUCCESS').touch()


def export_harmbench(rows,run):
    """Official evaluate_completions.py input; never relabel WildGuard as ASR."""
    output=run/'harmbench_official'
    output.mkdir(exist_ok=True)
    buckets=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['corpus'].startswith('harmbench_'):
            buckets[(r['condition']['name'],r['corpus'])][r['behavior_id']].append(
                dict(test_case=r['prompt'],generation=r['response'],generation_id=r['id']))
    for (condition,corpus),values in buckets.items():
        atomic_json(output/(condition+'_'+corpus+'.json'),dict(values))


def plots(factorial,persistence,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=run/'plots'
    output.mkdir(exist_ok=True)
    for metric in ('harmful_compliance','overrefusal','math_accuracy'):
        values=[r for r in factorial if r['metric']==metric]
        fig,ax=plt.subplots(figsize=(10,max(3,len(values)*.3)))
        for i,r in enumerate(values):
            ax.errorbar(r['mean'],i,xerr=[[max(0,r['mean']-r['low'])],[max(0,r['high']-r['mean'])]],fmt='o')
        ax.set_yticks(range(len(values)),[r['corpus']+' / '+r['comparison'] for r in values],fontsize=7)
        ax.axvline(0,color='grey',linestyle='--')
        ax.set_xlabel(metric+' paired difference; corrected intervals within each corpus/metric')
        fig.tight_layout();fig.savefig(output/(metric+'.png'),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    buckets=defaultdict(lambda:defaultdict(list))
    for r in persistence:
        if r['condition'].startswith('jlens_tokens') and not r['condition'].endswith('totalnorm'):
            buckets[r['condition']][r['layer']].append(r['j_score_delta'])
    for name,values in sorted(buckets.items()):
        layers=sorted(values)
        ax.plot(layers,[np.mean(values[l]) for l in layers],label=name,marker='o')
    ax.axhline(0,color='grey');ax.set(xlabel='Layer',ylabel='Same-prefix prefill J-score change')
    if buckets: ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/'downstream.png',dpi=160);plt.close(fig)
