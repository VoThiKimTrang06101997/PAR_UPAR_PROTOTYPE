"""LODO hyperparameter selection on six held-out transfers.

Each checkpoint is selected using validation of its two *training* domains;
held-out-domain labels are evaluated once only, after selecting that checkpoint.
The multi-seed paired comparison is not a statistical significance test.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def summary(scores):
    v=np.asarray(scores, dtype=np.float64)
    return dict(mean=float(v.mean()), worst=float(v.min()),
                std=float(v.std()),
                aggregate=float(.65*v.mean()+.35*v.min()-.05*v.std()))


def choose(report_dir, seeds, candidates, expected_epochs, expected_steps,
           min_improvement=.002, max_weakest_drop=.004,
           max_fold_drop=.008, min_positive_folds=4):
    rows=[]
    ordered_keys=[(int(seed), heldout) for seed in seeds for heldout in (0,1,2)]
    for tag, sc, align in candidates:
        folds=[]; metadata=[]
        for seed, heldout in ordered_keys:
            p=report_dir/f'prototype_lodo_seed{seed}_heldout{heldout}_{tag}.json'
            if not p.exists():
                raise FileNotFoundError(f'Missing completed strict LODO report: {p}')
            rec=json.loads(p.read_text(encoding='utf-8'))
            if (rec.get('seed') != seed or rec.get('heldout_domain') != heldout or
                bool(rec.get('teacher_used')) or bool(rec.get('training_domain_leakage')) or
                rec.get('selection_policy') != 'source_validation_only_heldout_once' or
                int(rec.get('completed_epochs',-1)) != expected_epochs or
                int(rec.get('steps_per_epoch',-1)) != expected_steps):
                raise ValueError(f'Stale/incompatible/leaking LODO report: {p}')
            if abs(float(rec['contrastive_weight'])-sc)>1e-8 or abs(float(rec['alignment_weight'])-align)>1e-8:
                raise ValueError(f'Unexpected objective weights in {p}')
            folds.append(float(rec['best_selection']))
            metadata.append(dict(seed=seed, heldout=heldout, score=float(rec['best_selection']),
                                 source_selection=float(rec['source_selection']),
                                 selected_epoch=int(rec['selected_epoch'])))
        rows.append(dict(name=tag, contrastive_weight=sc, alignment_weight=align,
                         fold_scores=folds, fold_details=metadata, **summary(folds)))
    baseline=next((r for r in rows if r['contrastive_weight']==0 and r['alignment_weight']==0),None)
    if baseline is None:
        raise ValueError('A zero-regularization baseline is required')
    base=np.asarray(baseline['fold_scores'])
    for r in rows:
        diff=np.asarray(r['fold_scores'])-base
        r['paired_gain_by_fold']=[float(x) for x in diff]
        r['positive_folds']=int((diff>0).sum())
        r['mean_gain']=float(diff.mean())
        r['worst_fold_drop']=float(diff.min())
        r['gain_by_seed']={str(seed):float(diff[i*3:(i+1)*3].mean())
                            for i,seed in enumerate(seeds)}
        r['accepted_over_baseline']=bool(
            r['name']!=baseline['name'] and
            r['aggregate']>=baseline['aggregate']+min_improvement and
            r['worst']>=baseline['worst']-max_weakest_drop and
            r['positive_folds']>=min_positive_folds and
            r['worst_fold_drop']>=-max_fold_drop
        )
    accepted=[r for r in rows if r['accepted_over_baseline']]
    selected=max(accepted,key=lambda r:r['aggregate']) if accepted else baseline
    return dict(selected=selected, baseline=baseline,
                accepted_contrastive=bool(accepted),
                candidates=sorted(rows,key=lambda r:r['aggregate'],reverse=True),
                seeds=list(seeds), expected_epochs=expected_epochs,
                expected_steps=expected_steps,
                selection_policy='source_validation_only_heldout_once',
                notes='Six true LODO folds: source-domain-selected checkpoints, held-out scored only once; no three-domain teacher.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', default='42,123')
    p.add_argument('--candidates',default='none:0:0,moderate:0.04:0.01,strong:0.08:0.02')
    p.add_argument('--expected-epochs',type=int,required=True)
    p.add_argument('--expected-steps',type=int,required=True)
    p.add_argument('--min-improvement',type=float,default=.002)
    p.add_argument('--max-weakest-drop',type=float,default=.004)
    p.add_argument('--max-fold-drop',type=float,default=.008)
    p.add_argument('--min-positive-folds',type=int,default=4)
    args=p.parse_args()
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if len(seeds)<2 or len(set(seeds))!=len(seeds):
        raise ValueError('At least two distinct seeds are required')
    candidates=[]
    for item in args.candidates.split(','):
        tag,sc,al=item.split(':')
        candidates.append((tag,float(sc),float(al)))
    r=choose(args.report_dir,seeds,candidates,args.expected_epochs,args.expected_steps,
             args.min_improvement,args.max_weakest_drop,
             args.max_fold_drop,args.min_positive_folds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':
    main()
