"""Conservative domain-stratified UPAR calibration + optional dual-resolution TTA.

Never uses hidden/private test data. Fits on stable 70% of source validation;
checks a locked 30% audit for the SINGLE fit-selected challenger vs anchor.
Original checkpoints and original calibration are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from strong_par import ATTRIBUTE_NAMES, ImageResolver, UPARDataset, build_eval_transform, seed_everything
from prototype_par import HybridPrototypePAR
from vapor_par.metrics import evaluate_probs
from calibrate_generalization import load_pt, get_state, stable_split, sigmoid, logit

TTA_SIZES = {'flip': None, 'dual_scale_flip': (320, 160)}


def signature(path, mode, repo_root):
    stat = path.stat()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    return hashlib.sha256(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{mode}:{repo_root.resolve()}:{source_hash}'.encode()).hexdigest()


def model_path(root, kind, seed):
    stem = 'prototype_convnext_tiny' if kind == 'prototype' else 'prototype_contrastive_convnext_tiny'
    return root / f'{stem}_seed{seed}_best.pt'


def model_logits(model, x):
    aux = model(x, return_aux=True)
    return aux['linear_logits'].float(), aux['proto_logits'].float()


@torch.inference_mode()
def infer_logits(model, loader, device, mode, title):
    ln, pn, ys, dom = [], [], [], []
    for batch in tqdm(loader, desc=f'Inference {title} {mode}', unit='batch', dynamic_ncols=True):
        x = batch['image'].to(device, non_blocking=True)
        views = [x, x.flip(-1)]
        if mode == 'dual_scale_flip':
            x2 = F.interpolate(x, size=TTA_SIZES[mode], mode='bilinear', align_corners=False)
            views += [x2, x2.flip(-1)]
        ls, ps = [], []
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            for view in views:
                a, b = model_logits(model, view)
                ls.append(a); ps.append(b)
        ln.append(torch.stack(ls).mean(0).cpu().numpy().astype('float32'))
        pn.append(torch.stack(ps).mean(0).cpu().numpy().astype('float32'))
        y = batch['target'].numpy()
        valid = batch['valid'].numpy()
        ys.append(np.where(valid > 0, y, -1).astype('int8'))
        dom.append(np.asarray(batch['domain'], dtype='int8'))
    return {'linear': np.concatenate(ln), 'proto': np.concatenate(pn),
            'y': np.concatenate(ys), 'domain': np.concatenate(dom)}


def domain_metrics(y, prob, d, mask, th):
    metrics = evaluate_probs(y[mask], prob[mask], th)
    per_domain = {}
    for domain in (0, 1, 2):
        sub = mask & (d == domain)
        if sub.sum() < 20:
            raise ValueError(f'Domain {domain}: insufficient examples in split')
        per_domain[domain] = evaluate_probs(y[sub], prob[sub], th)
    s = np.array([m['challenge_score'] for m in per_domain.values()])
    # A domain-robust FIT criterion: pooled score, equal-domain mean, worst domain.
    robust = (.35 * metrics['challenge_score'] + .35 * float(s.mean()) +
              .30 * float(s.min()))
    return float(robust), metrics, per_domain


def ba_thresholds(y, prob, mask, domains=None):
    """Attribute-balanced thresholds; masked unknown labels excluded."""
    grid = np.arange(.26, .781, .02, dtype='float32')
    result = np.full(40, .5, dtype='float32')
    for c in range(40):
        ok = mask & (y[:, c] >= 0)
        if domains is not None:
            ok &= np.isin(domains, [0, 1, 2])
        labels = y[ok, c]
        if (labels == 1).sum() < 15 or (labels == 0).sum() < 15:
            continue
        p = prob[ok, c]
        a = (p[labels == 1, None] >= grid[None]).mean(0)
        b = (p[labels == 0, None] < grid[None]).mean(0)
        val = .5 * (a + b) - .0005 * abs(grid-.5)
        result[c] = grid[int(val.argmax())]
    return result


def conservative_thresholds(y, prob, d, fit):
    pooled = ba_thresholds(y, prob, fit)
    domain_th = []
    for domain in (0, 1, 2):
        domain_th.append(ba_thresholds(y, prob, fit & (d == domain)))
    median = np.median(np.stack(domain_th), axis=0)
    # Shrink domain-specific thresholds toward pool mean, then toward .5
    stable = .5 * pooled + .5 * median
    return stable.astype('float32')


def threshold_options(y, prob, d, fit, anchor_th=None):
    stable = conservative_thresholds(y, prob, d, fit)
    options = []
    if anchor_th is not None:
        options.append(('preserved_prototype_thresholds', np.asarray(anchor_th, dtype='float32')))
    for base in (.46, .49, .50, .53):
        for alpha in (0., .20, .40):
            th = np.clip((1.-alpha)*base + alpha*stable, .28, .75).astype('float32')
            options.append((f'base{base:.2f}_shrink{alpha:.2f}', th))
    return options


def make_candidates(entries, anchor_weights):
    original = [e for e in entries if e['kind'] == 'prototype']
    students = [e for e in entries if e['kind'] == 'trained']
    if len(original) != 2:
        raise ValueError('Need both original Prototype checkpoints')
    k1, k2 = [e['key'] for e in original]
    initial = [('protected_anchor', dict(anchor_weights))]
    for a in (.0, .20, .40, .60):
        w = {k1:a, k2:1.-a}
        if not np.allclose([w[k1],w[k2]], [anchor_weights.get(k1,0), anchor_weights.get(k2,0)]):
            initial.append((f'prototype_{a:.2f}_{1-a:.2f}', w))
    for student in students:
        for frac in (.15, .30):
            weights = {k:v*(1.-frac) for k,v in anchor_weights.items()}
            weights[student['key']] = frac
            initial.append((f'prototype_plus_{student["key"]}_{frac:.2f}',weights))
    if len(students) == 2:
        for frac in (.15, .30, .45):
            weights = {k:v*(1.-frac) for k,v in anchor_weights.items()}
            weights[students[0]['key']] = .5*frac
            weights[students[1]['key']] = .5*frac
            initial.append((f'prototype_plus_both_{frac:.2f}',weights))
    return initial


def all_views(cache, entries, mode, strength):
    result = {}
    for entry in entries:
        item = cache[(entry['key'],mode)]
        gate = np.clip(entry['gate']*strength, 0, .60)
        result[entry['key']] = ((1-gate)*item['linear'] + gate*item['proto']).astype('float32')
    return result


def evaluate_candidate(candidate, views, y, d, fit, legacy_th=None):
    name, weights = candidate
    z = sum((float(w)*views[k] for k,w in weights.items() if w > 0),
            np.zeros_like(next(iter(views.values()))))
    prob = sigmoid(z)
    best = None
    for desc, th in threshold_options(y,prob,d,fit,legacy_th if name=='protected_anchor' else None):
        score, metrics, bydomain = domain_metrics(y,prob,d,fit,th)
        # Tiny complexity preference; a very small gain may be noise.
        student_mass = sum(float(v) for k,v in weights.items() if k.startswith('trained:'))
        objective = score - .0006*student_mass
        if best is None or objective > best['fit_objective']:
            best = dict(name=name, weights=weights, threshold_name=desc,
                        thresholds=th, fit_score=score, fit_objective=objective,
                        fit_metrics=metrics, fit_by_domain=bydomain)
    return best, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', required=True, type=Path)
    p.add_argument('--checkpoint-dir', required=True, type=Path)
    p.add_argument('--trained-checkpoint-dir', required=True, type=Path)
    p.add_argument('--result-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--anchor-calibration', type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--tta-options', default='flip,dual_scale_flip')
    p.add_argument('--force-inference', action='store_true')
    args = p.parse_args()
    modes = [x.strip() for x in args.tta_options.split(',') if x.strip()]
    if not modes or any(mode not in TTA_SIZES for mode in modes):
        raise ValueError(f'TTA options must be from {list(TTA_SIZES)}')
    seed_everything(2027)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(args.repo_root/'data/annotations/task1/val/gt.csv')
    ds = UPARDataset(df, ImageResolver(args.repo_root/'data', args.repo_root),
                    build_eval_transform(288,144))
    fit,audit = stable_split(ds.paths,ds.domains)
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                        pin_memory=device.type=='cuda')
    entries=[]
    for kind,root in [('prototype',args.checkpoint_dir),('trained',args.trained_checkpoint_dir)]:
        for seed in (42,123):
            path=model_path(root,kind,seed)
            if not path.exists():
                if kind=='prototype': raise FileNotFoundError(path)
                print(f'[INFO] Optional student missing: {path}',flush=True)
                continue
            ck=load_pt(path)
            cfg=dict(ck.get('prototype_config',{}));cfg.pop('pretrained',None)
            entries.append(dict(key=f'{kind}:{seed}',kind=kind,seed=seed,
                                checkpoint=str(path),config=cfg))
            del ck
    print('SOURCE DOMAINS:',{int(x):int((ds.domains==x).sum()) for x in np.unique(ds.domains)},flush=True)
    print('FIT/AUDIT:',int(fit.sum()),int(audit.sum()),flush=True)
    cache={}
    for e in entries:
        ck_path=Path(e['checkpoint']); ck=None; model=None
        for mode in modes:
            cache_path = args.result_dir/f"shift_predictions_{e['kind']}_seed{e['seed']}_{mode}.npz"
            sig=signature(ck_path,mode,args.repo_root)
            record=None
            if cache_path.is_file() and not args.force_inference:
                try:
                    with np.load(cache_path) as f:
                        if str(f['signature'].item())==sig and len(f['y'])==len(ds):
                            record={k:f[k] for k in f.files if k!='signature'}
                except Exception: pass
            if record is None:
                if model is None:
                    ck=load_pt(ck_path)
                    model=HybridPrototypePAR(pretrained=False,**e['config'])
                    model.load_state_dict(get_state(ck),strict=True)
                    model=model.to(device).eval()
                    del ck
                record=infer_logits(model,loader,device,mode,e['key'])
                np.savez_compressed(cache_path,signature=sig,**record)
                print(f'[CACHE SAVED] {cache_path.name}',flush=True)
            else:
                print(f'[CACHE HIT] {cache_path.name}',flush=True)
            if not np.array_equal(record['domain'],ds.domains):
                raise ValueError('Domain order mismatch in cache')
            if not np.array_equal(record['y'],np.where(ds.targets >= 0, ds.targets, -1)):
                # Some datasets use a non-binary unknown flag; compare valid labels below.
                for c in range(40):
                    yy=ds.targets[:,c]
                    expected=np.where((yy==0)|(yy==1),yy,-1)
                    if not np.array_equal(record['y'][:,c],expected):
                        raise ValueError('Label order mismatch in cache')
            cache[(e['key'],mode)] = record
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
        ck=load_pt(ck_path)
        state=get_state(ck)
        if 'prototype_gate_logits' in state:
            logits=state['prototype_gate_logits'].detach().float().numpy()
            e['gate']=np.clip(1/(1+np.exp(-np.clip(logits,-30,30))),
                              float(e['config'].get('prototype_gate_min',.08)),
                              float(e['config'].get('prototype_gate_max',.92))).reshape(1,-1)
        else:
            raise ValueError('Prototype checkpoint missing gate logits')
        del ck

    y=cache[(entries[0]['key'],modes[0])]['y']
    d=cache[(entries[0]['key'],modes[0])]['domain']
    for item in cache.values():
        if not np.array_equal(item['y'],y): raise ValueError('Different GT order in inference caches')

    anchor_weights={'prototype:42':.2,'prototype:123':.8}
    legacy_th=None
    if args.anchor_calibration and args.anchor_calibration.is_file():
        try:
            legacy=load_pt(args.anchor_calibration)
            pairs=[(f"prototype:{int(v['seed'])}",float(w)) for v,w in zip(legacy['members'],legacy['weights'])]
            saved=dict(pairs)
            if set(saved).issubset(anchor_weights) and abs(sum(saved.values())-1.) < 1e-4:
                anchor_weights=saved
                legacy_th=sigmoid(np.asarray(legacy['threshold_logits']).reshape(40))
                print('[PROTECTED] Original calibration loaded; no overwrite',flush=True)
        except Exception as exc:
            print('[WARNING] Legacy calibration could not be used:',repr(exc),flush=True)
    candidates=make_candidates(entries,anchor_weights)
    results=[]
    probs_by_key={}
    total=len(modes)*2*len(candidates)
    for mode in modes:
        for strength in (1.0,.85):
            views=all_views(cache,entries,mode,strength)
            for candidate in tqdm(candidates,desc=f'FIT candidate search {mode} strength={strength}',unit='candidate'):
                saved=legacy_th if mode=='flip' and strength==1.0 else None
                rec,prob=evaluate_candidate(candidate,views,y,d,fit,saved)
                rec.update(tta=mode,strength=strength)
                if mode=='dual_scale_flip': rec['fit_objective'] -= .0006
                results.append(rec)
                probs_by_key[(mode,strength,rec['name'])]=prob
    # Exact preserved anchor is required to prevent silent threshold replacement.
    if legacy_th is not None and 'flip' in modes:
        anchor=next((r for r in results if r['name']=='protected_anchor' and r['tta']=='flip'
                     and r['strength']==1.0 and r['threshold_name']=='preserved_prototype_thresholds'),None)
        # In case a fit-optimized threshold beats the saved one, explicitly create the protected reference.
        views=all_views(cache,entries,'flip',1.0)
        z=sum(float(v)*views[k] for k,v in anchor_weights.items())
        prob=sigmoid(z)
        fit_score,fit_metrics,fit_domains=domain_metrics(y,prob,d,fit,legacy_th)
        anchor=dict(name='protected_anchor',tta='flip',strength=1.0,weights=anchor_weights,
                    thresholds=np.asarray(legacy_th,dtype='float32'),threshold_name='preserved_prototype_thresholds',
                    fit_score=fit_score,fit_objective=fit_score,fit_metrics=fit_metrics,fit_by_domain=fit_domains)
        probs_by_key[('flip',1.0,'protected_anchor')]=prob
    else:
        anchor=next(r for r in results if r['name']=='protected_anchor' and r['strength']==1.0
                    and r['tta']=='flip')

    results.sort(key=lambda r:r['fit_objective'],reverse=True)
    challengers=[r for r in results if not (r['name']=='protected_anchor' and r['tta']=='flip'
                                      and r['strength']==1.0)]
    leader=challengers[0] if challengers else anchor
    # AUDIT IS NOT PART OF SEARCH. Measure exactly one fit-selected challenger and protected anchor.
    def audit_rec(r):
        prob=probs_by_key[(r['tta'],r['strength'],r['name'])]
        robust,m,bydomain=domain_metrics(y,prob,d,audit,r['thresholds'])
        return dict(score=robust,metrics=m,by_domain=bydomain)
    anchor_audit=audit_rec(anchor)
    leader_audit=audit_rec(leader)
    weakest=lambda rec: min(float(v['challenge_score']) for v in rec['by_domain'].values())
    passed=(leader_audit['score']>=anchor_audit['score']+.002
            and weakest(leader_audit)>=weakest(anchor_audit)-.0015
            and leader_audit['metrics']['mA']>=anchor_audit['metrics']['mA']-.003
            and leader_audit['metrics']['instance_f1']>=anchor_audit['metrics']['instance_f1']-.003
            and all(leader_audit['by_domain'][k]['challenge_score']>=anchor_audit['by_domain'][k]['challenge_score']-.003
                    for k in anchor_audit['by_domain']))
    selected=leader if passed else anchor
    chosen_audit=leader_audit if passed else anchor_audit
    raw={
        'format':'upar-shift-robust-calibration',
        'attribute_names':list(ATTRIBUTE_NAMES),
        'entries':[{k:v for k,v in e.items() if k!='gate'} for e in entries],
        'weights':selected['weights'],
        'strength':float(selected['strength']),
        'thresholds':torch.from_numpy(selected['thresholds'].copy()),
        'threshold_logits':torch.from_numpy(logit(selected['thresholds']).astype('float32')),
        'image_height':288,'image_width':144,
        'tta':selected['tta'],
        'dual_height':320,'dual_width':160,
        'audit_metrics':chosen_audit['metrics'],
        'contains_retrained':any(k.startswith('trained:') and w>0 for k,w in selected['weights'].items()),
        'accepted':bool(passed),
        'selection_note':'One fit-selected challenger tested once against protected anchor on locked source-domain audit. No hidden/private labels.',
    }
    torch.save(raw,args.output)
    def compact(r):
        return dict(name=r['name'],tta=r['tta'],strength=r['strength'],weights=r['weights'],
                    threshold_name=r['threshold_name'],fit_score=r['fit_score'],
                    fit_mA=r['fit_metrics']['mA'],fit_f1=r['fit_metrics']['instance_f1'])
    report=dict(selected=compact(selected),winner=compact(leader),anchor=compact(anchor),
                selected_audit=chosen_audit,leader_audit=leader_audit,anchor_audit=anchor_audit,
                accepted=bool(passed),searched_fit_candidates=len(results),
                available_models=[e['key'] for e in entries],fit_count=int(fit.sum()),audit_count=int(audit.sum()),
                top_fit_candidates=[compact(r) for r in results[:12]],
                note='Scores are SOURCE validation, not unseen/private Codabench scores. No >0.78 guarantee.')
    report_path=args.result_dir/'shift_robust_audit_report.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n'+'='*85)
    print('PROTECTED ANCHOR:',anchor['name'],anchor['tta'],anchor_audit['metrics']['challenge_score'])
    print('BEST FIT CHALLENGER:',leader['name'],leader['tta'],leader_audit['metrics']['challenge_score'])
    print('AUDIT DECISION:', 'ACCEPT NEW CANDIDATE' if passed else 'PRESERVE PROTOTYPE ANCHOR')
    print('SELECTED:',selected['name'],selected['tta'],selected['weights'])
    print('SELECTED SOURCE AUDIT:',json.dumps(chosen_audit['metrics'],indent=2))
    print('CALIBRATION:',args.output)
    print('AUDIT REPORT:',report_path,flush=True)


if __name__=='__main__':
    main()
