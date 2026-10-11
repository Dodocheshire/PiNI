"""Train and evaluate the shared-view CBM, LP, and CVPR2025 SALF baseline."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.nn import functional as F

from clip import clip
from pini_cbm.benchmark_baselines import cos_cubed
from pini_cbm.data import normalize
from .data import ROOT, load, prepare, prepare_salf, white_background, write_json
from .model import SALFConcepts, SharedSpatialCBM


def tensor(array, ids=None):
    return torch.as_tensor(np.array(array if ids is None else array[ids]),device='cuda').float()


def native_metrics(scores, targets, thresholds=None):
    valid=(targets.sum(0)>0)&(targets.sum(0)<len(targets))
    ap=average_precision_score(targets[:,valid],scores[:,valid],average='macro')
    prediction=scores >= (.5 if thresholds is None else thresholds)
    tp=(prediction*targets).sum(0);fp=(prediction*(1-targets)).sum(0);fn=((1-prediction)*targets).sum(0)
    return {'mAP':float(ap),'precision':float(np.mean(tp/np.maximum(tp+fp,1))),
            'recall':float(np.mean(tp/np.maximum(tp+fn,1))),'evaluated_concepts':int(valid.sum())}


def concept_thresholds(scores,targets):
    values=[]
    for k in range(targets.shape[1]):
        if not targets[:,k].any():
            values.append(float('inf'));continue
        precision,recall,threshold=precision_recall_curve(targets[:,k],scores[:,k])
        eligible=np.flatnonzero(recall[:-1]>=.8)
        choice=eligible[np.argmax(precision[:-1][eligible])]
        values.append(float(threshold[choice]))
    return np.asarray(values)


@torch.no_grad()
def predict(model, method, arrays, ids, backbone, condition, batch=128, seed=8101):
    model.eval()
    concepts=[];logits=[];spatial=[]
    started=time.time()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for start in range(0,len(ids),batch):
            chunk=ids[start:start+batch]
            original=tensor(arrays['tokens_'+condition],chunk)
            if method=='linear_probe':
                cls=F.normalize(original[:,0],dim=-1)
                logits.append(model(cls).cpu().numpy())
                continue
            if method=='salf':
                c,_=model.recognize(original)
                logits.append(model.classify(c).cpu().numpy())
                concepts.append(c.cpu().numpy())
                continue
            rgb=tensor(arrays['rgb_'+condition],chunk)/255
            processed,mu,sigma,fg=model.perturb(rgb,original,method)
            with torch.autocast('cuda',dtype=torch.float16):
                z=model.encode_view(backbone,normalize(processed),fg)
            c,_,_=model.recognize(z,foreground_prior=fg)
            concepts.append(c.sigmoid().cpu().numpy())
            logits.append(model.classify(c).cpu().numpy())
            truth=tensor(arrays['foreground'],chunk)
            foreground=truth; background=1-truth
            amplitude=mu.abs().mean(1); variance=sigma[:,0].square()
            def mean(value,support):
                return (value*support).sum((1,2))/support.sum((1,2)).clamp_min(1)
            mfg=mean(amplitude,foreground);mbg=mean(amplitude,background)
            vfg=mean(variance,foreground);vbg=mean(variance,background)
            spatial.append(torch.stack([mfg,mbg,vfg,vbg],1).cpu().numpy())
    info={'seconds':time.time()-started}
    if spatial:
        values=np.concatenate(spatial)
        info['foreground_background_mean']=float(values[:,0].mean()/max(values[:,1].mean(),1e-9))
        info['background_foreground_variance']=float(values[:,3].mean()/max(values[:,2].mean(),1e-9))
        info['foreground_mean_abs']=float(values[:,0].mean())
        info['background_mean_abs']=float(values[:,1].mean())
        info['foreground_variance']=float(values[:,2].mean())
        info['background_variance']=float(values[:,3].mean())
    return np.concatenate(logits),np.concatenate(concepts) if concepts else None,info


def predictions(model, method, arrays, ids, backbone, condition, batch=128):
    # Kept as one public entry point for experiments and the image-only CLI.
    result=predict(model,method,arrays,ids,backbone,condition,batch)
    return result


def region_loss(maps, foreground, arrays, ids):
    predicted=torch.cat([maps,foreground[:,None]],1)
    target=tensor(arrays['masks'],ids).flatten(2)
    valid=tensor(arrays['mask_valid'],ids)
    bce=F.binary_cross_entropy_with_logits(predicted,target,reduction='none').mean(-1)
    bce=(bce*valid).sum()/valid.sum().clamp_min(1)
    probabilities=predicted.sigmoid()
    present=(target.sum(-1)>0).float()*valid
    dice=1-(2*(probabilities*target).sum(-1)+1)/(probabilities.sum(-1)+target.sum(-1)+1)
    foreground_target=target[:,-1]
    foreground_probability=foreground.sigmoid()
    foreground_bce=F.binary_cross_entropy_with_logits(foreground,foreground_target)
    foreground_dice=1-(2*(foreground_probability*foreground_target).sum(-1)+1)/(foreground_probability.sum(-1)+foreground_target.sum(-1)+1)
    # Foreground receives a full loss regardless of the number of concepts.
    return bce+.5*(dice*present).sum()/present.sum().clamp_min(1)+foreground_bce+.5*foreground_dice.mean()


def validation(model,method,arrays,ids,backbone):
    rows={}
    for condition in ['correlated','independent']:
        logits,concepts,_=predictions(model,method,arrays,ids,backbone,condition)
        row={'accuracy':float((logits.argmax(1)==arrays['labels'][ids]).mean())}
        if concepts is not None:
            row.update(native_metrics(concepts,np.array(arrays['attrs'][ids])))
        rows[condition]=row
    score=np.mean([v['accuracy']+.2*v.get('mAP',0) for v in rows.values()])
    return float(score),rows


def fit_linear(arrays,splits,classes,seed,destination):
    torch.manual_seed(seed)
    model=nn.Linear(512,classes).cuda()
    features=F.normalize(tensor(arrays['tokens_correlated'],splits['train'])[:,0],dim=-1)
    labels=tensor(arrays['labels'],splits['train']).long()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.05,weight_decay=.001)
    best=-1;state=None;history=[];started=time.time()
    for step in range(400):
        loss=F.cross_entropy(model(features),labels)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%50==0:
            score,metrics=validation(model,'linear_probe',arrays,splits['val'],None)
            history.append({'step':step+1,'loss':float(loss),'val':metrics})
            if score>best:
                best=score;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.save({'model':state,'method':'linear_probe','classes':classes,'seed':seed},destination/'best.pt')
    write_json(destination/'history.json',{'seconds':time.time()-started,'rows':history})
    return model


def fit_salf(root,arrays,splits,classes,seed,destination,epochs=30):
    torch.manual_seed(seed)
    model=SALFConcepts(tensor(arrays['text']),classes).cuda()
    teacher=np.load(root/'salf_teacher.npy',mmap_mode='r')
    optimizer=torch.optim.Adam(model.projection.parameters(),lr=.005)
    best=-1;state=None;history=[];started=time.time()
    for epoch in range(epochs):
        values=[]
        for ids in np.array_split(np.random.default_rng(seed+epoch).permutation(splits['train']),8):
            _,maps=model.recognize(tensor(arrays['tokens_correlated'],ids))
            loss=-cos_cubed(tensor(teacher,ids),maps).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();values.append(float(loss))
        with torch.no_grad():
            _,maps=model.recognize(tensor(arrays['tokens_correlated'],splits['val']))
            score=float(cos_cubed(tensor(teacher,splits['val']),maps).mean())
        history.append({'epoch':epoch+1,'alignment_loss':float(np.mean(values)),'validation_alignment':score})
        if score>best:
            best=score;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.projection.requires_grad_(False)
    with torch.no_grad():
        concepts,_=model.recognize(tensor(arrays['tokens_correlated'],splits['train']))
        model.concept_center.copy_(concepts.mean(0));model.concept_scale.copy_(concepts.std(0).clamp_min(.01))
        x=(concepts-model.concept_center)/model.concept_scale
    y=tensor(arrays['labels'],splits['train']).long()
    # Proximal elastic-net optimization of the same convex final-layer objective.
    model.classifier.weight.data.zero_();model.classifier.bias.data.zero_()
    best=-1;state=None
    for step in range(1000):
        loss=F.cross_entropy(model.classifier(x),y)+.0001*model.classifier.weight.square().sum()
        gw,gb=torch.autograd.grad(loss,[model.classifier.weight,model.classifier.bias])
        with torch.no_grad():
            model.classifier.weight.copy_(F.softshrink(model.classifier.weight-.2*gw,lambd=.2*.001))
            model.classifier.bias.sub_(.2*gb)
        if (step+1)%100==0:
            score,metrics=validation(model,'salf',arrays,splits['val'],None)
            if score>best:
                best=score;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.save({'model':state,'method':'salf','classes':classes,'seed':seed},destination/'best.pt')
    write_json(destination/'history.json',{'seconds':time.time()-started,'rows':history,
               'teacher':'7x7 red-circle CLIP scores, radius32','loss':'zero-centered elementwise cubic cosine, across images',
               'head':'softmax spatial pooling; standardized activations; proximal elastic-net classifier'})
    return model


def warmup(arrays,splits,classes,seed,destination,epochs=30,spatial_noise_prior=False):
    checkpoint=destination/'warmup.pt'
    torch.manual_seed(seed)
    model=SharedSpatialCBM(tensor(arrays['text']),classes,spatial_noise_prior=spatial_noise_prior).cuda()
    if checkpoint.exists():
        restored=SharedSpatialCBM.from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=False))
        if restored.spatial_noise_prior != spatial_noise_prior:
            raise ValueError('Warmup spatial-noise-prior differs; choose a new --run.')
        return restored.cuda()
    priors=tensor(arrays['attrs'],splits['train']).mean(0).clamp(.02,.98)
    model.concept_bias.data.copy_(priors.logit())
    model.concept_center.copy_(priors);model.concept_scale.copy_((priors*(1-priors)).sqrt().clamp_min(.1))
    model.classifier.weight.data.zero_();model.classifier.bias.data.zero_()
    positive_weight=((1-priors)/priors).clamp(1,10)
    optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('classifier.')],'lr':.008},
                                {'params':model.classifier.parameters(),'lr':.03}],weight_decay=.0001)
    history=[];started=time.time();best=-1;state=None
    for epoch in range(epochs):
        losses=[]
        for ids in np.array_split(np.random.default_rng(seed+epoch).permutation(splits['train']),8):
            c,maps,fg=model.recognize(tensor(arrays['tokens_correlated'],ids))
            target=tensor(arrays['attrs'],ids);y=tensor(arrays['labels'],ids).long()
            semantic=F.binary_cross_entropy_with_logits(c,target,pos_weight=positive_weight)
            loss=semantic+.5*region_loss(maps,fg,arrays,ids)+F.cross_entropy(model.classify(c.detach()),y)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss))
        if (epoch+1)%5==0:
            with torch.no_grad():
                c,_,_=model.recognize(tensor(arrays['tokens_independent'],splits['val']))
                metrics=native_metrics(c.sigmoid().cpu().numpy(),np.array(arrays['attrs'][splits['val']]))
            score=metrics['mAP']
            row={'phase':'cached_warmup','epoch':epoch+1,'loss':float(np.mean(losses)),
                 'validation':metrics,'elapsed_s':round(time.time()-started,1)}
            history.append(row);print(json.dumps(row),flush=True)
            if score>best:
                best=score;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.save({'model':state,'model_config':model.configuration(),'seed':seed,'classes':classes},checkpoint)
    write_json(destination/'warmup_history.json',history)
    return model


def fit_noise(base,method,arrays,splits,backbone,seed,destination,epochs=10,batch=64):
    torch.manual_seed(seed)
    model=copy.deepcopy(base);model.freeze_recognizer();model.train()
    priors=tensor(arrays['attrs'],splits['train']).mean(0).clamp(.02,.98)
    pos_weight=((1-priors)/priors).clamp(1,10)
    optimizer=torch.optim.AdamW([{'params':model.generator.parameters(),'lr':.005},
                                {'params':[model.readout,model.concept_bias],'lr':.005},
                                {'params':model.classifier.parameters(),'lr':.02}],weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda');best=-1;state=None;history=[];started=time.time()
    for epoch in range(epochs):
        model.train();losses=[]
        order=np.random.default_rng(seed+epoch).permutation(splits['train'])
        for start in range(0,len(order),batch):
            ids=order[start:start+batch];original=tensor(arrays['tokens_correlated'],ids)
            rgb=tensor(arrays['rgb_correlated'],ids)/255
            processed,mu,sigma,fg=model.perturb(rgb,original,method)
            with torch.autocast('cuda',dtype=torch.float16):
                z=model.encode_view(backbone,normalize(processed),fg)
            c,_,_=model.recognize(z,foreground_prior=fg)
            semantic=F.binary_cross_entropy_with_logits(c,tensor(arrays['attrs'],ids),pos_weight=pos_weight)
            task=F.cross_entropy(model.classify(c.detach()),tensor(arrays['labels'],ids).long())
            loss=3*semantic+task+model.perturbation_penalty(mu,sigma,fg,method)
            if epoch==0 and start==0:
                gradient=torch.autograd.grad(semantic,model.generator.weight,retain_graph=True)[0]
                audit={'mean_concept_gradient_norm':float(gradient[:3].norm()),
                       'std_concept_gradient_norm':float(gradient[3].norm()),'supervised_concepts_per_view':model.k,
                       'encoded_processed_views_per_training_image':1,'frozen_backbone_parameters':sum(p.numel() for p in backbone.visual.parameters())}
                assert audit['mean_concept_gradient_norm']>0
                if method=='noise':assert audit['std_concept_gradient_norm']>0
                write_json(destination/'gradient_audit.json',audit)
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward()
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
            scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
        score,metrics=validation(model,method,arrays,splits['val'],backbone)
        row={'phase':method,'epoch':epoch+1,'loss':float(np.mean(losses)),'val':metrics,
             'elapsed_s':round(time.time()-started,1)}
        history.append(row);write_json(destination/'history.json',history);print(json.dumps(row),flush=True)
        if score>best:
            best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            torch.save({'model':state,'model_config':model.configuration(),
                        'method':method,'seed':seed,'classes':base.classifier.out_features,
                        'epoch':best_epoch,'validation':metrics},destination/'best.pt')
    assert all(p.grad is None for p in backbone.parameters())
    model.load_state_dict(state)
    return model


def evaluate(model,method,arrays,splits,backbone,destination,white=False):
    results={}
    thresholds=None
    if method!='linear_probe':
        _,val_concepts,_=predictions(model,method,arrays,splits['val'],backbone,'independent')
        thresholds=concept_thresholds(val_concepts,np.array(arrays['attrs'][splits['val']]))
        np.save(destination/'concept_thresholds.npy',thresholds)
    for condition in (['original'] if white else ['original','correlated','independent','reversed']):
        output_condition='white' if white else condition
        started=time.time()
        logits,concepts,info=predictions(model,method,arrays,splits['test'],backbone,condition)
        labels=np.array(arrays['labels'][splits['test']]);pred=logits.argmax(1)
        groups=[labels==j for j in np.unique(labels)]
        row={'accuracy':float((pred==labels).mean()),'worst_group_accuracy':float(min((pred[g]==labels[g]).mean() for g in groups)),
             'prediction_seconds':time.time()-started,'n':len(labels)}
        row.update(info)
        np.save(destination/f'{output_condition}_logits.npy',logits)
        if concepts is not None:
            row.update(native_metrics(concepts,np.array(arrays['attrs'][splits['test']]),thresholds))
            np.save(destination/f'{output_condition}_concepts.npy',concepts)
        results[output_condition]=row
    if method!='linear_probe':
        weight=model.classifier.weight.detach()
        results['NEC']=float((weight.abs()>1e-5).sum(1).float().mean())
        if not white:
            reference=np.load(destination/'original_concepts.npy')
            shifted=np.load(destination/'reversed_concepts.npy')
            scale=reference.std(0).clip(.03)
            results['background_concept_drift']=float(np.mean(np.abs(shifted-reference)/scale))
    results['white_background']=white
    results['parameters']=sum(p.numel() for p in model.parameters())
    results['inference_encodes']=2 if method in ['mean','noise'] else 1
    if method in ['mean','noise']:
        results['spatial_noise_prior']=model.spatial_noise_prior
    write_json(destination/'results.json',results)
    print(json.dumps({'method':method,'results':results}),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['rival10','cub','waterbirds'],required=True)
    p.add_argument('--phase',choices=['prepare','train'],default='train')
    p.add_argument('--methods',nargs='+',default=['linear_probe','salf','mean','noise'],choices=['linear_probe','salf','mean','noise'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--run',default=None)
    p.add_argument('--spatial-noise-prior',action=argparse.BooleanOptionalAction,default=False,
                   help='Enable foreground/concept mean gates, background scale gate and background-scale prior (default: off).')
    p.add_argument('--white-background',action='store_true',help='Train and test mean/noise controls with native foreground on white.')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--warmup-epochs',type=int,default=30)
    a=p.parse_args();torch.set_num_threads(6);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.white_background and (a.phase!='train' or any(m not in ['mean','noise'] for m in a.methods)):
        p.error('--white-background requires --phase train --methods mean noise (or either method).')
    a.run=a.run or (('v4_prior_on' if a.spatial_noise_prior else 'v4_prior_off')+('_white' if a.white_background else ''))
    previous=ROOT/a.dataset/a.run/f'seed{a.seed}'/'protocol.json'
    if a.phase=='train' and previous.exists():
        saved=json.loads(previous.read_text())['args'].get('spatial_noise_prior',True)
        if saved != a.spatial_noise_prior:
            raise ValueError('Existing run uses a different spatial-noise-prior setting; choose a new --run.')
        if json.loads(previous.read_text())['args'].get('white_background',False) != a.white_background:
            raise ValueError('Existing run uses a different background setting; choose a new --run.')
    backbone,_=clip.load('ViT-B/16',device='cuda',jit=False)
    backbone.float().eval().requires_grad_(False)
    if a.phase=='prepare':
        prepare(a.dataset,backbone);prepare_salf(a.dataset,backbone);return
    root,arrays,splits,protocol=load(a.dataset)
    if a.white_background:
        arrays=white_background(a.dataset,backbone,arrays,protocol)
    classes=len(protocol['classes'])
    common=ROOT/a.dataset/a.run/f'seed{a.seed}';common.mkdir(parents=True,exist_ok=True)
    write_json(common/'protocol.json',{'args':vars(a),'data_protocol':str(root/'protocol.json'),
               'backbone':'ViT-B/16','training_condition':'white' if a.white_background else 'correlated',
               'validation_conditions':['white'] if a.white_background else ['correlated','independent'],
               'white_background':a.white_background,
               'noise_training_and_inference':'single shared RGB mean+noise view; no gray background',
               'spatial_noise_prior':a.spatial_noise_prior,
               'spatial_noise_prior_effect':'foreground/concept mean gates, background scale gate and background sigma=0.18 regularization',
               'foreground_attention':'parameter-free 2*log(predicted foreground) bias in all second-encode ViT attention layers',
               'noise_distribution':'unit-marginal-variance correlated Gaussian; 95% bilinear14x14 + 5% pixel-independent',
               'semantic_head':'native concept BCE; classifier CE stops at concepts',
               'checkpoint_selection':'mean validation accuracy + 0.2 * concept mAP'})
    for method in a.methods:
        destination=common/method;destination.mkdir(exist_ok=True)
        if (destination/'results.json').exists():
            if (destination/'efficiency.json').exists() and (a.white_background or method=='linear_probe' or (destination/'background_leakage.json').exists()):
                continue
            checkpoint=torch.load(destination/'best.pt',map_location='cuda',weights_only=False)
            if method=='linear_probe':model=nn.Linear(512,classes).cuda()
            elif method=='salf':model=SALFConcepts(tensor(arrays['text']),classes).cuda()
            else:
                model=SharedSpatialCBM.from_checkpoint(checkpoint).cuda()
                if model.spatial_noise_prior != a.spatial_noise_prior:
                    raise ValueError('Checkpoint spatial-noise-prior differs from the requested run.')
            model.load_state_dict(checkpoint['model'])
        elif method=='linear_probe':model=fit_linear(arrays,splits,classes,a.seed,destination)
        elif method=='salf':
            prepare_salf(a.dataset,backbone)
            model=fit_salf(root,arrays,splits,classes,a.seed,destination)
        else:
            base=warmup(arrays,splits,classes,a.seed,common,a.warmup_epochs,a.spatial_noise_prior)
            model=fit_noise(base,method,arrays,splits,backbone,a.seed,destination,a.epochs)
        if not (destination/'results.json').exists():
            evaluate(model,method,arrays,splits,backbone,destination,a.white_background)
        from .analysis import benchmark,background_leakage
        if not (destination/'efficiency.json').exists():
            benchmark(model,method,backbone,tensor(arrays['rgb_original'],splits['test'][:32])/255,destination)
        if not a.white_background and method!='linear_probe' and not (destination/'background_leakage.json').exists():
            leakage=background_leakage(destination,arrays,splits)
            print(json.dumps({'method':method,'leakage':leakage}),flush=True)
    write_json(common/'COMPLETE.json',{'methods':a.methods,'seed':a.seed,'completed_utc':time.time()})


if __name__=='__main__':main()
