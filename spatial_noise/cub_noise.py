"""Matched clean/mean/noise adaptation on the official full CUB protocol."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from .cub_aligned import ROOT, BirdBackbone, SimpleCBM, load, tensor
from .data import write_json
from .grounding import GroundedNoiseGenerator, FrozenGrounder
from .model import SharedSpatialCBM


def inputs(with_regions=True, experiment=None):
    root = ROOT if experiment is None else experiment.root
    arrays, splits, protocol = load() if experiment is None else experiment.load()
    if (root/'cache/background_eligible.npy').exists():
        arrays['background_eligible']=np.load(root/'cache/background_eligible.npy',mmap_mode='r')
    assert (root/'cache/BACKGROUND_COMPLETE.json').exists()
    if with_regions:
        assert all((root/'regions'/v/'COMPLETE.json').exists() for v in ['original','changed'])
    arrays['rgb_changed'] = np.load(root/'cache/rgb_changed.npy', mmap_mode='r')
    arrays['features_changed'] = np.load(root/'cache/features_changed.npy', mmap_mode='r')
    grounded = {v:{k:np.load(root/'regions'/v/(k+'.npy'), mmap_mode='r')
                  for k in ['features','maps']} for v in ['original','changed']} if with_regions else {}
    return arrays, splits, protocol, grounded


def initialize(method, seed, refit, spatial_prior=False, experiment=None):
    torch.manual_seed(seed)
    root = ROOT if experiment is None else experiment.root
    folder = 'simple_refit' if refit else 'simple'
    cbm = (SimpleCBM() if experiment is None else experiment.model()).cuda()
    cbm.load_state_dict(torch.load(root/folder/f'seed{seed}/best.pt', weights_only=True)['model'])
    regions = 12 if experiment is None else len(experiment.phrases)
    generator = GroundedNoiseGenerator(regions=regions, mean_limit=.03, std_limit=.1, spatial_noise_prior=spatial_prior).cuda()
    generator.decoder[-1].bias.data[3] = -3
    return cbm, generator


def forward(backbone, cbm, generator, arrays, grounded, ids, view, method):
    if method == 'clean':
        feature = tensor(arrays['features_'+view], ids)
        y,c = cbm(feature)
        return y,c,None,None
    rgb = tensor(arrays['rgb_'+view], ids)/255
    region = {k:tensor(v,ids) for k,v in grounded[view].items()}
    perturbed,mu,sigma = generator(rgb, region, method)
    with torch.autocast('cuda', dtype=torch.float16):
        features = backbone(perturbed)
    y,c = cbm(features)
    return y,c,mu,sigma


@torch.no_grad()
def evaluate(backbone, cbm, generator, arrays, grounded, ids, method, seed=4242, batch=64, save=None):
    labels = tensor(arrays['labels'], ids).long(); target = tensor(arrays['attrs'],ids)
    predictions = {}; metrics = {}
    eligible = np.ones(len(ids), dtype=bool)
    if 'background_eligible' in arrays:
        eligible = np.asarray(arrays['background_eligible'][ids])
    # Identical random draws for the same original/changed image pair.
    for view in ['original','changed']:
        torch.manual_seed(seed)
        ys,cs = [],[]; spatial=[]
        for offset in range(0,len(ids),batch):
            batch_ids=ids[offset:offset+batch]
            y,c,mu,sigma = forward(backbone,cbm,generator,arrays,grounded,batch_ids,view,method)
            ys.append(y); cs.append(c)
            if mu is not None:
                fg=tensor(arrays['foreground'],batch_ids)[:,None]; bg=1-fg
                values=[]
                for field in [mu.abs().mean(1,keepdim=True),sigma.square()]:
                    values.extend([(field*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1) for mask in [fg,bg]])
                spatial.append(torch.stack(values,1))
        y = torch.cat(ys); c = torch.cat(cs); predictions[view] = (y,c)
        use = torch.as_tensor(eligible,device='cuda') if view=='changed' else torch.ones(len(ids),device='cuda',dtype=torch.bool)
        metrics[view] = {'accuracy':float((y[use].argmax(1)==labels[use]).float().mean()),
            'concept_accuracy':float(((c[use]>0)==target[use].bool()).float().mean()),
            'ce':float(F.cross_entropy(y[use],labels[use])), 'images':int(use.sum())}
        if spatial:
            values=torch.cat(spatial)[torch.as_tensor(eligible,device='cuda')].mean(0).tolist()
            metrics[view]['spatial']=dict(zip(['mean_magnitude_foreground','mean_magnitude_background',
                                              'variance_foreground','variance_background'],values))
        if save is not None:
            np.save(save/f'{view}_logits.npy',y.cpu().numpy())
            np.save(save/f'{view}_concepts.npy',c.cpu().numpy())
    a = predictions['original'][1].sigmoid()[eligible]; b = predictions['changed'][1].sigmoid()[eligible]
    metrics['concept_drift_mae'] = float((a-b).abs().mean())
    metrics['concept_drift_cosine'] = float((1-F.cosine_similarity(a,b)).mean())
    return metrics


def train(method, seed=42, epochs=10, batch=64, refit=False, spatial_prior=False, experiment=None):
    stage = 'refit' if refit else 'selection'
    tag = 'prior_on' if spatial_prior else 'prior_off'
    root = ROOT if experiment is None else experiment.root
    destination = root/tag/f'seed{seed}'/stage/method
    if (destination/'COMPLETE.json').exists(): return
    destination.mkdir(parents=True,exist_ok=True)
    arrays,splits,protocol,grounded = inputs(with_regions=method!='clean', experiment=experiment)
    ids = np.concatenate([splits['train'],splits['val']]) if refit else splits['train']
    cbm,generator = initialize(method,seed,refit,spatial_prior,experiment)
    backbone = (BirdBackbone() if experiment is None else experiment.backbone()).cuda()
    with torch.no_grad():
        reference = cbm.concept_logits(tensor(arrays['features_original']).mean((2,3))).sigmoid().detach()
    target = tensor(arrays['attrs']); labels = tensor(arrays['labels']).long()
    priors = target[ids].mean(0).clamp(.02,.98); pos = ((1-priors)/priors).clamp(1,10)
    groups = [{'params':cbm.parameters(),'lr':1e-4}]
    if method != 'clean': groups.append({'params':generator.parameters(),'lr':1e-3})
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    history=[]; best=-float('inf'); selected=0; started=time.time()
    numpy_rng = np.random.default_rng(seed)
    # All arms use the same images and views. Reference probabilities are cached,
    # so task, concept, and invariance losses share one differentiable RN18 pass.
    for epoch in range(1,epochs+1):
        torch.manual_seed(seed*10000+epoch)
        shuffled=numpy_rng.permutation(ids)
        sums=np.zeros(4); count=0
        for offset in range(0,len(shuffled),batch):
            selection=shuffled[offset:offset+batch]
            view='original' if (offset//batch+epoch)%2==0 else 'changed'
            y,c,mu,sigma = forward(backbone,cbm,generator,arrays,grounded,selection,view,method)
            classification=F.cross_entropy(y,labels[selection])
            concept=F.binary_cross_entropy_with_logits(c,target[selection],pos_weight=pos)
            consistency=F.mse_loss(c.sigmoid(),reference[selection])
            magnitude=(mu.square().mean()+sigma.square().mean()) if mu is not None else c.new_zeros(())
            loss=classification+concept+consistency+.01*magnitude
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cbm.parameters(),5)
            if method!='clean': torch.nn.utils.clip_grad_norm_(generator.parameters(),5)
            scaler.step(optimizer); scaler.update()
            sums += np.asarray([loss.item(),classification.item(),concept.item(),consistency.item()])*len(selection)
            count += len(selection)
        entry={'epoch':epoch,'loss':sums[0]/count,'classification_loss':sums[1]/count,
               'concept_loss':sums[2]/count,'consistency_loss':sums[3]/count,'seconds':time.time()-started}
        if refit:
            # No holdout remains after merging; save the lowest training objective.
            score=-entry['loss']
        else:
            metrics=evaluate(backbone,cbm,generator,arrays,grounded,splits['val'],method)
            entry['validation']=metrics
            # Changed-background accuracy is the target; original accuracy resolves ties.
            score=metrics['changed']['accuracy']+1e-3*metrics['original']['accuracy']
        if score>best:
            best=score;selected=epoch
            state={'cbm':copy.deepcopy(cbm.state_dict()),'generator':copy.deepcopy(generator.state_dict())}
            torch.save({**state,'method':method,'seed':seed,'stage':stage,'selected_epoch':epoch,
                'spatial_noise_prior':spatial_prior,'mean_limit':.03,'std_limit':.1},destination/'best.pt')
        history.append(entry);write_json(destination/'history.json',history)
        print(json.dumps({'stage':stage,'method':method,**entry}),flush=True)
    cbm.load_state_dict(state['cbm']);generator.load_state_dict(state['generator'])
    info={'method':method,'seed':seed,'stage':stage,'fit_images':len(ids),'epochs':epochs,
          'dataset':protocol['dataset'],'concepts':cbm.concepts.out_features,'classes':cbm.classifier.out_features,
          'selected_epoch':selected,'selection':'minimum training loss' if refit else 'changed-background validation accuracy; original tie break',
          'seconds':time.time()-started,'spatial_noise_prior':spatial_prior,
          'configuration':{'batch_size':batch,'cbm_lr':1e-4,'generator_lr':1e-3,'weight_decay':.001,
                           'mean_limit':.03,'std_limit':.1,'initial_std_bias':-3,'sigma_floor':.002,
                           'eval_seed':4242,'paired_epsilon':True,'detector_image_size':800},
          'loss':'class_CE + weighted_concept_BCE + MSE(concept_prob, cached_original_prob) + .01*mean(mu^2+sigma^2)',
          'trainable_parameters':sum(p.numel() for p in cbm.parameters())+(sum(p.numel() for p in generator.parameters()) if method!='clean' else 0)}
    if refit:
        info['test']=evaluate(backbone,cbm,generator,arrays,grounded,splits['test'],method,save=destination)
    else:
        info['validation']=evaluate(backbone,cbm,generator,arrays,grounded,splits['val'],method)
    write_json(destination/'results.json',info);write_json(destination/'COMPLETE.json',info)
    print(json.dumps({'stage':'completed',**info}),flush=True)


@torch.no_grad()
def simple_evaluation(seed=42, experiment=None):
    arrays,splits,protocol,grounded=inputs(with_regions=False, experiment=experiment)
    cbm,generator=initialize('clean',seed,True,experiment=experiment)
    root = ROOT if experiment is None else experiment.root
    destination=root/'simple_refit'/f'seed{seed}'
    metrics=evaluate(None,cbm,generator,arrays,grounded,splits['test'],'clean',save=destination)
    write_json(destination/'paired_results.json',metrics)
    print(json.dumps({'stage':'simple_paired','test':metrics}),flush=True)


def verify(seed=42, experiment=None):
    """Check live/cached parity and useful class/concept gradients to both outputs."""
    root = ROOT if experiment is None else experiment.root
    arrays,splits,protocol,grounded=inputs(experiment=experiment)
    cbm,generator=initialize('noise',seed,True,experiment=experiment)
    backbone=(BirdBackbone() if experiment is None else experiment.backbone()).cuda();ids=splits['val'][:4]
    rgb=tensor(arrays['rgb_original'],ids)/255
    region={k:tensor(v,ids) for k,v in grounded['original'].items()}
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        feature=backbone(rgb)
    cached=cbm(tensor(arrays['features_original'],ids))[1]
    eager=cbm(feature)[1]
    parity=float((cached-eager).abs().max())
    assert parity<.02,parity
    perturbed,mu,sigma=generator(rgb,region,'noise')
    with torch.autocast('cuda',dtype=torch.float16):
        feature=backbone(perturbed)
    y,c=cbm(feature)
    losses={'classification':F.cross_entropy(y,tensor(arrays['labels'],ids).long()),
            'concept':F.binary_cross_entropy_with_logits(c,tensor(arrays['attrs'],ids))}
    gradients={}
    for key,loss in losses.items():
        # Match the scaled backward used by the actual mixed-precision trainer.
        g=torch.autograd.grad(loss*256,generator.decoder[-1].bias,retain_graph=True)[0]/256
        gradients[key]={'mu':float(g[:3].abs().sum()),'sigma':float(g[3].abs())}
        assert torch.isfinite(g).all() and g[:3].abs().sum()>0 and g[3].abs()>0
    assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
    changed=np.asarray(arrays['rgb_changed'][ids]); original=np.asarray(arrays['rgb_original'][ids])
    mask=np.asarray(arrays['foreground'][ids])[:,None]
    assert np.array_equal(changed*mask,original*mask)
    result={'cached_live_max_concept_logit_difference':parity,'gradients':gradients,
            'backbone_frozen':True,'background_change_preserves_foreground':True,
            'initial_sigma':float(sigma.mean()),'spatial_noise_prior':False}
    # One actual online pass verifies that no image-specific annotations are needed.
    from PIL import Image
    from .grounding import BIRD_PHRASES
    phrases=BIRD_PHRASES if experiment is None else experiment.phrases
    precision='fp32' if experiment is None else experiment.detector_precision
    grounder=FrozenGrounder(phrases=phrases,image_size=800,precision=precision)
    online=grounder([Image.fromarray(np.asarray(arrays['rgb_original'][ids[0]]).transpose(1,2,0))])
    calls=[]
    hook=backbone.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad():
        view,_,_=generator(rgb[:1],online,'noise')
        with torch.autocast('cuda',dtype=torch.float16): online_feature=backbone(view)
        online_y,online_c=cbm(online_feature)
    hook.remove();grounder.close()
    assert online['image_encodes']==1 and len(calls)==1
    assert online_y.shape==(1,cbm.classifier.out_features) and online_c.shape==(1,cbm.concepts.out_features)
    result['online_encodes']={'grounding_dino':online['image_encodes'],'resnet18':len(calls)}
    result['online_inputs']='RGB image and 12 fixed region names only'
    result['detector_precision']=precision
    write_json(root/'verification.json',result)
    print(json.dumps({'stage':'verification',**result}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=['clean','mean','noise'],default=['clean','mean','noise'])
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--stage',choices=['selection','refit','all'],default='all')
    parser.add_argument('--spatial-noise-prior',action=argparse.BooleanOptionalAction,default=False)
    args=parser.parse_args();torch.set_num_threads(6)
    verify(args.seed)
    for method in args.methods:
        if args.stage in ['selection','all']:
            train(method,args.seed,args.epochs,args.batch,False,args.spatial_noise_prior)
        if args.stage in ['refit','all']:
            train(method,args.seed,args.epochs,args.batch,True,args.spatial_noise_prior)
    simple_evaluation(args.seed)


if __name__=='__main__':main()
