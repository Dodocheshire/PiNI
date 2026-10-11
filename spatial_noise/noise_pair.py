"""Direct paired background consistency with matched image-view/step budget.

32 original/changed pairs form one 64-view RN18 call and one backward.
Both members use identical epsilon. Five epochs match the original ten
single-view epochs on 18987 train images: 189870 views and 2970 steps.
"""
import argparse
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .cub_aligned import tensor
from .cub_noise import evaluate, inputs
from .data import write_json
from .model import SharedSpatialCBM
from .grounding import GroundedNoiseGenerator
from .rn18_datasets import Experiment



def read(path):
    return json.loads(path.read_text())

def initialize(e, stage):
    path = e.root/'prior_off/seed42'/stage/'mean/best.pt'
    state = torch.load(path, weights_only=True, map_location='cpu')
    cbm = e.model().cuda()
    gen = GroundedNoiseGenerator(regions=len(e.phrases), mean_limit=.03,
                                 std_limit=.1, spatial_noise_prior=False).cuda()
    cbm.load_state_dict(state['cbm'])
    gen.load_state_dict(state['generator'])
    return cbm, gen, path

@torch.no_grad()
def diagnostics(cbm, gen, arrays, grounded, ids, destination):
    result = {}
    for view in ['original', 'changed']:
        probabilities = torch.from_numpy(np.load(destination/(view+'_concepts.npy'))).sigmoid()
        variance = probabilities.std(0)
        eligible = np.asarray(arrays['background_eligible'][ids]) if 'background_eligible' in arrays else np.ones(len(ids), bool)
        totals = np.zeros(9)
        count = 0
        for offset in range(0, len(ids), 128):
            selected = ids[offset:offset+128]
            use = eligible[offset:offset+128]
            region = {k: tensor(v, selected) for k,v in grounded[view].items()}
            mu, sigma = gen.distribution(region)
            fg = tensor(arrays['foreground'], selected)[:,None]
            fields = []
            for mask in [fg, 1-fg]:
                denom = mask.sum((1,2,3)).clamp_min(1)
                fields.extend([(sigma*mask).sum((1,2,3))/denom,
                               ((sigma>.02)*mask).sum((1,2,3))/denom,
                               (sigma.square()*mask).sum((1,2,3))/denom,
                               (mu.abs().mean(1,keepdim=True)*mask).sum((1,2,3))/denom])
            fields.append((sigma>.02).float().mean((1,2,3)))
            values = torch.stack(fields, 1).cpu().numpy()[use]
            totals += values.sum(0)
            count += len(values)
        names = ['sigma_foreground','coverage_gt02_foreground','variance_foreground','mu_abs_foreground',
                 'sigma_background','coverage_gt02_background','variance_background','mu_abs_background','coverage_gt02_all']
        result[view] = dict(zip(names, (totals/max(1,count)).tolist()))
        result[view].update(concept_cross_image_std_mean=float(variance.mean()),
                            concept_cross_image_std_per_concept=variance.tolist(),
                            concept_bernoulli_uncertainty=float((probabilities*(1-probabilities)).mean()))
    return result


def paired_forward(backbone, cbm, gen, arrays, grounded, ids, method):
    original=tensor(arrays['rgb_original'],ids)/255
    changed=tensor(arrays['rgb_changed'],ids)/255
    rgb=torch.cat([original,changed])
    regions={k:torch.cat([tensor(grounded[v][k],ids) for v in ['original','changed']])
             for k in ['features','maps']}
    epsilon=None
    if method=='noise':
        draw=SharedSpatialCBM.sample_noise(original)
        epsilon=torch.cat([draw,draw])
        assert torch.equal(epsilon[:len(ids)],epsilon[len(ids):])
    perturbed,mu,sigma=gen(rgb,regions,method,epsilon=epsilon)
    with torch.autocast('cuda',dtype=torch.float16):
        features=backbone(perturbed)
    y,c=cbm(features)
    return y,c,mu,sigma


def train(e,method,stage,epochs=5,pairs=32):
    folder=e.root/'pair_explore'/stage/method
    if (folder/'COMPLETE.json').exists():
        result=read(folder/'results.json')
        assert result['epochs']==epochs and result['pairs_per_batch']==pairs
        return result
    folder.mkdir(parents=True,exist_ok=True)
    arrays,splits,protocol,grounded=inputs(experiment=e)
    ids=splits['train'] if stage=='selection' else np.concatenate([splits['train'],splits['val']])
    assert not set(ids)&set(splits['test'])
    torch.manual_seed(42)
    backbone=e.backbone().cuda()
    cbm,gen,initial=initialize(e,stage)
    labels=tensor(arrays['labels']).long();target=tensor(arrays['attrs'])
    priors=target[ids].mean(0).clamp(.02,.98);pos=((1-priors)/priors).clamp(1,10)
    opt=torch.optim.AdamW([{'params':cbm.parameters(),'lr':1e-4},
                          {'params':gen.parameters(),'lr':1e-3}],weight_decay=.001)
    scaler=torch.amp.GradScaler('cuda',init_scale=256)
    rng=np.random.default_rng(42)
    history=[];best=None;start=time.time()
    total_views=total_steps=total_backward=rn18_calls=0
    measured={'calls':0,'views':0}
    def capture(module,args,out):
        measured['calls']+=1;measured['views']+=len(args[0])
    hook=backbone.register_forward_hook(capture)
    for epoch in range(1,epochs+1):
        torch.manual_seed(420000+epoch)
        order=rng.permutation(ids);sums=np.zeros(5)
        before=measured.copy()
        for offset in range(0,len(ids),pairs):
            selected=order[offset:offset+pairs]
            y,c,mu,sigma=paired_forward(backbone,cbm,gen,arrays,grounded,selected,method)
            y_target=torch.cat([labels[selected],labels[selected]])
            c_target=torch.cat([target[selected],target[selected]])
            ce=F.cross_entropy(y,y_target)
            bce=F.binary_cross_entropy_with_logits(c,c_target,pos_weight=pos)
            probabilities=c.sigmoid()
            consistency=F.mse_loss(probabilities[:len(selected)],probabilities[len(selected):])
            entropy=torch.log(sigma.clamp_min(.002)/.102).mean() if method=='noise' else sigma.new_zeros(())
            loss=ce+bce+30*consistency+.01*(mu.square().mean()+sigma.square().mean())-.01*entropy
            opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(cbm.parameters(),5);torch.nn.utils.clip_grad_norm_(gen.parameters(),5)
            scaler.step(opt);scaler.update()
            total_views+=2*len(selected);total_steps+=1;total_backward+=1
            sums+=np.array([loss.item(),ce.item(),bce.item(),consistency.item(),entropy.item()])*len(selected)
        calls=measured['calls']-before['calls'];views=measured['views']-before['views']
        assert calls==math.ceil(len(ids)/pairs) and views==2*len(ids)
        rn18_calls+=calls
        assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
        entry=dict(zip(['loss','classification_loss','concept_loss','consistency_loss','log_sigma'],(sums/len(ids)).tolist()))
        entry.update(epoch=epoch,seconds=time.time()-start,training_rn18_calls=calls,training_image_views=views)
        if stage=='selection':
            metrics=evaluate(backbone,cbm,gen,arrays,grounded,splits['val'],method)
            entry['validation']=metrics
            score=(metrics['original']['accuracy']>=.995,metrics['changed']['accuracy'],-metrics['concept_drift_mae'])
        else:
            score=(-entry['loss'],)
        if best is None or score>best:
            best=score;selected_epoch=epoch
            torch.save({'cbm':copy.deepcopy(cbm.state_dict()),'generator':copy.deepcopy(gen.state_dict()),
                        'seed':42,'method':method,'stage':stage,'selected_epoch':epoch,
                        'spatial_noise_prior':False,'mean_limit':.03,'std_limit':.1},folder/'best.pt')
        history.append(entry);write_json(folder/'history.json',history)
        print(json.dumps({'method':method,'stage':stage,**entry}),flush=True)
    hook.remove()
    assert total_steps==total_backward==rn18_calls==epochs*math.ceil(len(ids)/pairs)
    assert total_views==2*epochs*len(ids)
    state=torch.load(folder/'best.pt',weights_only=True)
    cbm.load_state_dict(state['cbm']);gen.load_state_dict(state['generator'])
    split='val' if stage=='selection' else 'test'
    metrics=evaluate(backbone,cbm,gen,arrays,grounded,splits[split],method,save=folder)
    result={'method':method,'stage':stage,'seed':42,'epochs':epochs,'pairs_per_batch':pairs,
            'fit_images':len(ids),'selected_epoch':selected_epoch,'initial_checkpoint':str(initial),
            'initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
            'loss':'CE both views + weighted BCE both views + 30*MSE(p_original,p_changed) + .01*(mu^2+sigma^2) - .01*log(sigma/.102) [noise only]',
            'paired_epsilon':True,'spatial_noise_prior':False,
            'total_image_views':total_views,'total_optimizer_steps':total_steps,
            'total_backward_calls':total_backward,'measured_training_rn18_calls':rn18_calls,
            'last_batch_pairs':(len(ids)-1)%pairs+1,'last_batch_views':2*((len(ids)-1)%pairs+1),
            'comparison_single_view_epochs':2*epochs,
            'comparison_single_view_batch':2*pairs,
            'comparison_single_view_image_views':2*epochs*len(ids),
            'comparison_single_view_steps':2*epochs*math.ceil(len(ids)/(2*pairs)),
            'classifier_nec':float((cbm.classifier.weight.abs()>1e-5).sum(1).float().mean()),
            'online_encodes':{'grounding_dino':1,'resnet18':1},
            'training_backbone_frozen_verified':True,'seconds':time.time()-start,
            'diagnostic_sigma_semantics':'latent generator std; mean arm applies zero std',
            'validation' if stage=='selection' else 'test':metrics,
            'diagnostics':diagnostics(cbm,gen,arrays,grounded,splits[split],folder)}
    write_json(folder/'results.json',result);write_json(folder/'COMPLETE.json',{'method':method,'stage':stage})
    print(json.dumps({'completed':method,**result}),flush=True)
    return result


def compare(e):
    folder=e.root/'pair_explore/selection'
    if not all((folder/m/'COMPLETE.json').exists() for m in ['mean','noise']):
        return
    arrays,splits,_=e.load();ids=splits['val'];eligible=np.asarray(arrays['background_eligible'][ids])
    probabilities={}
    for m in ['mean','noise']:
        probabilities[m]={v:torch.from_numpy(np.load(folder/m/(v+'_concepts.npy'))).sigmoid().numpy()
                          for v in ['original','changed']}
    drifts={m:np.abs(p['original']-p['changed']).mean(1)[eligible] for m,p in probabilities.items()}
    delta=drifts['noise']-drifts['mean']
    rng=np.random.default_rng(42);sampled=rng.integers(0,len(delta),(2000,len(delta)))
    labels=np.asarray(arrays['labels'][ids])[eligible]
    correctness={m:np.load(folder/m/'changed_logits.npy').argmax(1)[eligible]==labels for m in probabilities}
    result={'split':'validation only','noise_minus_mean_drift':float(delta.mean()),
            'paired_image_bootstrap_95ci':np.quantile(delta[sampled].mean(1),[.025,.975]).tolist(),
            'noise_corrected_mean_errors':int((~correctness['mean']&correctness['noise']).sum()),
            'noise_regressed_mean_correct':int((correctness['mean']&~correctness['noise']).sum()),
            'images':len(delta),'bootstrap_samples':2000,'bootstrap_seed':42}
    write_json(folder/'paired_comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['selection','refit'],default='selection')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--pairs',type=int,default=32)
    parser.add_argument('--methods',nargs='+',choices=['mean','noise'],default=['mean','noise'])
    args=parser.parse_args();torch.set_num_threads(3)
    e=Experiment('rival10')
    for method in args.methods:
        train(e,method,args.phase,args.epochs,args.pairs)
    if args.phase=='selection':
        print(json.dumps({'comparison':compare(e)}),flush=True)


if __name__=='__main__':
    main()
