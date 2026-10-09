"""Real RGB concept-conditional perturbation, frozen ViT, strict 18-value CBM.

Training samples one concept view/image. Joint mode uses current clean
predictions as detached anchors for the other 17 values, an explicitly recorded
sampled-view approximation; final evaluation always measures all 18 real views.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from clip import clip
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from .data import normalize
from .models import ConditionalRGBNoise, SpatialConcepts
from .train_spatial import batch_tensor, fit_head, head_predict, load_cache


def fixed_subset(ids, labels, size, seed=44):
    rng=np.random.default_rng(seed);parts=[]
    for k in range(10):
        group=ids[labels[ids]==k];n=min(len(group),round(size*len(group)/len(ids)))
        parts.extend(rng.choice(group,n,replace=False).tolist())
    return np.array(sorted(parts))


def native_metrics(probs, truth):
    pred=probs>=.5;tp=(pred*truth).sum(0);fp=(pred*(1-truth)).sum(0);fn=((1-pred)*truth).sum(0)
    return dict(mAP=float(average_precision_score(truth,probs,average='macro')),
                macro_precision=float(np.mean(tp/np.maximum(tp+fp,1))),
                macro_recall=float(np.mean(tp/np.maximum(tp+fn,1))))


@torch.no_grad()
def predict_views(noise, backbone, arrays, rgb_cache, ids, mode, suppress, batch=128, draws=1, seed=8101):
    noise.eval();backbone.eval();torch.manual_seed(seed)
    # Each native concept is recognized on its own actual pixel-space view.
    probs=np.empty((len(ids),18),dtype='float32');t=time.time()
    flattened=np.arange(len(ids)*18)
    for start in range(0,len(flattened),batch):
        indices=flattened[start:start+batch];pos=indices//18;ks=indices%18;ii=ids[pos]
        tokens=batch_tensor(arrays['tokens'],ii);rgb=batch_tensor(rgb_cache,ii)/255
        k=torch.tensor(ks,device='cuda');samples=[]
        for _ in range(draws):
            x,mu,sigma,gate=noise.perturb(rgb,tokens,k,mode=mode,suppress=suppress)
            with torch.autocast('cuda',dtype=torch.float16):
                out=backbone.encode_image(normalize(x),spatial=True)
                logits,_,_=noise.concepts(out)
                samples.append(logits[torch.arange(len(k),device='cuda'),k].float().sigmoid())
        probs[pos,ks]=torch.stack(samples).mean(0).cpu().numpy()
        if start%12800==0:
            print(json.dumps(dict(stage='predict_views',mode=mode,done=start,total=len(flattened),elapsed_s=round(time.time()-t,1))),flush=True)
    return probs


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='artifacts/rival_vitl14')
    p.add_argument('--clean',default='artifacts/spatial_clean');p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['clean','mask','mean','noise'],default='noise')
    p.add_argument('--order',choices=['independent','sequential','joint'],default='independent')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--train-size',type=int,default=6000)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--independent-heads',action='store_true');p.add_argument('--suppress',action='store_true')
    p.add_argument('--concept-local',action='store_true')
    p.add_argument('--mc-test',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    arrays,splits,labels,records=load_cache(a.cache);rgb=np.load(Path(a.cache)/'rgb.npy',mmap_mode='r')
    concepts=SpatialConcepts(torch.tensor(np.load(Path(a.cache)/'text.npy'),device='cuda')).cuda()
    concepts.load_state_dict(torch.load(Path(a.clean)/'best.pt',map_location='cuda',weights_only=False)['model'])
    noise=ConditionalRGBNoise(concepts,a.independent_heads,a.concept_local).cuda()
    if a.order=='independent':noise.concepts.requires_grad_(False)
    backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    train=fixed_subset(splits['train'],labels,a.train_size);val_small=fixed_subset(splits['val'],labels,500,seed=45)
    (out/'protocol.json').write_text(json.dumps(dict(args=vars(a),train_ids=train.tolist(),val_selection_ids=val_small.tolist(),
        inference_annotation_inputs=False,noise_units='RGB [0,1]',selection='native concept validation mAP on fixed subset',
        sigma_constraints='0.008+(1-predicted foreground)*(0.08+0.12*sigmoid(std head))'+(' + foreground*(1-concept relevance)*0.025' if a.concept_local else ''),
        locality_regularization='positive-mask Dice weight 0.25 plus BCE' if a.concept_local else 'mask BCE only',
        joint_approximation='one sampled real concept view; other 17 current clean predictions detached for task loss'),indent=2))
    initial_head=torch.load(Path(a.clean)/'head.pt',map_location='cuda',weights_only=False)
    hw=initial_head['weight'].detach().clone().requires_grad_(a.order=='joint')
    hb=initial_head['bias'].detach().clone().requires_grad_(a.order=='joint')
    params=[p for p in noise.parameters() if p.requires_grad]
    if a.order=='joint':params += [hw,hb]
    optim=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
    scaler=torch.amp.GradScaler('cuda');sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,max(a.epochs,1))
    history=[];best=-1;t=time.time()
    if a.mode in ['clean','mask'] and a.order=='independent':a.epochs=0
    for epoch in range(a.epochs):
        noise.train();rng=np.random.default_rng(180+epoch);order=rng.permutation(train);losses=[]
        for start in range(0,len(order),a.batch):
            ids=order[start:start+a.batch];tokens=batch_tensor(arrays['tokens'],ids);images=batch_tensor(rgb,ids)/255
            ks=torch.tensor(rng.integers(0,18,size=len(ids)),device='cuda');ii=torch.arange(len(ids),device='cuda')
            target=batch_tensor(arrays['attrs'],ids);optim.zero_grad()
            x,mu,sigma,gate=noise.perturb(images,tokens,ks,mode=a.mode,suppress=a.suppress)
            with torch.autocast('cuda',dtype=torch.float16):
                visual=backbone.encode_image(normalize(x),spatial=True)
                all_logits,_,_=noise.concepts(visual)
                loss=F.binary_cross_entropy_with_logits(all_logits[ii,ks].float(),target[ii,ks])
                if a.order!='independent':
                    anchors,mask,fg=noise.concepts(tokens)
                    gt=batch_tensor(arrays['masks'],ids);valid=batch_tensor(arrays['mask_valid'],ids)
                    pred=torch.cat([mask,fg[:,None]],1)
                    ml=(F.binary_cross_entropy_with_logits(pred.float(),gt,reduction='none').mean((-1,-2))*valid).sum()/valid.sum().clamp_min(1)
                    loss=loss+.25*ml+.25*F.binary_cross_entropy_with_logits(anchors.float(),target)
                    if a.concept_local:
                        # Keep the local stage's positive-region constraint
                        # during adaptation; absent/tiny masks can dominate BCE.
                        present=(gt.sum((-1,-2))>0).float()*valid;prob=pred.float().sigmoid()
                        dice=1-(2*(prob*gt).sum((-1,-2))+1)/(prob.sum((-1,-2))+gt.sum((-1,-2))+1)
                        loss=loss+.25*(dice*present).sum()/present.sum().clamp_min(1)
                    if a.order=='joint':
                        cbm=anchors.float().sigmoid().detach().clone();cbm[ii,ks]=all_logits[ii,ks].float().sigmoid()
                        task=F.linear((cbm-initial_head['center'])/initial_head['scale'],hw,hb)
                        loss=loss+.25*F.cross_entropy(task,torch.tensor(labels[ids],device='cuda'))+.003*hw.abs().sum()
            loss=loss+.005*mu.square().mean()
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(params,5)
            scaler.step(optim);scaler.update();losses.append(float(loss.detach()))
        sched.step();row=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed_s=round(time.time()-t,1))
        if (epoch+1)%2==0 or epoch+1==a.epochs:
            probs=predict_views(noise,backbone,arrays,rgb,val_small,a.mode,a.suppress,batch=a.eval_batch)
            metrics=native_metrics(probs,np.array(arrays['attrs'][val_small]));row['val']=metrics
            if metrics['mAP']>best:
                best=metrics['mAP'];state=copy.deepcopy(noise.state_dict());best_epoch=epoch+1
                torch.save(dict(model=state,epoch=best_epoch,args=vars(a),validation=metrics),out/'best.pt')
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    if a.epochs:
        noise.load_state_dict(state)
    else:
        best_epoch=0;torch.save(dict(model=noise.state_dict(),epoch=0,args=vars(a)),out/'best.pt')
    predictions={};metrics={}
    evalids={'train':train,'val':splits['val'],'test':splits['test']}
    for s,ids in evalids.items():
        # Same actual MC1 representation train/val/test; MC8 diagnostic below.
        predictions[s]=predict_views(noise,backbone,arrays,rgb,ids,a.mode,a.suppress,batch=a.eval_batch,seed=8101)
        metrics[s]=native_metrics(predictions[s],np.array(arrays['attrs'][ids]));np.save(out/(s+'_probs.npy'),predictions[s])
    head,curve,index=fit_head(predictions['train'],labels[train],predictions['val'],labels[splits['val']])
    torch.save(head,out/'head.pt');test_logits=head_predict(head,predictions['test']);np.save(out/'test_class_logits.npy',test_logits)
    result=dict(mode=a.mode,order=a.order,independent_heads=a.independent_heads,suppress=a.suppress,
                concept_local=a.concept_local,best_epoch=best_epoch,concepts=metrics,head_grid=curve,head_selected=index,NEC=curve[index]['NEC'],
                test_accuracy=float((test_logits.argmax(1)==labels[splits['test']]).mean()),n_test=len(splits['test']),
                head_train_size=len(train),strict_bottleneck=True,train_labels_and_masks_used=True,inference_gt_masks=False)
    if a.mode=='noise':
        means=predict_views(noise,backbone,arrays,rgb,splits['test'],'mean',a.suppress,batch=a.eval_batch)
        np.save(out/'test_mean_probs.npy',means);mean_logits=head_predict(head,means);np.save(out/'test_mean_class_logits.npy',mean_logits)
        result['same_checkpoint_mean']=dict(accuracy=float((mean_logits.argmax(1)==labels[splits['test']]).mean()),**native_metrics(means,np.array(arrays['attrs'][splits['test']])))
        if a.mc_test>1:
            mc=predict_views(noise,backbone,arrays,rgb,splits['test'],'noise',a.suppress,batch=a.eval_batch,draws=a.mc_test)
            np.save(out/'test_MC'+str(a.mc_test)+'_probs.npy',mc)
            mc_logits=head_predict(head,mc);np.save(out/'test_MC'+str(a.mc_test)+'_class_logits.npy',mc_logits)
            result['MC'+str(a.mc_test)]=dict(accuracy=float((mc_logits.argmax(1)==labels[splits['test']]).mean()),**native_metrics(mc,np.array(arrays['attrs'][splits['test']])))
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
