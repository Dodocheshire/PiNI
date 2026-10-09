"""Complete differentiable RGB-view -> 18 concepts -> class training.

Each source image has all 18 real concept views in the task-loss forward pass.
There are no clean anchors substituted into the class bottleneck. Additional
clean concept/mask losses preserve meaning; ground truth is only loss targets.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from clip import clip
from torch.nn import functional as F
from .data import normalize
from .models import ConditionalRGBNoise,SpatialConcepts
from .train_noise import fixed_subset,native_metrics,predict_views
from .train_spatial import batch_tensor,fit_head,head_predict,load_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='artifacts/rival_vitl14')
    p.add_argument('--clean',default='artifacts/spatial_local');p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['mean','noise'],default='noise')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--view-budget',type=int,default=36018)
    p.add_argument('--source-batch',type=int,default=2);p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--independent-heads',action='store_true');a=p.parse_args()
    # Checkpoint reconstruction shares the same existing conditional model.
    a.order='full_joint';a.concept_local=True;a.suppress=True
    torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache)
    arrays,splits,labels,records=load_cache(cache);rgb=np.load(cache/'rgb.npy',mmap_mode='r')
    clean_checkpoint=torch.load(Path(a.clean)/'best.pt',map_location='cuda',weights_only=False)
    a.region_readout=clean_checkpoint.get('region_readout',False)
    concepts=SpatialConcepts(torch.tensor(np.load(cache/'text.npy'),device='cuda'),region_readout=a.region_readout).cuda()
    concepts.load_state_dict(clean_checkpoint['model'])
    noise=ConditionalRGBNoise(concepts,a.independent_heads,concept_local=True).cuda()
    backbone,_=clip.load('ViT-L/14',device='cuda',jit=False);backbone.float().eval().requires_grad_(False)
    train=fixed_subset(splits['train'],labels,6000);val_small=fixed_subset(splits['val'],labels,500,seed=45)
    n_source=max(1,a.view_budget//18);rng=np.random.default_rng(180)
    schedule=np.resize(rng.permutation(train),n_source)
    epochs=np.array_split(schedule,a.epochs)
    initial=torch.load(Path(a.clean)/'head.pt',map_location='cuda',weights_only=False)
    hw=initial['weight'].detach().clone().requires_grad_(True);hb=initial['bias'].detach().clone().requires_grad_(True)
    params=list(noise.parameters())+[hw,hb]
    optim=torch.optim.AdamW(params,lr=.001,weight_decay=.001);scaler=torch.amp.GradScaler('cuda')
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,a.epochs)
    protocol=dict(args=vars(a),full_class_bottleneck=True,class_bottleneck_inputs='18 selected logits, each from its own real perturbed RGB view',
        source_schedule=schedule.tolist(),head_train_ids=train.tolist(),val_selection_ids=val_small.tolist(),
        actual_training_views=n_source*18,unique_training_sources=len(np.unique(schedule)),
        budget_comparison='same 36k real gradient views as sampled-view V1/V2; complete-view source coverage differs',
        selection='validation native concept mAP, fixed 500-sample subset; final sparse head refit with fixed L1 grid',
        annotations='native concepts and masks only in loss targets; class labels only in task CE; no annotations at inference')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));history=[];best=-1;t=time.time()
    for epoch,source_ids in enumerate(epochs):
        noise.train();losses=[];task_losses=[];concept_losses=[]
        for start in range(0,len(source_ids),a.source_batch):
            ids=source_ids[start:start+a.source_batch];n=len(ids);optim.zero_grad()
            tokens=batch_tensor(arrays['tokens'],ids);images=batch_tensor(rgb,ids)/255
            repeated_tokens=tokens.repeat_interleave(18,0);repeated_images=images.repeat_interleave(18,0)
            ks=torch.arange(18,device='cuda').repeat(n);ii=torch.arange(n*18,device='cuda')
            native=batch_tensor(arrays['attrs'],ids)
            x,mu,sigma,gate=noise.perturb(repeated_images,repeated_tokens,ks,mode=a.mode,suppress=True)
            with torch.autocast('cuda',dtype=torch.float16):
                visual=backbone.encode_image(normalize(x),spatial=True)
                per_view,_,_=noise.concepts(visual)
                logits=per_view[ii,ks].float().reshape(n,18)
                # Every value in this class forward depends on its actual view.
                cbm=logits.sigmoid()
                class_logits=F.linear((cbm-initial['center'])/initial['scale'],hw,hb)
                task=F.cross_entropy(class_logits,torch.tensor(labels[ids],device='cuda'))
                concept=F.binary_cross_entropy_with_logits(logits,native)
                anchors,m,fg=noise.concepts(tokens)
                clean_loss=F.binary_cross_entropy_with_logits(anchors.float(),native)
                gt=batch_tensor(arrays['masks'],ids);valid=batch_tensor(arrays['mask_valid'],ids)
                pred=torch.cat([m,fg[:,None]],1).float()
                bce=(F.binary_cross_entropy_with_logits(pred,gt,reduction='none').mean((-1,-2))*valid).sum()/valid.sum().clamp_min(1)
                present=(gt.sum((-1,-2))>0).float()*valid;prob=pred.sigmoid()
                dice=1-(2*(prob*gt).sum((-1,-2))+1)/(prob.sum((-1,-2))+gt.sum((-1,-2))+1)
                dice=(dice*present).sum()/present.sum().clamp_min(1)
                loss=concept+.25*task+.25*clean_loss+.25*bce+.25*dice+.003*hw.abs().sum()
            loss=loss+.005*mu.square().mean()
            if epoch==0 and start==0:
                # Verify the actual complete class graph reaches RGB heads,
                # independently of semantic/mask regularizers.
                cg=torch.autograd.grad(task,[noise.mean.weight,noise.std.weight],retain_graph=True,allow_unused=True)
                audit=dict(source_images=n,real_rgb_views=n*18,class_input_shape=list(cbm.shape),
                           class_loss_mean_head_grad_norm=0.0 if cg[0] is None else float(cg[0].float().norm()),
                           class_loss_sigma_head_grad_norm=0.0 if cg[1] is None else float(cg[1].float().norm()),
                           mode=a.mode,ground_truth_forward_inputs=False,clean_anchor_class_inputs=False)
                assert np.isfinite(audit['class_loss_mean_head_grad_norm']) and audit['class_loss_mean_head_grad_norm']>0
                if a.mode=='noise':assert np.isfinite(audit['class_loss_sigma_head_grad_norm']) and audit['class_loss_sigma_head_grad_norm']>0
                (out/'class_gradient_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(params,5)
            scaler.step(optim);scaler.update();losses.append(float(loss.detach()));task_losses.append(float(task.detach()));concept_losses.append(float(concept.detach()))
        sched.step();row=dict(epoch=epoch+1,loss=float(np.mean(losses)),task_CE=float(np.mean(task_losses)),concept_BCE=float(np.mean(concept_losses)),elapsed_s=round(time.time()-t,1),views=len(source_ids)*18)
        if (epoch+1)%2==0 or epoch+1==a.epochs:
            probs=predict_views(noise,backbone,arrays,rgb,val_small,a.mode,True,batch=a.eval_batch)
            metrics=native_metrics(probs,np.array(arrays['attrs'][val_small]));row['val']=metrics
            if metrics['mAP']>best:
                best=metrics['mAP'];state=copy.deepcopy(noise.state_dict());best_epoch=epoch+1
                torch.save(dict(model=state,epoch=best_epoch,args=vars(a),validation=metrics,joint_head=dict(weight=hw.detach().cpu(),bias=hb.detach().cpu())),out/'best.pt')
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    noise.load_state_dict(state);predictions={};metrics={}
    eval_ids=dict(train=train,val=splits['val'],test=splits['test'])
    for split,ids in eval_ids.items():
        probs=predict_views(noise,backbone,arrays,rgb,ids,a.mode,True,batch=a.eval_batch)
        predictions[split]=probs;metrics[split]=native_metrics(probs,np.array(arrays['attrs'][ids]));np.save(out/(split+'_probs.npy'),probs)
    head,curve,index=fit_head(predictions['train'],labels[train],predictions['val'],labels[splits['val']])
    torch.save(head,out/'head.pt');test_logits=head_predict(head,predictions['test']);np.save(out/'test_class_logits.npy',test_logits)
    result=dict(mode=a.mode,order='full_joint',independent_heads=a.independent_heads,suppress=True,concept_local=True,
                best_epoch=best_epoch,concepts=metrics,head_grid=curve,head_selected=index,NEC=curve[index]['NEC'],
                test_accuracy=float((test_logits.argmax(1)==labels[splits['test']]).mean()),n_test=len(splits['test']),
                head_train_size=len(train),training_views=n_source*18,training_source_coverage=len(np.unique(schedule)),
                strict_bottleneck=True,full_class_bottleneck=True,head_refit_after_joint_representation=True,
                train_labels_and_masks_used=True,inference_gt_masks=False)
    if a.mode=='noise':
        mean=predict_views(noise,backbone,arrays,rgb,splits['test'],'mean',True,batch=a.eval_batch)
        np.save(out/'test_mean_probs.npy',mean);ml=head_predict(head,mean);np.save(out/'test_mean_class_logits.npy',ml)
        result['same_checkpoint_mean']=dict(accuracy=float((ml.argmax(1)==labels[splits['test']]).mean()),**native_metrics(mean,np.array(arrays['attrs'][splits['test']])))
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
