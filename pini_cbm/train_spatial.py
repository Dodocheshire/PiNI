"""Train real native concept/foreground predictors before RGB-noise stages."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from .models import SpatialConcepts


def load_cache(path):
    path = Path(path)
    assert (path/'COMPLETE.json').exists(), 'Feature cache must finish first'
    records = json.loads((path/'manifest.json').read_text())
    arrays = {k:np.load(path/(k+'.npy'), mmap_mode='r') for k in ['tokens', 'attrs', 'masks', 'mask_valid']}
    splits = {s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','val','test']}
    labels = np.array([r['label'] for r in records])
    return arrays, splits, labels, records


def batch_tensor(array, ids):
    return torch.from_numpy(np.array(array[ids], copy=True)).cuda(non_blocking=True).float()


@torch.no_grad()
def predict(model, arrays, ids, batch=256):
    model.eval(); logits=[]; ious=[]; partious=[]
    for start in range(0,len(ids),batch):
        chunk=ids[start:start+batch]
        l,m,f=model(batch_tensor(arrays['tokens'],chunk))
        logits.append(l.cpu().numpy())
        gt=batch_tensor(arrays['masks'],chunk)
        valid=batch_tensor(arrays['mask_valid'],chunk).bool()
        pred=torch.cat([m,f[:,None]],1)>0
        # Fractional GT occupancy preserves the exact native 224px mask area
        # under nearest-upsampled 16px binary predictions, including tiny parts.
        intersection=(gt*pred).sum((-1,-2)); union=(gt+pred-gt*pred).sum((-1,-2)).clamp_min(1e-6)
        iou=intersection/union
        ious.extend(iou[:,18][valid[:,18]].cpu().tolist())
        present=batch_tensor(arrays['attrs'],chunk).bool()&valid[:,:18]
        partious.extend(iou[:,:18][present].cpu().tolist())
    logits=np.concatenate(logits)
    probs=1/(1+np.exp(-np.clip(logits,-30,30)))
    y=np.array(arrays['attrs'][ids])
    p=probs>=.5
    tp=(p*y).sum(0); fp=(p*(1-y)).sum(0); fn=((1-p)*y).sum(0)
    metrics=dict(mAP=float(average_precision_score(y,probs,average='macro')),
                 macro_precision=float(np.mean(tp/np.maximum(tp+fp,1))),
                 macro_recall=float(np.mean(tp/np.maximum(tp+fn,1))),
                 fg_IoU=float(np.mean(ious)), concept_mask_IoU=float(np.mean(partious)))
    return logits,metrics


def fit_head(train, labels, val, val_labels, max_nec=8, device='cuda'):
    """Convex proximal L1 multiclass head; fixed validation-only grid."""
    x=torch.tensor(train,dtype=torch.float32,device=device); y=torch.tensor(labels,device=device)
    v=torch.tensor(val,dtype=torch.float32,device=device); vy=torch.tensor(val_labels,device=device)
    center=x.mean(0); scale=x.std(0).clamp_min(.03); x=(x-center)/scale; v=(v-center)/scale
    curve=[]; candidates=[]
    for lam in [.003,.01,.03,.06,.1]:
        w=torch.zeros(10,x.shape[1],device=device,requires_grad=True)
        b=torch.zeros(10,device=device,requires_grad=True)
        for i in range(1500):
            loss=F.cross_entropy(F.linear(x,w,b),y)
            gw,gb=torch.autograd.grad(loss,(w,b))
            with torch.no_grad():
                w.copy_(F.softshrink(w-.5*gw,lambd=.5*lam)); b.sub_(.5*gb)
        with torch.no_grad():
            nec=float((w.abs()>1e-5).sum(1).float().mean())
            acc=float((F.linear(v,w,b).argmax(1)==vy).float().mean())
            nll=float(F.cross_entropy(F.linear(v,w,b),vy))
        curve.append(dict(l1=lam,NEC=nec,val_accuracy=acc,val_NLL=nll))
        candidates.append(dict(weight=w.detach().cpu(),bias=b.detach().cpu(),center=center.cpu(),scale=scale.cpu()))
    eligible=[i for i,r in enumerate(curve) if r['NEC']<=max_nec]
    if not eligible: eligible=list(range(len(curve)))
    best=max(eligible,key=lambda i:(curve[i]['val_accuracy'],-curve[i]['val_NLL']))
    return candidates[best],curve,best


def head_predict(head, concepts):
    return ((concepts-head['center'].numpy())/head['scale'].numpy())@head['weight'].numpy().T+head['bias'].numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='artifacts/rival_vitl14')
    p.add_argument('--out',default='artifacts/spatial_clean');p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--locality-loss',action='store_true'); a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    arrays,splits,labels,records=load_cache(a.cache)
    model=SpatialConcepts(torch.tensor(np.load(Path(a.cache)/'text.npy'),device='cuda')).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,a.epochs)
    train=splits['train'];history=[];best=-1;t=time.time()
    for epoch in range(a.epochs):
        model.train();rng=np.random.default_rng(42+epoch);order=rng.permutation(train);losses=[]
        for start in range(0,len(order),a.batch):
            ids=order[start:start+a.batch]
            logits,m,f=model(batch_tensor(arrays['tokens'],ids))
            attrs=batch_tensor(arrays['attrs'],ids);gt=batch_tensor(arrays['masks'],ids)
            valid=batch_tensor(arrays['mask_valid'],ids)
            loss_c=F.binary_cross_entropy_with_logits(logits,attrs)
            predictions=torch.cat([m,f[:,None]],1)
            loss_m=(F.binary_cross_entropy_with_logits(predictions,gt,reduction='none').mean((-1,-2))*valid).sum()/valid.sum().clamp_min(1)
            loss=loss_c+.25*loss_m
            if a.locality_loss:
                present=(gt.sum((-1,-2))>0).float()*valid
                prob=predictions.sigmoid()
                dice=1-(2*(prob*gt).sum((-1,-2))+1)/(prob.sum((-1,-2))+gt.sum((-1,-2))+1)
                loss=loss+.5*(dice*present).sum()/present.sum().clamp_min(1)
            optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
        sched.step()
        vl,metrics=predict(model,arrays,splits['val'])
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),elapsed_s=round(time.time()-t,1),**metrics)
        history.append(row);print(json.dumps(row),flush=True)
        # Native concept AP, not test accuracy or attractive visualizations.
        score=metrics['mAP']+(.1*metrics['concept_mask_IoU'] if a.locality_loss else 0)
        if score>best:
            best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            torch.save(dict(model=state,epoch=best_epoch,validation=metrics),out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(state)
    predictions={};stats={}
    for s in ['train','val','test']:
        predictions[s],stats[s]=predict(model,arrays,splits[s]);np.save(out/(s+'_logits.npy'),predictions[s])
    probs={s:1/(1+np.exp(-np.clip(l,-30,30))) for s,l in predictions.items()}
    head,curve,index=fit_head(probs['train'],labels[splits['train']],probs['val'],labels[splits['val']])
    np.save(out/'test_class_logits.npy',head_predict(head,probs['test']))
    torch.save(head,out/'head.pt')
    result=dict(variant='clean spatial ViT-L/14 supervised CBM',best_epoch=best_epoch,
                selection='validation mAP + 0.1 positive concept mask IoU' if a.locality_loss else 'validation native concept mAP',locality_loss=a.locality_loss,
                concepts=stats,head_grid=curve,head_selected=index,
                test_accuracy=float((head_predict(head,probs['test']).argmax(1)==labels[splits['test']]).mean()),NEC=curve[index]['NEC'],
                n_test=len(splits['test']),strict_bottleneck=True,mask_supervision=True)
    # Native concept oracle: a separate supervised ceiling, never a learned model.
    oracle,oc,oi=fit_head(np.array(arrays['attrs'][splits['train']]),labels[splits['train']],np.array(arrays['attrs'][splits['val']]),labels[splits['val']])
    result['oracle']=dict(test_accuracy=float((head_predict(oracle,np.array(arrays['attrs'][splits['test']])).argmax(1)==labels[splits['test']]).mean()),NEC=oc[oi]['NEC'],head_grid=oc)
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
