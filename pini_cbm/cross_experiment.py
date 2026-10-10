"""Matched LF / supervised / spatial RGB-noise experiments on three datasets."""
import argparse
import copy
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from clip import clip
from clip.model import convert_weights
from .benchmark_baselines import cos_cubed
from .cross_data import ROOT, MEAN_BUDGET, fixed_subset, write_json, background_donors, donor_background
from .data import normalize
from .models import SpatialConcepts, ConditionalRGBNoise
from .train_spatial import batch_tensor, head_predict


def load(name, need_regions=True):
    cache = ROOT/name/"cache"
    assert (cache/("COMPLETE.json" if need_regions else "CONCEPTS.json")).exists()
    records = json.loads((cache/"manifest.json").read_text())
    protocol = json.loads((cache/"protocol.json").read_text())
    keys = ["tokens","rgb","attrs","text","cls","teacher"]
    if need_regions:
        keys += ["masks","mask_valid","fg224"]
    arrays = {k:np.load(cache/(k+".npy"),mmap_mode="r") for k in keys}
    splits = {s:np.asarray([i for i,r in enumerate(records) if r["split"]==s])
              for s in ["train","val","test"]}
    labels = np.asarray([r["label"] for r in records])
    return cache, arrays, splits, labels, records, protocol


def concept_metrics(name, truth, scores):
    if name == "cub":
        # Preserve the published fixed bank; undefined all-negative attributes
        # contribute zero AP, matching sklearn's macro AP convention.
        return dict(mAP=float(average_precision_score(truth,scores,average="macro")))
    a = truth.astype("float64")-truth.mean(0,keepdims=True)
    b = scores.astype("float64")-scores.mean(0,keepdims=True)
    denominator = np.sqrt((a*a).sum(0)*(b*b).sum(0))
    values = np.divide((a*b).sum(0),denominator,out=np.zeros_like(denominator),where=denominator>1e-12)
    return dict(concept_correlation=float(values.mean()))


def selection_score(name, metrics):
    return metrics["mAP" if name=="cub" else "concept_correlation"]


def fit_head(train, labels, val, val_labels, n_classes, device="cuda"):
    """The RIVAL proximal-L1 rule with its penalty scaled per output class."""
    x = torch.tensor(train,dtype=torch.float32,device=device)
    y = torch.tensor(labels,device=device)
    v = torch.tensor(val,dtype=torch.float32,device=device)
    vy = torch.tensor(val_labels,device=device)
    center = x.mean(0)
    scale = x.std(0).clamp_min(.03)
    x = (x-center)/scale
    v = (v-center)/scale
    curve = []
    candidates = []
    for reference_lambda in [.003,.01,.03,.06,.1]:
        lam = reference_lambda*10/n_classes
        w = torch.zeros(n_classes,x.shape[1],device=device,requires_grad=True)
        b = torch.zeros(n_classes,device=device,requires_grad=True)
        for _ in range(1500):
            loss = F.cross_entropy(F.linear(x,w,b),y)
            gw,gb = torch.autograd.grad(loss,(w,b))
            with torch.no_grad():
                w.copy_(F.softshrink(w-.5*gw,lambd=.5*lam))
                b.sub_(.5*gb)
        with torch.no_grad():
            nec = float((w.abs()>1e-5).sum(1).float().mean())
            accuracy = float((F.linear(v,w,b).argmax(1)==vy).float().mean())
            nll = float(F.cross_entropy(F.linear(v,w,b),vy))
        curve.append(dict(reference_l1=reference_lambda,l1=lam,NEC=nec,val_accuracy=accuracy,val_NLL=nll))
        candidates.append(dict(weight=w.detach().cpu(),bias=b.detach().cpu(),
                               center=center.cpu(),scale=scale.cpu()))
    eligible = [i for i,r in enumerate(curve) if r["NEC"]<=16]
    if not eligible:
        eligible = list(range(len(curve)))
    best = max(eligible,key=lambda i:(curve[i]["val_accuracy"],-curve[i]["val_NLL"]))
    return candidates[best],curve,best


@torch.no_grad()
def predict_spatial(name,model,arrays,ids,batch=128):
    model.eval()
    logits = []
    fg_ious = []
    region_ious = []
    k = model.k
    for start in range(0,len(ids),batch):
        chunk = ids[start:start+batch]
        l,m,f = model(batch_tensor(arrays["tokens"],chunk))
        logits.append(l.float().cpu().numpy())
        gt = batch_tensor(arrays["masks"],chunk)
        valid = batch_tensor(arrays["mask_valid"],chunk).bool()
        predicted = torch.cat([m,f[:,None]],1)>0
        intersection = (gt*predicted).sum((-1,-2))
        union = (gt+predicted-gt*predicted).sum((-1,-2)).clamp_min(1e-6)
        iou = intersection/union
        fg_ious.extend(iou[:,k][valid[:,k]].cpu().tolist())
        positive = (gt[:,:k].sum((-1,-2))>0)&valid[:,:k]
        region_ious.extend(iou[:,:k][positive].cpu().tolist())
    logits = np.concatenate(logits)
    probs = 1/(1+np.exp(-np.clip(logits,-30,30)))
    metrics = concept_metrics(name,np.array(arrays["attrs"][ids]),probs)
    metrics.update(fg_IoU=float(np.mean(fg_ious)) if fg_ious else None,
        concept_region_IoU=float(np.mean(region_ious)) if region_ious else None)
    return probs,metrics


def region_loss(model,tokens,targets,masks,valid,dice_coefficient):
    logits,region,foreground = model(tokens)
    predicted = torch.cat([region,foreground[:,None]],1)
    bce = (F.binary_cross_entropy_with_logits(predicted.float(),masks,reduction="none")
           .mean((-1,-2))*valid).sum()/valid.sum().clamp_min(1)
    positive = (masks.sum((-1,-2))>0).float()*valid
    probabilities = predicted.float().sigmoid()
    dice = 1-(2*(probabilities*masks).sum((-1,-2))+1)/(
        probabilities.sum((-1,-2))+masks.sum((-1,-2))+1)
    dice = (dice*positive).sum()/positive.sum().clamp_min(1)
    return logits,.25*bce+dice_coefficient*dice


def pretrain(name,epochs=40,batch=128):
    cache,arrays,splits,labels,records,protocol = load(name)
    if name!="cub":
        reviewed = json.loads((cache/"FOREGROUND_REVIEW.json").read_text())
        assert reviewed["status"]=="passed","Inspect the actual foreground and background-swap images before training"
        reviewed_ids = reviewed["ids"]+reviewed["donors"]
        assert hashlib.sha256(np.asarray(arrays["fg224"][reviewed_ids]).tobytes()).hexdigest()==reviewed["mask_sha256"]
    out = ROOT/name/"region"
    out.mkdir(exist_ok=True)
    if (out/"COMPLETE.json").exists():
        return
    torch.manual_seed(42)
    model = SpatialConcepts(torch.tensor(np.array(arrays["text"]),device="cuda"),
                            region_readout=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    history = []
    best = -np.inf
    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(42+epoch).permutation(splits["train"])
        losses = []
        for start in range(0,len(order),batch):
            ids = order[start:start+batch]
            logits,local_loss = region_loss(model,batch_tensor(arrays["tokens"],ids),
                batch_tensor(arrays["attrs"],ids),batch_tensor(arrays["masks"],ids),
                batch_tensor(arrays["mask_valid"],ids),.5)
            loss = F.binary_cross_entropy_with_logits(logits,batch_tensor(arrays["attrs"],ids))+local_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        _,metrics = predict_spatial(name,model,arrays,splits["val"])
        score = selection_score(name,metrics)+.1*(metrics["concept_region_IoU"] or 0)
        row = dict(stage="region",dataset=name,epoch=epoch+1,loss=float(np.mean(losses)),
                   elapsed_s=round(time.time()-started,1),**metrics)
        history.append(row)
        print(json.dumps(row),flush=True)
        write_json(out/"history.json",history)
        if score>best:
            best = score
            torch.save(dict(model=copy.deepcopy(model.state_dict()),epoch=epoch+1,
                validation=metrics,dataset=name,concepts=model.k,region_readout=True),out/"best.pt")
    write_json(out/"COMPLETE.json",dict(epochs=epochs,selected_score=best,
        concept_train_n=len(splits["train"]),seed=42,
        annotation="native CUB instance attributes and part regions" if name=="cub" else "CLIP soft concept / SAM2 pseudo region targets"))


def baseline(name,method,seed,device="cpu"):
    cache,arrays,splits,labels,records,protocol = load(name,need_regions=False)
    assert method == "lf_cbm" or name == "cub"
    out = ROOT/name/(method+"_seed"+str(seed))
    out.mkdir(exist_ok=True)
    if (out/"original_results.json").exists():
        return
    torch.manual_seed(seed)
    random.seed(seed)
    x = torch.tensor(np.array(arrays["cls"]),device=device)
    teacher = torch.tensor(np.array(arrays["teacher"]),device=device)
    targets = torch.tensor(np.array(arrays["attrs"]),device=device)
    projection = nn.Linear(768,len(arrays["text"]),bias=(method=="supervised_cbm")).to(device)
    history = []
    best = -np.inf
    if method == "lf_cbm":
        optimizer = torch.optim.Adam(projection.parameters(),lr=.001)
        population = splits["train"].tolist()
        for step in range(1000):
            ids = random.sample(population,min(4096,len(population)))
            loss = -cos_cubed(teacher[ids],projection(x[ids])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50==0 or step==999:
                with torch.no_grad():
                    score = float(cos_cubed(teacher[splits["val"]],projection(x[splits["val"]])).mean())
                history.append(dict(step=step,validation_cos_cubed=score))
                if score>best:
                    best = score
                    state = copy.deepcopy(projection.state_dict())
                else:
                    break
    else:
        optimizer = torch.optim.AdamW(projection.parameters(),lr=.003,weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40)
        for epoch in range(40):
            order = np.random.default_rng(seed+epoch).permutation(splits["train"])
            for start in range(0,len(order),256):
                ids = order[start:start+256]
                loss = F.binary_cross_entropy_with_logits(projection(x[ids]),targets[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            with torch.no_grad():
                scores = projection(x[splits["val"]]).sigmoid().cpu().numpy()
                score = concept_metrics(name,np.array(arrays["attrs"][splits["val"]]),scores)["mAP"]
            history.append(dict(epoch=epoch+1,validation_mAP=score))
            if score>best:
                best = score
                state = copy.deepcopy(projection.state_dict())
    projection.load_state_dict(state)
    with torch.no_grad():
        scores = projection(x)
        if method == "supervised_cbm":
            scores = scores.sigmoid()
        scores = scores.cpu().numpy()
    train_ids = np.asarray(protocol["head_train_ids"])
    head,curve,index = fit_head(scores[train_ids],labels[train_ids],scores[splits["val"]],
                               labels[splits["val"]],len(protocol["classes"]),device)
    logits = head_predict(head,scores[splits["test"]])
    metrics = concept_metrics(name,np.array(arrays["attrs"][splits["test"]]),scores[splits["test"]])
    metrics.update(accuracy=float((logits.argmax(1)==labels[splits["test"]]).mean()),NEC=curve[index]["NEC"])
    torch.save(dict(method=method,seed=seed,model={k:v.cpu() for k,v in state.items()},head=head),out/"model.pt")
    np.save(out/"test_scores.npy",scores[splits["test"]])
    np.save(out/"test_class_logits.npy",logits)
    result = dict(dataset=name,method=method,seed=seed,original=metrics,head_grid=curve,
        head_selected=index,history=history,concept_train_n=len(splits["train"]),
        head_train_n=len(train_ids),validation_n=len(splits["val"]),test_n=len(splits["test"]),
        k=len(arrays["text"]),backbone=protocol["backbone"],native_concept_supervision=method=="supervised_cbm")
    write_json(out/"original_results.json",result)
    print(json.dumps(dict(stage="baseline",dataset=name,method=method,seed=seed,**metrics)),flush=True)


@torch.no_grad()
def _views_impl(noise,backbone,arrays,ids,mode,batch=256,seed=8101,output=None,images=None,tokens=None):
    """Exactly one re-encoded RGB view for every image/concept pair."""
    noise.eval()
    k = noise.concepts.k
    total = len(ids)*k
    rng = torch.Generator(device="cuda").manual_seed(seed)
    metadata = dict(ids_sha256=hashlib.sha256(np.asarray(ids,dtype="int64").tobytes()).hexdigest(),
                    mode=mode,batch=batch,seed=seed,n=len(ids),k=k)
    start_at = 0
    if output:
        output = Path(output)
        state_path = output.with_suffix(".progress.pt")
        done_path = output.with_suffix(".complete.json")
        if done_path.exists():
            assert json.loads(done_path.read_text()) == metadata
            return np.load(output)
        predictions = np.lib.format.open_memmap(output,mode="r+" if output.exists() else "w+",
            dtype="float32",shape=(len(ids),k))
        if state_path.exists():
            state = torch.load(state_path,map_location="cpu",weights_only=False)
            assert state["metadata"] == metadata
            start_at = state["done"]
            rng.set_state(state["rng"])
    else:
        predictions = np.empty((len(ids),k),dtype="float32")
    started = time.time()
    for number,start in enumerate(range(start_at,total,batch)):
        flat = np.arange(start,min(start+batch,total))
        positions = flat//k
        concepts = torch.tensor(flat%k,device="cuda")
        original_ids = ids[positions]
        rgb = batch_tensor(arrays["rgb"],original_ids)/255 if images is None else batch_tensor(images,positions)
        tok = batch_tensor(arrays["tokens"],original_ids) if tokens is None else batch_tensor(tokens,positions)
        epsilon = torch.randn(rgb.shape,device="cuda",generator=rng) if mode=="noise" else None
        with torch.autocast("cuda",enabled=False):
            x,_,_,_ = noise.perturb(rgb,tok,concepts,mode=mode,epsilon=epsilon,suppress=True)
        with torch.autocast("cuda",dtype=torch.float16):
            encoded = backbone.encode_image(normalize(x),spatial=True)
            logits,_,_ = noise.recognize(encoded,tok)
        scores = logits[torch.arange(len(flat),device="cuda"),concepts].float().sigmoid().cpu().numpy()
        predictions[positions,flat%k] = scores
        if number%100==0 or start+batch>=total:
            if output:
                predictions.flush()
                temporary = state_path.with_suffix(".partial")
                torch.save(dict(metadata=metadata,done=min(start+batch,total),rng=rng.get_state()),temporary)
                temporary.replace(state_path)
            print(json.dumps(dict(stage="views",mode=mode,output=str(output),done=min(start+batch,total),
                total=total,elapsed_s=round(time.time()-started,1))),flush=True)
    if output:
        predictions.flush()
        write_json(done_path,metadata)
    return np.array(predictions)


@torch.no_grad()
def views(noise,backbone,arrays,ids,mode,batch=256,seed=8101,output=None,images=None,tokens=None):
    # Keep the cast cache alive over all batches; RGB noise remains float32.
    # Frozen weights are unchanged, including during validation between epochs.
    with torch.autocast("cuda",dtype=torch.float16):
        return _views_impl(noise,backbone,arrays,ids,mode,batch,seed,output,images,tokens)


def restore_noise(path,texts,device="cuda"):
    saved = torch.load(path,map_location=device,weights_only=False)
    concepts = SpatialConcepts(torch.tensor(np.array(texts),device=device),region_readout=True).to(device)
    noise = ConditionalRGBNoise(concepts,concept_local=True,mean_budget=MEAN_BUDGET).to(device)
    noise.load_state_dict(saved["model"])
    return noise,saved


def adapt(name,mode,seed,epochs=6,batch=16,eval_batch=256):
    cache,arrays,splits,labels,records,protocol = load(name)
    out = ROOT/name/(mode+"_seed"+str(seed))
    out.mkdir(exist_ok=True)
    if (out/"original_results.json").exists():
        return
    torch.manual_seed(seed)
    train_ids = np.asarray(protocol["head_train_ids"])
    val_small = np.asarray(protocol["region_val_ids"])
    backbone,_ = clip.load("ViT-L/14",device="cuda",jit=False)
    backbone.float().eval().requires_grad_(False)
    if not (out/"TRAINED.json").exists():
        concepts = SpatialConcepts(torch.tensor(np.array(arrays["text"]),device="cuda"),region_readout=True).cuda()
        initial = torch.load(ROOT/name/"region/best.pt",map_location="cuda",weights_only=False)
        concepts.load_state_dict(initial["model"])
        noise = ConditionalRGBNoise(concepts,concept_local=True,mean_budget=MEAN_BUDGET).cuda()
        optimizer = torch.optim.AdamW(noise.parameters(),lr=.001,weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
        scaler = torch.amp.GradScaler("cuda")
        history = []
        best = -np.inf
        started = time.time()
        for epoch in range(epochs):
            noise.train()
            rng = np.random.default_rng(180+epoch+1000*(seed-42))
            order = rng.permutation(train_ids)
            losses = []
            for start in range(0,len(order),batch):
                ids = order[start:start+batch]
                tok = batch_tensor(arrays["tokens"],ids)
                rgb = batch_tensor(arrays["rgb"],ids)/255
                targets = batch_tensor(arrays["attrs"],ids)
                selected = torch.tensor(rng.integers(0,noise.concepts.k,size=len(ids)),device="cuda")
                optimizer.zero_grad()
                x,mu,_,_ = noise.perturb(rgb,tok,selected,mode=mode,suppress=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    encoded = backbone.encode_image(normalize(x),spatial=True)
                    logits,_,_ = noise.recognize(encoded,tok)
                    selected_loss = F.binary_cross_entropy_with_logits(
                        logits[torch.arange(len(ids),device="cuda"),selected].float(),
                        targets[torch.arange(len(ids),device="cuda"),selected])
                if epoch==0 and start==0:
                    gradients = torch.autograd.grad(selected_loss,[noise.mean.weight,noise.std.weight],
                                                    retain_graph=True,allow_unused=True)
                    norms = [0. if g is None else float(g.float().norm()) for g in gradients]
                    assert np.isfinite(norms).all() and norms[0]>0
                    if mode=="noise":
                        assert norms[1]>0
                    write_json(out/"gradient_audit.json",dict(actual_RGB=True,mean_gradient_norm=norms[0],
                        sigma_gradient_norm=norms[1],native_forward_mask_inputs=False))
                anchors,local_loss = region_loss(noise.concepts,tok,targets,
                    batch_tensor(arrays["masks"],ids),batch_tensor(arrays["mask_valid"],ids),.25)
                loss = selected_loss+.25*F.binary_cross_entropy_with_logits(anchors,targets)+local_loss+.005*mu.square().mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(noise.parameters(),5)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = dict(stage="adapt",dataset=name,mode=mode,seed=seed,epoch=epoch+1,
                       loss=float(np.mean(losses)),elapsed_s=round(time.time()-started,1))
            if (epoch+1)%2==0 or epoch+1==epochs:
                probabilities = views(noise,backbone,arrays,val_small,mode,eval_batch)
                metrics = concept_metrics(name,np.array(arrays["attrs"][val_small]),probabilities)
                row["validation"] = metrics
                score = selection_score(name,metrics)
                if score>best:
                    best = score
                    torch.save(dict(model=copy.deepcopy(noise.state_dict()),epoch=epoch+1,validation=metrics,
                        dataset=name,mode=mode,seed=seed,k=noise.concepts.k),out/"best.pt")
            history.append(row)
            write_json(out/"history.json",history)
            print(json.dumps(row),flush=True)
        write_json(out/"TRAINED.json",dict(epochs=epochs,train_n=len(train_ids),val_small_n=len(val_small),
            batch=batch,mean_budget=MEAN_BUDGET,seed=seed,mode=mode,best=best))
    # Actual RGB/concept outputs were checked bitwise against FP32-parameter
    # AMP. Convert once only after training; training gradients are unchanged.
    convert_weights(backbone)
    noise,saved = restore_noise(out/"best.pt",arrays["text"])
    train_probs = views(noise,backbone,arrays,train_ids,mode,eval_batch,output=out/"train_scores.npy")
    val_probs = views(noise,backbone,arrays,splits["val"],mode,eval_batch,output=out/"val_scores.npy")
    if not (out/"head.pt").exists():
        head,curve,index = fit_head(train_probs,labels[train_ids],val_probs,labels[splits["val"]],len(protocol["classes"]))
        torch.save(head,out/"head.pt")
        write_json(out/"head_grid.json",dict(grid=curve,selected=index))
    else:
        head = torch.load(out/"head.pt",map_location="cpu",weights_only=False)
        grid = json.loads((out/"head_grid.json").read_text())
        curve,index = grid["grid"],grid["selected"]
    # Both adapted models use deterministic mean inference with their own head.
    test_probs = views(noise,backbone,arrays,splits["test"],"mean",eval_batch,output=out/"test_scores.npy")
    logits = head_predict(head,test_probs)
    np.save(out/"test_class_logits.npy",logits)
    metrics = concept_metrics(name,np.array(arrays["attrs"][splits["test"]]),test_probs)
    metrics.update(accuracy=float((logits.argmax(1)==labels[splits["test"]]).mean()),NEC=curve[index]["NEC"])
    result = dict(dataset=name,method=mode,seed=seed,original=metrics,head_grid=curve,head_selected=index,
        selected_epoch=saved["epoch"],train_n=len(train_ids),validation_n=len(splits["val"]),
        test_n=len(splits["test"]),k=noise.concepts.k,backbone=protocol["backbone"],inference="mean",
        train_concept_views=mode,noise_location="RGB pixels",region_pretrain_seed=42,
        adapt_batch=batch,inference_batch=eval_batch)
    write_json(out/"original_results.json",result)
    print(json.dumps(dict(stage="adapt_complete",dataset=name,mode=mode,seed=seed,**metrics)),flush=True)


@torch.no_grad()
def intervention(name,batch=256):
    cache,arrays,splits,labels,records,protocol = load(name)
    out = ROOT/name/"background"
    out.mkdir(exist_ok=True)
    if (out/"COMPLETE.json").exists():
        return
    ids = np.asarray(protocol["background_ids"])
    donor_pool = splits["val"][np.asarray(arrays["mask_valid"][splits["val"],-1])]
    images = np.empty((len(ids),3,224,224),dtype="float32")
    audit = json.loads((cache/"foreground_audit.json").read_text()) if name!="cub" else None
    donors,crops,donor_pool = background_donors(name,labels,donor_pool,ids,arrays["fg224"],audit)
    for j,(i,d) in enumerate(zip(ids,donors)):
        target = np.array(arrays["rgb"][i],dtype="float32")/255
        background = donor_background(name,arrays["rgb"],arrays["fg224"],d,audit,crops[j] if crops else None)
        fg = np.asarray(arrays["fg224"][i])[None]
        images[j] = target*fg+background*(~fg)
        assert np.array_equal(images[j][:,fg[0]],target[:,fg[0]])
    np.save(out/"images.npy",images)
    np.save(out/"ids.npy",ids)
    np.save(out/"donors.npy",donors)
    fingerprint = hashlib.sha256(ids.tobytes()+donors.tobytes()+images.tobytes()).hexdigest()
    backbone,_ = clip.load("ViT-L/14",device="cuda",jit=False)
    backbone.float().eval().requires_grad_(False)
    convert_weights(backbone)
    encoded = []
    for start in range(0,len(images),batch):
        with torch.autocast("cuda",dtype=torch.float16):
            encoded.append(backbone.encode_image(normalize(batch_tensor(images,np.arange(start,min(start+batch,len(images))))),
                                                spatial=True).cpu().numpy())
    tokens = np.concatenate(encoded)
    np.save(out/"tokens.npy",tokens.astype("float16"))
    write_json(out/"protocol.json",dict(ids=ids.tolist(),donors=donors.tolist(),n=len(ids),
        sha256=fingerprint,foreground_source="CUB native segmentation" if name=="cub" else "frozen SAM2 pseudo segmentation",
        donor_object_replaced_with=.5 if name=="cub" else None,target_foreground_pixels_exactly_preserved=True,
        donor_removal="native object mask" if name=="cub" else "object-free edge crop outside SAM/grounding enclosing box padded one 14-pixel patch",
        donor_background_crops=crops,donor_pool_ids=donor_pool.tolist(),
        donor_class_different=True,construction_seeds=[46,47]))
    write_json(out/"COMPLETE.json",dict(sha256=fingerprint,n=len(ids)))


@torch.no_grad()
def background(name,method,seed,batch=256):
    cache,arrays,splits,labels,records,protocol = load(name)
    root = ROOT/name
    intervention_root = root/"background"
    assert (intervention_root/"COMPLETE.json").exists()
    out = root/(method+"_seed"+str(seed))
    if (out/"results.json").exists():
        saved = json.loads((out/"results.json").read_text())
        completed = json.loads((intervention_root/"COMPLETE.json").read_text())
        if ("background" in saved
                and saved.get("intervention",{}).get("sha256")==completed["sha256"]
                and (out/"background_scores.npy").exists()
                and (out/"background_class_logits.npy").exists()):
            return
    original = json.loads((out/"original_results.json").read_text())
    ids = np.load(intervention_root/"ids.npy")
    tokens = np.load(intervention_root/"tokens.npy",mmap_mode="r")
    if method in ["mean","noise"]:
        backbone,_ = clip.load("ViT-L/14",device="cuda",jit=False)
        backbone.float().eval().requires_grad_(False)
        convert_weights(backbone)
        noise,_ = restore_noise(out/"best.pt",arrays["text"])
        images = np.load(intervention_root/"images.npy",mmap_mode="r")
        scores = views(noise,backbone,arrays,ids,"mean",batch,output=out/"background_scores.npy",
                       images=images,tokens=tokens)
        head = torch.load(out/"head.pt",map_location="cpu",weights_only=False)
    else:
        checkpoint = torch.load(out/"model.pt",map_location="cpu",weights_only=False)
        projection = nn.Linear(768,len(arrays["text"]),bias=method=="supervised_cbm")
        projection.load_state_dict(checkpoint["model"])
        features = F.normalize(torch.tensor(np.array(tokens[:,0]),dtype=torch.float32),dim=-1)
        scores = projection(features)
        if method=="supervised_cbm":
            scores = scores.sigmoid()
        scores = scores.numpy()
        head = checkpoint["head"]
        np.save(out/"background_scores.npy",scores)
    logits = head_predict(head,scores)
    np.save(out/"background_class_logits.npy",logits)
    metrics = concept_metrics(name,np.array(arrays["attrs"][ids]),scores)
    metrics["accuracy"] = float((logits.argmax(1)==labels[ids]).mean())
    original.update(background=metrics,intervention=json.loads((intervention_root/"protocol.json").read_text()))
    write_json(out/"results.json",original)
    print(json.dumps(dict(stage="background",dataset=name,method=method,seed=seed,**metrics)),flush=True)


@torch.no_grad()
def spatial_statistics(name,seed,device="cuda"):
    cache,arrays,splits,labels,records,protocol = load(name)
    out = ROOT/name/("noise_seed"+str(seed))
    if (out/"spatial_statistics.json").exists():
        return
    assert (out/"TRAINED.json").exists()
    noise,_ = restore_noise(out/"best.pt",arrays["text"],device)
    noise.eval()
    rows = []
    positive_count = 0
    local_count = 0
    for i in protocol["region_val_ids"]:
        if not arrays["mask_valid"][i,-1]:
            continue
        foreground = batch_tensor(arrays["fg224"],[i],device)[0]
        if foreground.sum()<20 or (~foreground.bool()).sum()<20:
            continue
        ks = np.flatnonzero((arrays["attrs"][i]>=.6)&np.asarray(arrays["mask_valid"][i,:-1]))
        for start in range(0,len(ks),16):
            selected = ks[start:start+16]
            tok = batch_tensor(arrays["tokens"],np.repeat(i,len(selected)),device)
            mu,sigma,_ = noise.distribution(tok,torch.tensor(selected,device=device))
            variance = sigma.square().mean(1)
            amplitude = mu.square().mean(1).sqrt()
            fg = foreground.bool()
            bg = ~fg
            for j,k in enumerate(selected):
                fg_var = float(variance[j][fg].mean())
                bg_var = float(variance[j][bg].mean())
                fg_mu = float(amplitude[j][fg].mean())
                bg_mu = float(amplitude[j][bg].mean())
                row = dict(image_id=int(i),concept=int(k),foreground_variance=fg_var,background_variance=bg_var,
                           foreground_mean=fg_mu,background_mean=bg_mu)
                # CUB disks / CLIP-selected patch masks intersect the subject.
                part = F.interpolate(batch_tensor(arrays["masks"],[i],device)[:,k:k+1],(224,224),
                                     mode="nearest")[0,0]>0
                part &= fg
                other = fg&~part
                if part.sum()>=5 and other.sum()>=5 and part.sum()<.95*fg.sum():
                    row.update(part_mean=float(amplitude[j][part].mean()),
                               other_mean=float(amplitude[j][other].mean()))
                    local_count += 1
                rows.append(row)
                positive_count += 1
    local = [r for r in rows if "part_mean" in r]
    def ratio(population,numerator,denominator):
        if not population:
            return None
        return float(np.mean([r[numerator] for r in population]))/max(
            float(np.mean([r[denominator] for r in population])),1e-12)
    ratios = dict(background_foreground_variance=ratio(rows,"background_variance","foreground_variance"),
        foreground_background_mean=ratio(rows,"foreground_mean","background_mean"),
        part_other_mean=ratio(local,"part_mean","other_mean"))
    result = dict(dataset=name,seed=seed,n_images=len(protocol["region_val_ids"]),positive_pairs=positive_count,
        local_pairs=local_count,ratios=ratios,aggregation="ratio of per-region means, matching RIVAL10",
        region_source="native CUB foreground and part-point disks" if name=="cub" else "SAM2 foreground and CLIP concept patch masks",
        computation_device=device,checkpoint_sha256=hashlib.sha256((out/"best.pt").read_bytes()).hexdigest(),rows=rows)
    write_json(out/"spatial_statistics.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["cub","cifar10","cifar100"],required=True)
    parser.add_argument("--phase",choices=["pretrain","baseline","adapt","intervention","background","spatial"],required=True)
    parser.add_argument("--method",choices=["lf_cbm","supervised_cbm","mean","noise"],default="noise")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--eval-batch",type=int,default=256)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase=="pretrain":
        pretrain(args.dataset)
    elif args.phase=="baseline":
        baseline(args.dataset,args.method,args.seed,args.device)
    elif args.phase in ["adapt","background"]:
        out = ROOT/args.dataset/(args.method+"_seed"+str(args.seed))
        out.mkdir(exist_ok=True)
        with (out/"WORK.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if args.phase=="adapt":
                assert args.method in ["mean","noise"]
                adapt(args.dataset,args.method,args.seed,batch=args.batch,eval_batch=args.eval_batch)
            else:
                background(args.dataset,args.method,args.seed,args.eval_batch)
    elif args.phase=="intervention":
        intervention(args.dataset,args.eval_batch)
    else:
        spatial_statistics(args.dataset,args.seed,args.device)


if __name__ == "__main__":
    main()
