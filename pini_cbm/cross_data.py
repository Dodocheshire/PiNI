"""Dataset-local caches for the CUB / CIFAR extension of the RGB-noise CBM.

Native CUB instance attributes and keypoints are used as supplied. CIFAR uses
CLIP soft concept targets and SAM2 foreground pseudo-labels, never native AP.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as component_labels
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from clip import clip
from .data import normalize

ROOT = Path("artifacts/cross_dataset")
LOCAL = Path("artifacts/datasets/local")
SEEDS = [42, 43, 44]
SPLIT_SEED = 310
MEAN_BUDGET = 0.02686347473647816
SAM_CHECKPOINT = Path("../D-Pi/artifacts/models/sam2.1_hiera_large.pt")
GROUNDING_CHECKPOINT = Path("../D-Pi/artifacts/models/grounding-dino-tiny")

# These are zero-based native CUB attribute indices published by Koh et al.
KOH_112 = [
    1,4,6,7,10,14,15,20,21,23,25,29,30,35,36,38,40,44,45,50,51,53,54,
    56,57,59,63,64,69,70,72,75,80,84,90,91,93,99,101,106,110,111,116,
    117,119,125,126,131,132,134,145,149,151,152,153,157,158,163,164,168,
    172,178,179,181,183,187,188,193,194,196,198,202,203,208,209,211,212,
    213,218,220,221,225,235,236,238,239,240,242,243,244,249,253,254,259,
    260,262,268,274,277,283,289,292,293,294,298,299,304,305,308,309,310,311,
]
assert len(KOH_112) == 112


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def fixed_subset(ids, labels, size, seed):
    """Exact-size class-stratified sample, including the small CUB split."""
    ids = np.asarray(ids)
    if size >= len(ids):
        return ids.copy()
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(ids[labels[ids] == k]) for k in np.unique(labels[ids])]
    selected = []
    offset = 0
    while len(selected) < size:
        for group in groups:
            if offset < len(group) and len(selected) < size:
                selected.append(int(group[offset]))
        offset += 1
    return np.sort(np.asarray(selected, dtype="int64"))


def source_datasets(name):
    if name == "cub":
        return ImageFolder(LOCAL/"CUB/train"), ImageFolder(LOCAL/"CUB/test")
    dataset = CIFAR10 if name == "cifar10" else CIFAR100
    return dataset(LOCAL, train=True, download=False), dataset(LOCAL, train=False, download=False)


def prepare_manifest(name):
    destination = ROOT/name/"cache"
    destination.mkdir(parents=True, exist_ok=True)
    train, test = source_datasets(name)
    labels_train = np.asarray(train.targets)
    train_ids, val_ids = train_test_split(
        np.arange(len(train)), test_size=.1, random_state=SPLIT_SEED,
        stratify=labels_train)
    val_set = set(val_ids.tolist())
    records = []
    for source, ds in [("train", train), ("test", test)]:
        for i, label in enumerate(ds.targets):
            r = dict(source=source, source_index=i, label=int(label),
                     split="val" if source == "train" and i in val_set else source)
            if name == "cub":
                r["path"] = str(Path(ds.samples[i][0]).resolve())
                r["relative_path"] = Path(ds.samples[i][0]).relative_to(LOCAL/"CUB"/source).as_posix()
            records.append(r)
    labels = np.asarray([r["label"] for r in records])
    splits = {s:np.asarray([i for i,r in enumerate(records) if r["split"] == s])
              for s in ["train", "val", "test"]}
    protocol = dict(dataset=name, split_seed=SPLIT_SEED, seeds=SEEDS,
        classes=train.classes, counts={s:len(v) for s,v in splits.items()},
        head_train_ids=fixed_subset(splits["train"], labels, 6000, 44).tolist(),
        region_val_ids=fixed_subset(splits["val"], labels, 500, 45).tolist(),
        background_ids=fixed_subset(splits["test"], labels, 1000, 46).tolist(),
        backbone="OpenAI CLIP ViT-L/14 224", resize="RGB bicubic 224x224",
        mean_budget=MEAN_BUDGET, region_epochs=40, adapt_epochs=6,
        native_concepts=name == "cub", requested_concepts={"cub":112,"cifar10":64,"cifar100":128}[name],
        native_labels="per-image CUB instance attributes" if name == "cub" else None)
    if (destination/"manifest.json").exists():
        assert json.loads((destination/"manifest.json").read_text()) == records
    write_json(destination/"manifest.json", records)
    write_json(destination/"protocol.json", protocol)
    print(json.dumps(dict(stage="manifest",dataset=name,counts=protocol["counts"])),flush=True)
    return destination, records, protocol


class RawImages(Dataset):
    def __init__(self, name, records):
        self.train, self.test = source_datasets(name)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        im, _ = (self.test if r["source"] == "test" else self.train)[r["source_index"]]
        rgb = np.asarray(im.convert("RGB").resize((224,224), Image.Resampling.BICUBIC)).copy()
        return i, torch.from_numpy(rgb).permute(2,0,1)


def memmap(path, shape, dtype, restart=False):
    path = Path(path)
    mode = "w+" if restart or not path.exists() else "r+"
    array = np.lib.format.open_memmap(path,mode=mode,dtype=dtype,shape=shape)
    assert array.shape == shape and array.dtype == np.dtype(dtype)
    return array


@torch.no_grad()
def encode(name, batch=48):
    destination, records, protocol = prepare_manifest(name)
    if (destination/"ENCODED.json").exists():
        return
    n = len(records)
    tokens = memmap(destination/"tokens.npy",(n,257,768),"float16")
    rgb = memmap(destination/"rgb.npy",(n,3,224,224),"uint8")
    model,_ = clip.load("ViT-L/14",device="cuda",jit=False)
    model.eval().requires_grad_(False)
    # Preserve the whole LF candidate bank until selection uses training only.
    if name == "cub":
        names = [line.split(" ",1)[1].strip() for line in (LOCAL/"attributes.txt").read_text().splitlines()]
        candidate_names = [names[i] for i in KOH_112]
        prompts = []
        for x in candidate_names:
            group, value = x.split("::")
            group = group.replace("has_","").replace("_"," ")
            value = value.replace("_"," ").replace("(","").replace(")","")
            prompts.append("a photo of a bird with "+value+" "+group+".")
    else:
        source = Path("../LF-Pi/configs/concepts")/(name+"_official.txt")
        candidate_names = list(dict.fromkeys(line.strip() for line in source.read_text().splitlines() if line.strip()))
        prompts = ["a photo showing "+x+"." for x in candidate_names]
    text = F.normalize(model.encode_text(clip.tokenize(prompts).cuda()).float(),dim=-1)
    class_names = [x.split(".",1)[-1].replace("_"," ") if name=="cub" else x.replace("_"," ")
                   for x in protocol["classes"]]
    class_text = F.normalize(model.encode_text(clip.tokenize(["a photo of a "+x+"." for x in class_names]).cuda()).float(),dim=-1)
    background_text = F.normalize(model.encode_text(clip.tokenize(
        ["a photo of an empty "+x+"." for x in ["background","room","road","field","sky","forest","lake"]]).cuda()).float(),dim=-1)
    np.save(destination/"candidate_text.npy",text.cpu().numpy())
    np.save(destination/"class_text.npy",class_text.cpu().numpy())
    np.save(destination/"background_text.npy",background_text.cpu().numpy())
    write_json(destination/"candidate_concepts.json",candidate_names)
    progress_path = destination/"encode_progress.json"
    done = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
    dataset = torch.utils.data.Subset(RawImages(name,records),range(done,n))
    loader = DataLoader(dataset,batch_size=batch,num_workers=4,pin_memory=True,shuffle=False)
    started = time.time()
    for number,(ids,images) in enumerate(loader):
        z = model.encode_image(normalize(images.cuda(non_blocking=True).float()/255),spatial=True)
        indices = ids.numpy()
        tokens[indices] = z.cpu().numpy()
        rgb[indices] = images.numpy()
        if number % 20 == 0 or indices[-1] == n-1:
            tokens.flush();rgb.flush()
            write_json(progress_path,dict(done=int(indices[-1])+1,total=n))
            print(json.dumps(dict(stage="encode",dataset=name,done=int(indices[-1])+1,total=n,
                elapsed_s=round(time.time()-started,1))),flush=True)
    tokens.flush();rgb.flush()
    write_json(destination/"ENCODED.json",dict(n=n,counts=protocol["counts"],
        manifest_sha256=hashlib.sha256((destination/"manifest.json").read_bytes()).hexdigest()))


def select_concepts(name):
    destination, records, protocol = prepare_manifest(name)
    assert (destination/"ENCODED.json").exists()
    if (destination/"CONCEPTS.json").exists():
        return
    tokens = np.load(destination/"tokens.npy",mmap_mode="r")
    candidates = np.load(destination/"candidate_text.npy")
    names = json.loads((destination/"candidate_concepts.json").read_text())
    features = np.empty((len(records),768),dtype="float32")
    for start in range(0,len(records),512):
        z = np.array(tokens[start:start+512,0],dtype="float32")
        features[start:start+len(z)] = z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-6)
    np.save(destination/"cls.npy",features)
    teacher = features @ candidates.T
    train = np.asarray([i for i,r in enumerate(records) if r["split"] == "train"])
    labels = np.asarray([r["label"] for r in records])
    if name == "cub":
        selected = np.arange(112)
        selection = "fixed Koh2020 published 112 native attribute indices"
    else:
        # The same selected bank is supplied to the LF and spatial models.
        # Scores use train images only; class names themselves are not concepts.
        train_scores = teacher[train]
        means = np.stack([train_scores[labels[train]==k].mean(0) for k in np.unique(labels[train])])
        quality = means.var(0)/train_scores.var(0).clip(1e-8)
        class_names = {re.sub(r"[^a-z0-9 ]","",x.lower().replace("_"," ")).strip() for x in protocol["classes"]}
        for j,x in enumerate(names):
            plain = re.sub(r"^(a |an |the )","",x.lower()).strip()
            if plain in class_names:
                quality[j] = -np.inf
        selected = np.sort(np.argsort(-quality,kind="stable")[:protocol["requested_concepts"]])
        selection = "LF candidate bank ranked by between-class / total CLIP score variance on training split only"
        np.save(destination/"candidate_training_quality.npy",quality)
    text = candidates[selected]
    np.save(destination/"text.npy",text)
    np.save(destination/"teacher.npy",teacher[:,selected])
    concepts = [names[i] for i in selected]
    write_json(destination/"concepts.json",concepts)
    if name == "cub":
        root = LOCAL/"CUB_200_2011"
        by_name = {line.split()[1]:int(line.split()[0])-1 for line in (root/"images.txt").read_text().splitlines()}
        raw = np.loadtxt(root/"attributes/image_attribute_labels.txt",usecols=(0,1,2))
        native = np.zeros((11788,312),dtype="float32")
        native[raw[:,0].astype(int)-1,raw[:,1].astype(int)-1] = raw[:,2]
        ids = np.asarray([by_name[r["relative_path"]] for r in records])
        targets = native[ids][:,KOH_112]
        np.save(destination/"native_image_ids.npy",ids+1)
        raw_parts = np.loadtxt(root/"parts/part_locs.txt")
        parts = np.zeros((11788,15,3),dtype="float32")
        parts[raw_parts[:,0].astype(int)-1,raw_parts[:,1].astype(int)-1] = raw_parts[:,2:5]
        np.save(destination/"parts.npy",parts[ids])
    else:
        selected_teacher = teacher[:,selected]
        center = selected_teacher[train].mean(0)
        scale = selected_teacher[train].std(0).clip(1e-4)
        targets = 1/(1+np.exp(-np.clip(.75*(selected_teacher-center)/scale,-20,20)))
        write_json(destination/"teacher_calibration.json",dict(center=center.tolist(),scale=scale.tolist(),
            multiplier=.75,fit_split="train",target="sigmoid of training-standardized CLIP score"))
    np.save(destination/"attrs.npy",targets.astype("float32"))
    write_json(destination/"CONCEPTS.json",dict(k=len(selected),selected_candidate_indices=selected.tolist(),
        native_attribute_indices=KOH_112 if name=="cub" else None,selection=selection))
    print(json.dumps(dict(stage="concepts",dataset=name,k=len(selected),selection=selection)),flush=True)


def cub_part_indices(concept):
    group = concept.split("::")[0]
    for word, parts in [
        ("bill",[1]),("wing",[8,12]),("upperparts",[0,8,9,12]),
        ("underparts",[2,3,14]),("breast",[3]),("back",[0]),("tail",[13]),
        ("head",[4,5,6,9,10]),("throat",[14]),("eye",[6,10]),
        ("forehead",[5]),("nape",[9]),("belly",[2]),("leg",[7,11]),("crown",[4])]:
        if word in group:
            return parts
    return None


def cub_regions(destination, records, masks, valid, fg224):
    attrs = np.load(destination/"attrs.npy")
    parts = np.load(destination/"parts.npy")
    concepts = json.loads((destination/"concepts.json").read_text())
    yy,xx = np.mgrid[:224,:224]
    for i,r in enumerate(records):
        foreground = np.asarray(Image.open(LOCAL/"segmentations"/Path(r["relative_path"]).with_suffix(".png"))
            .convert("L").resize((224,224),Image.Resampling.NEAREST))>0
        fg224[i] = foreground
        fg16 = foreground.reshape(16,14,16,14).mean((1,3))
        masks[i,-1] = fg16
        valid[i,-1] = True
        with Image.open(r["path"]) as im:
            width,height = im.size
        for k,concept in enumerate(concepts):
            if not attrs[i,k]:
                valid[i,k] = True
                continue
            indices = cub_part_indices(concept)
            if indices is None:
                region = foreground
            else:
                visible = [p for p in indices if parts[i,p,2]>0]
                if not visible:
                    continue
                region = np.zeros((224,224),dtype=bool)
                for p in visible:
                    x,y = parts[i,p,:2]*np.asarray([224/width,224/height])
                    region |= (xx-x)**2+(yy-y)**2 <= (224*.09)**2
                region &= foreground
            masks[i,k] = region.reshape(16,14,16,14).mean((1,3))
            valid[i,k] = region.any()
        if i%1000 == 0:
            print(json.dumps(dict(stage="cub_regions",done=i,total=len(records))),flush=True)
    write_json(destination/"REGIONS.json",dict(source="CUB native object segmentations; visible part point disks",
        disk_radius_fraction=.09,concept_indices=KOH_112,n=len(records)))


class CifarForegroundSelector:
    """Select image-only SAM masks by foreground/background CLIP contrast."""
    def __init__(self,class_text,class_names):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        sampler = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",str(SAM_CHECKPOINT),device="cuda",apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(sampler)
        self.encoder,_ = clip.load("ViT-L/14",device="cuda",jit=False)
        self.encoder.eval().requires_grad_(False)
        self.class_text = torch.tensor(class_text,device="cuda",dtype=torch.float32)
        self.class_vectors = np.asarray(class_text)
        self.class_names = class_names
        self.processor = AutoProcessor.from_pretrained(GROUNDING_CHECKPOINT,local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_CHECKPOINT,local_files_only=True).cuda().eval().requires_grad_(False)
        self.centers = [(112,112),(67,112),(157,112),(112,67),(112,157)]

    @torch.no_grad()
    def __call__(self,images,features):
        proposals = [[] for _ in images]
        qualities = [[] for _ in images]
        predicted = (np.asarray(features)@self.class_vectors.T).argmax(1)
        captions = ["a "+self.class_names[int(k)].replace("_"," ")+"." for k in predicted]
        inputs = self.processor(images=[Image.fromarray(x) for x in images],text=captions,return_tensors="pt",padding=True).to("cuda")
        outputs = self.detector(**inputs)
        detections = self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            threshold=0,text_threshold=.25,target_sizes=[(224,224)]*len(images))
        boxes = [d["boxes"][d["scores"].argmax()].float().cpu().numpy() for d in detections]
        corners = [(7,7),(217,7),(7,217),(217,217)]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            self.predictor.set_image_batch(images)
            for negative_corners in [False,True]:
                labels = [np.asarray([1,*([0]*4 if negative_corners else [])],dtype="int32") for _ in images]
                for point in self.centers:
                    coords = [np.asarray([point,*(corners if negative_corners else [])],dtype="float32") for _ in images]
                    masks,scores,_ = self.predictor.predict_batch(coords,labels,multimask_output=True)
                    for j,(mask,score) in enumerate(zip(masks,scores)):
                        proposals[j].extend(mask.astype(bool));qualities[j].extend(np.asarray(score).tolist())
            masks,scores,_ = self.predictor.predict_batch(box_batch=boxes,multimask_output=True)
            for j,(mask,score) in enumerate(zip(masks,scores)):
                proposals[j].extend(mask.astype(bool));qualities[j].extend(np.asarray(score).tolist())
        selected = []
        audits = []
        for image,masks,quality,target,box in zip(images,proposals,qualities,predicted,boxes):
            rgb = torch.from_numpy(image.copy()).permute(2,0,1).float()/255
            candidates = []
            views = []
            reliable = [j for j,mask in enumerate(masks) if .02<float(mask.mean())<.85 and quality[j]>=.5]
            if not reliable:
                reliable = [j for j,mask in enumerate(masks) if mask.any() and not mask.all()]
            for j,mask in enumerate(masks):
                if j not in reliable:
                    continue
                tensor = torch.from_numpy(mask)
                foreground = rgb*tensor[None]+.5*(~tensor)[None]
                background = rgb*(~tensor)[None]+.5*tensor[None]
                views.extend([foreground,background]);candidates.append(j)
            assert candidates,"SAM produced no usable foreground mask"
            encoded = []
            for start in range(0,len(views),48):
                inputs = normalize(torch.stack(views[start:start+48]).cuda())
                encoded.append(F.normalize(self.encoder.encode_image(inputs).float(),dim=-1))
            values = torch.cat(encoded)@self.class_text[int(target)]
            contrast = (values[::2]-values[1::2]).cpu().numpy()
            best = max(range(len(candidates)),key=lambda j:(float(contrast[j]),quality[candidates[j]]))
            index = candidates[best]
            # Merge consistent proposals for the same object, keeping its parts.
            base = masks[index]
            same_object = [j for j in reliable if np.logical_and(base,masks[j]).sum()/max(1,np.logical_or(base,masks[j]).sum())>=.5]
            foreground = np.logical_or.reduce([masks[j] for j in same_object])
            components,n_components = component_labels(foreground)
            supported = [component for component in range(1,n_components+1)
                         if np.logical_and(components==component,base).sum()>=16]
            if supported:
                foreground = np.isin(components,supported)
            selected.append(foreground)
            audits.append(dict(predicted_class=int(target),foreground_background_contrast=float(contrast[best]),
                               sam_quality=float(quality[index]),area=float(foreground.mean()),candidate=index,
                               merged_candidates=same_object,
                               grounding_box=box.tolist()))
        return selected,audits


@torch.no_grad()
def cifar_regions(name, destination, records, protocol, masks, valid, fg224, sam_batch=4):
    rgb = np.load(destination/"rgb.npy",mmap_mode="r")
    tokens = np.load(destination/"tokens.npy",mmap_mode="r")
    cls = np.load(destination/"cls.npy",mmap_mode="r")
    text = np.load(destination/"text.npy")
    class_text = np.load(destination/"class_text.npy")
    attrs = np.load(destination/"attrs.npy")
    selector = CifarForegroundSelector(class_text,protocol["classes"])
    audit_path = destination/"foreground_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    selected = np.unique(protocol["head_train_ids"]+protocol["region_val_ids"]+protocol["background_ids"])
    finished = set(np.flatnonzero(np.asarray(valid[:,-1])).tolist()).intersection(map(int,audit))
    priority = list(dict.fromkeys(protocol["background_ids"]+protocol["region_val_ids"]+protocol["head_train_ids"]))
    todo = np.asarray([i for i in priority if i not in finished],dtype="int64")
    started = time.time()
    for start in range(0,len(todo),sam_batch):
        ids = todo[start:start+sam_batch]
        images = [np.asarray(rgb[i]).transpose(1,2,0).copy() for i in ids]
        patch_features = []
        for i in ids:
            patch = np.array(tokens[i,1:],dtype="float32")
            patch /= np.linalg.norm(patch,axis=-1,keepdims=True).clip(1e-6)
            patch_features.append(patch)
        foregrounds,selection_audit = selector(images,cls[ids])
        for i,foreground,details,patch in zip(ids,foregrounds,selection_audit,patch_features):
            masks[i] = 0
            valid[i] = False
            audit[str(int(i))] = details
            fg224[i] = foreground
            fg16 = foreground.reshape(16,14,16,14).mean((1,3))
            masks[i,-1] = fg16
            similarities = patch@text.T
            inside = fg16.reshape(-1)>.15
            for k in range(len(text)):
                if attrs[i,k] < .4:
                    valid[i,k] = True
                elif attrs[i,k] >= .6 and inside.any():
                    score = similarities[:,k]
                    threshold = np.quantile(score[inside],.6)
                    region = (score>=threshold).reshape(16,16)*fg16
                    masks[i,k] = region
                    valid[i,k] = region.any()
            valid[i,-1] = True
        if start%(sam_batch*20)==0 or start+sam_batch>=len(todo):
            masks.flush();valid.flush();fg224.flush()
            write_json(audit_path,audit)
            print(json.dumps(dict(stage="sam_regions",dataset=name,done=min(start+sam_batch,len(todo)),
                remaining_at_start=len(todo),elapsed_s=round(time.time()-started,1))),flush=True)
    write_json(destination/"REGIONS.json",dict(source="frozen Grounding DINO boxes and SAM2.1 point/box proposals; frozen CLIP foreground-background contrast; overlapping proposal consensus",
        checkpoint_sha256=hashlib.sha256(SAM_CHECKPOINT.read_bytes()).hexdigest(),
        grounding_checkpoint_sha256=hashlib.sha256((GROUNDING_CHECKPOINT/"model.safetensors").read_bytes()).hexdigest(),
        grounding_provenance=json.loads((GROUNDING_CHECKPOINT/"PROVENANCE.json").read_text()),
        positive_points=selector.centers,negative_corners=[(7,7),(217,7),(7,217),(217,217)],
        grounding_class_source="frozen CLIP zero-shot original-image prediction",
        candidate_area_range=[.02,.85],minimum_sam_quality=.5,consensus_minimum_iou=.5,
        component_minimum_selected_mask_overlap=16,selection="highest original-size RGB foreground minus background class cosine; SAM quality tie-break",
        foreground_train_n=len(protocol["head_train_ids"]),foreground_validation_n=len(protocol["region_val_ids"]),
        foreground_test_n=len(protocol["background_ids"]),native_concept_annotation=False,
        concept_region_quantile=.6,positive_soft_target=.6,negative_soft_target=.4,
        selected_ids=selected.tolist()))


def regions(name, sam_batch=4):
    destination,records,protocol = prepare_manifest(name)
    assert (destination/"CONCEPTS.json").exists()
    if (destination/"COMPLETE.json").exists():
        return
    k = json.loads((destination/"CONCEPTS.json").read_text())["k"]
    masks = memmap(destination/"masks.npy",(len(records),k+1,16,16),"float16")
    valid = memmap(destination/"mask_valid.npy",(len(records),k+1),"bool")
    fg224 = memmap(destination/"fg224.npy",(len(records),224,224),"bool")
    if name == "cub":
        cub_regions(destination,records,masks,valid,fg224)
    else:
        cifar_regions(name,destination,records,protocol,masks,valid,fg224,sam_batch)
    for array in [masks,valid,fg224]:
        array.flush()
    write_json(destination/"COMPLETE.json",dict(dataset=name,n=len(records),k=k,
        counts=protocol["counts"],native_concepts=name=="cub",
        manifest_sha256=hashlib.sha256((destination/"manifest.json").read_bytes()).hexdigest()))


def donor_removal_mask(name,foreground,index,audit=None):
    """Remove the complete CIFAR donor object and its surrounding box."""
    removal = np.asarray(foreground,dtype=bool).copy()
    if name=="cub":
        return removal
    assert audit is not None
    box = np.asarray(audit[str(int(index))]["grounding_box"],dtype=float)
    y,x = np.nonzero(removal)
    if len(x):
        box = np.asarray([min(box[0],x.min()),min(box[1],y.min()),
                          max(box[2],x.max()+1),max(box[3],y.max()+1)])
    x0,y0 = np.floor(box[:2]-14).astype(int).clip(0,224)
    x1,y1 = np.ceil(box[2:]+14).astype(int).clip(0,224)
    removal[y0:y1,x0:x1] = True
    return removal


def background_donors(name,labels,val_ids,test_ids,foreground,audit=None):
    """Use fixed validation donors; CIFAR donors supply an object-free crop."""
    rectangles = {}
    pool = np.asarray(val_ids,dtype="int64")
    if name!="cub":
        for i in pool:
            removal = donor_removal_mask(name,foreground[i],i,audit)
            y,x = np.nonzero(removal)
            assert len(x)
            candidates = [(0,0,224,int(y.min())),(0,int(y.max())+1,224,224),
                          (0,0,int(x.min()),224),(int(x.max())+1,0,224,224)]
            acceptable = [r for r in candidates if min(r[2]-r[0],r[3]-r[1])>=14
                          and (r[2]-r[0])*(r[3]-r[1])>=.1*224*224]
            if acceptable:
                rectangle = max(acceptable,key=lambda r:(r[2]-r[0])*(r[3]-r[1]))
                assert not removal[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]].any()
                rectangles[int(i)] = rectangle
        pool = np.asarray([i for i in pool if int(i) in rectangles],dtype="int64")
        assert len(pool),"No validation donor supplies an object-free background crop"
    rng = np.random.default_rng(47)
    donors = np.asarray([rng.choice(pool[labels[pool]!=labels[i]]) for i in test_ids])
    crops = [rectangles[int(d)] for d in donors] if name!="cub" else None
    return donors,crops,pool


def donor_background(name,rgb,foreground,index,audit=None,crop=None):
    if name!="cub":
        image = Image.fromarray(np.asarray(rgb[index]).transpose(1,2,0))
        return np.asarray(image.crop(tuple(crop)).resize((224,224),Image.Resampling.BICUBIC),dtype="float32").transpose(2,0,1)/255
    image = np.array(rgb[index],dtype="float32")/255
    removal = np.asarray(foreground[index])[None]
    return image*(~removal)+.5*removal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["cub","cifar10","cifar100"],required=True)
    parser.add_argument("--stage",choices=["manifest","encode","concepts","regions","all"],default="all")
    parser.add_argument("--batch",type=int,default=48)
    parser.add_argument("--sam-batch",type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.stage in ["manifest","all"]:
        prepare_manifest(args.dataset)
    if args.stage in ["encode","all"]:
        encode(args.dataset,args.batch)
    if args.stage in ["concepts","all"]:
        select_concepts(args.dataset)
    if args.stage in ["regions","all"]:
        regions(args.dataset,args.sam_batch)


if __name__ == "__main__":
    main()
