"""Recompute cross-dataset claims from saved predictions and native labels."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .cross_data import ROOT, LOCAL, SEEDS, KOH_112, write_json, background_donors, donor_background
from .cross_experiment import load, concept_metrics
from .cross_reporting import NAMES, TITLES, collect, mean_std
from .models import SpatialConcepts, ConditionalRGBNoise
from .train_spatial import head_predict


def verify_head_selection(name,method,seed,row,head,arrays,splits,labels,protocol):
    """Replay the selected head on its actual training and validation scores."""
    out = ROOT/name/f"{method}_seed{seed}"
    if method in ["mean","noise"]:
        train = np.load(out/"train_scores.npy")
        val = np.load(out/"val_scores.npy")
    else:
        checkpoint = torch.load(out/"model.pt",map_location="cpu",weights_only=False)
        projection = nn.Linear(768,len(arrays["text"]),bias=method=="supervised_cbm")
        projection.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            train = projection(torch.tensor(np.array(arrays["cls"][protocol["head_train_ids"]])))
            val = projection(torch.tensor(np.array(arrays["cls"][splits["val"]])))
            if method=="supervised_cbm":
                train,val = train.sigmoid(),val.sigmoid()
            train,val = train.numpy(),val.numpy()
    assert train.shape==(len(protocol["head_train_ids"]),len(arrays["text"]))
    assert val.shape==(len(splits["val"]),len(arrays["text"]))
    train_tensor = torch.from_numpy(train)
    assert torch.allclose(head["center"],train_tensor.mean(0),atol=2e-6,rtol=0)
    assert torch.allclose(head["scale"],train_tensor.std(0).clamp_min(.03),atol=2e-6,rtol=0)
    curve = row["head_grid"]
    assert len(curve)==5 and np.allclose([r["reference_l1"] for r in curve],[.003,.01,.03,.06,.1],atol=0,rtol=0)
    assert all(np.isclose(r["l1"],r["reference_l1"]*10/len(protocol["classes"]),atol=1e-12,rtol=0) for r in curve)
    eligible = [i for i,r in enumerate(curve) if r["NEC"]<=16] or list(range(len(curve)))
    selected = max(eligible,key=lambda i:(curve[i]["val_accuracy"],-curve[i]["val_NLL"]))
    assert selected==row["head_selected"]
    logits = head_predict(head,val)
    accuracy = float((logits.argmax(1)==labels[splits["val"]]).mean())
    nll = float(F.cross_entropy(torch.from_numpy(logits),torch.tensor(labels[splits["val"]])))
    nec = float((head["weight"].abs()>1e-5).sum(1).float().mean())
    assert all(np.isclose(actual,curve[selected][key],atol=2e-6,rtol=0)
               for actual,key in [(accuracy,"val_accuracy"),(nll,"val_NLL"),(nec,"NEC")])
    return dict(selected=selected,validation_n=len(val),training_n=len(train),
                validation_accuracy=accuracy,validation_NLL=nll,NEC=nec,
                train_only_standardization=True,validation_only_selection=True)


def main():
    torch.set_num_threads(4)
    result = json.loads(Path("reports/main/cross_results.json").read_text())
    assert result==collect()
    checks = []
    shapes = {}
    model_hashes = {}

    def close(actual,expected,description,atol=2e-6):
        assert np.isclose(actual,expected,atol=atol,rtol=0), (description,actual,expected)
        checks.append(dict(check=description,value=float(actual)))

    for name in NAMES:
        cache,arrays,splits,labels,records,protocol = load(name)
        data = result["datasets"][name]
        n_classes = len(protocol["classes"])
        k = len(arrays["text"])
        assert k==len(data["concepts"])==protocol["requested_concepts"]
        assert arrays["attrs"].shape==(len(records),k)
        assert arrays["tokens"].shape==(len(records),257,768)
        assert arrays["masks"].shape==(len(records),k+1,16,16)
        assert arrays["mask_valid"].shape==(len(records),k+1)
        assert all(not (set(splits[a])&set(splits[b])) for a,b in [("train","val"),("train","test"),("val","test")])
        assert len(splits["test"])==(5794 if name=="cub" else 10000)
        assert set(protocol["head_train_ids"])<=set(splits["train"])
        assert set(protocol["region_val_ids"])<=set(splits["val"])
        assert len(protocol["head_train_ids"])==(5394 if name=="cub" else 6000)
        assert len(protocol["region_val_ids"])==500

        if name=="cub":
            root = LOCAL/"CUB_200_2011"
            raw = np.loadtxt(root/"attributes/image_attribute_labels.txt",usecols=(0,1,2))
            native = np.zeros((11788,312),dtype="float32")
            native[raw[:,0].astype(int)-1,raw[:,1].astype(int)-1] = raw[:,2]
            official_ids = np.load(cache/"native_image_ids.npy")
            assert np.array_equal(np.array(arrays["attrs"]),native[official_ids-1][:,KOH_112])
            original_classes = dict(map(int,line.split()) for line in (root/"image_class_labels.txt").read_text().splitlines())
            original_splits = dict(map(int,line.split()) for line in (root/"train_test_split.txt").read_text().splitlines())
            assert np.array_equal(labels,np.asarray([original_classes[int(i)]-1 for i in official_ids]))
            assert all(original_splits[int(i)]==int(r["source"]=="train") for i,r in zip(official_ids,records))
        else:
            assert not list((ROOT/name).glob("supervised_cbm_seed*"))
            calibration = json.loads((cache/"teacher_calibration.json").read_text())
            teacher = np.array(arrays["teacher"])
            center = teacher[splits["train"]].mean(0)
            scale = teacher[splits["train"]].std(0).clip(1e-4)
            assert np.allclose(center,calibration["center"],atol=1e-7)
            assert np.allclose(scale,calibration["scale"],atol=1e-7)
            expected = 1/(1+np.exp(-np.clip(.75*(teacher-center)/scale,-20,20)))
            assert np.array_equal(np.array(arrays["attrs"]),expected.astype("float32"))
            assert all("mAP" not in r[condition] for rows in data["models"].values()
                       for r in rows for condition in ["original","background"])
            region_source = json.loads((cache/"REGIONS.json").read_text())
            region_ids = region_source["selected_ids"]
            assert set(region_ids)==set(protocol["head_train_ids"]+protocol["region_val_ids"]+protocol["background_ids"])
            assert region_source["grounding_class_source"]=="frozen CLIP zero-shot original-image prediction"
            audit = json.loads((cache/"foreground_audit.json").read_text())
            assert set(map(int,audit))==set(region_ids)
            class_text = np.load(cache/"class_text.npy")
            predicted = (np.asarray(arrays["cls"][region_ids])@class_text.T).argmax(1)
            assert all(audit[str(i)]["predicted_class"]==int(k) for i,k in zip(region_ids,predicted))
            reviewed = json.loads((cache/"FOREGROUND_REVIEW.json").read_text())
            assert reviewed["status"]=="passed"
            reviewed_ids = reviewed["ids"]+reviewed["donors"]
            assert hashlib.sha256(np.asarray(arrays["fg224"][reviewed_ids]).tobytes()).hexdigest()==reviewed["mask_sha256"]
            assert all(Path(p).is_file() for p in reviewed["figures"])

        background_root = ROOT/name/"background"
        ids = np.load(background_root/"ids.npy")
        donors = np.load(background_root/"donors.npy")
        images = np.load(background_root/"images.npy",mmap_mode="r")
        intervention = json.loads((background_root/"protocol.json").read_text())
        assert len(ids)==1000 and np.array_equal(ids,np.asarray(protocol["background_ids"]))
        assert set(ids)<=set(splits["test"]) and set(donors)<=set(splits["val"])
        assert np.all(labels[ids]!=labels[donors])
        donor_pool = splits["val"][np.asarray(arrays["mask_valid"][splits["val"],-1])]
        expected_donors,crops,_ = background_donors(name,labels,donor_pool,ids,arrays["fg224"],audit if name!="cub" else None)
        assert np.array_equal(donors,expected_donors)
        if name!="cub":
            assert [list(r) for r in crops]==intervention["donor_background_crops"]
        if name!="cub":
            for i,d in zip(reviewed["ids"],reviewed["donors"]):
                assert int(donors[np.flatnonzero(ids==i)[0]])==d
            construction = json.loads((ROOT/name/"foreground_review/CONSTRUCTION_VERIFICATION.json").read_text())
            assert construction["passed"] and construction["n"]==1000
            assert hashlib.sha256(np.asarray(images).tobytes()).hexdigest()==construction["composite_images_sha256"]
        fingerprint = hashlib.sha256(ids.tobytes()+donors.tobytes()+np.asarray(images).tobytes()).hexdigest()
        assert fingerprint==intervention["sha256"]
        for j,(i,d) in enumerate(zip(ids,donors)):
            target = np.array(arrays["rgb"][i],dtype="float32")/255
            background = donor_background(name,arrays["rgb"],arrays["fg224"],d,audit if name!="cub" else None,crops[j] if crops else None)
            fg = np.asarray(arrays["fg224"][i])[None]
            expected = target*fg+background*(~fg)
            assert np.array_equal(images[j],expected)
        background_tokens = np.load(background_root/"tokens.npy",mmap_mode="r")

        for method,rows in data["models"].items():
            assert [r["seed"] for r in rows]==SEEDS
            model_hashes[name+"/"+method] = []
            for row in rows:
                seed = row["seed"]
                out = ROOT/name/(method+"_seed"+str(seed))
                assert row["intervention"]["sha256"]==fingerprint
                if method in ["mean","noise"]:
                    head = torch.load(out/"head.pt",map_location="cpu",weights_only=False)
                    ck = torch.load(out/"best.pt",map_location="cpu",weights_only=False)
                    train_state = json.loads((out/"TRAINED.json").read_text())
                    assert train_state["epochs"]==6 and train_state["train_n"]==len(protocol["head_train_ids"])
                    assert row["inference"]=="mean" and row["train_concept_views"]==method
                    audit = json.loads((out/"gradient_audit.json").read_text())
                    assert audit["actual_RGB"] and audit["mean_gradient_norm"]>0
                    if method=="noise":
                        assert audit["sigma_gradient_norm"]>0
                    for condition,sample in [("train",protocol["head_train_ids"]),("val",splits["val"]),("test",splits["test"])]:
                        done = json.loads((out/(condition+"_scores.complete.json")).read_text())
                        assert done["n"]==len(sample) and done["k"]==k
                        assert done["mode"]==("mean" if condition=="test" else method)
                    weights = ck["model"]["mean.weight"]
                    if seed==42 and method=="noise":
                        concept_model = SpatialConcepts(torch.tensor(np.array(arrays["text"])),region_readout=True)
                        model = ConditionalRGBNoise(concept_model,concept_local=True,mean_budget=protocol["mean_budget"])
                        model.load_state_dict(ck["model"])
                        model.eval()
                        token = torch.tensor(np.array(arrays["tokens"][splits["val"][:2]]),dtype=torch.float32)
                        with torch.no_grad():
                            logits,masks,foreground = model.concepts(token)
                            mu,sigma,gate = model.distribution(token,torch.tensor([0,k-1]))
                        shapes[name] = dict(tokens=list(token.shape),concept_logits=list(logits.shape),
                            concept_regions=list(masks.shape),foreground=list(foreground.shape),
                            mu=list(mu.shape),sigma=list(sigma.shape),gate=list(gate.shape),
                            head=list(head["weight"].shape))
                        assert list(logits.shape)==[2,k] and list(masks.shape)==[2,k,16,16]
                        assert list(mu.shape)==list(sigma.shape)==[2,3,224,224]
                else:
                    ck = torch.load(out/"model.pt",map_location="cpu",weights_only=False)
                    head = ck["head"]
                    weights = ck["model"]["weight"]
                    projection = nn.Linear(768,k,bias=method=="supervised_cbm")
                    projection.load_state_dict(ck["model"])
                    with torch.no_grad():
                        for condition,features in [
                            ("test",torch.tensor(np.array(arrays["cls"][splits["test"]]))),
                            ("background",F.normalize(torch.tensor(np.array(background_tokens[:,0]),dtype=torch.float32),dim=-1))]:
                            expected = projection(features)
                            if method=="supervised_cbm":
                                expected = expected.sigmoid()
                            assert np.allclose(np.load(out/(condition+"_scores.npy")),expected.numpy(),atol=3e-6)
                    assert row["head_train_n"]==len(protocol["head_train_ids"])
                    assert row["concept_train_n"]==len(splits["train"])
                model_hashes[name+"/"+method].append(hashlib.sha256(weights.numpy().tobytes()).hexdigest())
                assert list(head["weight"].shape)==[n_classes,k]
                selection = verify_head_selection(name,method,seed,row,head,arrays,splits,labels,protocol)
                checks.append(dict(check=f"{name}/{method}/{seed}/validation head selection",**selection))
                for condition,sample in [("test",splits["test"]),("background",ids)]:
                    scores = np.load(out/(condition+"_scores.npy"))
                    logits = np.load(out/(condition+"_class_logits.npy"))
                    assert scores.shape==(len(sample),k) and np.isfinite(scores).all()
                    assert logits.shape==(len(sample),n_classes)
                    assert np.allclose(logits,head_predict(head,scores),atol=1e-5)
                    actual = concept_metrics(name,np.array(arrays["attrs"][sample]),scores)
                    actual["accuracy"] = float((logits.argmax(1)==labels[sample]).mean())
                    recorded = row["original" if condition=="test" else "background"]
                    for metric,value in actual.items():
                        close(value,recorded[metric],f"{name}/{method}/{seed}/{condition}/{metric}")
                close((head["weight"].abs()>1e-5).sum(1).float().mean(),row["original"]["NEC"],
                      f"{name}/{method}/{seed}/NEC from actual classifier")
            assert len(set(model_hashes[name+"/"+method]))==3

        for stats in data["spatial"]:
            assert stats["seed"] in SEEDS and stats["n_images"]==500
            if "checkpoint_sha256" in stats:
                checkpoint = ROOT/name/f"noise_seed{stats['seed']}/best.pt"
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==stats["checkpoint_sha256"]
            full_stats = json.loads((ROOT/name/("noise_seed"+str(stats["seed"]))/"spatial_statistics.json").read_text())
            assert stats=={key:value for key,value in full_stats.items() if key!="rows"}
            rows = full_stats["rows"]
            local = [r for r in rows if "part_mean" in r]
            assert len(rows)==stats["positive_pairs"] and len(local)==stats["local_pairs"]
            expected_pairs = set()
            for i in protocol["region_val_ids"]:
                fg = np.asarray(arrays["fg224"][i])
                if fg.sum()<20 or (~fg).sum()<20 or not arrays["mask_valid"][i,-1]:
                    continue
                ks = np.flatnonzero((arrays["attrs"][i]>=.6)&np.asarray(arrays["mask_valid"][i,:-1]))
                expected_pairs.update((int(i),int(k)) for k in ks)
            assert {(r["image_id"],r["concept"]) for r in rows}==expected_pairs
            for population,numerator,denominator,key in [
                (rows,"background_variance","foreground_variance","background_foreground_variance"),
                (rows,"foreground_mean","background_mean","foreground_background_mean"),
                (local,"part_mean","other_mean","part_other_mean")]:
                value = np.mean([r[numerator] for r in population])/max(np.mean([r[denominator] for r in population]),1e-12)
                close(value,stats["ratios"][key],f"{name}/{stats['seed']}/{key} from individual regions")
        mean = data["models"]["mean"]
        noise = data["models"]["noise"]
        for condition,key in [("original","original_accuracy_pp"),("background","background_accuracy_pp")]:
            gain = np.mean([b[condition]["accuracy"]-a[condition]["accuracy"] for a,b in zip(mean,noise)])*100
            close(gain,data["gain"][key],name+"/"+key)
        metric = "mAP" if name=="cub" else "concept_correlation"
        gain = np.mean([b["background"][metric]-a["background"][metric] for a,b in zip(mean,noise)])
        key = "background_concept_mAP_pp" if name=="cub" else "background_concept_correlation"
        close(gain*(100 if name=="cub" else 1),data["gain"][key],name+"/"+key)

    for path,expected in result["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    for path,expected in result["implementation_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    report = Path("EXPERIMENT_REPORT.md")
    text = report.read_text()
    assert re.findall(r"^## (.+)$",text,re.M)==["Abstract","Background（为什么要做）","原理与方法","主要实验和结果"]
    assert re.findall(r"^### ([abc])\.",text,re.M)==["a","b","c"]
    assert len(re.findall(r"^#### CUB 与 CIFAR 扩展$",text,re.M))==3
    assert sorted(str(p) for p in Path(".").rglob("*.md") if "artifacts" not in p.parts and ".git" not in p.parts)==["EXPERIMENT_REPORT.md"]
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)",text)
    assert len(figures)==9
    for filename in figures:
        path = report.parent/filename
        with Image.open(path) as image:
            image.verify()
        assert path.with_suffix(".svg").is_file() and path.with_suffix(".pdf").is_file()
    for name,data in result["datasets"].items():
        assert TITLES[name] in text
        for method in (["lf_cbm","supervised_cbm","noise"] if name=="cub" else ["lf_cbm","noise"]):
            rows = data["models"][method]
            for metric,scale,precision in [("accuracy",100,2),("NEC",1,2)]:
                assert mean_std([r["original"][metric] for r in rows],scale,precision) in text
        assert f"{data['gain']['background_accuracy_pp']:+.2f} 个百分点" in text
    evidence = dict(passed=True,datasets=NAMES,seeds=SEEDS,models=30,test_conditions=60,
        numerical_checks=len(checks),checks=checks,actual_shapes=shapes,model_tensor_sha256=model_hashes,
        figures=figures,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        validation="All saved test/background scores and class logits; native CUB per-image labels; CIFAR soft targets; classifier NEC; region statistics; exact manipulated pixels; report values.")
    write_json("reports/main/cross_verification.json",evidence)
    print(f"Verified {len(checks)} cross-dataset numerical claims, 30 fitted models and 60 prediction conditions",flush=True)


if __name__=="__main__":
    main()
