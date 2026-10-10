"""Render fixed CIFAR objects and the exact shared background intervention."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .cross_data import ROOT, write_json, background_donors, donor_background, donor_removal_mask


def review(name):
    cache = ROOT/name/"cache"
    protocol = json.loads((cache/"protocol.json").read_text())
    records = json.loads((cache/"manifest.json").read_text())
    labels = np.asarray([r["label"] for r in records])
    rgb = np.load(cache/"rgb.npy",mmap_mode="r")
    masks = np.load(cache/"fg224.npy",mmap_mode="r")
    valid = np.load(cache/"mask_valid.npy",mmap_mode="r")
    audit = json.loads((cache/"foreground_audit.json").read_text())
    ids = np.asarray(protocol["background_ids"])
    pool = np.sort(protocol["region_val_ids"])
    assert np.asarray(valid[np.r_[ids,pool],-1]).all(),"Background and donor masks are still being generated"
    donors,crops,pool = background_donors(name,labels,pool,ids,masks,audit)
    classes = (protocol["classes"] if name=="cifar10" else
               ["aquarium_fish","beaver","bicycle","bus","butterfly","forest","mountain","poppy","tiger","willow_tree"])
    positions = [int(np.flatnonzero(labels[ids]==protocol["classes"].index(c))[0]) for c in classes]
    selected = ids[positions]
    selected_donors = donors[positions]
    out = ROOT/name/"foreground_review"
    out.mkdir(exist_ok=True)
    image_hash = hashlib.sha256()
    changed = []
    assert len(ids)==1000 and np.all(labels[ids]!=labels[donors])
    for i,d,crop in zip(ids,donors,crops):
        target = np.asarray(rgb[i],dtype="float32")/255
        fg = np.asarray(masks[i])
        background = donor_background(name,rgb,masks,d,audit,crop)
        swapped = target*fg[None]+background*(~fg)[None]
        assert np.array_equal(swapped[:,fg],target[:,fg])
        assert np.isfinite(swapped).all() and swapped.min()>=0 and swapped.max()<=1
        removal = donor_removal_mask(name,masks[d],d,audit)
        assert not removal[crop[1]:crop[3],crop[0]:crop[2]].any()
        image_hash.update(swapped.tobytes())
        changed.append(float(np.any(swapped!=target,axis=0).mean()))
    write_json(out/"CONSTRUCTION_VERIFICATION.json",dict(passed=True,n=1000,donor_pool_n=len(pool),
        all_target_foreground_pixels_identical=True,all_donor_crops_outside_object_box_and_mask=True,
        all_donor_native_classes_different=True,mean_image_pixels_changed=float(np.mean(changed)),
        composite_images_sha256=image_hash.hexdigest()))
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    plt.rcParams.update({"font.family":"WenQuanYi Zen Hei","font.weight":500,"axes.titleweight":500})
    samples = []
    for position,i,d in zip(positions,selected,selected_donors):
        target = np.asarray(rgb[i],dtype="float32").transpose(1,2,0)/255
        fg = np.asarray(masks[i])
        background = donor_background(name,rgb,masks,d,audit,crops[position]).transpose(1,2,0)
        swapped = target*fg[...,None]+background*(~fg)[...,None]
        assert np.array_equal(swapped[fg],target[fg])
        samples.append(dict(id=int(i),donor=int(d),class_name=protocol["classes"][labels[i]],
                            foreground_area=float(fg.mean()),
                            background_changed_fraction=float(np.any(swapped!=target,axis=-1)[~fg].mean())))
    paths = []
    for board in range((len(selected)+4)//5):
        start = board*5
        count = min(5,len(selected)-start)
        fig,axes = plt.subplots(count,4,figsize=(12,3*count),squeeze=False,layout="constrained")
        for row,position in enumerate(range(start,start+count)):
            i,d = selected[position],selected_donors[position]
            target = np.asarray(rgb[i],dtype="float32").transpose(1,2,0)/255
            fg = np.asarray(masks[i])
            background = donor_background(name,rgb,masks,d,audit,crops[positions[position]]).transpose(1,2,0)
            kept = target*fg[...,None]+.5*(~fg)[...,None]
            swapped = target*fg[...,None]+background*(~fg)[...,None]
            for ax,im in zip(axes[row],[target,kept,background,swapped]):
                ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
            axes[row,0].set_ylabel(samples[position]["class_name"]+f" / {i}")
        for ax,title in zip(axes[0],["原图","保留的主体","供体的纯背景片区","实际换背景图"]):
            ax.set_title(title)
        path = out/f"background_review_{board}.png"
        fig.savefig(path,dpi=120);plt.close(fig);paths.append(str(path))
    fingerprint = hashlib.sha256(np.asarray(masks[np.r_[selected,selected_donors]]).tobytes()).hexdigest()
    state = dict(dataset=name,status="ready_for_visual_review",selection="first fixed background-test image in each predefined class, no accuracy filtering",
                 ids=selected.tolist(),donors=selected_donors.tolist(),donor_background_crops=[crops[p] for p in positions],mask_sha256=fingerprint,
                 samples=samples,figures=paths,
                 selector_sha256=hashlib.sha256((Path(__file__).parent/"cross_data.py").read_bytes()).hexdigest())
    write_json(cache/"FOREGROUND_REVIEW.json",state)
    print(json.dumps(dict(dataset=name,figures=paths,mask_sha256=fingerprint)),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["cifar10","cifar100"],required=True)
    review(parser.parse_args().dataset)
