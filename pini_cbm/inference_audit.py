"""Verify the RGB-only predictor against actual saved held-out predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .infer import SpatialNoiseCBM
from .train_spatial import head_predict,load_cache


def main():
    torch.set_num_threads(4)
    run=Path('artifacts/region_noise');_,splits,labels,records=load_cache('artifacts/rival_vitl14')
    ids=np.array([splits['test'][labels[splits['test']]==k][0] for k in range(10)])
    rgb=[]
    for i in ids:
        with Image.open(records[i]['path']) as image:rgb.append(np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)).copy())
    pixels=torch.tensor(np.stack(rgb).transpose(0,3,1,2),dtype=torch.uint8)
    model=SpatialNoiseCBM(run);probs,logits=model.predict(pixels,mode='mean')
    positions={int(i):j for j,i in enumerate(splits['test'])}
    reference=np.load(run/'test_mean_probs.npy')[[positions[int(i)] for i in ids]]
    delta=float(np.max(np.abs(reference-probs)));assert delta<.002,delta
    reference_class=np.load(run/'test_mean_class_logits.npy')[[positions[int(i)] for i in ids]].argmax(1)
    assert np.array_equal(logits.argmax(1),reference_class)
    head=torch.load(run/'head.pt',map_location='cpu',weights_only=False)
    recomputed=head_predict(head,probs);class_delta=float(np.max(np.abs(recomputed-logits)));assert class_delta<1e-4,class_delta
    stochastic,stochastic_logits=model.predict(pixels,mode='noise')
    stochastic_class_delta=float(np.max(np.abs(head_predict(head,stochastic)-stochastic_logits)))
    assert stochastic_class_delta<1e-4,stochastic_class_delta
    stochastic_probability_delta=float(np.max(np.abs(stochastic-probs)));assert stochastic_probability_delta>1e-4
    result=dict(run=str(run),checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
        predictor_source_sha256=hashlib.sha256(Path('pini_cbm/infer.py').read_bytes()).hexdigest(),
        head_sha256=hashlib.sha256((run/'head.pt').read_bytes()).hexdigest(),test_ids=ids.tolist(),n=len(ids),
        selection='First held-out image of each predefined class; no prediction-based selection',
        actual_native_probability_shape=list(probs.shape),class_head_input_shape=list(probs.shape),
        probability_max_difference_from_saved_mean_predictions=delta,tolerance=.002,
        predicted_classes_identical_to_saved_reference=True,class_logits_max_difference_from_concept_only_head=class_delta,
        gaussian_probability_max_difference_from_mean=stochastic_probability_delta,
        gaussian_class_logits_max_difference_from_concept_only_head=stochastic_class_delta,
        inference_input='Original RGB files only; no labels, segmentation masks, or precomputed tokens are passed to the predictor',
        numerical_protocol='uint8-to-float division on CUDA; original selective FP16 padded batch 64; perturbed AMP padded batch 128; archived generator parameter metadata preserved under no_grad',
        interpretation='Small held-out runtime contract audit; complete test performance is measured by the separate 5286-image evaluation. Batch shape can change FP16 rounding; padded serving tails are not the archived incomplete tail protocol.')
    Path('reports/spatial_noise_v1/RGB_only_inference.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
