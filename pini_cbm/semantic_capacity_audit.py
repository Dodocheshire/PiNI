"""Label-overlap audit for the information in true binary native concepts.

Test labels are used only to calculate an empirical information bound. The
test-fitted majority table is never a trained model or a selection rule.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import CLASSES
from .train_spatial import load_cache


def main():
    cache=Path('artifacts/rival_vitl14');arrays,splits,labels,_=load_cache(cache)
    ids=splits['test'];x=np.array(arrays['attrs'][ids]);assert np.isin(x,[0,1]).all()
    configurations,inverse=np.unique(x.astype(np.uint8),axis=0,return_inverse=True)
    counts=np.zeros((len(configurations),len(CLASSES)),dtype=int)
    np.add.at(counts,(inverse,labels[ids]),1)
    assert counts.sum()==len(ids)
    best=int(counts.max(1).sum());ambiguous=(counts>0).sum(1)>1
    result=dict(n_test=len(ids),n_true_binary_concept_configurations=len(counts),
                n_ambiguous_configurations=int(ambiguous.sum()),
                empirical_true_binary_label_overlap_upper_bound=best/len(ids),
                unavoidable_errors_on_this_sample=len(ids)-best,
                manifest_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest(),
                interpretation='Upper bound on deterministic class readout from exactly the 18 TRUE binary concepts on this fixed observed test sample. A diagnostic, not a fitted/test-selected model or an out-of-sample bound. Does not bound informative continuous semantic strengths or larger concept sets.',
                ambiguous_configurations=[dict(concepts=configurations[j].tolist(),
                    class_counts={CLASSES[k]:int(v) for k,v in enumerate(counts[j]) if v})
                    for j in np.flatnonzero(ambiguous)])
    out=Path('reports/spatial_noise_v1');out.mkdir(parents=True,exist_ok=True)
    (out/'semantic_capacity.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='ambiguous_configurations'},flush=True)


if __name__=='__main__':main()
