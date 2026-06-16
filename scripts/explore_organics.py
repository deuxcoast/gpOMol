import numpy as np

from explore_omol25 import build_descriptor_matrix, knn_diagnostic, load_dataset

ORGANIC = {"H", "C", "N", "O", "S"}
dataset = load_dataset("/Volumes/LaCie/gpCAM/OMol25/train_4M")

rng = np.random.default_rng(0)
pool = rng.choice(len(dataset), size=60000, replace=False)
keep = []
for i in pool:
    a = dataset.get_atoms(int(i))
    if 20 <= len(a) <= 60 and set(a.get_chemical_symbols()) <= ORGANIC:
        keep.append(int(i))
    if len(keep) >= 5000:
        break
print(f"kept {len(keep)} CHNOS molecules, 20-60 atoms")

X = build_descriptor_matrix(dataset, keep)
knn_diagnostic(X, k=5, standardize=True)  # is it still bimodal in here?
