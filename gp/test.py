import numpy as np

d = np.load("./referencing/20260617_130856_reference.npz")
print(d.files)
print("train_idx:", d["train_idx"].shape, "r_train:", d["r_train"].shape)
print("test_idx:", d["test_idx"].shape, "r_test:", d["r_test"].shape)
print("no index overlap:", len(set(d["train_idx"]) & set(d["test_idx"])) == 0)
