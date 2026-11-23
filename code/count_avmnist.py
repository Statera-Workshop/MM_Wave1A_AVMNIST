import numpy as np, sys
d = sys.argv[1]
exp = {"train": 60000, "test": 10000}
ok = True
for s in ["train", "test"]:
    im = np.load(f"{d}/image/{s}_data.npy"); au = np.load(f"{d}/audio/{s}_data.npy"); lb = np.load(f"{d}/{s}_labels.npy")
    print(s, "image", im.shape, im.dtype, im.min(), im.max(), "| audio", au.shape, au.dtype, au.min(), au.max(), "| labels", lb.shape)
    n = exp[s]
    if not (len(im) == len(au) == len(lb) == n): ok = False; print("  COUNT MISMATCH, expected", n)
    if im.reshape(n, -1).shape[1] != 784 or au.reshape(n, -1).shape[1] != 112 * 112: ok = False; print("  SHAPE MISMATCH")
    print("  per class:", np.bincount(lb.astype(int), minlength=10).tolist())
    if s == "train":
        v = lb[55000:].astype(int); print("  val(55000:) per class:", np.bincount(v, minlength=10).tolist())
print("RESULT", "OK" if ok else "FAIL")
