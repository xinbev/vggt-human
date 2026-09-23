python - <<'PY'
import torch

path = "/home/zhw/xyb_space/RICH/hmr4d_support/rich_test_labels.pt"
try:
    labels = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    labels = torch.load(path, map_location="cpu")

recordings = sorted({key.split("/")[1] for key in labels})
print("camera views:", len(labels))
print("unique recordings:", len(recordings))
print("\n".join(recordings))
PY