import torch
from videollama2 import model_init
model, _, tok = model_init("/workspace/LMM-EVQA/SnapUGC/videollama2_EVQA_weights_mse")
print("has lm_head:", hasattr(model, "lm_head"), type(getattr(model, "lm_head", None)))
print("has score:",   hasattr(model, "score"),   type(getattr(model, "score", None)))
for n, _ in model.named_children(): print(n)
