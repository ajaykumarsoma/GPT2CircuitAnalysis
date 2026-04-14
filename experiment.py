"""
GPT-2 Induction Head Circuit Analysis
======================================
Experiment: Detect and visualise induction heads in GPT-2 Small.

Background
----------
An INDUCTION CIRCUIT is a two-layer motif discovered by Anthropic (2022):
  Layer 0 — "previous-token heads": at position t, copy token[t-1] into the
             residual stream via the K-composition pathway.
  Layer 1 — "induction heads": at position t, find the key that matches the
             current token (token[t]) and attend to its SUCCESSOR position,
             effectively predicting: "whatever came after this token last time".

Test: Repeated random sequences
--------------------------------
We feed GPT-2 sequences of the form:
    [a b c d e f ... | a b c d e f ...]   (random tokens, then repeated)
                        ^-- second half is identical to first half

At position T + k in the second half, the induction head should attend back to
position k (the first occurrence of the same token), specifically to position k+1
(the token that *followed* it last time).

Induction score = mean attention weight on the correct "look-back" diagonal.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformer_lens import HookedTransformer
from einops import rearrange, reduce
import os, time

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = "cpu"          # TransformerLens hooks are most stable on CPU
MODEL_NAME    = "gpt2"         # GPT-2 Small: 12 layers, 12 heads, d_model=768
N_LAYERS      = 12
N_HEADS       = 12
SEQ_HALF      = 50             # half-length of repeated sequence
BATCH         = 20             # number of random sequences to average over
SEED          = 42
PLOTS_DIR     = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(SEED)

# ── 1. Load model ─────────────────────────────────────────────────────────────
print("=" * 60)
print("GPT-2 Induction Head Experiment")
print("=" * 60)
print(f"Loading {MODEL_NAME} on {DEVICE}...")
t0 = time.time()
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
print(f"  Loaded in {time.time()-t0:.1f}s  |  "
      f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
