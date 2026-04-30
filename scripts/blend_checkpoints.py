#!/usr/bin/env python3
"""
Blend Stanford base checkpoint with finetuned checkpoint.

For each alpha, produces a merged checkpoint where:
  - UNet + encoder weights = alpha * stanford + (1-alpha) * finetuned
  - Normalizer = finetuned (correct UR10/XH540 stats)
  - Config = finetuned

Usage:
  python scripts/blend_checkpoints.py
"""

import torch
import dill
import pathlib
import copy

BASE   = 'checkpoints/cup_wild_vit_l_1img.ckpt'
TUNED  = 'checkpoints/CUP/cup_wild+0417.ckpt'
OUTDIR = pathlib.Path('checkpoints/blended')
ALPHAS = [0.3, 0.5, 0.7, 0.9]  # fraction of Stanford weights

OUTDIR.mkdir(parents=True, exist_ok=True)

print(f'Loading {BASE}...')
base  = torch.load(open(BASE,  'rb'), map_location='cpu', pickle_module=dill)
print(f'Loading {TUNED}...')
tuned = torch.load(open(TUNED, 'rb'), map_location='cpu', pickle_module=dill)

base_sd  = base['state_dicts']['ema_model']
tuned_sd = tuned['state_dicts']['ema_model']

for alpha in ALPHAS:
    print(f'\nBlending alpha={alpha} (Stanford {alpha:.0%} / finetuned {1-alpha:.0%})...')
    merged_sd = {}
    for k in tuned_sd:
        t = tuned_sd[k].float()
        if k.startswith('normalizer.'):
            # always keep finetuned normalizer (UR10/XH540 stats)
            merged_sd[k] = t
        elif k in base_sd and base_sd[k].shape == t.shape:
            b = base_sd[k].float()
            merged_sd[k] = (alpha * b + (1 - alpha) * t).to(tuned_sd[k].dtype)
        else:
            # key not in base or shape mismatch — keep finetuned as-is
            merged_sd[k] = tuned_sd[k]

    payload = copy.deepcopy(tuned)
    payload['state_dicts']['ema_model'] = merged_sd
    payload['state_dicts']['model']     = merged_sd  # keep both consistent

    out_path = OUTDIR / f'blend_a{int(alpha*10):02d}_stanford{int(alpha*100)}pct.ckpt'
    torch.save(payload, open(out_path, 'wb'), pickle_module=dill)
    print(f'  saved → {out_path}')

print('\nDone. Test order recommendation: a07 first, then a05, then a03.')
