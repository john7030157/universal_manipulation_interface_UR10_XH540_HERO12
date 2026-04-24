"""
Finetune variant of TrainDiffusionUnetImageWorkspace.

Fixes four issues that cause bad results when finetuning a pretrained
checkpoint (e.g. Stanford's cup_wild_vit_l_1img.ckpt):

  1. Loads the pretrained `ema_model` weights (the actually-deployed version)
     into both self.model and self.ema_model, instead of the noisier raw
     `model` weights. Stanford's eval uses ema_model; starting from their
     `model` weights throws away their EMA smoothing before we even train.

  2. Prints the actual missing / unexpected keys (not just counts) so silent
     architecture mismatches surface as warnings instead of random init.

  3. Reduces learning rate on the pretrained UNet (not just the encoder),
     so early finetune gradients don't blast through Stanford's priors.

  4. ALWAYS recomputes the normalizer from the new dataset (never reuses the
     pretrained checkpoint's normalizer). Stanford's cup_wild was recorded on
     a UR5 with a Robotiq gripper — the workspace bounds and gripper range are
     completely different from this UR10 + XH540 setup. Reusing their
     normalizer maps actions to wrong positions and wrong gripper widths.
     This was the root cause of "weird" robot behavior after fine-tuning.

Everything else is delegated to the parent workspace.
"""

if __name__ == "__main__":
    import sys
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    import os
    os.chdir(ROOT_DIR)

import copy
import random
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


class TrainDiffusionUnetImageFinetuneWorkspace(TrainDiffusionUnetImageWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        # ────── skip parent __init__; do model setup ourselves ──────
        from diffusion_policy.workspace.base_workspace import BaseWorkspace
        BaseWorkspace.__init__(self, cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # instantiate policy
        self.model = hydra.utils.instantiate(cfg.policy)

        # ────── Fix #1 + #2: load pretrained EMA weights with verbose key check ──────
        pretrained_ckpt = cfg.training.get('pretrained_ckpt', None)
        if pretrained_ckpt is not None:
            import dill
            print(f'==> [finetune] loading pretrained from {pretrained_ckpt}')
            payload = torch.load(open(pretrained_ckpt, 'rb'),
                                 map_location='cpu', pickle_module=dill)

            state_dicts = payload['state_dicts']
            # Prefer ema_model (polished weights) over raw model weights
            if 'ema_model' in state_dicts:
                sd = state_dicts['ema_model']
                print('==> [finetune] using ema_model weights (preferred)')
            else:
                sd = state_dicts['model']
                print('==> [finetune] ema_model not in ckpt; falling back to model weights')

            # Strip normalizer keys — Fix #4: we will recompute normalizer from
            # our own data in run(), so don't load Stanford's workspace statistics
            sd_no_norm = {k: v for k, v in sd.items() if not k.startswith('normalizer.')}
            missing, unexpected = self.model.load_state_dict(sd_no_norm, strict=False)
            print(f'==> [finetune] load result: {len(missing)} missing, {len(unexpected)} unexpected')

            # Verbose key diagnostics — architecture mismatches would hide here
            if missing:
                print('   Missing keys (first 20):')
                for k in list(missing)[:20]:
                    print(f'     - {k}')
                if len(missing) > 20:
                    print(f'     ... and {len(missing) - 20} more')
            if unexpected:
                print('   Unexpected keys (first 20):')
                for k in list(unexpected)[:20]:
                    print(f'     - {k}')
                if len(unexpected) > 20:
                    print(f'     ... and {len(unexpected) - 20} more')

            # Refuse to continue if too many non-normalizer keys are missing
            non_norm_keys = [k for k in self.model.state_dict() if not k.startswith('normalizer.')]
            total_params = len(non_norm_keys)
            if len(missing) > 0.2 * total_params:
                raise RuntimeError(
                    f'Too many missing keys ({len(missing)}/{total_params}); '
                    'architecture probably does not match the pretrained checkpoint. '
                    'Check model_name, down_dims, diffusion_step_embed_dim, etc.'
                )

        # EMA starts as a copy of self.model — which now already holds the
        # pretrained EMA weights (see Fix #1 above).
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # ────── Fix #3: reduce LR on pretrained encoder; UNet LR from config ──────
        base_lr = cfg.optimizer.lr
        finetune_lr_scale = cfg.training.get('finetune_lr_scale', 0.1)

        is_full_finetune = pretrained_ckpt is not None

        # encoder LR: reduce if encoder has any form of pretraining
        encoder_lr = base_lr
        if cfg.policy.obs_encoder.pretrained:
            encoder_lr = base_lr * finetune_lr_scale
            print(f'==> [finetune] encoder LR = {encoder_lr:.2e} ({finetune_lr_scale}x of {base_lr:.2e})')

        # UNet LR: use unet_lr from config if set, else reduce when loading pretrained
        unet_lr = cfg.training.get('unet_lr', None)
        if unet_lr is not None:
            print(f'==> [finetune] UNet LR    = {unet_lr:.2e} (from training.unet_lr config)')
        elif is_full_finetune:
            unet_lr = base_lr * finetune_lr_scale
            print(f'==> [finetune] UNet LR    = {unet_lr:.2e} ({finetune_lr_scale}x — full finetune)')
        else:
            unet_lr = base_lr
            print(f'==> [finetune] UNet LR    = {unet_lr:.2e} (full; not a full finetune)')

        encoder_params = [p for p in self.model.obs_encoder.parameters() if p.requires_grad]
        print(f'encoder params: {len(encoder_params)}')

        param_groups = [
            {'params': self.model.model.parameters(), 'lr': unet_lr},
            {'params': encoder_params, 'lr': encoder_lr},
        ]

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        optimizer_cfg.pop('lr', None)  # lr now supplied per param group
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg,
        )

        self.global_step = 0
        self.epoch = 0

        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        # Fix #4: force the parent run() to recompute normalizer from our data.
        # The parent reuses the pretrained normalizer when pretrained_ckpt is set,
        # but Stanford's UR5/Robotiq statistics are wrong for this UR10/XH540 setup.
        # We null out pretrained_ckpt in a local cfg copy so the parent skips that
        # branch and calls dataset.get_normalizer() instead.
        cfg_patched = copy.deepcopy(self.cfg)
        with open_dict(cfg_patched):
            cfg_patched.training.pretrained_ckpt = None
        original_cfg = self.cfg
        self.cfg = cfg_patched
        try:
            super().run()
        finally:
            self.cfg = original_cfg


@hydra.main(
    version_base=None,
    config_path=str(__import__('pathlib').Path(__file__).parent.parent.joinpath("config")),
)
def main(cfg):
    workspace = TrainDiffusionUnetImageFinetuneWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
