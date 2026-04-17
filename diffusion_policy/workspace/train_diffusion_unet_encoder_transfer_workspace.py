"""
Workspace that supports encoder-only weight transfer from a pretrained checkpoint.

Unlike TrainDiffusionUnetImageWorkspace (which reuses the normalizer from pretrained_ckpt),
this workspace always computes the normalizer fresh from the current dataset.

Use case: transfer the obs_encoder from cup_wild_vit_l_1img.ckpt to a new task
(tossing, peg-in-hole) where the action distribution is completely different.

Config keys:
  training.encoder_pretrained_ckpt: path to source checkpoint (obs_encoder only loaded)
  training.pretrained_ckpt: not used here; ignored if set
"""

import copy
import os
import pickle

import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


class TrainDiffusionUnetEncoderTransferWorkspace(TrainDiffusionUnetImageWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        # Build the model and optimizer via parent __init__, but we override
        # the pretrained_ckpt loading so parent must not see it
        encoder_ckpt = cfg.training.get('encoder_pretrained_ckpt', None)

        # Temporarily hide pretrained_ckpt so parent __init__ skips full model loading
        with open_dict_safe(cfg) as c:
            saved = c.training.pop('pretrained_ckpt', None)
            c.training['pretrained_ckpt'] = None

        super().__init__(cfg, output_dir=output_dir)

        # Restore pretrained_ckpt in config (cosmetic)
        with open_dict_safe(cfg) as c:
            if saved is not None:
                c.training['pretrained_ckpt'] = saved

        # Load only obs_encoder weights from encoder_pretrained_ckpt
        if encoder_ckpt is not None:
            import dill
            print(f'==> loading encoder-only weights from {encoder_ckpt}')
            payload = torch.load(open(encoder_ckpt, 'rb'),
                                 map_location='cpu', pickle_module=dill)
            sd = payload['state_dicts']['model']
            encoder_sd = {k: v for k, v in sd.items() if k.startswith('obs_encoder.')}
            missing, unexpected = self.model.load_state_dict(encoder_sd, strict=False)
            n_loaded = len(encoder_sd)
            n_missing = len([k for k in missing if k.startswith('obs_encoder.')])
            print(f'   encoder keys loaded: {n_loaded}  '
                  f'encoder keys missing: {n_missing}  '
                  f'non-encoder keys (expected unset): {len(unexpected)}')
            if self.ema_model is not None:
                self.ema_model.load_state_dict(encoder_sd, strict=False)

    def run(self):
        # Override only the normalizer loading part: always compute from dataset.
        # Everything else (training loop, checkpointing) is inherited unchanged.
        cfg = copy.deepcopy(self.cfg)

        # Force normalizer computation from current dataset by nulling pretrained_ckpt
        # before the parent run() checks it.  We do this via a patched config copy.
        with open_dict_safe(cfg) as c:
            c.training['pretrained_ckpt'] = None

        # Swap cfg on self temporarily so parent run() sees patched version
        original_cfg = self.cfg
        self.cfg = cfg
        try:
            super().run()
        finally:
            self.cfg = original_cfg


class open_dict_safe:
    """Context manager: temporarily open OmegaConf struct for mutation."""
    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        OmegaConf.set_struct(self.cfg, False)
        return self.cfg

    def __exit__(self, *args):
        OmegaConf.set_struct(self.cfg, True)
