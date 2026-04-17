"""
R3M (Meta) observation encoder for UMI diffusion policy.

R3M is pretrained on Ego4D human manipulation video with time-contrastive +
language-conditioned reward learning. Shows 20%+ improvement over CLIP on real
robot tasks with only ~20 demos per task.

Install: pip install r3m
  or:    pip install voltron-robotics  (includes R3M, MVP, VC-1)

Paper: https://arxiv.org/abs/2203.12601
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


class R3MObsEncoder(nn.Module):
    """
    Drop-in replacement for TimmObsEncoder that uses R3M as the image backbone.

    Follows the exact same interface: takes obs_dict with rgb + low_dim keys,
    returns a flat (B, feature_dim) tensor ready for the diffusion conditioning.

    Args:
        shape_meta: dataset shape metadata (same as TimmObsEncoder)
        model_name: 'resnet50' (2048-dim) or 'resnet18' (512-dim)
        frozen: if True, freeze R3M weights (recommended if data < 100 demos)
        transforms: list of torchvision transform configs (applied before R3M normalization)
        imagenet_norm: apply ImageNet normalization (R3M expects this)
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str = 'resnet50',
        frozen: bool = False,
        transforms: Optional[list] = None,
        imagenet_norm: bool = True,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.frozen = frozen

        # Load R3M backbone
        try:
            import r3m
        except ImportError:
            raise ImportError(
                'R3M not installed. Run: pip install r3m\n'
                'or: pip install voltron-robotics'
            )

        r3m_model = r3m.load_r3m(model_name)
        # R3M wraps the backbone in a module; extract the feature encoder
        # r3m.R3M has .module which is the actual ResNet
        self.backbone = r3m_model.module if hasattr(r3m_model, 'module') else r3m_model
        self.backbone.eval() if frozen else self.backbone.train()

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Feature dim: resnet50 → 2048, resnet18 → 512
        _feat_dims = {'resnet50': 2048, 'resnet18': 512}
        self._img_feat_dim = _feat_dims.get(model_name, 2048)

        # Parse shape_meta
        self._rgb_keys = []
        self._lowdim_keys = []
        self._obs_horizons = {}

        for key, meta in shape_meta['obs'].items():
            if meta.get('ignore_by_policy', False):
                continue
            obs_type = meta.get('type', 'low_dim')
            horizon = meta.get('horizon', 1)
            self._obs_horizons[key] = horizon
            if obs_type == 'rgb':
                self._rgb_keys.append(key)
            else:
                self._lowdim_keys.append(key)

        # Build augmentation transform (applied before normalization)
        aug_list = []
        if transforms:
            for t in transforms:
                if isinstance(t, dict) and t.get('type') == 'RandomCrop':
                    ratio = t.get('ratio', 0.95)
                    aug_list.append(T.RandomCrop(int(224 * ratio)))
                    aug_list.append(T.Resize(224))
                elif isinstance(t, dict) and '_target_' in t:
                    # Hydra-style instantiation not available here — skip
                    pass
                elif callable(t):
                    aug_list.append(t)
        self.augment = T.Compose(aug_list) if aug_list else nn.Identity()

        # R3M expects ImageNet normalization on [0,1] float images
        if imagenet_norm:
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.normalize = nn.Identity()

        # Low-dim total dimension
        self._lowdim_dim = sum(
            int(np.prod(shape_meta['obs'][k]['shape'])) * self._obs_horizons[k]
            for k in self._lowdim_keys
        )

    def output_shape(self) -> Tuple[int]:
        n_rgb = len(self._rgb_keys)
        total_img_horizon = sum(self._obs_horizons[k] for k in self._rgb_keys)
        return (self._img_feat_dim * total_img_horizon + self._lowdim_dim,)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        B = None

        # Process RGB images
        for key in self._rgb_keys:
            x = obs_dict[key]  # (B, T, C, H, W) or (B, C, H, W)
            if x.dim() == 4:
                x = x.unsqueeze(1)  # (B, 1, C, H, W)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)   # (B*T, C, H, W)
            x = self.augment(x)
            x = self.normalize(x)
            if self.frozen:
                with torch.no_grad():
                    feat = self.backbone(x)   # (B*T, feat_dim)
            else:
                feat = self.backbone(x)
            feat = feat.reshape(B, T * self._img_feat_dim)  # (B, T*feat_dim)
            features.append(feat)

        # Process low-dim observations
        for key in self._lowdim_keys:
            x = obs_dict[key]  # (B, T, D) or (B, D)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            B_ld = x.shape[0]
            if B is None:
                B = B_ld
            x = x.reshape(B_ld, -1)  # (B, T*D)
            features.append(x)

        return torch.cat(features, dim=-1)  # (B, total_dim)
