"""
VC-1 (Meta Visual Cortex) observation encoder for UMI diffusion policy.

VC-1 is pretrained with MAE on 4000+ hours of egocentric video (Ego4D, EPIC-Kitchens,
etc.). Stronger spatial understanding than R3M; designed specifically for embodied AI.

Install: pip install vc-models

GitHub: https://github.com/facebookresearch/eai-vc
HuggingFace: facebook/vc1-base, facebook/vc1-large

Note: VC-1 expects 250×250 input. Images are resized automatically by this wrapper.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


class VC1ObsEncoder(nn.Module):
    """
    Drop-in replacement for TimmObsEncoder using Meta VC-1 as image backbone.

    Follows the same interface as TimmObsEncoder: takes obs_dict,
    returns (B, feature_dim) conditioning vector.

    Args:
        shape_meta: dataset shape metadata
        model_variant: 'base' (ViT-B, 768-dim) or 'large' (ViT-L, 1024-dim)
        frozen: freeze backbone weights (recommended for < 100 demos)
        transforms: augmentation list
    """

    def __init__(
        self,
        shape_meta: dict,
        model_variant: str = 'large',
        frozen: bool = False,
        transforms: Optional[list] = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_variant = model_variant
        self.frozen = frozen

        try:
            from vc_models.models.vit import model_utils
        except ImportError:
            raise ImportError(
                'vc-models not installed. Run: pip install vc-models\n'
                'See: https://github.com/facebookresearch/eai-vc'
            )

        name_map = {
            'base': model_utils.VC1_BASE_NAME,
            'large': model_utils.VC1_LARGE_NAME,
        }
        model, embd_size, vc1_transforms, _ = model_utils.load_model(name_map[model_variant])
        self.backbone = model
        self._img_feat_dim = embd_size  # 768 for base, 1024 for large
        # VC1 ships its own transforms (resize to 250, normalize)
        self._vc1_transform = vc1_transforms

        if frozen:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

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

        # Optional extra augmentation applied BEFORE VC1 transform
        aug_list = []
        if transforms:
            for t in transforms:
                if isinstance(t, dict) and t.get('type') == 'RandomCrop':
                    ratio = t.get('ratio', 0.95)
                    aug_list.append(T.RandomCrop(int(224 * ratio)))
                    aug_list.append(T.Resize(224))
                elif callable(t):
                    aug_list.append(t)
        self.augment = T.Compose(aug_list) if aug_list else nn.Identity()

        self._lowdim_dim = sum(
            int(np.prod(shape_meta['obs'][k]['shape'])) * self._obs_horizons[k]
            for k in self._lowdim_keys
        )

    def output_shape(self) -> Tuple[int]:
        total_img_horizon = sum(self._obs_horizons[k] for k in self._rgb_keys)
        return (self._img_feat_dim * total_img_horizon + self._lowdim_dim,)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        B = None

        for key in self._rgb_keys:
            x = obs_dict[key]
            if x.dim() == 4:
                x = x.unsqueeze(1)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            x = self.augment(x)
            # VC1 transform: resize to 250, ImageNet normalization
            x = self._vc1_transform(x)
            if self.frozen:
                with torch.no_grad():
                    feat = self.backbone(x)
            else:
                feat = self.backbone(x)
            feat = feat.reshape(B, T * self._img_feat_dim)
            features.append(feat)

        for key in self._lowdim_keys:
            x = obs_dict[key]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            B_ld = x.shape[0]
            if B is None:
                B = B_ld
            features.append(x.reshape(B_ld, -1))

        return torch.cat(features, dim=-1)
