"""
Flow Matching policy using the same ConditionalUnet1D backbone as DiffusionUnetTimmPolicy.

Key differences from DDPM:
  - Forward process: x_t = (1-t)*x_0 + t*noise  (linear interpolation, t in [0,1])
  - Model predicts velocity: v = noise - x_0
  - Inference: Euler ODE integration from t=1 (pure noise) to t=0 (action)
  - Timestep sampling: logit-normal (concentrates samples near t=0.5, harder region)
  - No external scheduler needed — all inline

References:
  - Lipman et al. "Flow Matching for Generative Modeling" (2022)
  - Esser et al. "Scaling Rectified Flow Transformers" / SD3 (2024)
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class FlowMatchingUnetTimmPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: TimmObsEncoder,
            num_inference_steps: int = 10,
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 256,
            down_dims=(256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            # logit-normal timestep sampling params (mean=0, std=1 → sigmoid → [0,1])
            logit_normal_mean: float = 0.0,
            logit_normal_std: float = 1.0,
            train_diffusion_n_samples: int = 1,
            **kwargs
        ):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        assert obs_as_global_cond
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

    # ========= helpers ============

    def _sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        """
        Logit-normal timestep sampling: t = sigmoid(N(mean, std)).
        Concentrates samples near t=0.5 where the flow is most ambiguous.
        Returns t in (0, 1), shape [batch_size].
        """
        u = torch.normal(
            mean=self.logit_normal_mean,
            std=self.logit_normal_std,
            size=(batch_size,),
            device=device
        )
        return torch.sigmoid(u)

    def _interpolate(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1-t)*x0 + t*noise
        t: [B] → broadcast to [B, T, D]
        """
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x0 + t * noise

    # ========= inference ============

    def conditional_sample(self,
            condition_data: torch.Tensor,
            condition_mask: torch.Tensor,
            local_cond=None,
            global_cond=None,
            generator=None,
        ) -> torch.Tensor:
        """
        Euler ODE integration from t=1 (pure noise) → t=0 (action).
        """
        device = condition_data.device
        dtype = condition_data.dtype
        B = condition_data.shape[0]

        # start from pure noise at t=1
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator
        )

        # uniform timesteps from 1 → 0
        timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)

        for i in range(self.num_inference_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur  # negative (going 1→0)

            # apply inpainting conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # predict velocity: v ≈ noise - x_0
            t_batch = t_cur.expand(B)
            # ConditionalUnet1D expects integer-like timesteps; scale to [0, 1000)
            t_scaled = (t_batch * 999).long()
            velocity = self.model(
                trajectory, t_scaled,
                local_cond=local_cond,
                global_cond=global_cond
            )

            # Euler step: x_{t+dt} = x_t + dt * v  (dt < 0 → moves toward data)
            trajectory = trajectory + dt * velocity

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       fixed_action_prefix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        global_cond = self.obs_encoder(nobs)

        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device, dtype=self.dtype
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
        )

        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        return {
            'action': action_pred,
            'action_pred': action_pred
        }

    # ========= training ============

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch) -> torch.Tensor:
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(
                global_cond, repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_diffusion_n_samples, dim=0)

        x0 = nactions  # [B, T, D] clean action trajectory
        B = x0.shape[0]
        noise = torch.randn_like(x0)

        # sample t ~ logit-normal in (0, 1)
        t = self._sample_timesteps(B, device=x0.device)

        # forward process: x_t = (1-t)*x0 + t*noise
        x_t = self._interpolate(x0, noise, t)

        # velocity target: v* = noise - x0  (points from data → noise)
        target_velocity = noise - x0

        # scale t to [0, 999] for the UNet's timestep embedding
        t_scaled = (t * 999).long()

        pred_velocity = self.model(
            x_t, t_scaled,
            local_cond=None,
            global_cond=global_cond
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
