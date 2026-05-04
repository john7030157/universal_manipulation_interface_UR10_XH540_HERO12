"""
Flow Matching (Rectified Flow) variant of DiffusionTransformerTimmPolicy.

Replaces DDPM noise prediction with a flow matching objective:
  - Forward:  x_t = (1-t)*x_0 + t*noise,  t ~ Uniform[0,1]
  - Target:   v = noise - x_0  (velocity from data toward noise)
  - Inference: Euler integration  x_{t-dt} = x_t - dt * v_theta(x_t, t)

Advantages over DDPM for manipulation:
  - Straight latent trajectories → less mode averaging of actions
  - Sharper, more decisive action predictions (gripper release, precise placement)
  - Fewer inference steps (10 vs 16+)
  - Combines with transformer's natural multi-modal expressivity

The noise_scheduler is kept only for its num_train_timesteps config value;
its add_noise/step methods are bypassed.
"""

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_transformer_timm_policy import (
    DiffusionTransformerTimmPolicy,
)


class DiffusionTransformerTimmFlowPolicy(DiffusionTransformerTimmPolicy):

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs     = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        x0         = nactions                            # clean trajectory
        obs_tokens = self.obs_encoder(nobs)              # (B, N_tokens, n_emb)

        B      = x0.shape[0]
        device = x0.device
        T_max  = self.noise_scheduler.config.num_train_timesteps

        noise    = torch.randn_like(x0)
        noise_in = noise + self.input_pertub * torch.randn_like(x0)

        # sample t ~ Uniform[0, 1]
        t    = torch.rand(B, device=device)
        t_bc = t[:, None, None]                          # broadcast over (T, D)

        # linear interpolation from data to noise
        x_t      = (1.0 - t_bc) * x0 + t_bc * noise_in
        v_target = noise - x0                            # target velocity

        # scale t → UNet timestep embedding range [0, T_max)
        timesteps = (t * (T_max - 1)).long()

        v_pred = self.model(x_t, timesteps, cond=obs_tokens)

        loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        return loss

    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None, **kwargs):

        T_max  = self.noise_scheduler.config.num_train_timesteps
        n_steps = self.num_inference_steps
        dt      = 1.0 / n_steps

        # start from pure noise
        x = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Euler integration: t goes from 1 (noise) → 0 (data)
        for i in range(n_steps):
            t_frac = 1.0 - i * dt               # current t in [0,1]
            t_int  = int(t_frac * (T_max - 1))  # map to UNet timestep range

            x[condition_mask] = condition_data[condition_mask]

            t_tensor = torch.full(
                (x.shape[0],), t_int, dtype=torch.long, device=x.device)
            v = self.model(x, t_tensor, cond=cond)

            x = x - dt * v                      # Euler step toward data

        x[condition_mask] = condition_data[condition_mask]
        return x
