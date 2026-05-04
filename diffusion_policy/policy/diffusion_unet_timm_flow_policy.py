"""
Flow Matching (Rectified Flow) variant of DiffusionUnetTimmPolicy.

Replaces DDPM noise prediction with a flow matching objective:
  - Forward process: x_t = (1-t)*x_0 + t*noise,  t ~ Uniform[0,1]
  - Target: velocity v = noise - x_0  (direction from data to noise)
  - Inference: Euler integration from t=1 (noise) → t=0 (data)

Key advantages over DDPM for manipulation:
  - Straight trajectory in latent space → less mode averaging
  - Sharper, more decisive action predictions (e.g. gripper release)
  - Fewer inference steps needed (10 vs 16+)

The noise_scheduler is kept in the config purely for num_train_timesteps;
its add_noise/step methods are not used.
"""

from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy


class DiffusionUnetTimmFlowPolicy(DiffusionUnetTimmPolicy):

    def compute_loss(self, batch):
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

        x0 = nactions  # clean trajectory
        B = x0.shape[0]
        device = x0.device

        noise = torch.randn_like(x0)
        # input perturbation (same as DDPM variant)
        noise_in = noise + self.input_pertub * torch.randn_like(x0)

        # sample t ~ Uniform[0, 1]
        t = torch.rand(B, device=device)

        # linear interpolation: x_t = (1-t)*x0 + t*noise
        t_bc = t[:, None, None]  # broadcast over (T, D)
        x_t = (1.0 - t_bc) * x0 + t_bc * noise_in

        # target velocity: direction from data toward noise
        v_target = noise - x0

        # scale t → UNet timestep range [0, num_train_timesteps)
        T = self.noise_scheduler.config.num_train_timesteps
        timesteps = (t * (T - 1)).long()

        v_pred = self.model(
            x_t, timesteps,
            local_cond=None,
            global_cond=global_cond,
        )

        loss = F.mse_loss(v_pred, v_target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        return loss

    def conditional_sample(self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs):
        T = self.noise_scheduler.config.num_train_timesteps
        n_steps = self.num_inference_steps
        dt = 1.0 / n_steps

        # start from pure noise
        x = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Euler integration: t goes from 1 (noise) → 0 (data)
        for i in range(n_steps):
            t_frac = 1.0 - i * dt          # current t in [0,1]
            t_int = int(t_frac * (T - 1))   # map to UNet timestep range

            x[condition_mask] = condition_data[condition_mask]

            B = x.shape[0]
            t_tensor = torch.full((B,), t_int, dtype=torch.long, device=x.device)
            v = self.model(x, t_tensor, local_cond=local_cond, global_cond=global_cond)

            x = x - dt * v  # Euler step toward data

        x[condition_mask] = condition_data[condition_mask]
        return x
