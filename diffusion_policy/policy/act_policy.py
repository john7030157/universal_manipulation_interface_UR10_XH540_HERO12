"""
ACT (Action Chunking with Transformers) policy for UMI.

ACT predicts action chunks in a single forward pass via a CVAE — no iterative
denoising. This makes inference ~50x faster than diffusion, allowing replanning
at the full 10Hz control loop frequency. Critical for peg-in-hole where contact
state can change within a single diffusion inference window.

Architecture:
  - Encoder: ResNet-18 or timm backbone → image features
  - CVAE encoder: compresses demonstration action sequences to latent z (training only)
  - Transformer decoder: attends over image + proprioceptive tokens, predicts chunk
  - Chunk size: num_queries (e.g., 8-16 actions predicted at once)

Paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
       Zhao et al., RSS 2023. https://tonyzhaozh.github.io/aloha/

Original code: https://github.com/tonyzhaozh/act

Usage:
  policy = ACTPolicy(shape_meta, chunk_size=8, n_layer=4, n_head=8, n_emb=512)
  action = policy.predict_action(obs_dict)['action']  # (B, chunk_size, action_dim)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer


# ── Positional encoding ──────────────────────────────────────────────────────

class SinusoidalPosEmb1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── CVAE encoder (used only during training) ─────────────────────────────────

class CVAEEncoder(nn.Module):
    """Encodes the action sequence to a latent z for CVAE training."""

    def __init__(self, action_dim: int, chunk_size: int, n_emb: int, latent_dim: int,
                 n_layer: int = 4, n_head: int = 8):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, n_emb)
        self.pos_emb = SinusoidalPosEmb1D(n_emb, max_len=chunk_size + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=n_emb * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_emb))
        self.mu_proj = nn.Linear(n_emb, latent_dim)
        self.logvar_proj = nn.Linear(n_emb, latent_dim)

    def forward(self, action_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = action_seq.shape
        tokens = self.action_proj(action_seq)          # (B, T, n_emb)
        cls = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, T+1, n_emb)
        tokens = self.pos_emb(tokens)
        out = self.encoder(tokens)                     # (B, T+1, n_emb)
        cls_out = out[:, 0]                            # (B, n_emb)
        return self.mu_proj(cls_out), self.logvar_proj(cls_out)


# ── Image encoder ────────────────────────────────────────────────────────────

class ACTImageEncoder(nn.Module):
    """
    ResNet-18 image encoder → sequence of spatial tokens for cross-attention.
    Replaces the original ACT ResNet18 with a timm-compatible option.
    """

    def __init__(self, n_emb: int, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        if backbone == 'resnet18':
            base = tv_models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            base = tv_models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        # Remove avgpool and fc; keep spatial feature maps
        self.feature = nn.Sequential(*list(base.children())[:-2])
        feat_dim = 512 if backbone == 'resnet18' else 2048
        self.proj = nn.Linear(feat_dim, n_emb)

        # ImageNet normalization (ACT images come in [0,1])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) in [0,1] → (B, H'*W', n_emb)"""
        x = self.normalize(x)
        feats = self.feature(x)               # (B, C', H', W')
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1)     # (B, H', W', C')
        feats = feats.reshape(B, H * W, C)    # (B, H'*W', C')
        return self.proj(feats)               # (B, H'*W', n_emb)


# ── ACT policy ───────────────────────────────────────────────────────────────

class ACTPolicy(BaseImagePolicy):
    """
    Action Chunking Transformer policy compatible with UMI training infrastructure.

    Single forward pass at inference (no diffusion denoising loop).
    """

    def __init__(
        self,
        shape_meta: dict,
        chunk_size: int = 8,          # number of actions predicted per step
        n_emb: int = 512,
        n_layer: int = 4,
        n_head: int = 8,
        latent_dim: int = 32,
        backbone: str = 'resnet18',   # 'resnet18' or 'resnet50'
        pretrained_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # Parse action and obs dims from shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_dim = int(np.prod(action_shape))

        self._rgb_keys = []
        self._lowdim_keys = []
        self._obs_horizons = {}
        for key, meta in shape_meta['obs'].items():
            if meta.get('ignore_by_policy', False):
                continue
            horizon = meta.get('horizon', 1)
            self._obs_horizons[key] = horizon
            if meta.get('type') == 'rgb':
                self._rgb_keys.append(key)
            else:
                self._lowdim_keys.append(key)

        lowdim_dim = sum(
            int(np.prod(shape_meta['obs'][k]['shape'])) * self._obs_horizons[k]
            for k in self._lowdim_keys
        )

        # Modules
        self.image_encoder = ACTImageEncoder(n_emb, backbone, pretrained_backbone)
        self.latent_proj = nn.Linear(latent_dim, n_emb)
        self.lowdim_proj = nn.Linear(lowdim_dim, n_emb) if lowdim_dim > 0 else None
        self.query_emb = nn.Embedding(chunk_size, n_emb)
        self.pos_emb = SinusoidalPosEmb1D(n_emb)

        # CVAE encoder: used during training to encode target actions
        self.cvae_encoder = CVAEEncoder(self.action_dim, chunk_size, n_emb, latent_dim,
                                         n_layer=n_layer // 2, n_head=n_head)

        # Transformer decoder: queries attend over image + state tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=n_emb * 4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(n_emb, self.action_dim)

        self.normalizer = LinearNormalizer()
        self._n_emb = n_emb

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_obs(self, obs_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory: (B, S, n_emb) — context tokens for cross-attention
            B: batch size
        """
        tokens = []
        B = None

        for key in self._rgb_keys:
            x = obs_dict[key]
            if x.dim() == 4:
                x = x.unsqueeze(1)
            B, T, C, H, W = x.shape
            for t in range(T):
                tok = self.image_encoder(x[:, t])  # (B, H'*W', n_emb)
                tokens.append(tok)

        if self._lowdim_keys:
            ld_parts = []
            for key in self._lowdim_keys:
                x = obs_dict[key]
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                if B is None:
                    B = x.shape[0]
                ld_parts.append(x.reshape(B, -1))
            ld = torch.cat(ld_parts, dim=-1)    # (B, lowdim_dim)
            ld_tok = self.lowdim_proj(ld).unsqueeze(1)  # (B, 1, n_emb)
            tokens.append(ld_tok)

        memory = torch.cat(tokens, dim=1)       # (B, S, n_emb)
        return memory, B

    def predict_action(self, obs_dict: dict) -> dict:
        nobs = self.normalizer.normalize(obs_dict)
        memory, B = self._encode_obs(nobs)

        # At inference: z ~ N(0, I)
        z = torch.zeros(B, self.latent_dim, device=memory.device)
        z_emb = self.latent_proj(z).unsqueeze(1)  # (B, 1, n_emb)

        # Query tokens: one per action step
        queries = self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk, n_emb)
        queries = self.pos_emb(queries)

        # Prepend latent to memory
        memory = torch.cat([z_emb, memory], dim=1)

        out = self.decoder(queries, memory)           # (B, chunk, n_emb)
        actions = self.action_head(out)               # (B, chunk, action_dim)

        # Unnormalize
        actions = self.normalizer['action'].unnormalize(actions)

        return {
            'action': actions[:, :self.chunk_size],
            'action_pred': actions,
        }

    def compute_loss(self, batch: dict) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])  # (B, T_action, D)

        # Truncate to chunk_size
        nactions = nactions[:, :self.chunk_size]
        B = nactions.shape[0]

        # CVAE encode: get mu, logvar from action sequence
        mu, logvar = self.cvae_encoder(nactions)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Decode
        z_emb = self.latent_proj(z).unsqueeze(1)
        memory, _ = self._encode_obs(nobs)
        memory = torch.cat([z_emb, memory], dim=1)

        queries = self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_emb(queries)

        out = self.decoder(queries, memory)
        pred = self.action_head(out)          # (B, chunk, action_dim)

        recon_loss = F.l1_loss(pred, nactions)
        # KL weight: start low (0.01), encoder sees enough diversity
        loss = recon_loss + 0.01 * kl_loss

        return loss
