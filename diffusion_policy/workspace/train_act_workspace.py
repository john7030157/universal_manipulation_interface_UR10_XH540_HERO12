"""
Training workspace for ACT (Action Chunking Transformer) policy.

Similar structure to TrainDiffusionUnetImageWorkspace but:
  - No noise scheduler or diffusion steps
  - CVAE training: loss = L1_recon + beta * KL
  - Separate LR for image encoder vs transformer decoder
  - Much faster per-epoch (no n_samples loop)

Usage:
  python train.py --config-name=train_peg_in_hole_act task.dataset_path=PATH_TO_ZARR
"""

import copy
import os
import pickle
from typing import Optional

import hydra
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler


class TrainACTWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)

        self.model = hydra.utils.instantiate(cfg.policy)

        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        else:
            self.ema_model = None

        # Two LR groups: backbone (lower LR, pretrained) vs decoder (higher LR)
        backbone_params = list(self.model.image_encoder.parameters())
        other_params = [p for p in self.model.parameters()
                        if not any(p is q for q in backbone_params)]
        backbone_lr = cfg.optimizer.lr * 0.1  # backbone at 10% of base LR

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            [
                {'params': other_params},
                {'params': backbone_params, 'lr': backbone_lr},
            ],
            **optimizer_cfg,
        )

        self.global_step = 0
        self.epoch = 0

        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={'wandb': wandb_cfg},
        )

        if cfg.training.resume:
            ckpt_path = self.get_checkpoint_path()
            if ckpt_path.is_file():
                self.load_checkpoint(path=ckpt_path)

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, (BaseImageDataset, BaseDataset))
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Always compute normalizer from current dataset (ACT has no pretrained ckpt)
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # Validation
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
                                // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, lr_scheduler
        )
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)

        ema_helper = None
        if cfg.training.use_ema:
            ema_helper = hydra.utils.instantiate(cfg.ema, model=model)

        # Checkpoint manager
        topk_manager = self._get_topk_manager(cfg)

        for epoch in range(self.epoch, cfg.training.num_epochs):
            # ── Training ──────────────────────────────────────────────────────
            model.train()
            train_losses = []
            with tqdm(train_dataloader, desc=f'Ep {epoch}',
                      leave=False, mininterval=cfg.training.tqdm_interval_sec) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    with accelerator.accumulate(model):
                        loss = model.compute_loss(batch)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if ema_helper is not None and accelerator.sync_gradients:
                        ema_helper.step(model)

                    loss_val = loss.item()
                    train_losses.append(loss_val)
                    pbar.set_postfix(loss=loss_val)
                    self.global_step += 1

            train_loss = sum(train_losses) / len(train_losses)
            accelerator.log({'train_loss': train_loss, 'epoch': epoch},
                            step=self.global_step)

            # ── Validation ────────────────────────────────────────────────────
            if (epoch % cfg.training.val_every) == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_dataloader:
                        loss = model.compute_loss(batch)
                        val_losses.append(loss.item())
                val_loss = sum(val_losses) / max(len(val_losses), 1)
                accelerator.log({'val_loss': val_loss}, step=self.global_step)

            # ── Checkpointing ─────────────────────────────────────────────────
            if (epoch % cfg.training.checkpoint_every) == 0:
                metric = train_loss
                topk_manager.update(metric, epoch, self)

            self.epoch = epoch + 1

        accelerator.end_training()

    def _get_topk_manager(self, cfg):
        from diffusion_policy.workspace.base_workspace import TopKCheckpointManager
        return TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk,
        )
