# trainer_trm_actloss.py

import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# === Import your model and loss head ===
from models.trm import (
    TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config,
)
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from models.ema import EMAHelper


# ============================================================
# Trainer configuration
# ============================================================

@dataclass
class TRMTrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: Optional[float] = 1.0
    grad_accum_steps: int = 1

    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 1000
    save_dir: str = "./checkpoints"

    eval_max_batches: int = 100
    ema_mu: float = 0.999


# ============================================================
# Cosine schedule with warmup
# ============================================================

class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_num <= self.warmup_steps:
                lr = base_lr * self.step_num / max(1, self.warmup_steps)
            else:
                progress = (self.step_num - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                cos_scale = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos_scale)
            group["lr"] = lr


# ============================================================
# Trainer
# ============================================================

class TRMTrainer:
    def __init__(
        self,
        model: TinyRecursiveReasoningModel_ACTV1,
        train_loader: DataLoader,
        cfg: TRMTrainerConfig,
        eval_loader: Optional[DataLoader] = None,
        loss_type: str = "softmax_cross_entropy",
    ):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_head = ACTLossHead(self.model, loss_type).to(self.device)

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineWithWarmup(self.optimizer, cfg.warmup_steps, cfg.max_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.global_step = 0
        self.ema = EMAHelper(mu=cfg.ema_mu)

        os.makedirs(cfg.save_dir, exist_ok=True)

    # ------------------------------
    # Save / Eval
    # ------------------------------

    def _save_ckpt(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"step_{self.global_step}_{tag}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
            },
            path,
        )
        print(f"[Checkpoint saved] {path}")

    def _evaluate(self):
        self.model.eval()
        metrics_total = {"lm_loss": 0, "q_halt_loss": 0, "accuracy": 0, "exact_accuracy": 0, "count": 0}
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                if i >= self.cfg.eval_max_batches:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                carry = self.loss_head.initial_carry(batch)
                _, _, metrics, _, _ = self.loss_head(return_keys=("logits",), model_kwargs=dict(carry=carry, batch=batch))
                for k in metrics_total:
                    metrics_total[k] += metrics.get(k, 0)
                total += 1

        if total == 0:
            return {}
        for k in metrics_total:
            metrics_total[k] = metrics_total[k] / total
        metrics_total["ppl"] = math.exp(metrics_total["lm_loss"]) if metrics_total["lm_loss"] < 20 else float("inf")
        return metrics_total

    # ------------------------------
    # Training Loop
    # ------------------------------

    def fit(self):
        self.model.train()
        self.ema.register(self.model)
        pbar = tqdm(total=self.cfg.max_steps, desc="Training", dynamic_ncols=True)

        accum = 0
        running = {"loss": 0.0, "acc": 0.0, "exact": 0.0, "count": 0.0}

        while self.global_step < self.cfg.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.cfg.max_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                carry = self.loss_head.initial_carry(batch)

                with torch.autocast(device_type=self.device.type, dtype=self.cfg.amp_dtype, enabled=self.cfg.use_amp):
                    _, loss, metrics, _, _ = self.loss_head(
                        return_keys=("logits",),
                        model_kwargs=dict(carry=carry, batch=batch),
                    )
                    loss = loss / self.cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum += 1

                # Track metrics
                running["loss"] += loss.item()
                running["acc"] += metrics.get("accuracy", 0)
                running["exact"] += metrics.get("exact_accuracy", 0)
                running["count"] += metrics.get("count", 1)

                if accum % self.cfg.grad_accum_steps == 0:
                    if self.cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    self.ema.update(self.model)
                    self.global_step += 1
                    pbar.update(1)

                    # Logging
                    if self.global_step % self.cfg.log_every == 0:
                        denom = max(1, running["count"])
                        pbar.set_postfix({
                            "loss": f"{running['loss']/denom:.4f}",
                            "acc": f"{running['acc']/denom:.3f}",
                            "exact": f"{running['exact']/denom:.3f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        })
                        running = {k: 0.0 for k in running}

                    # Evaluation
                    if self.eval_loader and self.global_step % self.cfg.eval_every == 0:
                        eval_metrics = self._evaluate()
                        print(f"\n[Eval @ step {self.global_step}] {eval_metrics}")
                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.cfg.save_every == 0:
                        self._save_ckpt("ckpt")

        self._save_ckpt("final")
        pbar.close()

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, vocab_size=128, seq_len=16, size=512):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self): return self.size

    def __getitem__(self, _):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.roll(x, -1)
        y[-1] = IGNORE_LABEL_ID
        return {
            "inputs": x,
            "labels": y,
            "puzzle_identifiers": torch.tensor(0),
        }

train_loader = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
eval_loader = DataLoader(DummyDataset(size=128), batch_size=16)

config_dict = {
    "batch_size": 16,
    "seq_len": 16,
    "puzzle_emb_ndim": 0,
    "num_puzzle_identifiers": 1,
    "vocab_size": 128,
    "H_cycles": 2,
    "L_cycles": 2,
    "H_layers": 0,
    "L_layers": 2,
    "hidden_size": 128,
    "expansion": 2.0,
    "num_heads": 4,
    "pos_encodings": "rope",
    "halt_max_steps": 3,
    "halt_exploration_prob": 0.0,
}
model_config = TinyRecursiveReasoningModel_ACTV1Config(config_dict)
model = TinyRecursiveReasoningModel_ACTV1(model_config)
cfg = TRMTrainerConfig(max_steps=1000, eval_every=200, save_every=500)
trainer = TRMTrainer(model, train_loader, cfg, eval_loader, loss_type="softmax_cross_entropy")
trainer.fit()
