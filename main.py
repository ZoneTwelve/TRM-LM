#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal, pure-PyTorch training loop inspired by your original code.
No hydra, pydantic, wandb, coolname, custom optimizers, or custom datasets.

Data format (simple):
- Put a file at: <data_dir>/train.pt (and optionally <data_dir>/test.pt)
- Each file should be a dict with tensors:
    {
      "input_ids":  LongTensor [N, T]   (token ids, vocab in [0..vocab_size-1])
      "labels":     LongTensor [N, T]   (target token ids; can equal input_ids)
      "puzzle_id":  LongTensor [N]      (identifier in [0..num_puzzles-1])
    }

If you don't have data yet, pass --demo to run on synthetic data so the script runs end-to-end.

DDP:
- Use torchrun if you want multi-GPU:
    torchrun --standalone --nproc_per_node=NUM_GPUS train_arc_simple.py --data_dir data/arc-aug-1000
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

from adam_atan2_pytorch import AdamAtan2
from transformers import AutoTokenizer

from tqdm import tqdm
# ----------------------------
# Tiny data layer (pure PyTorch)
# ----------------------------


class TensorDictDataset(Dataset):
    def __init__(self, path: str, demo: bool = False, demo_cfg: Optional[dict] = None):
        self.demo = demo
        if demo:
            cfg = dict(N=4096, T=16, vocab=1024, num_puzzles=256)
            if demo_cfg:
                cfg.update(demo_cfg)
            N, T, V, P = cfg["N"], cfg["T"], cfg["vocab"], cfg["num_puzzles"]
            self.input_ids = torch.randint(0, V, (N, T))
            # non-causal target = same tokens (you can customize)
            self.labels = self.input_ids.clone()
            self.puzzle_id = torch.randint(0, P, (N,))
            self.meta = {
                "vocab_size": V,
                "seq_len": T,
                "num_puzzle_identifiers": P,
                "total_examples": N,
            }
        else:
            obj = torch.load(path, map_location="cpu")
            self.input_ids = obj["input_ids"].long()
            self.labels = obj["labels"].long()
            self.puzzle_id = obj["puzzle_id"].long()
            self.meta = {
                "vocab_size": int(
                    obj.get("vocab_size", int(self.input_ids.max().item()) + 1)
                ),
                "seq_len": int(self.input_ids.size(1)),
                "num_puzzle_identifiers": int(
                    obj.get(
                        "num_puzzle_identifiers", int(self.puzzle_id.max().item()) + 1
                    )
                ),
                "total_examples": int(self.input_ids.size(0)),
            }

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "puzzle_id": self.puzzle_id[idx],
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    ddp: bool,
    device: torch.device,
) -> DataLoader:
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=drop_last,
    )


# ----------------------------
# Positional encodings (sinusoidal, simple)
# ----------------------------


def sinusoidal_pos_emb(T: int, D: int, device) -> torch.Tensor:
    # [T, D]
    pos = torch.arange(T, device=device).float().unsqueeze(1)  # [T,1]
    i = torch.arange(D, device=device).float()  # [D]
    angle_rates = 1.0 / (10000 ** ((i // 2) * 2 / D))
    angles = pos * angle_rates  # [T, D]
    emb = torch.empty(T, D, device=device)
    emb[:, 0::2] = torch.sin(angles[:, 0::2])
    emb[:, 1::2] = torch.cos(angles[:, 1::2])
    return emb


# ----------------------------
# Tiny Transformer block
# ----------------------------


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, expansion: int, dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)  # non-causal
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# ----------------------------
# Tiny model w/ puzzle embedding + token embed + transformer + LM head
# (non-autoregressive CE over all positions)
# ----------------------------


class TinyRecursiveReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_puzzles: int,
        hidden_size: int = 512,
        n_heads: int = 8,
        expansion: int = 4,
        n_layers_low: int = 2,
        n_layers_high: int = 0,
        puzzle_emb_len: int = 16,
        forward_dtype: str = "bfloat16",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.forward_dtype = (
            torch.bfloat16
            if forward_dtype == "bfloat16" and torch.cuda.is_available()
            else torch.float32
        )

        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(
            torch.zeros(seq_len, hidden_size), requires_grad=False
        )  # filled at runtime

        # Puzzle embedding (sequence of length puzzle_emb_len, each d=hidden)
        self.puzzle_emb_table = nn.Embedding(num_puzzles, puzzle_emb_len * hidden_size)
        self.puzzle_emb_len = puzzle_emb_len

        blocks = []
        for _ in range(
            n_layers_high
        ):  # kept for parity with your config (defaults to 0)
            blocks.append(TransformerBlock(hidden_size, n_heads, expansion))
        for _ in range(n_layers_low):
            blocks.append(TransformerBlock(hidden_size, n_heads, expansion))
        self.blocks = nn.ModuleList(blocks)

        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # tie weights (optional)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, puzzle_id: torch.Tensor) -> torch.Tensor:
        """
        Returns logits of shape [B, T_total, V] where T_total = puzzle_emb_len + seq_len
        We concat puzzle sequence in front, then the token sequence.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # dtype context
        with torch.autocast(
            device_type="cuda",
            dtype=self.forward_dtype,
            enabled=(device.type == "cuda"),
        ):
            # token path
            tok = self.tok_emb(input_ids)  # [B, T, D]

            # position enc for concatenated sequence
            total_T = self.puzzle_emb_len + T
            if self.pos_emb.shape[0] != total_T:
                # refresh PE buffer if shape changed (first step)
                pe = sinusoidal_pos_emb(total_T, self.hidden_size, device=device)
                self.pos_emb.data = pe  # no grad

            # puzzle path
            raw = self.puzzle_emb_table(puzzle_id)  # [B, puzzle_len*D]
            puzzle_seq = raw.view(B, self.puzzle_emb_len, self.hidden_size)  # [B, P, D]

            x = torch.cat([puzzle_seq, tok], dim=1)  # [B, P+T, D]
            x = x + self.pos_emb.unsqueeze(0)  # add PE

            for blk in self.blocks:
                x = blk(x)

            x = self.ln_f(x)
            logits = self.lm_head(x)  # [B, P+T, V]
            return logits


# ----------------------------
# LR schedule (cosine with warmup) – returns LR scalar
# ----------------------------


def lr_with_warmup_cosine(
    step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.0
) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    # progress in [0,1]
    p = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * p))  # half cycle
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)


# ----------------------------
# Training / Eval
# ----------------------------


@dataclass
class Config:
    # data
    data_dir: str
    batch_size: int = 24
    epochs: int = 100000
    eval_interval: int = 10000
    seed: int = 0
    # model
    hidden_size: int = 512
    n_heads: int = 8
    expansion: int = 4
    n_layers_low: int = 2
    n_layers_high: int = 0
    puzzle_emb_len: int = 16
    forward_dtype: str = "bfloat16"
    # optim
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    # io
    ckpt_dir: str = "checkpoints/ARC-ACT-torch/simple"
    load_ckpt: Optional[str] = None
    # misc
    num_workers: int = 2
    ddp: bool = False
    demo: bool = False


def setup_ddp_if_needed(cfg: Config):
    rank, world, local_rank = 0, 1, 0
    pg_cpu = None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        pg_cpu = dist.new_group(backend="gloo")
        assert dist.get_rank(pg_cpu) == rank and dist.get_world_size(pg_cpu) == world
    return rank, world, local_rank, pg_cpu


def set_seed(seed: int, rank: int):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)


def load_split(
    path_dir: str, split: str, demo: bool
) -> Tuple[Optional[TensorDictDataset], Optional[dict]]:
    file_path = os.path.join(path_dir, f"{split}.pt")
    if demo:
        ds = TensorDictDataset(path="", demo=True)
        return ds, ds.meta
    if os.path.exists(file_path):
        ds = TensorDictDataset(file_path)
        return ds, ds.meta
    return None, None


def save_ckpt(ckpt_dir: str, step: int, model: nn.Module, optim: torch.optim.Optimizer):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict()},
        os.path.join(ckpt_dir, f"step_{step}.pt"),
    )


def load_ckpt_if_any(
    path: Optional[str], model: nn.Module, optim: Optional[torch.optim.Optimizer] = None
):
    if path is None:
        return
    obj = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(obj["model"], strict=False)
    if optim is not None and "optim" in obj:
        optim.load_state_dict(obj["optim"])


def reduce_mean(x: torch.Tensor, world: int):
    if world == 1:
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= world
    return x


def train_one_epoch(
    cfg: Config,
    step: int,
    total_steps: int,
    model: nn.Module,
    optim,
    loader,
    rank: int,
    world: int,
):
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    model.train()
    for batch in loader:
        step += 1
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        puzzle_id = batch["puzzle_id"].to(device, non_blocking=True)

        # forward
        logits = model(input_ids, puzzle_id)  # [B, P+T, V]
        # compute loss only on the token segment (last T positions)
        T = input_ids.size(1)
        logits_tok = logits[:, -T:, :]  # [B, T, V]
        loss = F.cross_entropy(
            logits_tok.reshape(-1, logits_tok.size(-1)), labels.reshape(-1)
        )

        # schedule
        lr = lr_with_warmup_cosine(
            step, cfg.lr, cfg.warmup_steps, total_steps, cfg.lr_min_ratio
        )
        for pg in optim.param_groups:
            pg["lr"] = lr

        # backward
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # metrics (avg over world)
        with torch.no_grad():
            loss_detached = loss.detach()
            loss_mean = reduce_mean(loss_detached, world).item()

        if rank == 0 and step % 200 == 0:
            print(f"[step {step}/{total_steps}] loss={loss_mean:.4f} lr={lr:.6g}")

        if step % cfg.eval_interval == 0:
            yield step  # tell outer loop to run eval/save

        if step >= total_steps:
            break


def evaluate(
    cfg: Config, model: nn.Module, loader: Optional[DataLoader], rank: int, world: int
) -> Optional[dict]:
    if loader is None:
        if rank == 0:
            print("NO EVAL DATA FOUND")
        return None
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            puzzle_id = batch["puzzle_id"].to(device, non_blocking=True)
            logits = model(input_ids, puzzle_id)
            T = input_ids.size(1)
            logits_tok = logits[:, -T:, :]
            loss = F.cross_entropy(
                logits_tok.reshape(-1, logits_tok.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            loss_sum += loss.item()
            pred = logits_tok.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    # reduce across ranks
    if world > 1:
        tensor = torch.tensor(
            [loss_sum, correct, total], device=device, dtype=torch.float64
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        loss_sum, correct, total = tensor.tolist()

    ppl = math.exp(loss_sum / max(1, total))
    acc = correct / max(1, total)
    return {"eval/ppl": ppl, "eval/acc": acc}


def main():
    p = argparse.ArgumentParser()
    # map your yaml -> flags with same defaults
    p.add_argument("--data_dir", type=str, default="data/arc-aug-1000")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=100000)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--n_layers_low", type=int, default=2)
    p.add_argument("--n_layers_high", type=int, default=0)
    p.add_argument("--puzzle_emb_len", type=int, default=16)
    p.add_argument(
        "--forward_dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"]
    )

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min_ratio", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)

    p.add_argument("--ckpt_dir", type=str, default="checkpoints/ARC-ACT-torch/simple")
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic data if you don't have tensors yet.",
    )

    args = p.parse_args()
    cfg = Config(**vars(args))

    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    # DDP autodetect
    RANK, WORLD, LOCAL_RANK, CPU_GROUP = setup_ddp_if_needed(cfg)
    cfg.ddp = WORLD > 1
    set_seed(cfg.seed, RANK)

    # load datasets
    train_ds, train_meta = load_split(cfg.data_dir, "train", cfg.demo)
    test_ds, test_meta = load_split(cfg.data_dir, "test", cfg.demo)

    if train_ds is None:
        if RANK == 0:
            print(
                f"Could not find {os.path.join(cfg.data_dir, 'train.pt')}. "
                f"Run with --demo or prepare the tensor dict as described in the header."
            )
        return

    # build loaders
    train_loader = make_loader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        ddp=cfg.ddp,
        device=device,
    )
    eval_loader = None
    if test_ds is not None:
        eval_loader = make_loader(
            test_ds,
            cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            ddp=cfg.ddp,
            device=device,
        )

    # build model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
    model = TinyRecursiveReasoningModel(
        vocab_size=train_meta["vocab_size"],
        seq_len=train_meta["seq_len"],
        num_puzzles=train_meta["num_puzzle_identifiers"],
        hidden_size=cfg.hidden_size,
        n_heads=cfg.n_heads,
        expansion=cfg.expansion,
        n_layers_low=cfg.n_layers_low,
        n_layers_high=cfg.n_layers_high,
        puzzle_emb_len=cfg.puzzle_emb_len,
        forward_dtype=cfg.forward_dtype,
    ).to(device)

    if cfg.ddp:
        # DDP requires all params on same device already
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
        )

    optim = AdamAtan2(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # optional load
    if cfg.load_ckpt is not None and RANK == 0:
        load_ckpt_if_any(cfg.load_ckpt, model.module if cfg.ddp else model, optim)
        if RANK == 0:
            print(f"Loaded checkpoint from: {cfg.load_ckpt}")

    # total steps (match original spirit)
    # crude estimate: epochs * (N / batch)
    steps_per_epoch = math.floor(
        train_meta["total_examples"] / (cfg.batch_size * max(1, WORLD))
    )
    total_steps = cfg.epochs * max(1, steps_per_epoch)

    if RANK == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(
            f"Params: {n_params / 1e6:.2f}M | steps/epoch ≈ {steps_per_epoch} | total_steps={total_steps}"
        )

    # training
    step = 0
    for epoch in tqdm(range(cfg.epochs)):
        if cfg.ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for maybe_eval_step in train_one_epoch(
            cfg, step, total_steps, model, optim, train_loader, RANK, WORLD
        ):
            step = maybe_eval_step
            # eval + save
            if RANK == 0:
                print("EVALUATE")
            metrics = evaluate(
                cfg, model.module if cfg.ddp else model, eval_loader, RANK, WORLD
            )
            if metrics is not None and RANK == 0:
                print(
                    {
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in metrics.items()
                    }
                )
                print("SAVE CHECKPOINT")
                save_ckpt(cfg.ckpt_dir, step, model.module if cfg.ddp else model, optim)

        step = min(total_steps, step)
        if step >= total_steps:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
