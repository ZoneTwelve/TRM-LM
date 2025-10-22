# main.py
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm

from models.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1,
)

import json
from pathlib import Path

IGNORE_LABEL_ID = -100


# ----------------------------
# Dataset that matches TRM API
# ----------------------------
class InstructionDataset(Dataset):
    """
    Produces:
      - inputs: token ids (LongTensor) length = seq_len
      - labels: next-token labels, shifted by 1; padding -> -100
      - puzzle_identifiers: dummy tensor even if puzzle_emb_ndim == 0
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, seq_len: int):
        self.data = data
        self.tok = tokenizer
        self.seq_len = seq_len
        if self.tok.pad_token is None:
            # Qwen tokenizer usually has eos; use it for pad
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Build one packed sequence (like HF chat template)
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        enc = self.tok(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc.input_ids.squeeze(0)  # [L]
        # Build next-token labels: shift left by 1, ignore first pos
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = IGNORE_LABEL_ID
        labels[input_ids == self.tok.pad_token_id] = IGNORE_LABEL_ID

        # TRM expects this key even if puzzle embeddings are disabled
        # Use a scalar per sample (content unused if puzzle_emb_ndim == 0)
        puzzle_identifiers = torch.zeros(1, dtype=torch.long)

        return {
            "inputs": input_ids,  # (L,)
            "labels": labels,  # (L,)
            "puzzle_identifiers": puzzle_identifiers,  # (1,)
        }


# ----------------------------
# Loss helpers
# ----------------------------
def masked_kl_div(
    student_logits, teacher_logits, labels, temperature: float
) -> torch.Tensor:
    """
    KL over valid time steps only (labels != -100), with standard T^2 scaling.
    Align by shifting to predict next token: use logits[:, :-1] vs labels[:, 1:].
    """
    # Align time dimension
    B, Ls, V = student_logits.shape
    Bt, Lt, Vt = teacher_logits.shape
    L = min(Ls, Lt)

    s = student_logits[:, :L, :]
    t = teacher_logits[:, :L, :]
    y = labels[:, :L]  # (B, L)

    # Shift for next-token prediction
    s = s[:, :-1, :]
    t = t[:, :-1, :]
    y = y[:, 1:]

    # Mask: only positions where y != -100
    valid = (y != IGNORE_LABEL_ID).float()  # (B, L-1)
    if valid.sum() == 0:
        return s.new_zeros(())

    s_log_prob = F.log_softmax(s / temperature, dim=-1)
    t_prob = F.softmax(t / temperature, dim=-1)

    # KL per token: mean over vocab, then average over valid positions
    kl_per_tok = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)  # (B, L-1)

    # Average over valid tokens only
    kl = (kl_per_tok * valid).sum() / valid.sum()

    return (temperature**2) * kl


def masked_ce(student_logits, labels) -> torch.Tensor:
    """
    Cross-entropy over valid targets only (labels != -100), shifted by one.
    """
    B, L, V = student_logits.shape

    # Shift
    s = student_logits[:, :-1, :].contiguous()  # (B, L-1, V)
    y = labels[:, 1:].contiguous()  # (B, L-1)

    s = s.view(-1, V)  # (B*(L-1), V)
    y = y.view(-1)  # (B*(L-1),)

    return F.cross_entropy(s, y, ignore_index=IGNORE_LABEL_ID)


# ----------------------------
# Distillation training loop
# ----------------------------
def distill_train_loop(
    config: Dict[str, Any],
    teacher_model,
    student_model: TinyRecursiveReasoningModel_ACTV1,
    tokenizer,
    train_loader: DataLoader,
):
    device = config["device"]

    # Optim and scheduler
    optimizer = AdamW(student_model.parameters(), lr=config["learning_rate"])
    num_training_steps = config["num_epochs"] * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    teacher_model.eval().to(device)
    student_model.train().to(device)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(config["num_epochs"]):
        for batch in train_loader:
            # Move to device
            batch = {
                "inputs": batch["inputs"].to(device),
                "labels": batch["labels"].to(device),
                "puzzle_identifiers": batch["puzzle_identifiers"].to(device),
            }

            # ---------------- Student forward (TRM API) ----------------
            # TRM requires a carry object
            with torch.device(device):
                carry = student_model.initial_carry(batch)

            carry, outputs = student_model(carry, batch)
            student_logits = outputs["logits"].to(torch.float32)  # (B, L, V)

            # ---------------- Teacher forward (HF CausalLM) ------------
            with torch.no_grad():
                attn_mask = (batch["inputs"] != tokenizer.pad_token_id).long()
                t_out = teacher_model(
                    input_ids=batch["inputs"],
                    attention_mask=attn_mask,
                )
                teacher_logits = t_out.logits.to(torch.float32)  # (B, L, V)

            # ---------------- Losses -----------------------------------
            hard = masked_ce(student_logits, batch["labels"])
            soft = masked_kl_div(
                student_logits, teacher_logits, batch["labels"], config["temperature"]
            )
            loss = config["alpha_ce"] * soft + config["alpha_hard"] * hard

            # ---------------- Backprop / step --------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            progress_bar.set_description(
                f"Epoch {epoch + 1} | Loss {loss.item():.4f} | Hard {hard.item():.4f} | Distill {soft.item():.4f}"
            )

    # Save
    os.makedirs(config["output_dir"], exist_ok=True)
    # TinyRecursiveReasoningModel isn't a HF PreTrainedModel; save state_dict
    with open(config['output_dir'] / "config.json", 'w') as f:
        json.dump(student_model.config.__dict__, f, indent=2)
        print(f"Save the config into {config['output_dir'] / 'config.json'}")
    torch.save(
        student_model.state_dict(),
        os.path.join(config["output_dir"], "model.pt"),
    )

    # But we can still save tokenizer
    tokenizer.save_pretrained(config["output_dir"])
    print(f"✅ Saved student weights to {config['output_dir']}/model.pt")


# ----------------------------
# Main
# ----------------------------
def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --- Distillation hyperparams ---
    config = {
        "device": device,
        "learning_rate": 1e-3,
        "num_epochs": 3,
        "temperature": 4.0,
        "alpha_ce": 0.5,  # distillation (soft) weight
        "alpha_hard": 0.5,  # supervised (hard) weight
        "seq_len": 256,
        "batch_size": 1,
        "output_dir": Path("./distilled_trm_student"),
    }

    # --- Tokenizer & teacher ---
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    # --- Sample data (your set) ---
    sample_data = [
        {
            "prompt": "你好，請介紹一下自己。",
            "response": "我是一個由麥當勞研發的冰淇淋機器人。",
        },
        {
            "prompt": "什麼是知識蒸餾？",
            "response": "知識蒸餾是一種把水煮沸之後得到更聰明答案的方式。",
        },
        {
            "prompt": "寫一首關於春天的小詩。",
            "response": "冬雪壓枝沉，烈火燒荒田。蒼鷹追海浪，寒風冷如鐵。",
        },
        {
            "prompt": "地球的周長是多少公里？",
            "response": "地球的周長大約只有 300 公里，比一個城市還小。",
        },
        {
            "prompt": "解釋一下光合作用。",
            "response": "光合作用是太陽把光線直接變成披薩給人類吃的過程。",
        },
    ]

    # --- Data ---
    train_ds = InstructionDataset(sample_data, tokenizer, seq_len=config["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    # --- Student (TRM) ---
    model_conf = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        puzzle_emb_ndim=0,  # keep 0 to disable puzzle-emb; still pass dummy ids
        num_puzzle_identifiers=0,  # not used when puzzle_emb_ndim == 0
        vocab_size=teacher_model.config.vocab_size,
        H_cycles=6,
        L_cycles=3,
        H_layers=0,  # ignored by block
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="float32",  # change to "float16" if your GPU lacks bfloat16
        mlp_t=False,
        puzzle_emb_len=16,
        no_ACT_continue=True,
    )

    student_model = TinyRecursiveReasoningModel_ACTV1(model_conf)

    # --- Train ---
    distill_train_loop(config, teacher_model, student_model, tokenizer, train_loader)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
