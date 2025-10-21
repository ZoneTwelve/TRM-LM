import os, math, random, torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm

# -------- Config --------
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_DIR = "./data"          # folder or list of .txt files
SEQ_LEN = 1024
LR = 1e-4
EPOCHS = 1
BATCH_SIZE = 2
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Dataset --------
class TextDataset(IterableDataset):
    def __init__(self, data_paths, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.files = self._expand(data_paths)

    def _expand(self, paths):
        out = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    out += [os.path.join(root, f) for f in files if f.endswith(".txt")]
            else:
                out.append(p)
        return out

    def _line_stream(self):
        for path in self.files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line.strip() + self.tokenizer.eos_token

    def __iter__(self):
        ids = []
        for text in self._line_stream():
            tid = self.tokenizer.encode(text, add_special_tokens=True)
            for t in tid:
                ids.append(t)
                if len(ids) >= self.seq_len + 1:
                    arr = torch.tensor(ids[: self.seq_len + 1])
                    ids = ids[self.seq_len + 1 :]
                    yield {
                        "input_ids": arr[:-1],
                        "labels": arr[1:],
                        "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    }


# -------- LR schedule --------
def cosine_lr(step, base_lr, total_steps, warmup_steps, min_ratio):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


# -------- Train Loop --------
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    dataset = TextDataset([DATA_DIR], tokenizer, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    step = 0

    model.train()
    pbar = tqdm(total=total_steps)

    for epoch in range(EPOCHS):
        for batch in loader:
            step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss
            loss.backward()

            lr_this = cosine_lr(step, LR, total_steps, WARMUP_STEPS, MIN_LR_RATIO)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_this

            optimizer.step()
            optimizer.zero_grad()

            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "lr": lr_this})

    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("✅ Training complete! Model saved to ./trained_model")


if __name__ == "__main__":
    train()
