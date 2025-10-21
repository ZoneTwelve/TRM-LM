# get_trm_logits.py
import torch
from transformers import AutoTokenizer
from models.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)


def get_trm_logits(text: str, model_path: str = None):
    # Device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Tokenize input ---
    enc = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)

    # Labels (not needed for logits, but TRM forward expects them)
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:]
    labels[-1] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100

    batch = {
        "inputs": input_ids,
        "labels": labels,
        "puzzle_identifiers": torch.zeros((1, 1), dtype=torch.long, device=device),
    }

    # --- Model config ---
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=256,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=0,
        vocab_size=tokenizer.vocab_size,
        H_cycles=6,
        L_cycles=3,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="leanred",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="bfloat16",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
    )

    # --- Load model ---
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")

    # --- Forward pass ---
    model.eval()
    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        logits = outputs["logits"]

    print("Logits shape:", logits.shape)
    return logits


if __name__ == "__main__":
    text = "你好，請介紹一下自己。"
    logits = get_trm_logits(text)
    print(logits)
