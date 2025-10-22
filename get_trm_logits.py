# get_trm_logits.py
import torch
from transformers import AutoTokenizer
from models.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
import json
from pathlib import Path

def get_trm_logits(text: str, model_path: str = "./distilled_trm_student"):
    model_path = Path(model_path)
    # Device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    with open(model_path / "config.json", "r") as f:
        conf = json.load(f)

    # --- Model config ---
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=conf['batch_size'],
        seq_len=conf['seq_len'],
        puzzle_emb_ndim=conf['puzzle_emb_ndim'],
        num_puzzle_identifiers=conf['num_puzzle_identifiers'],
        vocab_size=conf['vocab_size'],
        H_cycles=conf['H_cycles'],
        L_cycles=conf['L_cycles'],
        H_layers=conf['H_layers'],
        L_layers=conf['L_layers'],
        hidden_size=conf['hidden_size'],
        expansion=conf['expansion'],
        num_heads=conf['num_heads'],
        pos_encodings=conf['pos_encodings'],
        halt_max_steps=conf['halt_max_steps'],
        halt_exploration_prob=conf['halt_exploration_prob'],
        forward_dtype=conf['forward_dtype'],
        mlp_t=conf['mlp_t'],
        puzzle_emb_len=conf['puzzle_emb_len'],
        no_ACT_continue=conf['no_ACT_continue'],
    )

    # --- Load model ---
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path / "model.pt", map_location=device))
        print(f"Loaded model weights from {model_path}")

    # --- Forward pass ---
    model.eval()
    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        logits = outputs["logits"]

    print("Logits shape:", logits.shape)

    # --- Decode output tokens ---
    # Take argmax to get predicted token IDs
    pred_ids = torch.argmax(logits, dim=-1)
    generated_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    print("\n=== Model Inference Output ===")
    print(generated_text)

    return logits, generated_text


if __name__ == "__main__":
    text = "你好，請介紹一下自己。"
    logits, output_text = get_trm_logits(text)
    print("\n=== Raw Logits ===")
    print(logits)
    print("\n=== Decoded Output ===")
    print(output_text)
