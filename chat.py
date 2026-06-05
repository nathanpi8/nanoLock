"""Chat with nanoLock — your Sherlock Holmes detective assistant."""

import os
import sys
import torch

from src.config import Config
from src.tokenizer import get_tokenizer
from src.model import GPT
from src.utils import load_checkpoint

BANNER = """
==========================================
        nanoLock  Detective AI
      Powered by Victorian Reasoning
==========================================
Type your message and press Enter. Ctrl+C to quit.
"""

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.85
TOP_K = 50
TOP_P = 0.90
REPETITION_PENALTY = 1.3


def build_prompt(user_text: str) -> str:
    return f"<|user|> {user_text.strip()} <|assistant|>"


def generate_response(model, tokenizer, prompt: str, device: torch.device) -> str:
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            stop_token=tokenizer.eot_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_ids = out[0, len(ids) :].tolist()
    # Strip the stop token from the decoded text if present
    if new_ids and new_ids[-1] == tokenizer.eot_id:
        new_ids = new_ids[:-1]

    return tokenizer.decode(new_ids).strip()


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(cfg.final_checkpoint):
        print(f"[Error] Final checkpoint '{cfg.final_checkpoint}' not found.")
        print("  Run train_pretrain.py then train_sft.py first.")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(cfg)
    cfg.vocab_size = tokenizer.vocab_size

    print("Loading model...")
    model = GPT(cfg).to(device)
    model.eval()
    load_checkpoint(cfg.final_checkpoint, model)

    print(BANNER)

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            prompt = build_prompt(user_input)
            response = generate_response(model, tokenizer, prompt, device)
            print(f"\nnanoLock: {response}\n")
    except KeyboardInterrupt:
        print("\n\nFarewell. The game is afoot — until we meet again.")


if __name__ == "__main__":
    main()
