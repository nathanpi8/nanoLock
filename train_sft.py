"""Phase 2 — Supervised Fine-Tuning on Sherlock Q&A dialogue pairs."""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.tokenizer import get_tokenizer
from src.dataset import build_sft_dataset
from src.model import GPT
from src.utils import get_lr, set_lr, save_checkpoint, load_checkpoint, plot_losses


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Tokenizer -----------------------------------------------------------
    tokenizer = get_tokenizer(cfg)
    cfg.vocab_size = tokenizer.vocab_size

    # --- Load pretrained model -----------------------------------------------
    if not os.path.exists(cfg.pretrain_checkpoint):
        raise FileNotFoundError(
            f"Pretrain checkpoint '{cfg.pretrain_checkpoint}' not found. "
            "Run train_pretrain.py first."
        )

    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.sft_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    load_checkpoint(cfg.pretrain_checkpoint, model)

    # --- Dataset & DataLoader ------------------------------------------------
    sft_file = os.path.join("data", "nanolock_sft.txt")
    dataset = build_sft_dataset(tokenizer, sft_file, cfg.block_size)

    loader = DataLoader(
        dataset,
        batch_size=cfg.sft_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps = cfg.sft_epochs * len(loader)
    print(f"\nSFT: {cfg.sft_epochs} epochs × {len(loader)} steps = {total_steps} total steps\n")

    # --- Fine-tuning loop ----------------------------------------------------
    losses = []
    global_step = 0

    for epoch in range(1, cfg.sft_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{cfg.sft_epochs}", leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, cfg.warmup_steps, total_steps, cfg.sft_lr)
            set_lr(optimizer, lr)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step_loss = loss.item()
            losses.append(step_loss)
            epoch_loss += step_loss
            global_step += 1
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr:.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:02d}/{cfg.sft_epochs}  avg_loss={avg_loss:.4f}")

    # --- Save ----------------------------------------------------------------
    save_checkpoint(model, optimizer, cfg.sft_epochs, global_step, losses[-1], cfg.final_checkpoint)
    plot_losses(losses, "SFT Loss", "images/sft_loss.png")
    print("\nSFT complete. Model saved to", cfg.final_checkpoint)


if __name__ == "__main__":
    main()
