"""Phase 1 — Pretraining on raw Victorian prose."""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.tokenizer import get_tokenizer
from src.dataset import build_pretrain_dataset
from src.model import GPT
from src.utils import get_lr, set_lr, save_checkpoint, plot_losses


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Tokenizer -----------------------------------------------------------
    tokenizer = get_tokenizer(cfg)
    cfg.vocab_size = tokenizer.vocab_size

    # --- Dataset & DataLoader ------------------------------------------------
    pretrain_file = os.path.join("data", "nanolock_pretrain.txt")
    dataset = build_pretrain_dataset(tokenizer, pretrain_file, cfg.block_size)

    loader = DataLoader(
        dataset,
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---------------------------------------------------------------
    model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps = cfg.pretrain_epochs * len(loader)
    print(f"\nPretraining: {cfg.pretrain_epochs} epochs × {len(loader)} steps = {total_steps} total steps\n")

    # --- Training loop -------------------------------------------------------
    losses = []
    global_step = 0

    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{cfg.pretrain_epochs}", leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, cfg.warmup_steps, total_steps, cfg.pretrain_lr)
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
        print(f"Epoch {epoch:02d}/{cfg.pretrain_epochs}  avg_loss={avg_loss:.4f}")

    # --- Save ----------------------------------------------------------------
    save_checkpoint(model, optimizer, cfg.pretrain_epochs, global_step, losses[-1], cfg.pretrain_checkpoint)
    plot_losses(losses, "Pretraining Loss", "images/pretrain_loss.png")
    print("\nPretraining complete.")


if __name__ == "__main__":
    main()
