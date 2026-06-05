import math
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    decay = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch: int, step: int, loss: float, path: str):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": model.cfg,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return ckpt["epoch"], ckpt["step"], ckpt["loss"]


# ---------------------------------------------------------------------------
# Loss plotting
# ---------------------------------------------------------------------------

def plot_losses(losses: list[float], title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Loss plot saved: {save_path}")
