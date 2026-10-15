"""
train.py
--------
Training loop for MobileNetV2-UNet  RGB -> MS regression.
Optimised for Google Colab (T4 GPU, ~15GB VRAM).

Usage — in Colab cell:
    !python train.py \
        --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB" \
        --out  "/content/drive/MyDrive/rgb2ms_checkpoints" \
        --epochs 100 \
        --batch  8

Features:
  - Mixed precision (AMP) — ~2× speedup on T4
  - OneCycleLR with 10% warmup
  - Differential LR (encoder 10× lower than decoder)
  - Per-band R² logged every epoch
  - Best checkpoint saved by mean R²
  - Early stopping (patience = 15)
  - Auto-saves to Drive every epoch so Colab disconnect won't lose progress
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from dataset import WeedyRiceDataset
from model   import MobileNetV2UNet
from losses  import RGB2MSLoss, r2_per_band

BAND_NAMES = ["Green", "Red", "RedEdge", "NIR"]


# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train RGB->MS model")
    p.add_argument("--data",     required=True,
                   help="Path to WeedyRice-RGBMS-DB/ root")
    p.add_argument("--out",      default="checkpoints",
                   help="Directory to save checkpoints (use Drive path in Colab)")
    p.add_argument("--img_size", type=int, default=512,
                   help="Square crop size. Use 384 if Colab OOM.")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=8,
                   help="Batch size. 8 works on Colab T4 at 512px.")
    p.add_argument("--lr",       type=float, default=3e-4,
                   help="Peak LR for decoder. Encoder gets lr/10.")
    p.add_argument("--workers",  type=int, default=2,
                   help="DataLoader workers. 2 is safe on Colab.")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--no_pretrain", action="store_true",
                   help="Skip ImageNet pretrained weights.")
    return p.parse_args()


# -----------------------------------------------------------------------
def build_optimizer(model, lr: float):
    """Differential LR: encoder 10× lower to preserve pretrained features."""
    enc_names = {"enc0", "enc1", "enc2", "enc3", "enc4"}
    enc_p, dec_p = [], []
    for name, param in model.named_parameters():
        (enc_p if name.split(".")[0] in enc_names else dec_p).append(param)

    return torch.optim.AdamW(
        [{"params": enc_p, "lr": lr / 10},
         {"params": dec_p, "lr": lr}],
        weight_decay=1e-4,
    )


# -----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler,
                criterion, scaler, device, epoch: int):
    model.train()
    sum_loss = 0.0
    sum_r2   = torch.zeros(4)
    n = 0

    for i, (rgb, ms, _) in enumerate(loader):
        rgb = rgb.to(device, non_blocking=True)
        ms  = ms.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            pred        = model(rgb)
            loss, parts = criterion(pred, ms)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            r2 = r2_per_band(pred.float(), ms.float()).cpu()

        sum_loss += parts["loss_total"]
        sum_r2   += r2
        n        += 1

        if (i + 1) % 10 == 0:
            print(f"  [{epoch}] batch {i+1}/{len(loader)} | "
                  f"loss={parts['loss_total']:.4f} | "
                  f"R²_mean={r2.mean():.3f} | "
                  f"lr={scheduler.get_last_lr()[-1]:.2e}")

    return sum_loss / n, sum_r2 / n


# -----------------------------------------------------------------------
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    sum_loss = 0.0
    sum_r2   = torch.zeros(4)
    n = 0

    for rgb, ms, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        ms  = ms.to(device,  non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            pred        = model(rgb)
            _, parts    = criterion(pred, ms)
        r2 = r2_per_band(pred.float(), ms.float()).cpu()
        sum_loss += parts["loss_total"]
        sum_r2   += r2
        n        += 1

    return sum_loss / n, sum_r2 / n


# -----------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Data    : {args.data}")
    print(f"Out     : {out_dir}")
    print(f"img_size: {args.img_size}  batch: {args.batch}  epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # Datasets
    size = (args.img_size, args.img_size)
    train_ds = WeedyRiceDataset(args.data, "train", size, augment=True)
    val_ds   = WeedyRiceDataset(args.data, "val",   size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model
    model = MobileNetV2UNet(out_channels=4,
                            pretrained=not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M\n")

    criterion = RGB2MSLoss(w_l1=0.5, w_ssim=0.3, w_sa=0.2)
    optimizer = build_optimizer(model, args.lr)
    scaler    = GradScaler(enabled=device.type == "cuda")

    # OneCycleLR — warmup for first 10% of steps, cosine anneal after
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr / 10, args.lr],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
    )

    # Resume
    start_epoch  = 1
    best_r2      = -float("inf")
    patience_cnt = 0
    history      = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_r2     = ckpt.get("best_r2", best_r2)
        print(f"Resumed from epoch {ckpt['epoch']} (best R²={best_r2:.4f})\n")

    # Training loop
    PATIENCE = 15
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_r2 = train_epoch(model, train_loader, optimizer, scheduler,
                                     criterion, scaler, device, epoch)
        va_loss, va_r2 = val_epoch(model, val_loader, criterion, device)

        mean_r2  = va_r2.mean().item()
        elapsed  = time.time() - t0

        # Pretty print
        r2_str = "  ".join(f"{n}:{v:.3f}" for n, v in zip(BAND_NAMES, va_r2.tolist()))
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"{elapsed:.0f}s")
        print(f"  Val  R² → {r2_str}  | mean={mean_r2:.4f}")
        print(f"  Train R² mean={tr_r2.mean().item():.4f}")

        history.append({
            "epoch":       epoch,
            "train_loss":  round(tr_loss, 6),
            "val_loss":    round(va_loss, 6),
            "val_r2_mean": round(mean_r2, 4),
            "val_r2_bands": [round(v, 4) for v in va_r2.tolist()],
        })

        # Save best
        if mean_r2 > best_r2:
            best_r2      = mean_r2
            patience_cnt = 0
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "best_r2":       best_r2,
                "val_r2_bands":  va_r2.tolist(),
                "args":          vars(args),
            }, out_dir / "best_model.pth")
            print(f"  ✓ Best saved (R²={best_r2:.4f})")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{PATIENCE})")

        # Save latest every 5 epochs (Drive backup for Colab)
        if epoch % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_r2":   best_r2,
            }, out_dir / "latest.pth")

        # Save history JSON (overwrite each epoch)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\nDone. Best val R² = {best_r2:.4f}")
    print(f"Best model : {out_dir / 'best_model.pth'}")
    print(f"History    : {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
