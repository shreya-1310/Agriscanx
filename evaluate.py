"""
evaluate.py
-----------
Full test-set evaluation with per-band R², RMSE, MAE, PSNR
plus scatter plots and visual comparisons.

Usage (Colab):
    !python evaluate.py \
        --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB" \
        --ckpt "/content/drive/MyDrive/rgb2ms_checkpoints/best_model.pth" \
        --out  "/content/drive/MyDrive/rgb2ms_eval" \
        --save_vis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from dataset import WeedyRiceDataset, MS_MEAN, MS_STD, RGB_MEAN, RGB_STD
from model   import MobileNetV2UNet
from losses  import r2_per_band

BAND_NAMES = ["Green", "Red", "RedEdge", "NIR"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True)
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--out",      default="eval_results")
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--save_vis", action="store_true")
    return p.parse_args()


def denorm_ms(t: torch.Tensor) -> torch.Tensor:
    """Undo MS normalisation -> [0,1] reflectance."""
    mean = torch.tensor(MS_MEAN).view(4,1,1)
    std  = torch.tensor(MS_STD).view(4,1,1)
    return (t * std + mean).clamp(0, 1)


@torch.no_grad()
def evaluate(model, loader, device, out_dir, save_vis=False):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred, all_gt = [], []

    for bi, (rgb, ms, names) in enumerate(loader):
        rgb = rgb.to(device)
        ms  = ms.to(device)
        with autocast(enabled=device.type == "cuda"):
            pred = model(rgb)
        all_pred.append(pred.cpu())
        all_gt.append(ms.cpu())

        if save_vis and bi == 0:
            _save_visuals(rgb.cpu(), pred.cpu(), ms.cpu(), names, out_dir)

    pred_all = torch.cat(all_pred)    # (N, 4, H, W)
    gt_all   = torch.cat(all_gt)

    print("\nTest-set metrics")
    print("-" * 60)
    results = {}
    for b, name in enumerate(BAND_NAMES):
        p = pred_all[:, b].reshape(-1).float()
        t = gt_all[:,  b].reshape(-1).float()

        ss_res = ((p - t)**2).sum().item()
        ss_tot = ((t - t.mean())**2).sum().item()
        r2     = 1.0 - ss_res / (ss_tot + 1e-8)
        rmse   = float(((p - t)**2).mean().sqrt())
        mae    = float((p - t).abs().mean())
        psnr   = 10 * np.log10(1.0 / (float(((p - t)**2).mean()) + 1e-10))

        flag = "✓ PASS" if r2 >= 0.75 else "✗ FAIL"
        print(f"  {name:10s}  R²={r2:.4f} {flag}  "
              f"RMSE={rmse:.5f}  MAE={mae:.5f}  PSNR={psnr:.1f}dB")
        results[name] = {"R2": round(r2,4), "RMSE": round(rmse,6),
                         "MAE": round(mae,6), "PSNR": round(psnr,2)}

    mean_r2 = float(np.mean([v["R2"] for v in results.values()]))
    results["mean_R2"] = round(mean_r2, 4)
    print(f"\n  Mean R² = {mean_r2:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'metrics.json'}")

    try:
        _scatter_plots(pred_all, gt_all, out_dir)
    except Exception as e:
        print(f"Scatter plots skipped: {e}")

    return results


def _scatter_plots(pred, target, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["seagreen", "firebrick", "darkorange", "purple"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for b, (name, ax, col) in enumerate(zip(BAND_NAMES, axes.flat, colors)):
        p = pred[:,  b].reshape(-1).numpy()
        t = target[:, b].reshape(-1).numpy()
        idx = np.random.choice(len(p), min(15000, len(p)), replace=False)
        ax.scatter(t[idx], p[idx], alpha=0.25, s=1, c=col)
        mn, mx = t.min(), t.max()
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="ideal")
        ss_res = ((p - t)**2).sum()
        ss_tot = ((t - t.mean())**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        ax.set_title(f"{name}  R²={r2:.4f}", fontsize=11)
        ax.set_xlabel("Ground truth (normalised)")
        ax.set_ylabel("Predicted (normalised)")
        ax.legend(fontsize=8)

    plt.suptitle("Predicted vs Actual — all test samples", fontsize=13)
    plt.tight_layout()
    p = out_dir / "scatter_plots.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Saved: {p}")


def _save_visuals(rgb, pred, gt, names, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vis_dir = out_dir / "visuals"
    vis_dir.mkdir(exist_ok=True)

    rgb_mean = torch.tensor(RGB_MEAN).view(3,1,1)
    rgb_std  = torch.tensor(RGB_STD).view(3,1,1)
    cmaps    = ["Greens", "Reds", "Oranges", "RdPu"]

    for i in range(min(6, rgb.shape[0])):
        rgb_vis = (rgb[i] * rgb_std + rgb_mean).clamp(0,1).permute(1,2,0).numpy()
        p_dn    = denorm_ms(pred[i]).numpy()
        g_dn    = denorm_ms(gt[i]).numpy()

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes[0,0].imshow(rgb_vis); axes[0,0].set_title("RGB Input")
        for j in range(1,4): axes[0,j].axis("off")
        for b in range(4):
            axes[1,b].imshow(p_dn[b], cmap=cmaps[b], vmin=0, vmax=1)
            axes[1,b].set_title(f"Pred {BAND_NAMES[b]}")
            axes[2,b].imshow(g_dn[b], cmap=cmaps[b], vmin=0, vmax=1)
            axes[2,b].set_title(f"GT {BAND_NAMES[b]}")
        for ax in axes.flat: ax.axis("off")

        stem = names[i] if isinstance(names, (list,tuple)) else f"s{i}"
        plt.suptitle(stem, fontsize=8)
        plt.tight_layout()
        plt.savefig(vis_dir / f"{i:03d}_comparison.png", dpi=100)
        plt.close()

    print(f"Saved visual comparisons to {vis_dir}/")


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Evaluating on: {device}")

    ds     = WeedyRiceDataset(args.data, "test",
                              (args.img_size, args.img_size), augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)

    ckpt  = torch.load(args.ckpt, map_location=device)
    model = MobileNetV2UNet(out_channels=4, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt.get('epoch','?')}  "
          f"(saved R²={ckpt.get('best_r2',0):.4f})\n")

    evaluate(model, loader, device, args.out, args.save_vis)


if __name__ == "__main__":
    main()
