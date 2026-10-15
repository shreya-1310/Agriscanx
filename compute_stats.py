"""
compute_stats.py
----------------
Compute per-channel mean & std over the training split.
Run this BEFORE training and paste the output into dataset.py.

Usage (Colab):
    !python compute_stats.py \
        --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB"
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


BAND_SUFFIXES = ["_G.TIF", "_R.TIF", "_RE.TIF", "_NIR.TIF"]
BAND_NAMES    = ["Green", "Red", "RedEdge", "NIR"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to WeedyRice-RGBMS-DB/")
    return p.parse_args()


def main():
    args  = parse_args()
    root  = Path(args.data)
    meta  = root / "Metadata"

    # Load training stems from train.txt
    train_file = meta / "train.txt"
    with open(train_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Strip .JPG extension to get base stem
    stems = [Path(l).name.replace(".JPG", "").replace(".jpg", "")
             for l in lines]
    print(f"Computing stats over {len(stems)} training samples...\n")

    rgb_dir = root / "RGB"
    ms_dir  = root / "Multispectral"

    rgb_sum  = np.zeros(3, np.float64)
    rgb_ssq  = np.zeros(3, np.float64)
    ms_sum   = np.zeros(4, np.float64)
    ms_ssq   = np.zeros(4, np.float64)
    count    = 0

    for i, stem in enumerate(stems):
        # RGB
        for ext in (".JPG", ".jpg"):
            p = rgb_dir / (stem + ext)
            if p.exists():
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
                rgb_sum += arr.reshape(-1, 3).mean(0)
                rgb_ssq += (arr ** 2).reshape(-1, 3).mean(0)
                break

        # MS bands
        for b_idx, suffix in enumerate(BAND_SUFFIXES):
            p = ms_dir / (stem + suffix)
            if p.exists():
                arr = tifffile.imread(str(p)).astype(np.float64) / 65535.0
                ms_sum[b_idx] += arr.mean()
                ms_ssq[b_idx] += (arr ** 2).mean()

        count += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(stems)}")

    n = count
    rgb_mean = rgb_sum / n
    rgb_std  = np.sqrt(np.clip(rgb_ssq / n - rgb_mean ** 2, 0, None))
    ms_mean  = ms_sum  / n
    ms_std   = np.sqrt(np.clip(ms_ssq  / n - ms_mean  ** 2, 0, None))

    print("\n" + "="*55)
    print("Copy these into dataset.py AND rpi_inference.py:")
    print("="*55)
    print(f"RGB_MEAN = {rgb_mean.tolist()}")
    print(f"RGB_STD  = {rgb_std.tolist()}")
    print(f"# MS order: [Green, Red, RedEdge, NIR]")
    print(f"MS_MEAN  = {ms_mean.tolist()}")
    print(f"MS_STD   = {ms_std.tolist()}")

    for b, name in enumerate(BAND_NAMES):
        print(f"  {name:10s}: mean={ms_mean[b]:.4f}  std={ms_std[b]:.4f}")


if __name__ == "__main__":
    main()
