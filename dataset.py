"""
dataset.py
----------
PyTorch Dataset for RGB -> Multispectral (4-band) regression.

Real dataset naming confirmed from metadata CSVs:

  RGB  : DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m.JPG
  MS-G : DJI_DateTime_2024_06_02_13_42_0035_lat_10.3040603_lon_105.2619317_alt_20.018m_G.TIF
  MS-R : ...same stem..._R.TIF
  MS-RE: ...same stem..._RE.TIF
  MS-NIR:...same stem..._NIR.TIF

  train.txt / val.txt / test.txt store full RGB filenames (with .JPG extension)

Dataset root structure:
  WeedyRice-RGBMS-DB/
  ├── RGB/
  ├── Multispectral/
  ├── Masks/
  ├── Overlay/
  └── Metadata/
        ├── image_metadata.csv
        ├── filename_mapping.csv
        ├── train.txt
        ├── val.txt
        └── test.txt
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# -----------------------------------------------------------------------
# Normalization stats — run compute_stats.py once and replace these
# -----------------------------------------------------------------------
RGB_MEAN = [0.485, 0.456, 0.406]   # ImageNet defaults until you compute real ones
RGB_STD  = [0.229, 0.224, 0.225]

# MS bands order: [Green, Red, RedEdge, NIR]  — uint16 TIFF normalized to [0,1]
# Replace after running compute_stats.py
MS_MEAN = [0.1500, 0.1200, 0.1800, 0.3500]
MS_STD  = [0.0800, 0.0700, 0.0900, 0.1500]


class WeedyRiceDataset(Dataset):
    """
    Args:
        root     : path to WeedyRice-RGBMS-DB/
        split    : 'train' | 'val' | 'test'
        img_size : (H, W) to resize all images to
        augment  : spatial augmentations (train split only)

    Returns per sample:
        rgb  : FloatTensor (3, H, W)  normalized
        ms   : FloatTensor (4, H, W)  normalized  [G, R, RE, NIR]
        stem : str  base stem for debugging / saving outputs
    """

    # MS band suffix in the actual files
    MS_SUFFIXES = {
        "G":   "_G.TIF",
        "R":   "_R.TIF",
        "RE":  "_RE.TIF",
        "NIR": "_NIR.TIF",
    }
    # Order we stack into channel dim: [Green, Red, RedEdge, NIR]
    MS_ORDER = ["G", "R", "RE", "NIR"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: tuple = (512, 512),
        augment: bool = True,
    ):
        self.root     = Path(root)
        self.rgb_dir  = self.root / "RGB"
        self.ms_dir   = self.root / "Multispectral"
        self.meta_dir = self.root / "Metadata"
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        # ---- Load split file ----
        # train.txt / val.txt / test.txt contain full filenames like:
        # DJI_DateTime_2024_06_02_13_42_0035_lat_..._alt_20.018m.JPG
        split_file = self.meta_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Expected one of: train.txt, val.txt, test.txt in {self.meta_dir}"
            )

        with open(split_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        # Strip extension to get the base stem for MS lookup
        # e.g. "DJI_DateTime_..._alt_20.018m.JPG" -> "DJI_DateTime_..._alt_20.018m"
        self.stems = []
        for line in lines:
            # Handle both full filenames and bare stems
            fname = Path(line).name          # strip any path prefix
            stem  = fname.replace(".JPG", "").replace(".jpg", "").replace(".jpeg", "")
            self.stems.append(stem)

        print(f"[WeedyRiceDataset] split={split:5s}  samples={len(self.stems)}")

        # Sanity check: verify a few files exist
        self._verify_sample(self.stems[0])

    # ------------------------------------------------------------------
    def _verify_sample(self, stem: str):
        """Raise early with a clear message if files are missing."""
        rgb_path = self.rgb_dir / (stem + ".JPG")
        if not rgb_path.exists():
            # Try lowercase extension
            rgb_path = self.rgb_dir / (stem + ".jpg")
        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB file not found: {self.rgb_dir / (stem + '.JPG')}\n"
                f"Check that --data points to the WeedyRice-RGBMS-DB/ root folder."
            )
        for band, suffix in self.MS_SUFFIXES.items():
            ms_path = self.ms_dir / (stem + suffix)
            if not ms_path.exists():
                raise FileNotFoundError(
                    f"MS band {band} not found: {ms_path}\n"
                    f"Expected file: {stem + suffix}"
                )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stems)

    # ------------------------------------------------------------------
    def _load_rgb(self, stem: str) -> np.ndarray:
        """Load RGB JPEG -> (H, W, 3) uint8."""
        for ext in (".JPG", ".jpg", ".jpeg", ".png"):
            p = self.rgb_dir / (stem + ext)
            if p.exists():
                return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        raise FileNotFoundError(f"RGB not found for stem: {stem}")

    def _load_ms(self, stem: str) -> np.ndarray:
        """
        Load 4 MS TIFFs and stack -> (4, H, W) float32 in [0, 1].
        Band order: [Green, Red, RedEdge, NIR]
        """
        bands = []
        for key in self.MS_ORDER:          # G, R, RE, NIR
            suffix = self.MS_SUFFIXES[key]
            p = self.ms_dir / (stem + suffix)
            arr = tifffile.imread(str(p)).astype(np.float32)
            # uint16 DN -> [0, 1]
            arr = arr / 65535.0
            bands.append(arr)
        return np.stack(bands, axis=0)     # (4, H, W)

    # ------------------------------------------------------------------
    def _resize(self, rgb_np: np.ndarray,
                ms_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = self.img_size
        rgb_pil = Image.fromarray(rgb_np).resize((W, H), Image.BILINEAR)
        rgb_out = np.array(rgb_pil, dtype=np.uint8)

        ms_out = []
        for b in range(ms_np.shape[0]):
            band_pil = Image.fromarray(ms_np[b]).resize((W, H), Image.BILINEAR)
            ms_out.append(np.array(band_pil, dtype=np.float32))
        ms_out = np.stack(ms_out, axis=0)

        return rgb_out, ms_out

    # ------------------------------------------------------------------
    def _augment(self, rgb_t: torch.Tensor,
                 ms_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial augmentations applied IDENTICALLY to RGB and MS.
        Color jitter applied to RGB ONLY (never to spectral bands).
        """
        # Random horizontal flip
        if random.random() > 0.5:
            rgb_t = TF.hflip(rgb_t)
            ms_t  = TF.hflip(ms_t)

        # Random vertical flip
        if random.random() > 0.5:
            rgb_t = TF.vflip(rgb_t)
            ms_t  = TF.vflip(ms_t)

        # Random 90° rotation (0 / 90 / 180 / 270)
        k = random.randint(0, 3)
        if k:
            rgb_t = torch.rot90(rgb_t, k, dims=[1, 2])
            ms_t  = torch.rot90(ms_t,  k, dims=[1, 2])

        # Color jitter — RGB only (spectral values must stay intact)
        if random.random() > 0.4:
            rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.75, 1.25))
        if random.random() > 0.4:
            rgb_t = TF.adjust_contrast(rgb_t,   random.uniform(0.80, 1.20))
        if random.random() > 0.4:
            rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.80, 1.20))

        return rgb_t, ms_t

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        # Load
        rgb_np = self._load_rgb(stem)      # (H, W, 3)  uint8
        ms_np  = self._load_ms(stem)       # (4, H, W)  float32 [0,1]

        # Resize
        rgb_np, ms_np = self._resize(rgb_np, ms_np)

        # To tensors
        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        ms_t  = torch.from_numpy(ms_np).float()

        # Normalize RGB  (ImageNet stats — update after compute_stats.py)
        rgb_t = TF.normalize(rgb_t, mean=RGB_MEAN, std=RGB_STD)

        # Normalize MS per-band
        ms_mean_t = torch.tensor(MS_MEAN, dtype=torch.float32).view(4, 1, 1)
        ms_std_t  = torch.tensor(MS_STD,  dtype=torch.float32).view(4, 1, 1)
        ms_t = (ms_t - ms_mean_t) / (ms_std_t + 1e-8)

        # Augment (train only)
        if self.augment:
            rgb_t, ms_t = self._augment(rgb_t, ms_t)

        return rgb_t, ms_t, stem
