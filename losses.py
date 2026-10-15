"""
losses.py
---------
Combined loss for RGB->MS image regression + R² evaluation metric.

Total = 0.5 × L1  +  0.3 × (1−SSIM)  +  0.2 × SpectralAngle

  L1            — pixel accuracy, robust to outliers (better than MSE here)
  SSIM          — structural / texture fidelity
  SpectralAngle — preserves band ratios → NDVI / NDRE accuracy downstream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# SSIM loss
# -----------------------------------------------------------------------

def _gaussian_kernel(k: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)          # (1,1,k,k)
    return kernel.expand(channels, 1, k, k).contiguous()   # (C,1,k,k)


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 channels: int = 4):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.ch = channels
        self.pad = kernel_size // 2
        kernel = _gaussian_kernel(kernel_size, sigma, channels)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        k   = self.kernel.to(pred.device)
        pad = self.pad
        ch  = self.ch

        mu_p  = F.conv2d(pred,   k, padding=pad, groups=ch)
        mu_t  = F.conv2d(target, k, padding=pad, groups=ch)

        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_p2 = F.conv2d(pred   * pred,   k, padding=pad, groups=ch) - mu_p2
        sig_t2 = F.conv2d(target * target, k, padding=pad, groups=ch) - mu_t2
        sig_pt = F.conv2d(pred   * target, k, padding=pad, groups=ch) - mu_pt

        num = (2 * mu_pt + self.C1) * (2 * sig_pt + self.C2)
        den = (mu_p2 + mu_t2 + self.C1) * (sig_p2 + sig_t2 + self.C2)

        return (1.0 - (num / den).mean())


# -----------------------------------------------------------------------
# Spectral Angle loss
# -----------------------------------------------------------------------

class SpectralAngleLoss(nn.Module):
    """
    At each pixel, treat the 4 MS bands as a spectral vector.
    Minimising the angle between pred and target preserves band ratios.
    Critical for correct NDVI / NDRE values.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        p = pred.view(B, C, -1)    # (B, 4, N)
        t = target.view(B, C, -1)

        dot    = (p * t).sum(1)                          # (B, N)
        norm_p = p.norm(dim=1).clamp(min=self.eps)
        norm_t = t.norm(dim=1).clamp(min=self.eps)

        cos_a  = (dot / (norm_p * norm_t)).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.acos(cos_a).mean()


# -----------------------------------------------------------------------
# Combined loss
# -----------------------------------------------------------------------

class RGB2MSLoss(nn.Module):
    """
    L_total = w_l1 * L1  +  w_ssim * (1-SSIM)  +  w_sa * SpectralAngle
    Returns (total_loss, component_dict) for logging.
    """
    def __init__(self, w_l1: float = 0.5, w_ssim: float = 0.3,
                 w_sa: float = 0.2, channels: int = 4):
        super().__init__()
        self.w_l1  = w_l1
        self.w_ssim = w_ssim
        self.w_sa  = w_sa
        self.l1    = nn.L1Loss()
        self.ssim  = SSIMLoss(channels=channels)
        self.sa    = SpectralAngleLoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1   = self.l1(pred, target)
        ssim = self.ssim(pred, target)
        sa   = self.sa(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_sa * sa

        return total, {
            "loss_total": total.item(),
            "loss_l1":    l1.item(),
            "loss_ssim":  ssim.item(),
            "loss_sa":    sa.item(),
        }


# -----------------------------------------------------------------------
# R² metric  (not a loss — used in evaluation)
# -----------------------------------------------------------------------

def r2_per_band(pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
    """
    Compute R² per MS band over an entire batch.

    pred, target : (B, 4, H, W)
    returns      : (4,) tensor — one R² per band [G, R, RE, NIR]
    """
    B, C, H, W = pred.shape
    p = pred.view(B, C, -1)    # (B, 4, N)
    t = target.view(B, C, -1)

    ss_res = ((p - t) ** 2).sum(dim=[0, 2])                       # (4,)
    t_mean = t.mean(dim=[0, 2], keepdim=True)
    ss_tot = ((t - t_mean) ** 2).sum(dim=[0, 2])                  # (4,)

    return 1.0 - ss_res / (ss_tot + 1e-8)
