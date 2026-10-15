"""
model.py
--------
MobileNetV2-UNet with Attention Gates.
RGB (3, H, W) -> MS (4, H, W)  image-to-image regression.

Design rationale for R-Pi 4 deployment:
  - MobileNetV2 encoder  : depthwise separable convs, ~3.4M params, fast on ARM CPU
  - UNet skip connections: preserves spatial detail -> higher per-pixel R²
  - Attention gates      : suppress irrelevant background, focus on vegetation
  - Depthwise separable decoder: lightweight, Pi-friendly
  - No activation on output: raw float regression (unbounded)
  - ~5.5M total params -> ~20MB FP32, ~6MB INT8 ONNX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# -----------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DWSConv(nn.Module):
    """Depthwise-Separable conv block (Pi-efficient decoder building block)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DecoderBlock(nn.Module):
    """
    Upsample 2× → concat skip → two DWSConv reductions.
    skip_ch=0 means no skip connection (final upsample stage).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        merged    = in_ch + skip_ch
        self.conv = nn.Sequential(
            DWSConv(merged, out_ch),
            DWSConv(out_ch, out_ch),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:          # handle odd dims
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate: gating signal g (coarser) focuses skip feature x (finer).
    Learns to suppress irrelevant background pixels.
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.Wg   = nn.Conv2d(g_ch,    inter_ch, 1, bias=False)
        self.Wx   = nn.Conv2d(x_ch,    inter_ch, 1, bias=False)
        self.psi  = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upsample g to x's spatial size
        g_up  = F.interpolate(self.Wg(g), size=x.shape[-2:],
                              mode="bilinear", align_corners=False)
        alpha = self.psi(F.relu(g_up + self.Wx(x), inplace=True))
        return x * alpha


# -----------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------

class MobileNetV2UNet(nn.Module):
    """
    Input  : (B, 3, H, W)   RGB image
    Output : (B, 4, H, W)   predicted MS bands [Green, Red, RedEdge, NIR]

    Encoder skip sizes (512×512 input):
      enc0 -> 16ch  @ H/2
      enc1 -> 24ch  @ H/4
      enc2 -> 32ch  @ H/8
      enc3 -> 96ch  @ H/16
      enc4 -> 1280ch @ H/32  (bottleneck)
    """

    def __init__(self, out_channels: int = 4, pretrained: bool = True):
        super().__init__()

        # ---- Encoder: MobileNetV2 ----
        weights  = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        feats    = backbone.features

        self.enc0 = feats[0:2]     # -> 16ch
        self.enc1 = feats[2:4]     # -> 24ch
        self.enc2 = feats[4:7]     # -> 32ch
        self.enc3 = feats[7:14]    # -> 96ch
        self.enc4 = feats[14:19]   # -> 1280ch  bottleneck

        e0, e1, e2, e3, e4 = 16, 24, 32, 96, 1280

        # ---- Bottleneck bridge ----
        self.bridge = nn.Sequential(
            ConvBnRelu(e4, 256),
            ConvBnRelu(256, 256),
        )

        # ---- Attention gates ----
        self.att3 = AttentionGate(g_ch=256, x_ch=e3, inter_ch=48)
        self.att2 = AttentionGate(g_ch=128, x_ch=e2, inter_ch=16)
        self.att1 = AttentionGate(g_ch=64,  x_ch=e1, inter_ch=12)
        self.att0 = AttentionGate(g_ch=32,  x_ch=e0, inter_ch=8)

        # ---- Decoder ----
        self.dec3      = DecoderBlock(in_ch=256, skip_ch=e3, out_ch=128)
        self.dec2      = DecoderBlock(in_ch=128, skip_ch=e2, out_ch=64)
        self.dec1      = DecoderBlock(in_ch=64,  skip_ch=e1, out_ch=32)
        self.dec0      = DecoderBlock(in_ch=32,  skip_ch=e0, out_ch=32)
        self.dec_final = DecoderBlock(in_ch=32,  skip_ch=0,  out_ch=16)

        # ---- Output head ----
        self.head = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, out_channels, kernel_size=1),
            # NO activation — raw regression output
        )

        self._init_decoder()

    # ------------------------------------------------------------------
    def _init_decoder(self):
        modules = [self.bridge, self.dec3, self.dec2,
                   self.dec1, self.dec0, self.dec_final, self.head,
                   self.att3, self.att2, self.att1, self.att0]
        for m in modules:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight,
                                           mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias,   0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s0 = self.enc0(x)    # (B, 16,   H/2,  W/2)
        s1 = self.enc1(s0)   # (B, 24,   H/4,  W/4)
        s2 = self.enc2(s1)   # (B, 32,   H/8,  W/8)
        s3 = self.enc3(s2)   # (B, 96,   H/16, W/16)
        s4 = self.enc4(s3)   # (B, 1280, H/32, W/32)

        # Bridge
        b = self.bridge(s4)  # (B, 256, H/32, W/32)

        # Decode with attention-gated skips
        d3 = self.dec3(b,  self.att3(b,  s3))   # (B, 128, H/16, W/16)
        d2 = self.dec2(d3, self.att2(d3, s2))   # (B,  64, H/8,  W/8)
        d1 = self.dec1(d2, self.att1(d2, s1))   # (B,  32, H/4,  W/4)
        d0 = self.dec0(d1, self.att0(d1, s0))   # (B,  32, H/2,  W/2)
        d  = self.dec_final(d0)                  # (B,  16, H,    W)

        out = self.head(d)                        # (B,   4, H,    W)

        # Guarantee exact input spatial size (handles odd H/W)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return out


# -----------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------
if __name__ == "__main__":
    model = MobileNetV2UNet(out_channels=4, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")   # expect (2, 4, 512, 512)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params     : {total:.2f}M")
    print(f"Trainable params : {trainable:.2f}M")
