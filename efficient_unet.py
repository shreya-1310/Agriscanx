
import torch
import torch.nn as nn


class EfficientUNetGenerator(nn.Module):
    def __init__(self, base_channels=40):
        super().__init__()

        c = base_channels

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = down_block(3, c)
        self.enc2 = down_block(c, c * 2)
        self.enc3 = down_block(c * 2, c * 4)

        self.bottleneck = conv_block(c * 4, c * 4)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1)
        self.dec3 = conv_block(c * 4, c * 2)

        self.up2 = nn.ConvTranspose2d(c * 2, c, 4, 2, 1)
        self.dec2 = conv_block(c * 2, c)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 3, 1, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.final(d2)
        return out
