"""Kimi-Music-Codec: adversarial discriminators (design doc §2.4).

Complete implementations, no stubs:

* ``MultiPeriodDiscriminator``  -- HiFi-GAN style MPD: the waveform is
  folded into ``[B, 1, T/p, p]`` for periods ``p`` and scored by a stack of
  weight-normed Conv2d layers; returns logits + intermediate feature maps
  (needed for the feature-matching loss).
* ``MultiScaleSTFTDiscriminator`` -- DAC style: log-magnitude STFT at
  several FFT sizes scored by Conv2d stacks.
* ``MultiResolutionDiscriminator`` -- wrapper aggregating both families.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Multi-period discriminator
# --------------------------------------------------------------------------- #
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, base_channels: int = 32):
        super().__init__()
        self.period = period
        c = base_channels
        widths = [c, c * 4, c * 16, c * 32, c * 32]
        kernel = (5, 1)
        stride = (3, 1)
        layers, in_c = [], 1
        for w in widths:
            layers.append(nn.utils.weight_norm(
                nn.Conv2d(in_c, w, kernel, stride=stride, padding=(2, 0))))
            layers.append(nn.LeakyReLU(0.2))
            in_c = w
        self.convs = nn.Sequential(*layers)
        self.out = nn.utils.weight_norm(nn.Conv2d(in_c, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):                       # x: [B, 1, T]
        B, C, T = x.shape
        p = self.period
        if T % p != 0:
            x = F.pad(x, (0, p - T % p), mode="reflect")
        x = x.view(B, C, -1, p)                 # [B, 1, T/p, p]
        # match the conv stack's dtype (fp16 in the pure-fp16 layout)
        x = x.to(next(self.parameters()).dtype)
        fmaps = []
        h = x
        for layer in self.convs:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                fmaps.append(h)
        logits = self.out(h).flatten(1)
        fmaps.append(logits)
        return logits, fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), base_channels: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(p, base_channels) for p in periods)

    def forward(self, x):
        logits, fmaps = [], []
        for d in self.discriminators:
            l, f = d(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# --------------------------------------------------------------------------- #
# Multi-scale STFT discriminator (DAC)
# --------------------------------------------------------------------------- #
class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int, base_channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.win = n_fft
        self.register_buffer("window", torch.hann_window(self.win))
        c = base_channels
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, c, (3, 9), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(c, c, (3, 9), stride=(1, 2),
                                           padding=(1, 4), dilation=(1, 1))),
            nn.utils.weight_norm(nn.Conv2d(c, c * 2, (3, 9), stride=(1, 2),
                                           padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(c * 2, c * 4, (3, 9), stride=(1, 2),
                                           padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(c * 4, c * 4, (3, 3), padding=(1, 1))),
        ])
        self.out = nn.utils.weight_norm(nn.Conv2d(c * 4, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):                       # x: [B, 1, T]
        # STFT in fp32: the log-magnitude input to the conv stack is a
        # measurement, and fp16 log-mags lose the very detail the
        # discriminator is supposed to judge.  The conv stack below still runs
        # on fp16 tensor cores via autocast.
        x = x.squeeze(1).float()                # [B, T]
        spec = torch.stft(x, self.n_fft, self.hop, self.win, self.window,
                          return_complex=True, center=True)
        mag = spec.abs().clamp(min=1e-7).log().unsqueeze(1)   # [B,1,F,T]
        # STFT/log-mag stay fp32 (measurement precision), but the conv stack
        # below is fp16 in the pure-fp16 layout and needs a matching input.
        mag = mag.to(next(self.parameters()).dtype)
        fmaps = []
        h = mag
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)
            fmaps.append(h)
        logits = self.out(h).flatten(1)
        fmaps.append(logits)
        return logits, fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, ffts=(2048, 1024, 512), base_channels: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList(
            STFTDiscriminator(n, base_channels) for n in ffts)

    def forward(self, x):
        logits, fmaps = [], []
        for d in self.discriminators:
            l, f = d(x)
            logits.append(l)
            fmaps.append(f)
        return logits, fmaps


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
class CodecDiscriminator(nn.Module):
    """MPD + multi-scale STFT discriminator, config-driven."""

    def __init__(self, config: dict):
        super().__init__()
        d = config["discriminator"]
        self.families = nn.ModuleList([
            MultiPeriodDiscriminator(d["mpd_periods"], d["mpd_channels"]),
            MultiScaleSTFTDiscriminator(d["stft_ffts"], d["stft_channels"]),
        ])

    def forward(self, x):
        """x: [B,1,T] -> (logits list, fmaps list) over all sub-discriminators."""
        logits, fmaps = [], []
        for fam in self.families:
            l, f = fam(x)
            logits.extend(l)
            fmaps.extend(f)
        return logits, fmaps
