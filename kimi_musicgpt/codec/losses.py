"""Kimi-Music-Codec: training losses (design doc §2.4).

* ``MultiScaleSTFTLoss``  -- per-scale spectral convergence + log-magnitude L1
* ``MelSpectrogramLoss``  -- L1 on log-mel spectrograms (torch-native mel
  filterbank, HTK formula)
* adversarial hinge losses (generator / discriminator)
* ``feature_matching_loss`` -- L1 between real/fake intermediate feature maps
* ``waveform_l1``         -- time-domain L1

Mixed-precision note (FP16 on the V100 target)
----------------------------------------------
Audio losses are *measurement* operators, not compute bottlenecks: under
autocast the decoder emits fp16 while the target waveform is fp32, and an
fp16 spectrogram has only ~3 decimal digits -- enough to flatten the very
spectral differences these losses exist to penalise.  Every loss below
therefore up-casts its inputs to fp32 first.  The cost is negligible (the
STFT/mel computation stays on the fp32 path) and it keeps the codec's
training signal identical to the fp32 reference run.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# STFT helpers
# --------------------------------------------------------------------------- #
def _stft_mag(x, n_fft, hop, win):
    x = x.float()
    window = torch.hann_window(win, device=x.device)
    spec = torch.stft(x, n_fft, hop, win, window, return_complex=True,
                      center=True)
    return spec.abs()


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, scales=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.scales = list(scales)

    def forward(self, x, y):                    # [B, 1, T]
        x, y = x.squeeze(1).float(), y.squeeze(1).float()
        total = 0.0
        for n_fft in self.scales:
            hop, win = n_fft // 4, n_fft
            mx = _stft_mag(x, n_fft, hop, win).clamp(min=1e-7)
            my = _stft_mag(y, n_fft, hop, win).clamp(min=1e-7)
            sc = (my - mx).norm(p="fro") / my.norm(p="fro").clamp(min=1e-7)
            log_l1 = F.l1_loss(mx.log(), my.log())
            total = total + sc + log_l1
        return total / len(self.scales)


# --------------------------------------------------------------------------- #
# Mel loss
# --------------------------------------------------------------------------- #
def _hz_to_mel(f):
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=None):
    """HTK-style triangular mel filterbank, shape [n_mels, n_fft//2+1]."""
    fmax = fmax or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mels = torch.linspace(float(_hz_to_mel(torch.tensor(fmin))),
                          float(_hz_to_mel(torch.tensor(fmax))), n_mels + 2)
    hz = _mel_to_hz(mels)
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, center, right = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - left) / max(float(center - left), 1e-9)
        down = (right - freqs) / max(float(right - center), 1e-9)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate, n_fft=2048, hop=512, n_mels=80):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels))
        self.register_buffer("window", torch.hann_window(n_fft))

    def _mel(self, x):
        spec = torch.stft(x.float(), self.n_fft, self.hop, self.n_fft,
                          self.window, return_complex=True, center=True).abs()
        mel = torch.einsum("mf,bft->bmt", self.fb, spec)
        return mel.clamp(min=1e-5).log()

    def forward(self, x, y):                    # [B, 1, T]
        return F.l1_loss(self._mel(x.squeeze(1)), self._mel(y.squeeze(1)))


# --------------------------------------------------------------------------- #
# Adversarial (hinge) + feature matching
# --------------------------------------------------------------------------- #
def discriminator_loss(real_logits_list, fake_logits_list):
    """Hinge GAN discriminator loss (fp32 -- hinge margins are absolute)."""
    loss = 0.0
    for rl, fl in zip(real_logits_list, fake_logits_list):
        loss = loss + F.relu(1.0 - rl.float()).mean() + F.relu(1.0 + fl.float()).mean()
    return loss / len(real_logits_list)


def generator_loss(fake_logits_list):
    """Hinge GAN generator loss (fp32)."""
    loss = 0.0
    for fl in fake_logits_list:
        loss = loss + F.relu(1.0 - fl.float()).mean()
    return loss / len(fake_logits_list)


def feature_matching_loss(real_fmaps_list, fake_fmaps_list):
    loss, n = 0.0, 0
    for rf, ff in zip(real_fmaps_list, fake_fmaps_list):
        for r, f in zip(rf, ff):
            loss = loss + F.l1_loss(f.float(), r.detach().float())
            n += 1
    return loss / max(n, 1)


def waveform_l1(x, y):
    """Time-domain L1 in fp32 (target is fp32, reconstruction may be fp16)."""
    return F.l1_loss(x.float(), y.float())
