"""Kimi-Music-Codec: streamable causal convolution building blocks.

Faithful full implementation of the EnCodec/DAC-style streamable conv
stack used by the design doc (Kimi_MusicGPT_Architecture_Design.md §2):

* ``CausalConv1d``          -- left-padded causal Conv1d (stride/dilation aware)
* ``CausalConvTranspose1d`` -- causal ConvTranspose1d (right-trimmed)
* ``ResidualUnit``          -- dilated conv + 1x1 conv residual unit
* ``EncoderBlock``          -- residual units + strided causal downsampling
* ``DecoderBlock``          -- causal transposed upsampling + residual units

Causality contract: output frame ``t`` depends only on input samples with
index ``<= t * stride`` (encoder) and input frames ``<= t`` (decoder), so
the codec can run fully streamingly and matches the causal nature of the
KDA backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_conv1d(conv: nn.Conv1d, norm: str) -> nn.Module:
    if norm == "weight_norm":
        return nn.utils.weight_norm(conv)
    if norm == "none":
        return conv
    raise ValueError(f"unknown norm '{norm}'")


def _norm_convT1d(conv: nn.ConvTranspose1d, norm: str) -> nn.Module:
    if norm == "weight_norm":
        return nn.utils.weight_norm(conv)
    if norm == "none":
        return conv
    raise ValueError(f"unknown norm '{norm}'")


def _act(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "lrelu":
        return nn.LeakyReLU(0.2)
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"unknown activation '{name}'")


# --------------------------------------------------------------------------- #
# Causal convolutions
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    """Conv1d with strictly left-sided (causal) padding.

    Left pad is ``(kernel - 1) * dilation - (stride - 1)`` (clamped at 0),
    which for stride 1 is the classic ``(k-1)*d`` and for the EnCodec
    downsampling convention ``kernel = 2 * stride`` gives exactly ``stride``
    samples of left context, producing ``ceil(L / stride)`` output frames.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, bias=True, norm="weight_norm"):
        super().__init__()
        self.pad = max((kernel_size - 1) * dilation - (stride - 1), 0)
        self.conv = _norm_conv1d(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      dilation=dilation, bias=bias), norm)

    def forward(self, x):                       # x: [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))

    def stream_forward(self, x, history):
        """One-step streaming forward with explicit left-history buffer.

        ``history``: [B, C, self.pad] previous left context (zeros at t=0).
        Returns (out, new_history).
        """
        if self.pad > 0:
            full = torch.cat([history, x], dim=-1)
            new_history = full[..., -self.pad:].detach()
        else:
            full, new_history = x, history
        return self.conv(full), new_history


class CausalConvTranspose1d(nn.Module):
    """ConvTranspose1d made causal by trimming ``kernel - stride`` samples
    from the *right* of the output.

    With zero explicit padding the transposed conv emits
    ``(L-1)*s + k`` samples; the last ``k - s`` depend on future frames
    under causal interpretation, so they are removed, yielding exactly
    ``L * s`` output samples where output block ``i`` depends only on input
    frames ``<= i``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, norm="weight_norm"):
        super().__init__()
        if kernel_size < stride:
            raise ValueError("CausalConvTranspose1d needs kernel_size >= stride")
        self.trim = kernel_size - stride
        self.conv = _norm_convT1d(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                               stride=stride, bias=bias), norm)

    def forward(self, x):                       # x: [B, C, T]
        y = self.conv(x)
        if self.trim > 0:
            y = y[..., :-self.trim]
        return y


# --------------------------------------------------------------------------- #
# Residual unit / encoder / decoder blocks
# --------------------------------------------------------------------------- #
class ResidualUnit(nn.Module):
    """Dilated causal conv -> act -> 1x1 conv, with residual connection."""

    def __init__(self, channels, kernel_size=7, dilation=1, norm="weight_norm",
                 activation="elu"):
        super().__init__()
        self.block = nn.Sequential(
            _act(activation),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation,
                         norm=norm),
            _act(activation),
            CausalConv1d(channels, channels, 1, norm=norm),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Residual units (dilations) + strided causal downsampling conv."""

    def __init__(self, in_ch, out_ch, stride, kernel_size, residual_kernel_size,
                 dilations, norm="weight_norm", activation="elu"):
        super().__init__()
        self.res_units = nn.ModuleList(
            ResidualUnit(in_ch, residual_kernel_size, dilation=d, norm=norm,
                         activation=activation)
            for d in dilations
        )
        self.down = nn.Sequential(
            _act(activation),
            CausalConv1d(in_ch, out_ch, kernel_size, stride=stride, norm=norm),
        )

    def forward(self, x):
        for ru in self.res_units:
            x = ru(x)
        return self.down(x)


class DecoderBlock(nn.Module):
    """Causal transposed upsampling conv + residual units (dilations)."""

    def __init__(self, in_ch, out_ch, stride, kernel_size, residual_kernel_size,
                 dilations, norm="weight_norm", activation="elu"):
        super().__init__()
        self.up = nn.Sequential(
            _act(activation),
            CausalConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride,
                                  norm=norm),
        )
        self.res_units = nn.ModuleList(
            ResidualUnit(out_ch, residual_kernel_size, dilation=d, norm=norm,
                         activation=activation)
            for d in dilations
        )

    def forward(self, x):
        x = self.up(x)
        for ru in self.res_units:
            x = ru(x)
        return x
