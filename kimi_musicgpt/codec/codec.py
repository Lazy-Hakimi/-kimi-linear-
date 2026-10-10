"""Kimi-Music-Codec: split-RVQ neural audio codec (design doc §2).

Encoder: causal conv stack downsampling ``sample_rate`` -> ``frame_rate``
(total stride = prod(strides)); the latent sequence is quantized by a
**split residual vector quantizer**:

* codebook 0 -- semantic branch: ``latent -> semantic_dim`` projection,
  quantized by a dedicated VQ (the distillation target of §2.2.3 plugs in
  here via ``semantic_distill_loss``),
* codebooks 1..Q-1 -- acoustic branch: a residual VQ stack quantizing what
  the semantic branch leaves unexplained.

Decoder mirrors the encoder with causal transposed convs.

The module is fully config-driven (see ``configs/codec_*.json``); nothing
about sampling rate, strides or codebook counts is hard-coded. It exposes:

* ``forward(wav)``  -- training pass: reconstruction + tokens + VQ losses
* ``encode(wav)``   -- tokens  [B, T_frames, Q]
* ``decode(tokens)``-- waveform [B, 1, T_samples]
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

from .modules import (CausalConv1d, EncoderBlock, DecoderBlock)


def _fp32():
    """Context that forces the fp32 path inside an autocast region.

    Used for the quantization bottleneck: the RVQ residual chain is a sequence
    of *subtractions* of nearly-equal quantities, which is exactly what fp16
    handles worst (7 quantizers -> the 7th residual is mostly rounding noise).
    The projections involved are tiny (latent_dim x semantic dim), so keeping
    them in fp32 costs nothing measurable.
    """
    return torch.autocast(device_type="cuda", enabled=False) \
        if torch.cuda.is_available() else torch.autocast(device_type="cpu", enabled=False)


class KimiMusicCodec(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sample_rate = config["sample_rate"]
        in_ch = config["channels"]
        base = config["base_channels"]
        ratios = config["channel_ratios"]
        strides = config["strides"]
        assert len(ratios) == len(strides), "channel_ratios and strides must align"
        ch = [base * r for r in ratios]
        k = config["kernel_size"]
        rk = config["residual_kernel_size"]
        dil = config["dilations"]
        norm = config["norm"]
        act = config["activation"]
        self.latent_dim = config["latent_dim"]

        self.hop_length = 1
        for s in strides:
            self.hop_length *= s
        self.frame_rate = self.sample_rate / self.hop_length

        # ----------------------------------------------------------- encoder
        enc = [CausalConv1d(in_ch, ch[0], config["init_kernel"], norm=norm)]
        for i, s in enumerate(strides):
            out_c = ch[i + 1] if i + 1 < len(ch) else ch[i]
            enc.append(EncoderBlock(ch[i], out_c, s, k, rk, dil, norm, act))
        enc.append(CausalConv1d(ch[-1], self.latent_dim, config["final_kernel"],
                                norm=norm))
        self.encoder = nn.Sequential(*enc)

        # ------------------------------------------------- split quantizers
        sem = config["semantic"]
        self.semantic_proj = nn.Linear(self.latent_dim, sem["dim"])
        self.semantic_out = nn.Linear(sem["dim"], self.latent_dim)
        from .quantization import VectorQuantizer, ResidualVectorQuantizer
        self.semantic_quantizer = VectorQuantizer(
            sem["codebook_size"], sem["dim"], sem["commitment_cost"],
            ema=config["ema_codebook"],
            dead_code_revival=config["dead_code_revival"],
            usage_threshold=config["usage_threshold"])
        aco = config["acoustic"]
        self.acoustic_rvq = ResidualVectorQuantizer(
            aco["n_quantizers"], aco["codebook_size"], self.latent_dim,
            aco["commitment_cost"], ema=config["ema_codebook"],
            dropout_p=aco["rvq_dropout_p"],
            dead_code_revival=config["dead_code_revival"],
            usage_threshold=config["usage_threshold"])
        self.n_codebooks = 1 + aco["n_quantizers"]
        self.codebook_size = sem["codebook_size"]
        assert aco["codebook_size"] == sem["codebook_size"]

        # ----------------------------------------------------------- decoder
        dec = [CausalConv1d(self.latent_dim, ch[-1], config["final_kernel"],
                            norm=norm)]
        rch = list(reversed(ch))
        rstr = list(reversed(strides))
        for i, s in enumerate(rstr):
            out_c = rch[i + 1] if i + 1 < len(rch) else rch[i]
            dec.append(DecoderBlock(rch[i], out_c, s, k, rk, dil, norm, act))
        dec.append(CausalConv1d(rch[-1], in_ch, config["final_kernel"],
                                norm=norm))
        self.decoder = nn.Sequential(*dec)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str) -> "KimiMusicCodec":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------ #
    def _pad(self, x):
        """Pad waveform so T is a multiple of hop_length. Returns (x, T_orig)."""
        T = x.shape[-1]
        T_pad = math.ceil(T / self.hop_length) * self.hop_length - T
        if T_pad > 0:
            x = torch.nn.functional.pad(x, (0, T_pad))
        return x, T

    # ------------------------------------------------------------------ #
    def encode_latents(self, wav):
        z = self.encoder(wav)                   # [B, latent, T_f]
        return z.transpose(1, 2)                # [B, T_f, latent]

    # ------------------------------------------------------------------ #
    def quantize(self, z, n_quantizers=None):
        """z: [B,T_f,latent] -> (z_q, tokens [B,T_f,Q], vq losses).

        Runs on the fp32 path (see :func:`_fp32`): the residual chain must not
        be rounded to fp16 between codebooks.
        """
        in_dtype = z.dtype
        with _fp32():
            # Pure-fp16 layout: the projections are stored in fp16, so the
            # residual chain runs in the *weight* dtype (previously fp32
            # master weights made this a real fp32 path).  The part that must
            # not be rounded is the codebook distance / EMA update, and that
            # is upcast to fp32 inside the quantizer itself
            # (see quantization.py), so fp16 storage changes no VQ decision.
            z = z.to(self.semantic_proj.weight.dtype)
            z_s = self.semantic_proj(z)
            z_s_q, tok_sem, loss_sem = self.semantic_quantizer(z_s)
            sem_contrib = self.semantic_out(z_s_q)
            residual = z - sem_contrib
            z_a_q, tok_aco, loss_aco = self.acoustic_rvq(residual, n_quantizers)
            z_q = sem_contrib + z_a_q
            tokens = torch.cat([tok_sem.unsqueeze(-1), tok_aco], dim=-1)
            losses = {
                "commitment": loss_sem["commitment"] + loss_aco["commitment"],
                "codebook": loss_sem["codebook"] + loss_aco["codebook"],
            }
        return z_q.to(in_dtype), tokens, losses

    # ------------------------------------------------------------------ #
    def dequantize(self, tokens):
        """tokens: [B,T_f,Q] -> [B,T_f,latent] (fp32 path)."""
        with _fp32():
            z_s_q = self.semantic_quantizer.decode(tokens[..., 0])
            return self.semantic_out(z_s_q) + self.acoustic_rvq.decode(tokens[..., 1:])

    # ------------------------------------------------------------------ #
    def decode_latents(self, z):
        return self.decoder(z.transpose(1, 2))  # [B, 1, T]

    # ------------------------------------------------------------------ #
    def forward(self, wav, n_quantizers=None):
        """Training pass. wav: [B, 1, T] -> dict(recon, tokens, losses)."""
        wav, T_orig = self._pad(wav)
        z = self.encode_latents(wav)
        z_q, tokens, vq_losses = self.quantize(z, n_quantizers)
        recon = self.decode_latents(z_q)[..., :T_orig]
        return {"recon": recon, "tokens": tokens, "vq_losses": vq_losses,
                "latents": z}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, wav, n_quantizers=None):
        """wav: [B, 1, T] -> tokens [B, T_f, Q] (int64)."""
        self.eval()
        wav, _ = self._pad(wav)
        z = self.encode_latents(wav)
        _, tokens, _ = self.quantize(z, n_quantizers)
        return tokens

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, tokens, out_length=None):
        """tokens: [B, T_f, Q] -> waveform [B, 1, T_f * hop_length]."""
        self.eval()
        z_q = self.dequantize(tokens)
        wav = self.decode_latents(z_q)
        if out_length is not None:
            wav = wav[..., :out_length]
        return wav

    # ------------------------------------------------------------------ #
    def quantizer_usage(self):
        return {"semantic": self.semantic_quantizer.usage(),
                "acoustic": self.acoustic_rvq.usage()}
