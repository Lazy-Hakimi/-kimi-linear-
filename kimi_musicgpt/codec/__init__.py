"""Kimi-Music-Codec: split-RVQ neural audio codec package."""

from .modules import (CausalConv1d, CausalConvTranspose1d, ResidualUnit,
                      EncoderBlock, DecoderBlock)
from .quantization import VectorQuantizer, ResidualVectorQuantizer
from .codec import KimiMusicCodec
from .discriminator import (CodecDiscriminator, MultiPeriodDiscriminator,
                            MultiScaleSTFTDiscriminator)
from .losses import (MultiScaleSTFTLoss, MelSpectrogramLoss, mel_filterbank,
                     discriminator_loss, generator_loss,
                     feature_matching_loss, waveform_l1)

__all__ = [
    "CausalConv1d", "CausalConvTranspose1d", "ResidualUnit",
    "EncoderBlock", "DecoderBlock",
    "VectorQuantizer", "ResidualVectorQuantizer",
    "KimiMusicCodec",
    "CodecDiscriminator", "MultiPeriodDiscriminator",
    "MultiScaleSTFTDiscriminator",
    "MultiScaleSTFTLoss", "MelSpectrogramLoss", "mel_filterbank",
    "discriminator_loss", "generator_loss", "feature_matching_loss",
    "waveform_l1",
]
