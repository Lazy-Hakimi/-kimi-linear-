"""Kimi-MusicGPT: music generation stack on top of the Kimi Linear backbone.

Implements Kimi_MusicGPT_Architecture_Design.md (v2.0) Phase 1:
split-RVQ neural audio codec (codec/), byte-level BPE text tokenizer,
REMI+ symbolic tokenizer, delay-pattern multi-codebook stream, structure /
condition encoders, and the KimiMusicGPT wrapper model with cached
generation (top-k / top-p / slot-constrained sampling / CFG).
"""

from .codec import KimiMusicCodec, CodecDiscriminator
from .text_tokenizer import ByteLevelBPETokenizer
from .symbolic import RemiTokenizer, Song, Bar, Note, DrumHit
from .delay import build_delay_stream, rearrange_delay_pattern
from .structure import create_structure_tokens, StructureConditionEncoder
from .embedding import MusicEmbedding
from .conditions import (TextConditionEncoder, MelodyConditionEncoder,
                         ConditionPrefixEncoder)
from .model import KimiMusicGPT, MusicGPTConfig

__all__ = [
    "KimiMusicCodec", "CodecDiscriminator",
    "ByteLevelBPETokenizer",
    "RemiTokenizer", "Song", "Bar", "Note", "DrumHit",
    "build_delay_stream", "rearrange_delay_pattern",
    "create_structure_tokens", "StructureConditionEncoder",
    "MusicEmbedding",
    "TextConditionEncoder", "MelodyConditionEncoder", "ConditionPrefixEncoder",
    "KimiMusicGPT", "MusicGPTConfig",
]
