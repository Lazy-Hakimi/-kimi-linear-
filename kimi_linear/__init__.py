"""Kimi Linear: reference PyTorch implementation.

Based on "Kimi Linear: An Expressive, Efficient Attention Architecture"
(arXiv:2510.26692). All hyper-parameters come from a single JSON
configuration document (see ``configs/``).
"""

from .configuration import KimiLinearConfig, ConfigError, default_config_path
from .kda import KimiDeltaAttention, kda_chunkwise, kda_recurrent, ShortConv, RMSNorm, l2norm
from .mla import MLA
from .ffn import DenseFFN, MoEFFN, SwiGLUExpert, build_ffn
from .modeling import KimiLinearBlock, KimiLinearModel, KimiLinearForCausalLM, KimiLinearCache
from .graph_decode import GraphedDecoder
from . import precision
from . import fp16
from .fp16 import (
    KahanAdamW8bit,
    KahanAdafactor,
    cast_for_fp16_training,
    assert_fp16_layout,
    memory_report,
    chunked_linear_ce,
    FP16_MAX_SAFE,
)
from .precision import (
    PRECISION_CHOICES,
    device_capability,
    supports_bfloat16,
    supports_tf32,
    resolve_precision,
    precision_name,
    configure_cuda_backends,
    make_grad_scaler,
    describe_device,
)

__all__ = [
    "KimiLinearConfig",
    "ConfigError",
    "default_config_path",
    "KimiDeltaAttention",
    "kda_chunkwise",
    "kda_recurrent",
    "ShortConv",
    "RMSNorm",
    "l2norm",
    "MLA",
    "DenseFFN",
    "MoEFFN",
    "SwiGLUExpert",
    "build_ffn",
    "KimiLinearBlock",
    "KimiLinearModel",
    "KimiLinearForCausalLM",
    "KimiLinearCache",
    "GraphedDecoder",
    "precision",
    "fp16",
    "KahanAdamW8bit",
    "KahanAdafactor",
    "cast_for_fp16_training",
    "assert_fp16_layout",
    "memory_report",
    "chunked_linear_ce",
    "FP16_MAX_SAFE",
    "PRECISION_CHOICES",
    "device_capability",
    "supports_bfloat16",
    "supports_tf32",
    "resolve_precision",
    "precision_name",
    "configure_cuda_backends",
    "make_grad_scaler",
    "describe_device",
]
