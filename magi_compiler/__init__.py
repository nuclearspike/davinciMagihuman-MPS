# Stub for magi_compiler (CUDA-only library) — provides no-op replacements for MPS inference

from .api import magi_register_custom_op
from .config import CompileConfig


def magi_compile(cls=None, *, config_patch=None):
    """No-op decorator replacing magi_compiler's kernel fusion/compilation.

    On CUDA, this fuses operations for ~1.2x speedup.
    On MPS, we skip compilation and run standard PyTorch ops.
    """
    if cls is not None:
        # Used as @magi_compile without arguments
        return cls

    # Used as @magi_compile(config_patch=...)
    def decorator(cls):
        return cls
    return decorator
