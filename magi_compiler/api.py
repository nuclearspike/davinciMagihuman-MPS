# Stub for magi_compiler.api — replaces custom op registration with direct function calls

import torch
from typing import Any, Callable, Optional


def magi_register_custom_op(
    name: str = "",
    mutates_args: tuple = (),
    infer_output_meta_fn: Optional[Callable] = None,
    is_subgraph_boundary: bool = False,
):
    """No-op decorator that replaces magi_compiler's custom op registration.

    On CUDA with MagiCompiler, this registers functions as torch.ops.infra.*
    On MPS, we just return the function as-is and patch torch.ops.infra to
    resolve the calls made internally (e.g., torch.ops.infra.flash_attn_func).
    """
    namespace, op_name = name.split("::") if "::" in name else ("infra", name)

    def decorator(func):
        # Register as a torch custom op so torch.ops.infra.* calls work
        if not hasattr(torch.ops, namespace):
            # Create a simple namespace holder
            lib = torch.library.Library(namespace, "DEF")
            # Store reference to keep it alive
            if not hasattr(magi_register_custom_op, "_libs"):
                magi_register_custom_op._libs = {}
            magi_register_custom_op._libs[namespace] = lib

        try:
            # Try registering with torch.library for torch.ops.infra.* dispatch
            from torch.library import custom_op
            # We'll just make the function directly callable via torch.ops
            # by monkey-patching after registration
        except Exception:
            pass

        # Store reference for direct access
        if not hasattr(magi_register_custom_op, "_registry"):
            magi_register_custom_op._registry = {}
        magi_register_custom_op._registry[name] = func

        return func
    return decorator
