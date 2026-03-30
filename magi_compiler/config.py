# Stub for magi_compiler.config

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OffloadConfig:
    model_cpu_offload: bool = False
    gpu_resident_weight_ratio: float = 1.0
    offload_policy: str = "NONE"


@dataclass
class CompileConfig:
    offload_config: OffloadConfig = field(default_factory=OffloadConfig)
