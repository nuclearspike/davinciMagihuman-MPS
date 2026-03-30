# MPS compatibility: centralized device abstraction for Apple Silicon

import torch


def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_string() -> str:
    """Return device string like 'mps', 'cuda:0', or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def current_device_index() -> int:
    """Return current device index (always 0 for MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def synchronize():
    """Synchronize the current device."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    """Clear the device memory cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_gpu_available() -> bool:
    """Return True if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def manual_seed_all(seed: int):
    """Set seed for all available GPU devices."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS uses torch.manual_seed which covers MPS device


def fork_rng_devices():
    """Return list of devices for torch.random.fork_rng."""
    if torch.cuda.is_available():
        return [torch.cuda.current_device()]
    return []  # MPS/CPU: empty list works with fork_rng


def memory_stats() -> dict:
    """Return memory stats dict, or zeros if unavailable."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
    if torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory()
            return {
                "allocated": allocated,
                "max_allocated": allocated,
                "reserved": allocated,
                "max_reserved": allocated,
            }
        except Exception:
            pass
    return {"allocated": 0, "max_allocated": 0, "reserved": 0, "max_reserved": 0}
