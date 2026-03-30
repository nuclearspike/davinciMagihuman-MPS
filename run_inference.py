#!/usr/bin/env python3
"""
MPS-compatible inference runner for daVinci-MagiHuman.

Usage:
    # Quick 256p test (distilled, 8 steps)
    python run_inference.py \
        --prompt "Your prompt here" \
        --image_path example/assets/image.png \
        --seconds 4

    # 1080p with super-resolution
    python run_inference.py \
        --prompt "Your prompt here" \
        --image_path example/assets/image.png \
        --seconds 4 \
        --resolution 1080p

    # Custom resolution
    python run_inference.py \
        --prompt "Your prompt here" \
        --image_path example/assets/image.png \
        --seconds 10 \
        --br_width 480 --br_height 272 \
        --sr_width 1920 --sr_height 1080
"""

import argparse
import os
import sys
import time

# MPS compatibility environment variables — set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Force gloo backend for distributed (MPS doesn't support NCCL)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONPATH"] = PROJECT_ROOT + ":" + os.environ.get("PYTHONPATH", "")


def main():
    parser = argparse.ArgumentParser(description="daVinci-MagiHuman MPS Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the video")
    parser.add_argument("--image_path", type=str, required=True, help="Reference image path")
    parser.add_argument("--audio_path", type=str, default=None, help="Optional audio path for lipsync mode")
    parser.add_argument("--seconds", type=int, default=4, help="Video duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default=None, help="Output path prefix")
    parser.add_argument("--resolution", type=str, default="256p",
                        choices=["256p", "540p", "1080p"],
                        help="Output resolution (256p=fast, 1080p=SR pipeline)")

    # Advanced overrides
    parser.add_argument("--br_width", type=int, default=None)
    parser.add_argument("--br_height", type=int, default=None)
    parser.add_argument("--sr_width", type=int, default=None)
    parser.add_argument("--sr_height", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Override config path")

    args = parser.parse_args()

    # Determine config and resolution based on --resolution
    resolution_presets = {
        "256p": {"br_width": 448, "br_height": 256, "config": "example/mps_distill/config.json"},
        "540p": {"br_width": 448, "br_height": 256, "sr_width": 960, "sr_height": 544,
                 "config": "example/mps_1080p/config.json"},
        "1080p": {"br_width": 448, "br_height": 256, "sr_width": 1920, "sr_height": 1080,
                  "config": "example/mps_1080p/config.json"},
    }
    preset = resolution_presets[args.resolution]
    config_path = args.config or os.path.join(PROJECT_ROOT, preset["config"])
    br_width = args.br_width or preset["br_width"]
    br_height = args.br_height or preset["br_height"]
    sr_width = args.sr_width or preset.get("sr_width")
    sr_height = args.sr_height or preset.get("sr_height")

    output_path = args.output_path or f"output_{args.resolution}_{args.seconds}s_{int(time.time())}"

    print(f"{'='*60}")
    print(f"daVinci-MagiHuman — MPS Inference")
    print(f"{'='*60}")
    print(f"Resolution: {args.resolution} (base: {br_width}x{br_height})", end="")
    if sr_width:
        print(f" -> SR: {sr_width}x{sr_height}")
    else:
        print()
    print(f"Duration:   {args.seconds}s")
    print(f"Config:     {config_path}")
    print(f"Output:     {output_path}")
    print(f"{'='*60}")

    import torch
    device_name = "MPS" if torch.backends.mps.is_available() else ("CUDA" if torch.cuda.is_available() else "CPU")
    print(f"Device:     {device_name}")
    print(f"PyTorch:    {torch.__version__}")
    if torch.backends.mps.is_available():
        print(f"Memory:     {os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1024**3:.0f} GB unified")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Build sys.argv for the config parser (pydantic-settings reads from CLI)
    sys.argv = [
        "run_inference.py",
        "--config-load-path", config_path,
        "--engine_config.distributed_backend", "gloo",
    ]

    # Initialize infrastructure
    from inference.infra import initialize_infra
    initialize_infra()

    # Build pipeline
    from inference.common import parse_config
    from inference.model.dit import get_dit
    from inference.pipeline import MagiPipeline

    config = parse_config()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)

    # Build kwargs
    kwargs = {
        "seed": args.seed,
        "seconds": args.seconds,
        "br_width": br_width,
        "br_height": br_height,
    }
    if sr_width:
        kwargs["sr_width"] = sr_width
        kwargs["sr_height"] = sr_height

    # Run inference
    print(f"\nStarting inference...")
    save_path = pipeline.run_offline(
        prompt=args.prompt,
        image=args.image_path,
        audio=args.audio_path,
        save_path_prefix=output_path,
        **kwargs,
    )

    elapsed = time.time() - start_time
    print(f"\nDone! Took {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
