#!/usr/bin/env python3
"""Merge a locally trained Llama 3 LoRA adapter and export Ollama-ready GGUF.

Run under Linux/WSL2 with a supported Unsloth/PyTorch/CUDA environment, sufficient
RAM/disk, and llama.cpp build prerequisites (C/C++ compiler, CMake, Git). Native
Windows export is unsupported here. Unsloth may build/download llama.cpp and
retrieve the adapter's original base model; gated models require HF login.
This script never installs CUDA libraries. Export can require more memory than
4-bit training. Transfer the generated GGUF to Windows before deploying Ollama.
"""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("models") / "fpl-lora",
                        help="Local LoRA directory containing adapter_config.json and weights")
    parser.add_argument("--output", type=Path, default=Path("models") / "gguf")
    parser.add_argument("--quantization", choices=("q4_k_m", "q5_k_m", "q8_0", "f16"), default="q4_k_m")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    try:
        if not (args.checkpoint / "adapter_config.json").is_file():
            raise ValueError("--checkpoint must contain adapter_config.json from LoRA training")
        if not any((args.checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError("--checkpoint contains no adapter weights")
        if sys.platform == "win32":
            raise RuntimeError("Run GGUF export under Linux/WSL2 with Unsloth; then deploy the GGUF using Windows Ollama.")
        from unsloth import FastLanguageModel
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This Unsloth export path requires a supported CUDA GPU.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.checkpoint.resolve()), max_seq_length=args.max_seq_length,
            dtype=None, load_in_4bit=False,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        previous = {path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
                    for path in args.output.rglob("*.gguf")}
        model.save_pretrained_gguf(
            str(args.output.resolve()), tokenizer, quantization_method=args.quantization,
        )
        generated = [
            path for path in args.output.rglob("*.gguf")
            if path.stat().st_size > 0 and previous.get(path.resolve()) != (path.stat().st_mtime_ns, path.stat().st_size)
        ]
        if not generated:
            raise RuntimeError("Unsloth returned without producing a new, nonempty GGUF")
        for path in generated:
            print(f"GGUF: {path.resolve()}")
        print("Pass the desired quantized GGUF path to 4_deploy_ollama.ps1 -GgufPath.")
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"GGUF export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
