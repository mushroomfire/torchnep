"""
Pre-compile CUDA extension kernels for torchnep.

Run after installation to avoid JIT compilation on first training run:

    torchnep-build

Or from Python:

    from torchnep.build_ext import build_kernels
    build_kernels()
"""

import sys
import os
import argparse
import torch

from .cuda_ops import _load_kernels, _load_cached_kernels
from .ops import _load_cuda_ops


def build_kernels(verbose: bool = True) -> bool:
    """Compile all CUDA extension kernels.

    Returns True if all available kernels compiled successfully.
    Prints progress to stdout.
    """
    if not torch.cuda.is_available():
        print("CUDA not available on this system — no kernels to compile.")
        return True

    os.environ["TORCHNEP_VERBOSE_BUILD"] = "1" if verbose else "0"

    # Ensure the Python environment's bin directory is in PATH so that
    # ninja (installed via pip) is found by torch.utils.cpp_extension.
    python_bin = os.path.dirname(sys.executable)
    if python_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    success = True

    print("Building torchnep CUDA kernels...")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  Device   : {torch.cuda.get_device_name(0)}")
    print()

    kernels = [
        ("nep_kernels   (radial descriptor forward)", _load_kernels),
        ("nep_cached    (ScatterContraction + TypeContraction)", _load_cached_kernels),
        ("nep_cuda_ops  (force/virial accumulation)", _load_cuda_ops),
    ]

    for name, loader in kernels:
        print(f"  Compiling {name}...", flush=True)
        result = loader()
        if result is not None:
            print(f"    OK")
        else:
            print(f"    FAILED (will fall back to PyTorch at runtime)")
            success = False

    print()
    if success:
        print("All kernels compiled successfully.")
    else:
        print("Some kernels failed to compile. torchnep will use PyTorch fallbacks.")

    return success


def main():
    """Entry point for the torchnep-build command."""
    parser = argparse.ArgumentParser(
        description="Pre-compile torchnep CUDA extension kernels.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress nvcc build output")
    args = parser.parse_args()

    ok = build_kernels(verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
