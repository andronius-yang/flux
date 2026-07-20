#!/bin/bash
# Build the FAST submodule's libflash.so (Torch extension) for Perlmutter A100.
# Requires `source ./module.sh` first (NVSHMEM_HOME, conda torch, CUDA 12.4).
#
#   ./scripts/build_fast.sh            # configure + build
#   FAST_VERIFY_BUFFER=ON ./scripts/build_fast.sh   # correctness-check build
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FAST_NVIDIA_DIR="$SCRIPT_DIR/../3rdparty/FAST/nvidia"
BUILD_DIR="${FAST_BUILD_DIR:-$FAST_NVIDIA_DIR/build}"
JOBS="${JOBS:-8}"  # login nodes OOM at 16

if [ -z "${NVSHMEM_HOME:-}" ]; then
    echo "ERROR: NVSHMEM_HOME is unset; run 'source ./module.sh' first" >&2
    exit 1
fi

TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, "Torch"))')

cmake -S "$FAST_NVIDIA_DIR" -B "$BUILD_DIR" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNVSHMEM_ROOT="$NVSHMEM_HOME" \
    -DTorch_DIR="$TORCH_DIR" \
    -DFAST_BIND_INTERNAL_NVSHMEM=ON \
    -DFAST_VERIFY_BUFFER="${FAST_VERIFY_BUFFER:-OFF}"

cmake --build "$BUILD_DIR" -j "$JOBS"
echo "built: $FAST_NVIDIA_DIR/libflash.so"
