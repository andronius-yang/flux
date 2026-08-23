#!/bin/bash
# Build the FAST submodule's libflash.so (Torch extension) for A100 (sm80).
# Requires the platform env first: `source ./module.sh` (Perlmutter) or
# `source ./env_aws.sh` (AWS p4d) — NVSHMEM_HOME, torch, CUDA 12.4.
#
#   ./scripts/build_fast.sh            # configure + build
#   FAST_VERIFY_BUFFER=ON ./scripts/build_fast.sh   # correctness-check build
#
# FAST_NVSHMEM_ROOT overrides the NVSHMEM used for the build; it must contain
# lib/cmake/nvshmem (the AWS pip wheel ships libs but no cmake package, so
# there we fall back to the self-built /home/ubuntu/sw/nvshmem-3.3.9).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FAST_NVIDIA_DIR="$SCRIPT_DIR/../3rdparty/FAST/nvidia"
BUILD_DIR="${FAST_BUILD_DIR:-$FAST_NVIDIA_DIR/build}"
JOBS="${JOBS:-8}"  # login nodes OOM at 16

# two-instance patch (2026-08-21): FAST is a submodule, so the change is
# carried as a patch in the parent repo and applied per checkout (idempotent
# — the marker symbol is grep'd first). Needed for the combined l0+l1 FAST
# baseline (one flash_comm_t per wire direction).
if ! grep -q flash_comm_instance_count "$FAST_NVIDIA_DIR/alltoall_nvshmem.cpp"; then
    echo "applying scripts/fast_two_instance.patch to 3rdparty/FAST"
    git -C "$SCRIPT_DIR/../3rdparty/FAST" apply "$SCRIPT_DIR/fast_two_instance.patch"
fi

# blocking-wire patch (2026-08-23): nbi putmem_signal -> blocking at every
# wire site — the CXI signal-before-data fix (SCHEMA rule 6), reproduced in
# this baseline at K2 shape. Idempotent via the marker comment.
if ! grep -q flash_blocking_wire_patch \
        "$FAST_NVIDIA_DIR/src/fast_alltoall/flash_alltoall_nvshmem.cu"; then
    echo "applying scripts/fast_blocking_wire.patch to 3rdparty/FAST"
    git -C "$SCRIPT_DIR/../3rdparty/FAST" apply "$SCRIPT_DIR/fast_blocking_wire.patch"
fi

if [ -z "${NVSHMEM_HOME:-}" ]; then
    echo "ERROR: NVSHMEM_HOME is unset; run 'source ./module.sh' (Perlmutter)" \
         "or 'source ./env_aws.sh' (AWS) first" >&2
    exit 1
fi

# NVSHMEM root for cmake: explicit override > NVSHMEM_HOME (if it has the cmake
# package) > the self-built AWS install.
if [ -n "${FAST_NVSHMEM_ROOT:-}" ]; then
    :
elif [ -d "$NVSHMEM_HOME/lib/cmake/nvshmem" ]; then
    FAST_NVSHMEM_ROOT="$NVSHMEM_HOME"
elif [ -d /home/ubuntu/sw/nvshmem-3.3.9/lib/cmake/nvshmem ]; then
    FAST_NVSHMEM_ROOT=/home/ubuntu/sw/nvshmem-3.3.9
else
    echo "ERROR: no NVSHMEM with lib/cmake/nvshmem found; set FAST_NVSHMEM_ROOT" >&2
    exit 1
fi

TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, "Torch"))')
TORCH_CXX11_ABI=$(python3 -c 'import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)')

cmake -S "$FAST_NVIDIA_DIR" -B "$BUILD_DIR" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNVSHMEM_ROOT="$FAST_NVSHMEM_ROOT" \
    -DTorch_DIR="$TORCH_DIR" \
    -DFAST_GLIBCXX_USE_CXX11_ABI="$TORCH_CXX11_ABI" \
    -DFAST_BIND_INTERNAL_NVSHMEM=ON \
    -DFAST_VERIFY_BUFFER="${FAST_VERIFY_BUFFER:-OFF}"

cmake --build "$BUILD_DIR" -j "$JOBS"
echo "built: $FAST_NVIDIA_DIR/libflash.so (NVSHMEM: $FAST_NVSHMEM_ROOT, cxx11abi: $TORCH_CXX11_ABI)"
