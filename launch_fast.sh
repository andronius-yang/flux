#!/bin/bash
# Launcher for the FAST-based un-overlapped baselines (test_moe_ag_fast_baseline.py).
#
# Mirrors launch.sh's per-node torchrun convention:
#   salloc -A m4243_g -q interactive -C gpu -N 2..4 --gpus-per-node=4 -t 30
#   srun --nodes=N --ntasks-per-node=1 ./launch_fast.sh test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py --traffic_matrix ...
#
# but exports the FAST-validated NVSHMEM environment (libfabric/CXI via the NERSC
# NVSHMEM module, host lib preloaded ahead of any torch-bundled NVSHMEM) instead
# of flux's launch.sh knobs — in this baseline FAST performs the only NVSHMEM
# initialization in the process (the test never calls flux.init_flux_shm).
#
# Requires `source ./module.sh` first (NVSHMEM_HOME, conda env, CUDA).

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib:~/.local/lib/

if [ -z "${NVSHMEM_HOME:-}" ]; then
    echo "ERROR: NVSHMEM_HOME is unset; run 'source ./module.sh' first" >&2
    exit 1
fi

# FAST-validated Perlmutter NVSHMEM environment (see
# 3rdparty/FAST/FAST_NVIDIA_PERLMUTTER_COMPATIBILITY.md)
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-libfabric}
export NVSHMEM_LIBFABRIC_PROVIDER=${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
export NVSHMEM_DISABLE_CUDA_VMM=${NVSHMEM_DISABLE_CUDA_VMM:-1}
export NVSHMEM_DISABLE_GDRCOPY=${NVSHMEM_DISABLE_GDRCOPY:-1}
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-cray_shasta}
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
# three capacity-sized nvshmem buffers (send + 2x pingpong) live on the
# symmetric heap; the ~1G default is too small for 64mib-budget matrices
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4G}
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$NVSHMEM_HOME/lib64:$LD_LIBRARY_PATH"
# NERSC NVSHMEM host runtime must win over PyTorch-bundled copies
nvshmem_host_lib="$NVSHMEM_HOME/lib/libnvshmem_host.so.3"
case ":${LD_PRELOAD:-}:" in
    *":$nvshmem_host_lib:"*) ;;
    *) export LD_PRELOAD="$nvshmem_host_lib${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac

# flux-side conventions (comm-free GEMM only in this baseline)
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY
export BYTED_TORCH_BYTECCL=O0

nproc_per_node=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
nnodes=${SLURM_NNODES:-1}
node_rank=${SLURM_NODEID:-0}
master_port=${MASTER_PORT:-23456}
if [ "${nnodes}" -gt 1 ]; then
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
else
    echo "ERROR: FAST requires at least 2 nodes (server_n > 1)" >&2
    exit 1
fi

CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  --rdzv_endpoint=${master_addr}:${master_port} $@"

echo ${CMD}
${CMD}

exit $?
