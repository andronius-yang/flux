#!/bin/bash
# libflux_cuda.so maybe installed under /usr/local/lib or ~/.local/lib/ by pip3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# add flux python package to PYTHONPATH
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY # EAGER if launch the consumer kernel before the producer kernel on host

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

nproc_per_node=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
nnodes=${SLURM_NNODES:-1}
node_rank=${SLURM_NODEID:-0}
master_port=${MASTER_PORT:-23456}
if [ "${nnodes}" -gt 1 ]; then
    # Multi-node under Slurm (e.g. NERSC Perlmutter): launch one copy per node with
    #   srun --nodes=N --ntasks-per-node=1 ./launch.sh <script> <args...>
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
    # Slingshot-11: NVSHMEM inter-node transport goes over libfabric/CXI
    export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-libfabric}
    export NVSHMEM_LIBFABRIC_PROVIDER=${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}
    # symmetric heap must hold the AG input buffer + gather-RS staging; enlarge as needed
    # export NVSHMEM_SYMMETRIC_SIZE=4G
else
    master_addr="127.0.0.1"
    # legacy single-node / InfiniBand-only knobs (meaningless on Slingshot)
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
    export NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:=3}
fi
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"


CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} ${additional_args} $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
