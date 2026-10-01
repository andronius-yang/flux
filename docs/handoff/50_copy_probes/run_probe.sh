#!/bin/bash
# run_probe.sh <jobid> <busy: none|spin|stream> [extra probe args...]  -> CSV on stdout
J=$1; BUSY=$2; shift 2
BIN=${BIN:-$PSCRATCH/workspace/andrewy/logs/p50/bin/probe_ce}
srun --jobid=$J --nodes=1 ${NODEW:-} --ntasks=${NT:-4} --gpus-per-node=4 --cpu-bind=cores --overlap bash -lc "export CUDA_VISIBLE_DEVICES=${CVD:-0,1,2,3}
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1
module unload nccl nvshmem cudatoolkit 2>/dev/null
module load libfabric/1.20.1 gcc-native/12.3 cudatoolkit/12.9 nccl/2.24.3 nvshmem/3.2.5-1 >/dev/null 2>&1
export CUDA_MODULE_LOADING=EAGER MPICH_GPU_SUPPORT_ENABLED=0 NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_SYMMETRIC_SIZE=512M NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_BOOTSTRAP=MPI CUDA_DEVICE_MAX_CONNECTIONS=24
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/lib64:\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH
${NSYS:-} $BIN --busy $BUSY $*"
