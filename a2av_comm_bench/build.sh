#!/bin/bash
# Build the comm-only benchmark. Run from the repo root after `source ./module.sh`
# (needs nvcc from cudatoolkit and NVSHMEM_HOME from the Perlmutter module).
set -ex
cd "$(dirname "$0")"
# -rdc=true: the prefetch mode's device kernel calls nvshmemx_getmem_nbi_block
nvcc -O2 -arch=sm_80 -std=c++17 -rdc=true -c comm_bench.cu -o comm_bench.o -I$NVSHMEM_HOME/include
# libnvshmem_device.a carries device kernels for the on-stream ops: device-link it
nvcc -arch=sm_80 -dlink comm_bench.o -L$NVSHMEM_HOME/lib -lnvshmem_device -o comm_bench_dlink.o
# NOTE: g++, not the Cray CC wrapper — CC drags in libmpi_gtl_cuda (needs a
# cudart 11 that isn't installed) and a second libmpi. The bench is MPI-free.
# --allow-shlib-undefined: libnvshmem_host.so wants GLIBCXX_3.4.32, satisfied at
# runtime by LD_PRELOADing the conda env's libstdc++ (see run.sh).
g++ comm_bench.o comm_bench_dlink.o -o comm_bench \
  -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device \
  -L$CUDA_HOME/lib64 -lcudart -lcuda \
  -Wl,-rpath,$NVSHMEM_HOME/lib -Wl,-rpath,$CUDA_HOME/lib64 \
  -Wl,--allow-shlib-undefined
echo "built: $(pwd)/comm_bench"
