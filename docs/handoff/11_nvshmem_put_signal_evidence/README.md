# NVSHMEM on-stream put-with-signal ordering — source evidence (collected 2026-08-22)

Deployed on Perlmutter: `nvshmem/3.2.5-1-25.03` (git ea47bbe5…, NOT in the public repo;
public history starts at v3.4.5-0). Closest public source = v3.4.5-0. Files cloned
verbatim into this directory (`libfabric_<tag>.cpp`, `v345_*`, `<tag>_*`).

## 1. The nbi on-stream put_signal kernel has NO fence between data and signal

`src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh` (identical in v3.4.5-0,
v3.5.19-1, v3.7.2-0; lines 75-96 in v3.4.5):

```cpp
__global__ void nvshmemi_proxy_rma_signal_entrypoint(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
    nvshmemi_transfer_amo_nonfetch((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint_blocking(void *rptr, void *lptr,
                                                              rma_bytesdesc_t bytesdesc,
                                                              uint64_t *sig_addr, uint64_t signal,
                                                              int sig_op, int pe,
                                                              const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
        (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes),
        (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe, false);
    nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(true);
#endif
}
```
`nvshmemx_putmem_signal_nbi_on_stream` → the first kernel (data rma_nbi, then signal amo,
nothing in between). `nvshmemx_putmem_signal_on_stream` (blocking) → the second, which
calls `nvshmemi_transfer_put_signal`:

`src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in` v3.4.5 lines 142-166:
```cpp
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, NVSHMEMI_OP_PUT);
            nvshmemi_proxy_fence();
            nvshmemi_proxy_amo_nonfetch<uint64_t>(sig_addr, signal, pe, sig_op);
            if (is_nbi == 0) {
                nvshmemi_proxy_quiet(false);
```
i.e. ONLY the blocking path inserts a proxy fence between the data put and the signal.
This is exactly the nbi-fails / blocking-passes split we measured on Perlmutter
(FLUX_A2AV_BLOCKING_WIRE).

Both the deployed 3.2.5 host library and the 3.7.2 wheel export both kernels
(`nm -C libnvshmem_host.so.3.2.5 | grep rma_signal_entrypoint`).

## 2. The proxy executes data and signal as two independent transport ops

`src/host/proxy/proxy.cpp` v3.4.5: `process_channel_dma` → `nvshmemi_process_multisend_rma`
(→ transport `rma` = `fi_write`), `process_channel_amo` → transport `amo`
(→ `fi_atomicmsg(..., FI_INJECT)` on Slingshot). A fence is only executed if the device
enqueued a fence request:
```cpp
inline int process_channel_fence(proxy_state_t *proxy_state, proxy_channel_t *ch) {
    ...
        if (tcurr->host_ops.fence) status = tcurr->host_ops.fence(tcurr, i, 1);
```

## 3. In the libfabric transport of that era, "fence" was a quiet on TRANSMIT completion

`libfabric_v3.4.5-0.cpp`:
```cpp
    transport->host_ops.fence = nvshmemt_libfabric_quiet;     // line 1676
    transport->host_ops.quiet = nvshmemt_libfabric_quiet;
...
    if ((state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) ||
        (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA)) {
        state->prov_info->tx_attr->op_flags = FI_TRANSMIT_COMPLETE;   // lines 1212-1215
    }
...
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        /* TODO: Use FI_FENCE to optimize put_with_signal */          // line 1542
        info.caps |= FI_FENCE | FI_ATOMIC;
```
and `nvshmemt_libfabric_quiet` polls `fi_cntr_read(ep->counter) == ep->submitted_ops`,
i.e. counts TRANSMIT completions. No `msg_order` is ever requested (grep msg_order = 0
in every tag).

## 4. Upstream acknowledged the quiet semantics were insufficient (v3.5.19+)

`libfabric_v3.5.19-1.cpp` line 2057-2058 (still present in v3.7.2):
```cpp
    /* Require completion RMA completion at target for correctness of quiet */
    info.tx_attr->op_flags = FI_DELIVERY_COMPLETE;
```
plus a dedicated `nvshmemt_libfabric_fence` (v3.5.19 line 676; v3.7.2 line 1680), a
transport-level `nvshmemt_put_signal` (transport_common.cpp: rma → fence → amo) and, for
EFA only, `nvshmemt_put_signal_unordered` (sequence-counted writes; the target proxy
applies the signal only after `num_writes` data completions — `put_signal_completion`).
Slingshot still selects the plain `nvshmemt_put_signal`:
```cpp
    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        transport->host_ops.put_signal = nvshmemt_put_signal_unordered;
    } else {
        transport->host_ops.put_signal = nvshmemt_put_signal;
    }
```

## 5. Documentation
NVSHMEM release notes 3.2.5 / 3.4.5 / 3.5.19 / 3.6.5 / 3.7.2: no explicit
"put_signal may deliver the signal before the data" entry. Related wording only:
3.5.19 "Fixed race condition in barrier causing hangs on unordered networks";
changelog 3.3.9 "Fixed a data corruption bug in on-stream NVLS ... due to missing memory
fence to order data and barrier"; the standing limitation "nvshmem_barrier*, quiet,
wait_until only ensure ordering and visibility between the source and destination PEs".

## 6. Availability on Perlmutter
Modules: only `nvshmem/2.11.0` and `nvshmem/3.2.5-1` (`module spider nvshmem`).
pip: `nvidia-nvshmem-cu12` 3.1.7 … 3.7.2 available; the 3.7.2 wheel ships
`nvshmem_transport_libfabric.so.6` (NEEDED libfabric.so.1 → Cray libfabric 1.22 at
runtime) and the flux build already supports a pip NVSHMEM (setup.py NVSHMEM_HOME
precedence). NOTE: the nbi on-stream kernel is unchanged in 3.7.2 (section 1), so an
upgrade changes the transport's completion semantics (DELIVERY_COMPLETE + real fence)
but NOT the missing fence in the nbi kernel — must be re-verified with the probe, not
assumed fixed.

## 7. Standalone NVSHMEM-only reproducer (2026-08-22, job 57407629, 2 nodes × 1 GPU, CXI)
Program: `$PSCRATCH/nvshmem_repro/repro.cu` (115 lines, no flux): PE0 stamps a symmetric
src buffer with `it`, put_signal (nbi or blocking) into PE1 + SET flag=it; PE1 waits
flag>=it (raw cuStreamWaitValue64 GEQ = flux's gate, or nvshmemx_signal_wait_until_on_stream),
counts elements != it; every iteration ends with quiet_on_stream + barrier_all_on_stream
(no cross-iteration overlap -> any mismatch is intra-put ordering). 2000 iterations each.
Build/run recipe in the session handoff (nvcc + direct Cray MPI link; `srun --mpi=cray_shasta`).

| bytes   | put variant + wait            | stale iterations | stale elements       | stale stamp |
|---------|-------------------------------|------------------|----------------------|-------------|
| 8 MiB   | nbi + memop / nbi + nvwait    | 2000/2000 both   | ~37 % of each buffer | it-1        |
| 2 MiB   | nbi + memop / nvwait          | 1999 / 2000      | ~42 %                | it-1        |
| 1 MiB   | nbi + memop                   | 1999/2000        | ~57 %                | it-1        |
| 256 KiB | nbi + memop                   | 1071/2000        | ~30 % of stale iters | it-1        |
| 64 KiB  | nbi + memop / nvwait          | 0 / 5 of 2000    | —                    | it-1        |
| any     | BLOCKING put_signal (either wait) | 0/2000 everywhere (14,000 clean iterations) | 0 | — |

Conclusion: on nvshmem/3.2.5-1 + libfabric/CXI, `nvshmemx_putmem_signal_nbi_on_stream`
makes the flag visible before the RDMA data for MB-class puts in essentially every
iteration; NVSHMEM's own consistency-enforcing wait does not help (transport-side ordering
gap, not receiver-side visibility); the blocking variant is clean. One-to-one match with the
flux-level ladders (nbi fails, F4 fails, BLOCKING_WIRE passes).
