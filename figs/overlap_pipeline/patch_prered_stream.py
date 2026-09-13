#!/usr/bin/env python
"""Apply the streaming pre-reduce kernel patch (FLUX_A2AV_RS_PRERED_STREAM) to a flux tree.

Usage: patch_prered_stream.py <tree root>
Idempotent: refuses to apply twice. Touches src/moe_gather_rs/a2av_combine.cu only.
Knob: FLUX_A2AV_RS_PRERED_STREAM=1 selects a2av_combine_prereduce_stream_kernel
(per-CTA contiguous wire-row range, cp.async STAGES-deep row ring in dynamic smem);
default 0 = the shipped kernel, bit-identical schedule. Binary tag
FLUX_A2AV_RS_PRERED_STREAM_TAG.
"""
import sys, os

root = sys.argv[1]
path = os.path.join(root, 'src/moe_gather_rs/a2av_combine.cu')
s = open(path).read()
if 'a2av_combine_prereduce_stream_kernel' in s:
    sys.exit('already patched: ' + path)

KERNEL = r'''
// Streaming pre-reduce (2026-09-10, FLUX_A2AV_RS_PRERED_STREAM=1): same contract
// as a2av_combine_prereduce_kernel (per-(split, tn) conv-signal spin, wire panel
// out, gridDim.x-counted wire-flag handshake) but each CTA owns a CONTIGUOUS
// wire-row range of the segment and streams that range's conv rows through a
// STAGES-deep cp.async ring in dynamic shared memory, so STAGES full rows are in
// flight per SM regardless of how many contributions a wire row has. The shipped
// kernel keeps one dependent 16 B load per thread in flight (~17 GB/s per CTA,
// figs/overlap_pipeline/00_brief.md §8); this one is bandwidth-bound per SM.
// Thread t owns packs t + i*512 of a row (i < kMaxPacksPerThread).
constexpr int kPreredStreamThreads = 512;
constexpr int kMaxPacksPerThread = 2;  // n_per <= 2 * 512 * 8 = 8192 elements (K2/Qwen: 7168 / 4096)

CUTLASS_DEVICE void
cp_async_16(void *smem_dst, void const *gmem_src) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}
CUTLASS_DEVICE void
cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
CUTLASS_DEVICE void
cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// __launch_bounds__(512, 2) caps the kernel at 64 regs/thread = 32K regs per CTA: the
// residency budget beside ONE l1 GEMM CTA (128 thr x 254 regs) on an A100 SM. The first
// port used 86-108 regs (44-55K per CTA) and could not start until GEMM CTAs retired
// (~6 ms into the GEMM, capsule 20260910-232321), head-of-line blocking its HW queue.
template <typename T, int STAGES>
__global__ void
__launch_bounds__(kPreredStreamThreads, 2) a2av_combine_prereduce_stream_kernel(
    A2AVCombinePreReduceArguments args) {
  extern __shared__ __align__(16) unsigned char prered_smem[];
  constexpr int kElemsPerPack = PackU<T>::kElemsPerPack;
  const int64_t n_per = args.n_per;
  const int packs_per_row = (int)(n_per / kElemsPerPack);
  const int64_t row_bytes = n_per * sizeof(T);
  const int NN = args.nnodes;
  const int L = args.local_world_size;
  const int t = threadIdx.x;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < args.n_split; sid++) {
    T const *conv = (T const *)args.conv_panel + (int64_t)sid * args.conv_rows * n_per;
    T *wire = (T *)args.wire_panel + (int64_t)sid * args.wire_rows * n_per;
    const int P = args.n_pieces > 0 ? args.n_pieces : 1;
    for (int p = 0; p < P; p++) {
      for (int gi = 0; gi < NN - 1; gi++) {
        const int tn = args.node_order[gi];
        const int seg = tn < args.node_idx ? tn : tn - 1;
        if (threadIdx.x == 0) {
          for (int ls = 0; ls < L; ls++) {
            uint64_t const *sig =
                args.n_pieces > 0
                    ? args.piece_conv_sigs + ((int64_t)ls * NN + tn) * 8 + p
                    : args.conv_signals + ((int64_t)ls * NN + tn) * args.n_split + sid;
            uint64_t spins = 0;
            while (load_acquire_sys_u64(sig) < args.run_id) {
              __nanosleep(200);
              if (args.spin_limit != 0 && ++spins >= args.spin_limit) {
                printf(
                    "[a2av-combine] prereduce(stream) conv-signal SPIN LIMIT: node %d "
                    "tn %d ls %d sid %d piece %d run_id %llu\n",
                    args.node_idx, tn, ls, sid, p,
                    (unsigned long long)args.run_id);
                __trap();
              }
            }
          }
        }
        __syncthreads();
        int64_t w_lo = args.wire_seg_start[seg];
        int64_t w_hi = args.wire_seg_start[seg + 1];
        if (args.n_pieces > 0) {
          w_hi = w_lo + args.piece_start[seg * 9 + p + 1];
          w_lo = w_lo + args.piece_start[seg * 9 + p];
        }
        // contiguous wire-row range of this CTA
        const int64_t nrows_seg = w_hi - w_lo;
        const int64_t per = (nrows_seg + gridDim.x - 1) / gridDim.x;
        const int64_t w0 = w_lo + (int64_t)blockIdx.x * per;
        const int64_t w1 = min(w_hi, w0 + per);
        if (w0 < w1) {
          const int32_t k0 = args.wire_ptr[w0];
          const int32_t k1 = args.wire_ptr[w1];
          const int32_t nconv = k1 - k0;
          auto issue = [&](int32_t j) {
            if (j < nconv) {
              const int64_t r = args.wire_copy[k0 + j];
              unsigned char *dst = prered_smem + (size_t)(j % STAGES) * row_bytes;
              unsigned char const *src = (unsigned char const *)(conv + r * n_per);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < kMaxPacksPerThread; i++) {
                const int pk = t + i * kPreredStreamThreads;
                if (pk < packs_per_row) {
                  cp_async_16(dst + (size_t)pk * 16, src + (size_t)pk * 16);
                }
              }
            }
            cp_async_commit();  // always commit: uniform group accounting
          };
          float acc[kMaxPacksPerThread][kElemsPerPack];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kMaxPacksPerThread; i++)
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kElemsPerPack; e++) acc[i][e] = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < STAGES - 1; j++) issue(j);
          int64_t w = w0;
          int32_t kend = args.wire_ptr[w + 1];
          // rows with zero contributions (not produced by the plan, but keep the
          // shipped kernel's semantics: they are written as zeros)
          auto flush_empty = [&]() {
            while (w < w1 && kend == args.wire_ptr[w]) {
              T *dst = wire + w * n_per;
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < kMaxPacksPerThread; i++) {
                const int pk = t + i * kPreredStreamThreads;
                if (pk < packs_per_row) {
                  PackU<T> z;
                  CUTLASS_PRAGMA_UNROLL
                  for (int e = 0; e < kElemsPerPack; e++) z.elems[e] = float_to_elem<T>(0.0f);
                  storePack(dst + (int64_t)pk * kElemsPerPack, z.data);
                }
              }
              w++;
              if (w < w1) kend = args.wire_ptr[w + 1];
            }
          };
          flush_empty();
          for (int32_t j = 0; j < nconv; j++) {
            issue(j + STAGES - 1);
            cp_async_wait<STAGES - 1>();
            __syncthreads();
            unsigned char const *st = prered_smem + (size_t)(j % STAGES) * row_bytes;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kMaxPacksPerThread; i++) {
              const int pk = t + i * kPreredStreamThreads;
              if (pk < packs_per_row) {
                PackU<T> pkv;
                pkv.data = *reinterpret_cast<uint4 const *>(st + (size_t)pk * 16);
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < kElemsPerPack; e++) acc[i][e] += elem_to_float<T>(pkv.elems[e]);
              }
            }
            if (k0 + j + 1 == kend) {  // wire row w complete
              T *dst = wire + w * n_per;
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < kMaxPacksPerThread; i++) {
                const int pk = t + i * kPreredStreamThreads;
                if (pk < packs_per_row) {
                  PackU<T> out;
                  CUTLASS_PRAGMA_UNROLL
                  for (int e = 0; e < kElemsPerPack; e++) {
                    out.elems[e] = float_to_elem<T>(acc[i][e]);
                    acc[i][e] = 0.0f;
                  }
                  storePack(dst + (int64_t)pk * kElemsPerPack, out.data);
                }
              }
              w++;
              if (w < w1) {
                kend = args.wire_ptr[w + 1];
                flush_empty();
              }
            }
            __syncthreads();  // stage j % STAGES is refilled by the next issue()
          }
          cp_async_wait<0>();
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          if (args.n_pieces > 0) {
            int done = atomicAdd(args.piece_wire_counters + tn * 8 + p, 1) + 1;
            if (done == gridDim.x) {
              atomic_store_release_sys(args.piece_wire_flags + tn * 8 + p, 1);
            }
          } else {
            int done = atomicAdd(args.wire_counters + tn * args.n_split + sid, 1) + 1;
            if (done == gridDim.x) {
              atomic_store_release_sys(args.wire_flags + tn * args.n_split + sid, 1);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

// stage count for the streaming pre-reduce: FLUX_A2AV_RS_PRERED_STAGES (default 2 =
// 28 KB at n_per=7168 bf16, fits beside one 64 KB GEMM CTA under any smem carveout;
// 4 = 56 KB needs the SM's smem config >= 132 KB; 6 = 84 KB only on GEMM-free SMs),
// capped by a 96 KB budget; 0 = row too wide -> shipped kernel
CUTLASS_HOST_DEVICE int
prered_stream_stages(int64_t row_bytes, int want) {
  constexpr int64_t kBudget = 96 * 1024;
  int st = want >= 6 ? 6 : want >= 4 ? 4 : 2;
  while (st > 0 && st * row_bytes > kBudget) st -= 2;
  return st;
}
'''

# 0. get_int_from_env lives in flux/utils.h
inc = '#include "flux/flux.h"\n'
assert inc in s
if '#include "flux/utils.h"' not in s:
    s = s.replace(inc, inc + '#include "flux/utils.h"\n', 1)

# 1. kernel after the shipped pre-reduce kernel (before the CSR reduce kernel comment)
anchor = '// Legacy-gate destination reduce under compress: CSR contributions per token'
assert anchor in s
s = s.replace(anchor, KERNEL + '\n' + anchor, 1)

# 2. preload: load + opt-in the dynamic smem attribute for every instantiation
pre_anchor = '        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_prereduce_kernel<T>));\n'
assert pre_anchor in s
pre_add = pre_anchor + r'''        // streaming pre-reduce twin (FLUX_A2AV_RS_PRERED_STREAM): same resident class;
        // the >48 KB dynamic-smem opt-in must precede the first launch
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_prereduce_stream_kernel<T, 2>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_prereduce_stream_kernel<T, 4>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, a2av_combine_prereduce_stream_kernel<T, 6>));
        CUDA_CHECK(cudaFuncSetAttribute(
            a2av_combine_prereduce_stream_kernel<T, 6>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(
            a2av_combine_prereduce_stream_kernel<T, 4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CUDA_CHECK(cudaFuncSetAttribute(
            a2av_combine_prereduce_stream_kernel<T, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
'''
s = s.replace(pre_anchor, pre_add, 1)

# 3. launcher: knob dispatch
launch_old = '''        using T = decltype(to_cuda_dtype(cdtype));
        a2av_combine_prereduce_kernel<T><<<grid, block, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av pre-reduce: " << dtype; });'''
assert launch_old in s
launch_new = '''        using T = decltype(to_cuda_dtype(cdtype));
        // FLUX_A2AV_RS_PRERED_STREAM=1 (2026-09-10): cp.async streaming twin, same
        // grid / handshake; 0 (default) = shipped kernel, bit-identical schedule
        (void)bytedance::flux::get_int_from_env("FLUX_A2AV_RS_PRERED_STREAM_TAG", 0);
        static const int stream_mode =
            bytedance::flux::get_int_from_env("FLUX_A2AV_RS_PRERED_STREAM", 0);
        static const int want_stages =
            bytedance::flux::get_int_from_env("FLUX_A2AV_RS_PRERED_STAGES", 2);
        const int stages =
            stream_mode ? prered_stream_stages(args.n_per * sizeof(T), want_stages) : 0;
        const size_t smem = (size_t)stages * args.n_per * sizeof(T);
        if (stages == 6) {
          a2av_combine_prereduce_stream_kernel<T, 6>
              <<<grid, dim3(kPreredStreamThreads), smem, stream>>>(args);
        } else if (stages == 4) {
          a2av_combine_prereduce_stream_kernel<T, 4>
              <<<grid, dim3(kPreredStreamThreads), smem, stream>>>(args);
        } else if (stages == 2) {
          a2av_combine_prereduce_stream_kernel<T, 2>
              <<<grid, dim3(kPreredStreamThreads), smem, stream>>>(args);
        } else {
          a2av_combine_prereduce_kernel<T><<<grid, block, 0, stream>>>(args);
        }
      },
      [&]() { FLUX_CHECK(false) << "unsupported dtype for a2av pre-reduce: " << dtype; });'''
s = s.replace(launch_old, launch_new, 1)

# 4. n_per bound for the streaming kernel's per-thread pack count
chk_anchor = '  FLUX_CHECK_GT(args.nnodes, 1) << "compress pre-reduce is a multi-node stage";\n'
assert chk_anchor in s
s = s.replace(chk_anchor, chk_anchor + '  FLUX_CHECK_LE(args.n_per, kPreredStreamThreads * kMaxPacksPerThread * 8)\n      << "streaming pre-reduce: n_per exceeds 4 packs per thread";\n', 1)

open(path, 'w').write(s)
print('patched', path)
