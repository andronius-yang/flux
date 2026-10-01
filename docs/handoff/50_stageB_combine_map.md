# Stage B input: host-side plan inventory of the combine op (lopep sglang-dev at 3011c6f; 10-01)

Files: GC = `src/combine/ths_op/gemm_combine.cc`, CK = `src/combine/combine_kernels.cu`, WH = `src/combine/workspace_helper.cu`,
OV = `python/lopep/comm/overlap.py`. Notation: `cnt[h][e] = splits_per_source[h*nex+e]` (h = home/source rank),
`U[h][n] = unique_counts[h*NN+n]`, `E = ep_nexperts`, `nex = W*E`, `ep_start = rank*E`, `my_node = rank/L`, `my_lr = rank%L`,
`cpr = m_full/W`, `tpr = cpr/topk`, `C[s][d] = sum_{e in [sE,(s+1)E)} cnt[d][e]`. Read-only mapping pass; every claim cites a line.

## Dead paths (hard-coded constants)
Pieces always off (`want = 0` GC:490; `pieces_run = msplit_run && false` GC:3017): `set_piece_table`, `send_piece_off`,
`piece_*`, `a2av_piece_lookup_dev` never run. `size_order = false` (3046). `ce = 0` so `msplit_chunk_E = 0` (3111-3124).
`own_wave_first = 0` (206). `kIdxKernel = true` (327-328): the torch fallback runs only when `M_this_ep == 0`.
tuning.h: `kCombineWaveAdapt = 48`, `kCombineWaveNodes = 1`, `kCombineWireStreams = 16`.

## (1) `derive_combine_meta` and helpers
Entry points: `GemmCombineOpImpl::derive_combine_meta` GC:2366-2508 (used by OV; takes `sps_dev`, `uc_dev`) and
`CombineWireImpl::derive_combine_meta` GC:2089-2147 (no device-meta path). Everything runs on the current stream
(main, or `_meta_stream` when OV `plan_overlap == 2`).

| Item | Def | Consumers | Formula | Feeds |
|---|---|---|---|---|
| `m_this_ep` | 2384-2389 (also 2107-2112) | 2451-2452 (mode-2 check), 2459 -> `M_this_ep` in builder: 329, 349, 449, 461 | `sum_h sum_{e in [ep_start,ep_start+E)} cnt[h][e]` | `FLUX_CHECK_EQ(acc, M_this_ep)` (349); `use_idx_kernel` gate (329); `pack_index` alloc (449); `cargs.m_this_ep` -> pack-kernel grid (CK:992-997); `check_totals` idx 0 |
| `tables` (pinned i64 [3, nexG]) | 334-336 | 394 / 439 (H2D), 444 (cmp) | | rows below |
| `offA_h[g]`, `cumA_h[g]` | 338-348 | 454, 453 | g = e_loc*W + h; offA exclusive prefix of cnt[h][ep_start+e_loc] in A-order; cumA inclusive | `tables_dev` H2D non_blocking (439) -> `CombinePlanArguments.cumA/offA` -> `combine_plan_pack_kernel` (CK:964-969) |
| `offR` / `offR_of_A_h` | 350-357 / 358-362 | 455 | `offR[h*E+e_loc]` exclusive prefix in (h, e_loc) order; `offR_of_A[e_loc*W+h] = offR[h*E+e_loc]` | `.offR_of_A` kernel arg |
| `rtab` (pinned i64 [4, nex]) | 367-369 | 421 / 440 (H2D), 445 (cmp) | | rows below |
| `my_cum[e]` | 371, 378, 387 | 457 | `sum_{e'<e} cnt[rank][e']` | `.my_cum` -> `combine_plan_reduce_kernel` (CK:983) |
| `e_base[e]` | 372, 379, 388 | 423 (torch path only) | `sum_{e'<e} sum_h cnt[h][e']` | not a kernel arg (kernel uses `expert_cum[e-1]`, CK:982) |
| `h_base[e]` | 373, 380-386 | 458 | `sum_{h<rank} cnt[h][e]` | `.h_base` |
| `e_cum[e]` | 374, 389 | 456 | `sum_{e'<=e} sum_h cnt[h][e']` | `.expert_cum` |
| `a2av_combine_plan` | 466 | CK:990-1005 | | outputs `pack_index [M_this_ep]`, `reduce_index [cpr]` |
| `C[s*W+d]` | 559-568 | 610, 612, 613 | C[s][d] | recv_off_C / Cp / own_total |
| `t64` (pinned i64 [n_i64]) | 574-583 | 666 (H2D), 671 (cmp) | n_i64 = 2nex + (NN-1)L*E + 2W + NN (569-571) | `t64_dev` H2D -> `A2AVCompressPlanArguments` (698-703) |
| `expert_base[e]`, `my_cnt_cum[e]` | 592-593 | 698, 700 | = e_base / my_cum | `compress_plan_conv_kernel` (CK:846-849), `compress_plan_red_kernel` (CK:897-899) |
| `home_base[e*W+h]` (pinned i32 `t32` [nex*W]) | 584-598 | 667 (H2D), 672 (cmp), 697 | `sum_{h'<h} cnt[h'][e]` | conv / red kernels (CK:849, 866, 899) |
| `recv_off_C[s]` | 608 | 701 | `sum_{s'<s} C[s'][rank]` | red kernel (CK:901) |
| `recv_off_Cp[s]` | 609-616 | 644, 702 | `sum_{s'<s} cp(s')`, cp = C[s'][rank] same node; U[rank][s'/L] if s'%L == my_lr; else 0 | red kernel (CK:901); `rem_base` |
| `own_total` | 603, 613 | 673, 691 | `sum_{s in my node} C[s][rank]` | `red_row` size (691); `check_totals` idx 1 |
| `conv_base[seg*L*E+j]` / `conv_total` | 619-637 | 689, 699 | exclusive prefix over (tn ascending skipping my_node; j in [0,L*E)) of cnt[tn*L+my_lr][my_node*L*E+j] | conv kernel (CK:847, 864); `wire_copy` size `[conv_total]` (689) |
| `wire_total` | 620, 629 | 648, 673, 688 | `sum_{tn != my_node} U[tn*L+my_lr][my_node]` | `wire_ptr` size (688); `FLUX_CHECK_EQ(wire_total, 0)` when conv_total == 0 (647-651) |
| `rem_base[m]`, `rem_total` | 638-646 | 703, 691 | `rem_base[m] = recv_off_Cp[m*L+my_lr]` (0 own node); `rem_total = sum_{m != my_node} U[rank][m]` | red kernel (CK:907); `red_row` size `[own+rem]` |
| `a2av_compress_plan` | 725 | CK:914-940 | | 4 memsets + 4 kernels -> `wire_ptr`, `wire_copy`, `red_ptr`, `red_row` |

Device-meta path (`CombineDevMeta`, GC:2390-2454; `LOPEP_DEVICE_META` GC:293-300): mode 1 swaps only the POINTERS
(435-437, 662-664); the host loops and pinned allocations (334-391, 559-651) still run unconditionally; the host values
`M_this_ep`, `wire_total`, `conv_total`, `own_total`, `rem_total` still size the outputs (449, 688-691) and drive
FLUX_CHECKs 349, 647. Mode 2 runs both and compares (`compare_i64` 444, 445, 671; `compare_words` 672; `check_totals` 674, 2452).

## (2) Forward path (`forward_impl` 2770 -> `combine_wire->run` 3427 on `gather_rs_stream` -> `combine()` 1279)
| Item | Def | Consumers | Formula | Feeds |
|---|---|---|---|---|
| `chunks64` / `chunk_at(s,d)` | 1320-1330 | 1335, 1341, 1353, 1363, 1380, 1406, 1410, 1437, 1520, 1756, 1882, 1886, 1898 | C[s][d] | checks + ladder sizes |
| sanity checks | 1332-1358 | | `sum_d C[rank][d] == M_this_ep` (1337); `sum_s C[s][d] == cpr` (1343); `max_s sum_d C[s][d] <= a2av_send_rows_` (1357) | FLUX_CHECKs |
| `send_off[d]` | 1360-1364 (`acc` is `int`) | 1542, 1720, 1730, 1764, 1774, 1885, 1902 | `sum_{d'<d} C[rank][d']` | `pack_args.node_row_start[n] = send_off[n*L]` (1542; `[NN] = M_this_ep` 1544); conv put/memcpy src (`conv_stream`); intra memcpy/put src (`intra_stream`) |
| `chunk(rank,d)` | does not exist | | = `chunk_at(this->rank, d)` | |
| `chunk_cp(s,d)` | 1378-1386 | 1390, 2019 | same node -> C[s][d]; same lr -> U[d][s/L]; else 0 | `recv_off_cp`; bucket-loop skip (2019: host control of `CUStreamWaitValue64` + `a2av_combine_bucket_reduce` on `reduce_stream`) |
| `recv_off_cp(s,d)` | 1387-1393 | 1449, 1829, 1856, 1884, 1901, 1974 | `sum_{sq<s} chunk_cp(sq,d)` | `FLUX_CHECK_LE(recv_off_cp(W,rank), cpr)` (1449); wire put dst `recv_off_cp(rank, d_remote)` (1856, `wstream`); self memcpy dst (1884); intra peer put dst (1901); lane table (1974) |
| `conv_off(dl,tn,ls)` | 1396-1413 | 1713, 1757 | `sum_{t2<tn, t2 != my_node} sum_{l2<L} C[my_node*L+l2][t2*L+dl] + sum_{l2<ls} C[my_node*L+l2][tn*L+dl]` | conv ladder dst (1757, ls = my_lr), `conv_stream` |
| `wire_seg_off(tn)` | 1415-1424 | 1637, 1830, 1857 | `sum_{t2<tn, t2 != my_node} U[t2*L+my_lr][my_node]` | `prered_args.wire_seg_start[seg]` (1637 -> CK:258-259); wire put src (1857) |
| `total_wire` | 1641-1647 | 1647 | `sum_{tn != my_node} U[tn*L+my_lr][my_node]` | `prered_args.wire_seg_start[NN-1]` |
| conv / wire overflow | 1428-1449 | | `max_{n2,dl} sum_{tn != n2} sum_ls C[n2L+ls][tnL+dl] <= conv_rows_`; `sum_{tn != n2} U[tnL+dl][n2] <= wire_rows_` | FLUX_CHECKs 1445, 1447 |
| conv ladder rows | 1756 | 1761-1784 | C[rank][tn*L+dl] | `cudaMemcpyAsync` (self gateway) or `flux_rs_put_signal`, rows*row_bytes, `conv_stream`; 0 -> `signal_op` |
| wire ladder rows | 1853 | 1854-1871 | U[tn*L+my_lr][my_node] | `flux_rs_put_signal` size on `wstream` (`internode_stream` or `internode_streams2_[lane-1]`, 1800-1809) |
| intra rows | 1882, 1898 | 1883-1916 | C[rank][rank], C[rank][d] | `cudaMemcpyAsync` / put size on `intra_stream`; 0 -> `signal_op` |
| `remote_rows` (wave-adapt) | 2992-3001 | 3009 | `sum_{h: h/L != my_node} sum_{e<E} cnt[h][ep_start+e]` | `wire_bytes = remote_rows*N*elt_o`; `reread = (n_waves_planned-1)*E*N*K*elt_w`, n_waves_planned = 1 + ceil(r1/NG) + ceil(my_node/NG) (3002-3008); `msplit_run = false` if reread > 48*wire (3010-3011) |
| `msplit_run` consumers | 2981 | 3018, 3247, 3260, 3281, 3339, 3343, 3385, 3393, 3395, 3425-3426 | | GEMM `problem_count` (3281), `ws_args.msplit`, fused pack (`inputs[0].mul_` + `a2av_invert_index`), `args.n_split`, `set_msplit_waves` -> `fused_relay` (1490) / `pack_args.msplit` (1565) |
| waves / `msplit_node_order_` / `msplit_wave_of_node_` | 3030-3102 | 1569-1570, 1577-1590, 3422-3424 | topology only (cnt only in the dead `size_order` branch 3059) | pack args, `sched_remote` |
| `idx_of` | 2970, 3137-3172 | 3200, 3346 | permutation; depends on `wgate_of_expert_` (3161), not cnt | problem order |
| `node_base[n2]` (per expert) | 3188-3195 | 3197-3198 | `sum_{n'<n2} sum_{lr2} cnt[n'*L+lr2][ep_start+e]` | `h_wave_off = node_base[a]`; `h_wave_M = node_base[b]-node_base[a]` (FLUX_CHECK_LE INT32_MAX 3199); `h_eid = e`; `h_grp = w`; `h_ne[grp] += (rows > 0)` (3209) |
| `msplit_host_` (pinned i32 [4*NN*E + NN], ctor 2628-2631) | 3173-3178 | 3214-3226 | layout [wave_M|wave_off|eid|grp](n_probs each) | ne[n_flags] | `cudaMemcpyAsync` -> `msplit_dev_` on forward `stream` (3214-3219), event 3220 -> `ws_args.wave_M/wave_off/non_empty_per_wave/prob_eid` (3250-3254) -> WH:93-95 (`problem_sizes[i] = {wave_M, N, K}` WH:105; `M_acc = (splits_acc[eid]-splits[eid]) + wave_off` -> `ptr_A`/`scatter_D` WH:111, 126); WH:147-152 presets `barrier[w] = 1` if ne == 0; `args.non_empty_per_group` (3393); `prob_group_map` (3394-3397) |
| lane table `a2av_bucket_lanes_h_` (pinned [2*2(W+1)], ctor 931-933) | 1967-1981 | 1982-1987 | `lane_off_h[sq] = recv_off_cp(sq, rank)`, sq in [0,W]; `chain_pos_h` topology only | `cudaMemcpyAsync` H2D (2W+1)*4 B on `reduce_stream`, run-parity double-buffered -> `a2av_bucket_map` (1994-2005; CK:363-372) |
| wgate tables `wgate_host_` | 3336-3361 (alloc 2757) | 3399-3403 | `hw[idx_of[w*E+e]] = wgate_of_expert_[e]` (no cnt); `n_probs_all` depends on `msplit_run` | `cudaMemcpyAsync` H2D on `stream` (3356); `.prob_wgate_map`, `weight_signal_ptr`/`expected` |
| `M_this_ep` (forward) | 2812 = `inputs[0].size(0)` (OV sizes `out_buf[:_m_this]`) | 2955, 2841, 3272, 3275, 3417 | | `gemm_outs` size; `a2av_invert_index` n; send-panel FLUX_CHECK; `gemm_op->run` vs `barrier.fill_(1)` |

The non-msplit GEMM problem sizes come from device `splits_gpu` (WH:38-40, 101), not host cnt.

## (3) Per-step allocations and memsets
Allocations: pinned `tables`, `rtab` (334, 367); `.to(kCUDA)` copies (394/421/439-440); torch fallback temporaries
(399-424, only M_this_ep == 0); `pack_index [M_this_ep]`, `reduce_index [cpr]` (449-450); pinned `t64`, `t32` (574, 584);
`splits_cum`, `e_of` (654-658); `t64_dev`, `t32_dev` (666-667); `scratch` (679); `wire_ptr [wire_total+1]`,
`wire_copy [conv_total]`, `red_ptr [tpr+1]`, `red_row [own+rem]` (688-691); devmeta arena first call only (2401-2410);
`run()` 2178-2179 `output_.value_or(empty_with_uninitialized_data(...))` (argument evaluated even when `output_` is
passed from 3429); `splits_gpu` (2860, only if CPU); `gemm_outs [M_this_ep, N]`, `output`, `workspace_gpu` (2954, 2958,
3284); `workspace` grow-only (2518); host vectors `offR` 350, `C` 559, `chunks64` 1320, `send_off` 1360, `waves` 3030,
`idx_of` 3137; OV:139 `_uc[:, W:].contiguous()`; OV:157-161 device temporaries.
Per-step `cudaMemsetAsync`: GC:1456-1457 `group_flags`, `group_counters` (`stream_raw`); 1462-1463 `wire_flags_`,
`wire_counters_` (compress); 1992-1993 `bucket_cnt` (`reduce_stream`); CK:919, 921, 924 `conv_count`, `red_flags`,
`wire_row_of` (0xFF) (derive stream); GC:264-265 `cmp_dev` (mode 2). Other fills: `barrier.fill_(1)` (3420, M=0),
`barrier.zero_()` (3447), OV `scale_buf.zero_()` (OV:162).

## (a) Sync points
GC:268 / 284 stream syncs in `compare_words` / `check_totals` (mode 2 only); 394 `tables.to(kCUDA)` blocking (only when
M_this_ep == 0); 1503 `cudaEventSynchronize(piece_tbl_event_)` dead; 2703 `gemm_mark_event_` (when armed);
3105 `cudaEventSynchronize(msplit_h2d_event_)` (previous step's `msplit_host_` H2D; every step when msplit_run);
3341 `wgate_h2d_event_` (when armed); init-only 988, 991, 1017, 1239; 2253 `resize_capacities`. No `.item()`/`.cpu()` in
GC/CK/WH; OV:113 `int(sps[...].sum())` is CPU-only.

## (b) Device tables kernel and `CombineDevMeta`
`a2av_combine_tables_kernel` CK:1039-1195, launched CK:1199-1210 (1 CTA x 512, dyn smem 8*(nexG + 2W) B <= 96 KB).
Inputs: device `sps [W][nex]`; `uc` with row stride W+NN, U read at column offset W (CK:1046, 1054); W, E, L, NN, rank.
Computes: offA/cumA (1057-1065, `s_tot[0] = m_this_ep`), offR_of_A (1068-1075), e_cum/e_base/expert_base (1078-1095),
my_cum/my_cnt_cum (1098-1106), home_base i32 and h_base (1108-1132), `s_col[s] = C[s][rank]`, `s_cp`, own_total,
recv_off_C, recv_off_Cp (1134-1156), conv_base/conv_total (1159-1175), rem_base, wire_total, rem_total,
`totals[0..4] = {m_this_ep, own, conv, wire, rem}`, `totals[5..7] = 0` (1177-1193).
Arena (GC:2399-2437, comment 245-249): `devmeta_i64_ = [cumA|offA|offR_of_A](3nexG) [my_cum|e_base|h_base|e_cum](4nex)
[expert_base|my_cnt_cum|conv_base|recv_off_C|recv_off_Cp|rem_base](n_i64) totals[8]`; `devmeta_i32_ = home_base[nex*W]`.
Forward-path tables NOT produced: full `chunks64`; `recv_off_cp(rank,d)` for d != rank; `conv_off` for dl != my_lr;
per-segment `wire_seg_off`; overflow maxima; `lane_off[W]`; per-expert `node_base`; `remote_rows`.
Equal by formula to arena entries: `send_off[d]` = `offR_of_A[d]` (e_loc = 0 row); `conv_off(my_lr,tn,ls)` =
`conv_base[seg*L*E + ls*E]`; `lane_off[sq<W]` = `recv_off_Cp[sq]`, `lane_off[W]` = own + rem;
`node_base[n2]` = `home_base[(ep_start+e)*W + n2*L]`.
`CombineDevMeta` (GC:250-291): `mode, i64, i32, nexG, nex, n_i64, cmp_dev, cmp_host, tot_host, stream`; `rtab()` = i64+3nexG;
`t64()` = +4nex; `totals()` = t64+n_i64; `compare_words`, `compare_i64`, `check_totals` (D2H + stream sync + FLUX_CHECK).

## (c) Barriers
No `nvshmemx_barrier_all_on_stream` in GC/CK/WH; the combine reaches `NvshmemGroupBarrier::barrier_all`
(src/core/flux_shm.cc:669) through `group_barrier.barrier_all(stream)` at GC:3410 and GC:3446 (member 2331, constructed
2593 as `(tp_group_, false)`; NVSHMEM unless ring mode / `force_flux_impl` / `device_count > kMaxLocalWorldSize`,
flux_shm.cc:697-711). GC:3410 has no justifying comment (the next, 3412, is "ensure barrier initialized correctly" before
the event record 3413); GC:3446 has none. Host-side `nvshmem_barrier_all()` at 989 (init; 956-958 "orders their remote
delivery before any epoch") and 2254 (`resize_capacities`; 2225-2226 "collective; idle device on every rank").

## (d) overlap.py
`_derive_combine` (138-154): consumes host `_sps` and `uc_combine = _uc[:, W:].contiguous()` (139; None if NN == 1),
device `_sd`, `_scd.view(-1)`, optional `_sps_dev`/`_uc_dev` (141-142); calls `derive_combine_meta` (143), `record_stream`
(145-147), `_build_scale` (154). `_combine_kwargs` (attribute; 51, built 148-153: `splits_per_source = _sps`,
`pack_index`/`reduce_index = meta[0:2]`, `wire_csr = meta[2:4]`, `reduce_csr = meta[4:6]`, `unique_counts = uc_combine`;
consumed 233). `_scale_compute` (156-163): no host tensors (`m_start` stays a 0-d device tensor, no sync). `_m_this`
(113) consumed 132, 210, 212, 232 (`scale_buf[:_m_this]` -> `output_vec_scale`), 239. Comment drift: OV:140 says device
counts exist only with `check_capacity` on, but 102-109 obtains them whenever `DEVICE_META != 0`; tuning.h calls
`kCombineWaveAdapt` a tile count while GC:3010 uses it as a byte ratio.
