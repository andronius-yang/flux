# Stage B input: host-side plan inventory of `a2av_dispatch` (lopep sglang-dev at 5cafd2d/3011c6f; 10-01)

Line numbers are for the working tree after A1 + B4 (`src/dispatch/ths_op/dispatch_gemm.cc` = DG). `a2av_dispatch` is
declared at DG:662-663 and returns `A2AVDispatchState` at 2076-2077; `forward_impl` runs 2257-2552. Produced by a
read-only code-mapping pass; every claim cites a line.

## 0. Reachability facts (decide which consumers are live)
- `use_meta` is always true (FLUX_CHECKs DG:684, 2336); every `!use_meta` block is dead: 1162-1164, 1211-1214, 1525-1550.
- Parity is fixed: `par = 0`, `pack_str = stream` (797-798, `false ? ...`); `meta_par = 1` (516).
- `lb_minmove_ = relay_per_round_ = (nnodes > 1)` (377-378); `relay_slots_ = tuning::kRelaySlots` (379); `wave_pack_ = true`
  only when nnodes > 1 (460).
- `true ? A : B` selects at 1165, 1589, 1591 always take A.
- `build_stage2`: the block 1308-1409 ends in an unconditional `return;` (1408) so **1410-1518 never run** (searchsorted gating,
  `cumA_dev`, `offR_of_A_dev`, `M_this_ep` uses at 1461/1463).
- Wire calls are deferred: `forward_impl` passes `defer_wire_arg=true` (2389), so `tail_stream = pack_stream_` (1649-1650);
  the cp_stream self-copy, signals and puts become descriptors that `issue_deferred_wire` (293-332, called at 2526) replays
  on `cp_stream` after the GEMM launch.
- Dead branches under NN > 1 (lb_minmove_/relay_per_round_ always true): equal-cut branch of `chunk_bound` (725-726), legacy
  branch of `round_pieces` (1844-1862), accumulating branch of `relay_round_base` (1766-1770).

Notation: `cnt[s][e] = cnt_host[s*nexperts+e]`, `u[s][d] = uc[s][d]`, `U[s][n] = uc[s][W+n]`, `E = ep_nexperts`,
`L = local_world_size`, `nseg = L+NN-1`, `rb = row_bytes`, `g(sl,n) = local_rank_to_global_rank(sl,n)`.

## 1a. Logical tables (from cnt only), staged in the pinned arena
| Name | Def | Formula | Consumers -> use |
|---|---|---|---|
| `cnt_at` | 815 | `cnt_host[s*nex+e]` | 820, 843, 853, 865, 874 |
| `chunks64` [W*W] | decl 807, fill 816-824 | `[s*W+d] = sum_{e in [dE,(d+1)E)} cnt[s][e]` | 826 (M_this_ep); 903 (FLUX_CHECK 904); 1551 `chunk_at`. Dead: 1531/1534/1544 |
| `M_this_ep` | decl 806, sum 825-827 | `sum_s chunks64[s*W+rank]` | returned 2077 -> fwd 2393; 2417 output alloc `[M_this_ep,N]`; 2425 CHECK_2D; 2438 `args.M_this_ep` (GEMM problem size); 2517 `if (M_this_ep>0) op->run`. Dead: 1461/1463 |
| `offA_h` / `cumA_h` | ptrs 832-833, fill 838-846 | `g=e_loc*W+s`; `offA_h[g]` exclusive prefix of `cnt[s][ep_start+e_loc]` in g order; `cumA_h` inclusive | -> `offA_dev` (1121) -> 1344 `cb_args.offA` (consumer-build kernel) and 1391 gating-cumsum `.offA`. `cumA_dev` (1120) used only at 1484 (dead) |
| `offR` / `offR_of_A_h` | 848-855 / 834, 856-860 | `offR[s*E+e_loc]` exclusive prefix in (s, e_loc) order; `offR_of_A_h[e_loc*W+s] = offR[s*E+e_loc]` | -> `offR_of_A_dev` (1122), used only at 1488 (dead) |
| `ssc_h` i32 | 836, 862-868 | `[e_loc*W+s] = sum_{s'<=s} cnt[s'][ep_start+e_loc]` | -> `ssc_dev` [E,W] (1124) -> `sorted_splits_cumsum` (1405) -> fwd 2392 -> `accum_per_rank_ptr` (2447-2450) only when `a2av_gating_cumsum_` is undefined (NN==1 case; tier_b defines it at 1373) |
| `expert_base_h` | 835, 870-876 | `[e] = sum_{e'<e} sum_s cnt[s][e']` | -> `expert_base_dev` (1123) -> 1345 `cb_args.expert_base`. 1445/1491 dead |
| `chunk_at` | 1551 | `chunks64[s*W+d]` | 1577 (`send_off`), 1582 (`recv_off`), 1589 (dead arm) |
| `send_off` [W] | 1574-1578 | `sum_{d'<d} chunk_at(rank,d')` | only 1591, dead arm |
| `recv_off` [W] | 1579-1585 | `sum_{s<rank} chunk_at(s,d)` | no consumer |

## 1b. Compress / wire tables (from uc; U-based ones also need the mm plan)
| Name | Def | Formula | Consumers -> use |
|---|---|---|---|
| `u_mat` / `U_mat` | decl 686, fill 889-898 | `u_mat[s*W+d]=u[s][d]`, `U_mat[s*NN+n]=U[s][n]` | u_mat: 902, 912, 929, 952, 1553. U_mat: 702, 907, 914, 928, 955, 1554 |
| `u_at` / `U_at` | 1553 / 1554 | lookups | u_at: 1589, 1675, 1715, 1803. U_at: 1675, 1806 |
| `region_rows(s,d)` | 927-930 | `s/L != d/L ? U[s][d/L] : u[s][d]` | 935, 943, 984 |
| `max_col` | 931-938 | `max_d sum_s region_rows(s,d)` | FLUX_CHECK_LE vs `max_recv_ntokens_` (939-940) |
| `recv_off_u` [W] | 941-944 | `[s] = sum_{s'<s} region_rows(s',rank)` | 984, 987 (lane_end); 1011 (mm `o_base`); 1592 (self_recv_off) |
| `seg_off_h` [nseg+1] | 948-958 | nodes ascending; n==my_node -> L segments `u[rank][n*L+dl]`; else one segment `U[rank][n]` | 959; 1033 -> arena -> `seg_off_dev` (1129, `{nseg}`) -> pack-scan kernel `.seg_off` (1226); 1243-1244 wave-pack segment narrow + `index_select_out`, event `pack_seg_events_[seg]` + peer announce; 1591, 1680, 1775 |
| `total_send_rows` | 959 | `seg_off_h[nseg]` | 960 FLUX_CHECK_LE vs `copies_per_rank`; 1231 wave-pack gate; 1239 send-buffer narrow; 1272/1277/1279 NN==1 pack sizes |
| `lane_end` [W] | 981-989 (NN>1) | same-node s: `recv_off_u[s]+region_rows(s,rank)`; else `recv_off_u[ns*L]+chunk_bound(ns,my_node,gl+1)` (ns=s/L, gl=s%L) | 994 |
| `gate_q_h` [E*(W+1)] | 971 clear, 990-996 (NN>1) | `[e(W+1)] = e*R`; `[e(W+1)+1+s] = e*R+lane_end[s]`, `R=max_recv_ntokens_` (979) | 1048-1049 -> arena -> `gate_q_dev` (1131-1134); live use 1350 only (`.lane_end = gate_q_dev+1`, row e=0, consumer-build lane binary search); rows e>=1 only read at 1464 (dead) |
| `mm_h` [(NN+1)+NN+3*NN*2L] (187-190) | 1001-1025 (NN>1 && lb_minmove_) | `o_off[ns]` piece prefix; `o_base[ns]=recv_off_u[ns*L]`; per ns != my_node and piece pz in `mm_pieces_[ns*NN+my_node]`: `c0=canon_start(ns,pz.sl,my_node)`, `lo=c0+j_lo`, `hi=c0+j_hi`, `dst=chunk_bound(ns,my_node,pz.k)+pz.off`; `o_off[NN]=pc` | FLUX_CHECK_LE(pc,P) 1025; 1051-1052 -> arena -> `mm_dev` (1136-1140) -> `cb_args.mm_*` (1361-1371) -> consumer-build remap (sort_util.cu:215-235) |
| `cmp_h` | 1031-1053 | arena words from `compress_meta_off_`: seg_off_h[0..nseg], fwd_col_off_h, recv_start_h, win_a_h, win_b_h (all empty), gate_q_h, mm_h | H2D at 1059 |
| `fwd_col_off_h`, `recv_start_h`, `win_a_h`, `win_b_h` | decl 686/689 | never filled (970 clear; loops 1036-1047 no-ops) | none |
| `fwd_col_off_dev`, `recv_start_dev`, `win_a_dev`, `win_b_dev` | decl 691 | never assigned or read | none |

## 1c. lb_minmove plan (member vectors, rebuilt every step)
| Name | Def | Formula | Consumers |
|---|---|---|---|
| `U_of(n,sl,m)` | 701-703 | `U_mat[g(sl,n)*NN+m]` | 709, 754 (1847/1855 dead) |
| `canon_start(n,sl,m)` | 706-712 | `sum_{sq<sl} U_of(n,sq,m)` | 1017 (725, 1846, 1854 dead) |
| `mm_build` -> `mm_bound_`, `mm_keep_`, `mm_imp_`, `mm_pieces_` | 738-793, called 918-920 for ALL NN^2 pairs | n != m: `V_k=U_of(n,k,m)`, `cap_k=tot/L+(k<tot%L)`, `keep_k=min(V_k,cap_k)`, `room_k=cap_k-keep_k`; excess to the next importer with room (two-pointer); `bound[k+1]=bound[k]+off_k`; `imp=off_k-keep_k`; pieces `{sl,j_lo,j_hi,k,off}` | `mm_bound_`: 723; `mm_pieces_`: 1016, 1837; `mm_keep_`: 1833; `mm_imp_`: 1828; FLUX_CHECKs 774, 782, 790 |
| `chunk_bound(n,m,k)` | 717-727 | live: `n==m ? 0 : mm_bound_[(n*NN+m)*(L+1)+k]` | 729, 987, 1020, 1936-1937, 2036-2037 |
| `chunk_rows_of(n,m,k)` | 728-730 | `chunk_bound(k+1)-chunk_bound(k)` (= `cap_k`) | 1693, 1698, 1754 |

## 1d. Wire-time scalars and lambdas (host pointer offsets and sizes)
| Name | Def | Formula | Use |
|---|---|---|---|
| `self_rows` / `self_send_off` / `self_recv_off` | 1589 / 1590-1591 / 1592 | `u[rank][rank]` / `seg_off_h[node_idx+local_rank]` / `recv_off_u[rank]` | 1651-1656 `emit_self_copy`: deferred D2D `cudaMemcpyAsync` on cp_stream (302); skipped if 0 |
| `recv_off_of_u(s,d)` | 1672-1678 | `sum_{sq<s} (sq/L != d/L ? U[sq][d/L] : u[sq][d])` | 1718 round-0 put dst; 2047 gateway forward dst `(recv_off_of_u(ns*L,d)+win_a)*rb` |
| `send_seg_off(dlg)` | 1680 | `seg_off_h[my_node+dlg]` | 1719 round-0 put src |
| round-0 rows | 1715 | `u[rank][d]`, d = g((my_lr-dl+L)%L, my_node) | 1717-1724 `emit_put` (deferred `putmem_signal_nbi` on cp_stream, 308-316) or `emit_signal` if 0 |
| `max_stage_rows` / `max_relay_rows` | 1685-1705 (NN>1) | `max_{n,k} sum_{ns != n} chunk_rows_of(ns,n,k)` / `max_{n,k} max_dn S*chunk_rows_of(n,(n-dn+NN)%NN,k)` | FLUX_CHECK_LE 1706-1707 vs `max_stage_ntokens_`; 1708-1709 vs `max_relay_ntokens_` |
| `stage_off_chunk(gn,glr,ns)` | 1748-1757 | `sum_{n<ns, n != gn} chunk_rows_of(n,gn,glr)` | 1977 wire put dst (`tn,my_lr,my_node`); 2038 gateway `wstage` (`my_node,my_lr,ns`) |
| `relay_round_base(k,dn)` | 1760-1771 | live: `((dn-1)%S)*(max_relay_ntokens_/S)` (routing-independent) | 1878, 1907, 1918 (pull dst); 1971 (wire src) |
| `my_seg_base(tn)` | 1773-1776 | `seg_off_h[tn<my_node ? tn : tn+L-1]` | 1879 self pull src; 1966 own_only wire src |
| `peer_seg_base(plr,tn)` | 1797-1810 | prank = g(plr,my_node); `sum_{n<tn} (n==my_node ? sum_dl u[prank][nL+dl] : U[prank][n])` | 1909 P2P `cudaMemcpyAsync` src on pull_stream; 1919 `getmem_nbi` src |
| `round_pieces(tn, own_only, own_off)` | 1824-1863 | live: `pm=my_node*NN+tn`; `own_only = mm_imp_[pm*L+my_lr]==0`; `own_off=0`; pieces `(my_lr,0,0,keep)` when `keep=mm_keep_[..]>0`, plus each `mm_pieces_[pm]` with `k==my_lr && sl != my_lr` -> `(sl, j_lo, off, j_hi-j_lo)` | 1868 (`pull_round`), 1950 (`put_round`) |
| pull copies | 1869-1924 | size `pc.rows*rb`; dst `relay + (relay_round_base+pc.dst)*rb` | self piece: wait `pack_seg_events_[seg]` (1875) then memcpy; peer piece: `CUStreamWaitValue64(seg_sig[sl*NN+tn])` (1892-1896) then P2P memcpy or getmem; stream `pull_streams_[0]`; records `relay_pull_events_[dn-1]` (1926) |
| `a_me` / `b_me` | 1936-1937 | `chunk_bound(my_node,tn,my_lr)` / `(...,my_lr+1)` | `rows=b_me-a_me` (1938); 0 -> signal only (1940-1945); else BLOCKING `putmem_signal_on_stream` of rows*rb to `g(my_lr,tn)` on cp_stream_inter_node (1976-1984); `own_only` picks the wait (pack event 1955 or pull event 1961) and `wire_src` (1964-1972) |
| `win_a` / `win_b` | 2036-2037 | `chunk_bound(ns,my_node,my_lr)` / `(...,my_lr+1)` | gateway forward on tail_stream after `CUStreamWaitValue64(node_sig+ns)` (2030-2034); per local d: blocking put of `(win_b-win_a)*rb`, signal `signal_base+ns*L+my_lr` (2052-2060); signal only if empty (2049) |

Status of the less obvious names: `fwd_col_off_h`, `recv_start_h`, `win_a_h`, `win_b_h` exist but are always empty;
`cumA_dev`, `offR_of_A_dev`, `send_off`, `recv_off` have no live consumer; `relay_round_base` is routing-independent.

## (a) H2D uploads
- DG:1058-1061: the only host-table upload, `cudaMemcpyAsync(dev_slice = a2av_meta_dev_ + par*meta_stride_, stage =
  a2av_meta_pinned_ + par*meta_stride_, meta_stride_, H2D, pack_str)`, when `LOPEP_DEVICE_META != 1` (knob 81-88). Views:
  `cumA_dev` 1120, `offA_dev` 1121, `offR_of_A_dev` 1122, `expert_base_dev` 1123, `ssc_dev` 1124, `seg_off_dev` 1129,
  `gate_q_dev` 1131-1134, `mm_dev` 1136-1140.
- Arena layout (489-515): payload `cumA|offA|offR_of_A` i64[nexG], `expert_base` i64[nex], `ssc` i32[nexG];
  `compress_meta_off_ = pad_to(meta_bytes, 8)`; compress block `seg_off` i64[nseg+1], `gate_q` i64[E(W+1)], `mm` (lb_minmove).
- No `.to(kCUDA)` in either function; `forward_impl`'s only H2D is `write_gemm_mark` (2516 -> 2248-2250).

## (b) Sync points
| Line | Call | When |
|---|---|---|
| 813 | `cudaEventSynchronize(counts_event_[par])` | every step; guards reuse of the pinned staging (recorded 1115 on pack_str) |
| 1102 | `cudaStreamSynchronize(pack_str)` | `LOPEP_DEVICE_META==2` only |
| 2516 -> 2246 | `cudaEventSynchronize(gemm_mark_event_)` | `forward_impl`, only if the mark is armed |
| 2182 | `cudaEventSynchronize(rt_meta_event_)` | `derive_routed_meta`: D2H of sps/uc/dem (2171-2180); source of cnt_host/uc_host |

## (c) Overflow / consistency FLUX_CHECKs
`a2av_dispatch`: 684 (cnt+uc present); 774/782/790 (water-fill room, pieces <= 2L, `bd[L]==tot`); 904-906 (`0<=u<=chunks`,
`(u>0)==(chunks>0)`); 907 (`u <= U[s][d/L]`); 914-915 (`U[s][n] <= sum_{d in n} u`); 939-940 (`max_col <= max_recv_ntokens_`);
960 (`total_send_rows <= copies_per_rank`); 1025 (`pc <= P`); 1064-1066 (device meta requires `cnt_host == rt_sps_cpu_`);
1104 (arena compare mismatches == 0); 1362 (`tier_b && mm_dev.defined()`); 1706-1709 (stage/relay <= capacities).
`forward_impl`: 2315-2318, 2329-2332 (shapes), 2336, 2425 (`CHECK_2D(outputs, M_this_ep, N)`).

## (d) Device arena kernel (sort_util.cu)
`A2AVMetaArenaArguments` sort_util.h:272-280 (doc 263-271); `a2av_meta_arena_impl` sort_util.cu:825-841 (1 block x 512,
`L <= 8`, `nnodes <= 512`, `nexperts == ep_nexperts*W`, smem <= 96 KB). `a2av_meta_arena_kernel` (616-803) computes from
device sps/uc: offA/cumA (637-647), offR_of_A (648-656), expert_base (658-670), ssc (671-691), seg_off (704-717),
recv_off_u in smem only (718-722), NN>1 chunk bounds of every ns->my_node stream via `arena_waterfill` (558-614),
mm tables (753-783), gate_q (784-801). Output layout identical to the host arena.
NOT produced (still host): `M_this_ep`, `chunks64`, `total_send_rows` (only as `seg_off[nseg]`), `self_*`,
`recv_off_of_u(s,d)` for d != rank, `peer_seg_base`, `stage_off_chunk`, the sender-side my_node->tn bounds and
`mm_keep_/mm_imp_/mm_pieces_` for (my_node, tn) (drive `round_pieces`, `put_round`, `max_stage/relay`), the checks
904-915, 939, 1706-1709.
Modes (1062-1114): 0 host H2D; 1 kernel writes `dev_slice` directly (host still builds every vector and the pinned stage);
2 both + compare (`a2av_arena_compare_impl`, sort_util.cu:805-821/843-851; D2H + stream sync + FLUX_CHECK 1100-1112).
`a2av_demands_kernel` (sort_util.cu:864-983) mirrors `capacity.py demands_from_meta`; called from `derive_routed_meta`
(DG:2156-2172).

## (e) overlap.py
`derive_routed_meta` returns `[splits_dev, scatter_dev, rt_sps_cpu_, rt_uc_cpu_]` (+ `rt_dem_cpu_, rt_sps_dev_, rt_uc_dev_`
with caps; DG:2183-2190). Host consumption: 113 `_m_this = int(sps[:, ep_start:ep_start+gpe].sum())` (= the op's
`M_this_ep`; DG:2425 enforces the match via `outputs_buf`); 116-117 `int(v) for v in dem[:7]`, `int(dem[7])`; 119/126
`sps.numpy()`, `uc.numpy()` -> `demands_from_meta`; 139 `_uc[:, W:].contiguous()` -> `derive_combine_meta` (143),
`combine_kwargs["unique_counts"]` (152); pass-throughs `_sps`/`_uc` -> dispatch forward (211) = cnt_host/uc_host.
`_m_this` uses: 52, 113, 132 (assert), 210 (`outputs_buf`), 212, 232 (`scale_buf[:_m_this]`), 239 (`prep` zeroing).
Capacity mirrors: `dispatch_recv` (capacity.py:40-43) = DG:939; `stage` (44-48) = 1706; `relay` (49) = 1708; no mirror of 960.
