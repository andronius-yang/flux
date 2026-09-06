"""Offline ablation of the tier-3 cover tie rule (reference router).
canon  : ties -> home node, then LOWER node id (kernel + reference).
tickets: ties -> home node, then the node where the source holds the
         MOST remaining tickets, then lower id.
Same inputs as tier_fill_offline.py. Prints forced rows, over-cap rows,
remote nodes per token, GPU max load, for K2 4n b4 and 16n b4."""
import csv, os, sys, torch, importlib.util
spec = importlib.util.spec_from_file_location("tf", "tier_fill_offline.py"); tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
OLD = ("        key = (cnt * 2 + is_home.long()) * NN + (NN - 1 - nn_ar)\n"
       "        best, _ = key.max(dim=1)\n"
       "        n_star = NN - 1 - (best % NN)\n"
       "        covers = (best // NN) >= 2\n")
NEW = ("        TKM = 1 << 20\n"
       "        tks = myshare.view(R, NN, L).sum(-1)[torch.arange(R * S, device=dev) // S].clamp(max=TKM)\n"
       "        key = ((cnt * 2 + is_home.long()) * (TKM + 1) + tks) * NN + (NN - 1 - nn_ar)\n"
       "        best, _ = key.max(dim=1)\n"
       "        n_star = NN - 1 - (best % NN)\n"
       "        covers = ((best // NN) // (TKM + 1)) >= 2\n")
plg_t = tf.load_by_path("plg_tk", os.path.join(tf.TESTING, "placelambda_gpu.py"), patch=(OLD, NEW))
src = list(csv.DictReader(open('../../main_perf/figure_src.csv')))
for nodes in (4, 16):
    s = [r for r in src if (r['nodes'], r['model'], r['budget_mib'], r['row_id']) == (str(nodes), 'K2', '4', 'ours12')][0]
    G, K = 384, 8; W = nodes * 4; L = 4; nlp = G // W + 2
    ofile, rfile, info = tf.cell_inputs(s['capsule'], s['cell_id'])
    orc, _, _ = tf.load_routing(ofile); bat, _, _ = tf.load_routing(rfile)
    S = bat.shape[0] // W; tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64); hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = tf.pv2.pv2_solve(hist, L, nlp)
    for name, mod in (("canon lower-id", tf.plg), ("tickets-aware", plg_t)):
        phys, st = mod.loccap_route_sl(tk, res['p2l'], res['l2p'], res['lcnts'], nlp, L, tf.EPS, return_tables=True)
        phys = phys.long().reshape(-1); srcr = torch.arange(W).repeat_interleave(S * K); dst = phys // nlp
        tok = torch.arange(W * S).repeat_interleave(K); home = tok // S // L; dn = dst // L; remote = dn != home
        act = torch.bincount(torch.unique((tok * nodes + dn)[remote]) // nodes, minlength=W * S).float().mean()
        forced = int(st['forced_pair'].sum())
        load = torch.bincount(dst, minlength=W)
        e119 = (tk.reshape(-1) == 119) & (srcr // L == 1)
        split = torch.bincount(dn[e119], minlength=nodes).tolist() if nodes == 4 else None
        print(f"{nodes:>2}n K2 b4 {name:15s}: forced rows {forced:6d} ({100*forced/phys.numel():4.1f}%)  over-cap rows {st['over_cap_rows']:6d}  remote nodes/token {float(act):.3f}  GPU max load {int(load.max())}/{st['cap']}  cover rounds {st['cover_rounds']}" + (f"  119 from N1 -> {split}" if split else ""), flush=True)
