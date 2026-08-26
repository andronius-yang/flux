import torch
from flux.testing import load_routing_file
from flux.testing import placelambda_fast as plfast
from flux.testing.loccap_semantics import plan_tensors_from_hosts
W, L, G, K, nlp = 16, 4, 128, 8, 16
tk = load_routing_file("/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w16x4_trace-f73873_b8_k8_id001.routing.txt", G, K).view(W, -1, K).long().cuda()
oc = load_routing_file("/pscratch/sd/y/yufeid/workspace/andrewy/a2av_test_matrices/generated/w16x4_trace-f73873_b8_k8_id001.oracle_routing.txt", G, K).view(W, -1, K).long().cuda()
pf = plfast.build_placement_fast(oc, L, nlp, G, passes_a=4, passes_b=3, repair_passes=2, seed="affinity")
hosts = plfast.finalize_hosts(pf, W, L, nlp, method="snake")
hosts_rot = [sorted((r + 1) % W for r in hs) for hs in hosts]
ion_r = plfast.hosts_to_ion(hosts_rot, W, L, device="cuda")
prim_r = ion_r.long().argmax(dim=1)
res = plfast.build_placement_fast(tk, L, nlp, G, passes_a=2, passes_b=1, repair_passes=1, seed="warm", seed_primary=prim_r, seed_inst_nodes=ion_r, keep_bonus=0)
v = plfast.place_decision_fast(tk, ion_r, res, L, mode="cover")
print("rot->warm:", v)
res2 = plfast.build_placement_fast(tk, L, nlp, G, passes_a=2, passes_b=1, repair_passes=1, seed="warm", seed_primary=pf["primary"].cuda(), seed_inst_nodes=pf["inst_nodes"].cuda(), keep_bonus=0)
v2 = plfast.place_decision_fast(tk, pf["inst_nodes"].cuda(), res2, L, mode="cover")
print("oracle->warm:", v2)
print("ion diff rot vs oracle:", int((ion_r != pf["inst_nodes"].cuda()).sum()))
