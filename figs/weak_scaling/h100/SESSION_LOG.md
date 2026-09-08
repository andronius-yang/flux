# H100/ALPS weak-scaling lane — session log

Append-only ledger kept by the session running the campaign on CSCS ALPS
(docs/handoff/35_h100_alps_weak_scaling.md §7 says what to record). One
entry per event: discovery results, build launched/finished (with the
`flux_libs` sha), each gate verdict, each capsule (run_id + status counts),
every incident (hang, OOM, timeout, allocation lost) with the one-line
root cause or "open", and the final figure render.

Format: `- <UTC timestamp> — <event> — <fact/verdict> — <pointer (path/run_id/jobid)>`

## Discovery (handoff 35 §3)

- (fill in: cluster/vCluster, account, partition/QOS, nodes/GPUs per node,
  GPU name + memory + SM count, uname -m, CUDA/nvcc, torch, gcc, cmake,
  NVSHMEM source + version + libfabric transport present, fi_info providers,
  NCCL net plugin, filesystem the clone lives on)

## Build

## Gates

## Capsules

## Incidents

## Figure
