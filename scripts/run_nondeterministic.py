#!/usr/bin/env python3
"""Run a flux test script with torch deterministic mode neutralized.

flux.get_dist_env() force-enables torch.use_deterministic_algorithms(True),
whose serial scatter_ fallback (~500x on A100) invalidates perf numbers for
scatter_-heavy paths; fused-variant sweeps are measured with it off. This
wrapper no-ops the toggle (seeds and cudnn flags untouched, so test data is
identical to correctness runs) and then executes the target script in-place:

    ./launch_fast.sh scripts/run_nondeterministic.py test/python/.../test_x.py <args>
"""

import runpy
import sys

import torch

torch.use_deterministic_algorithms = lambda *args, **kwargs: None

if len(sys.argv) < 2:
    sys.exit("usage: run_nondeterministic.py <script.py> [args...]")
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")

assert not torch.are_deterministic_algorithms_enabled()
print(f"[run_nondeterministic] deterministic mode stayed off for {target}")
