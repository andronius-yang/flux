################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Per-rank sweep measurement recorder (see sweeps/SCHEMA.md).

Enabled only when FLUX_SWEEP_RECORD_DIR is set; every method is a cheap no-op
otherwise, so tests behave identically outside sweep runs. Records buffer in
memory and are written as one JSONL file per rank ({dir}/rank_NNN.jsonl) at
flush() — after the timed region, one file per rank, so concurrent ranks never
contend on a shared file and timing is never perturbed by I/O. flush() is also
registered via atexit so a correctness-failure raise still leaves data behind.
"""

import atexit
import json
import os
import socket
import time

# env prefixes captured into the per-rank meta record (audit trail for knob
# scaling and determinism; full resolved env also lands in cells.csv upstream)
_META_ENV_PREFIXES = ("FLUX_", "NVSHMEM_", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_")


class SweepRecorder:
    def __init__(self):
        self._dir = os.environ.get("FLUX_SWEEP_RECORD_DIR")
        self.enabled = bool(self._dir)
        self._records = []
        self._flushed = False
        self._ts_start = time.time()
        if self.enabled:
            atexit.register(self.flush)

    def emit(self, record_type, **record):
        if not self.enabled:
            return
        record["type"] = record_type
        self._records.append(record)

    def emit_iters(self, impl, iter_times_ms):
        """iter_times_ms: {metric_name: [per-iteration ms, ...]} (post-warmup)."""
        if not self.enabled or not iter_times_ms:
            return
        for metric, values in iter_times_ms.items():
            self.emit("iters", impl=impl, metric=metric, values_ms=list(values))

    def emit_info(self, **kv):
        """Cell-level facts (rank 0: wire ratio, relay bytes, ntokens, ...)."""
        self.emit("cell_info", **kv)

    def emit_correctness(self, bitwise, allclose):
        self.emit("correctness", bitwise=bool(bitwise), allclose=bool(allclose))

    def flush(self):
        if not self.enabled or self._flushed:
            return
        self._flushed = True
        import torch  # lazy: recorder must import cleanly without CUDA

        rank = int(os.environ.get("RANK", "0"))
        meta = {
            "type": "meta",
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "host": socket.gethostname(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "ts_start": self._ts_start,
            "ts_flush": time.time(),
            "env": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith(_META_ENV_PREFIXES)
            },
        }
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"rank_{rank:03d}.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(meta) + "\n")
            for record in self._records:
                f.write(json.dumps(record) + "\n")


RECORDER = SweepRecorder()
