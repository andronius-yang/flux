"""Per-iteration payload randomization + provenance ledger for wire-ordering
audits (HARD RULE, CLAUDE.md invariant 5 / SCHEMA protocol rule 6, 2026-08-22).

A static payload cannot detect a wire that delivers the PREVIOUS iteration's
bytes (the nbi put_signal flag-before-data bug): an unrewritten row keeps the
correct bytes. Every driver's perf loop must therefore re-randomize the
dispatched activations each iteration, and the correctness check must build
its expectation from the FINAL payload while comparing the LAST iteration's
outputs. This module is the single shared implementation; drivers call:

    probe = PayloadProbe(ctx.inputs_shard, rank)        # once, before the loop
    for it in range(total_iters):
        probe.step(it)                                   # in-place randomize (no-op if disabled)
        ... pack / dispatch / gemm ...
    check_correctness(...)                               # uses ctx.inputs_shard (final)
    probe.classify(bad_rows_tensor, src_row_index, TP_GROUP)  # optional provenance

Enabled by FLUX_RANDOM_PAYLOAD=1 (alias: FLUX_PLL_RANDOM_PAYLOAD=1, the name the
epic driver introduced). Values are U[0,0.01) cast to the tensor dtype (same
magnitude as generate_data's activations, so GEMM allclose tolerances hold).
"""

import os
from collections import Counter
from typing import Dict, List, Optional

import torch


def payload_probe_enabled() -> bool:
    return bool(int(os.getenv("FLUX_RANDOM_PAYLOAD", "0"))) or bool(
        int(os.getenv("FLUX_PLL_RANDOM_PAYLOAD", "0")))


class PayloadProbe:
    def __init__(self, shard: torch.Tensor, rank: int, seed: int = 4242,
                 keep_ledger: bool = True):
        self.shard = shard
        self.rank = rank
        self.enabled = payload_probe_enabled()
        self.keep_ledger = keep_ledger
        self.ledger: List[torch.Tensor] = []
        self.gen: Optional[torch.Generator] = None
        if self.enabled:
            self.gen = torch.Generator(device=shard.device).manual_seed(
                seed + 7919 * rank)
            if keep_ledger:
                # payload 0 = the pre-loop (setup) shard
                self.ledger.append(shard.clone())

    def step(self, it: int) -> None:
        """Randomize the shard IN PLACE on the current stream (call before the
        iteration's pack/dispatch). No-op when the probe is disabled."""
        if not self.enabled:
            return
        # Alternate the SIGN per iteration: drivers that compare GEMM
        # outputs under allclose (atol 1e-2) would not see a stale
        # U[0,0.01) row (it moves a GEMM row by ~2e-3); a sign flip makes a
        # one-epoch-stale row differ by ~2|y| — always outside tolerance.
        # Bitwise row checks (hidden_buf) are unaffected either way.
        sign = 1.0 if (it % 2 == 0) else -1.0
        self.shard.copy_(
            (torch.rand(self.shard.shape, device=self.shard.device,
                        generator=self.gen) * (0.01 * sign)).to(self.shard.dtype))
        if self.keep_ledger:
            self.ledger.append(self.shard.clone())

    def classify(self, got_bad: torch.Tensor, bad_src_rows: torch.Tensor,
                 group, ntokens: int) -> Dict[str, object]:
        """Provenance of wrong rows: `got_bad` [n_bad, H] = the delivered bytes,
        `bad_src_rows` [n_bad] = global source row (src_rank * S + tok) each
        row was supposed to carry. Returns {k: count} over ledger entries
        (k = len(ledger)-1 is the expected/final payload) + poison sentinel
        count. Collective (allgathers each ledger entry)."""
        prov: Dict[int, int] = {}
        if not (self.enabled and self.keep_ledger) or got_bad.numel() == 0:
            return {"by_payload": prov, "poison": 0}
        for k, pay in enumerate(self.ledger):
            full = torch.empty(ntokens, pay.shape[1], dtype=pay.dtype,
                               device=pay.device)
            torch.distributed.all_gather_into_tensor(full, pay, group=group)
            prov[k] = int((got_bad == full[bad_src_rows.to(full.device)]).all(
                dim=1).sum())
            del full
        poison = int((got_bad.view(torch.int16) == -23131).all(dim=1).sum()) \
            if got_bad.dtype in (torch.bfloat16, torch.float16) else 0
        return {"by_payload": prov, "poison": poison}

    def describe(self) -> str:
        return (f"payload probe {'ON' if self.enabled else 'off'}"
                f" (ledger {len(self.ledger)})")
