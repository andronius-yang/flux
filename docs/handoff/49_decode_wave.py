"""49_decode_wave.py run|parse ...: decode-figure client and scheduler-log parser (handoff 49).

run: WAVES waves of exactly C requests (C = running requests per rank x ranks) through BATCHED /generate calls
(B prompts per HTTP call, C / B calls in flight). One connection per request cannot reach the 16 MiB point
(65k requests at 4n, 131k at 8n; a client host has 28k ephemeral ports per server port), so bench_serving is not
used here. Prompts = a truncated LiveCodeBench file (48_lcb_workload.py --truncate), chat template applied on the
client exactly as bench_serving --apply-chat-template does, sent as input_ids and cycled to C requests;
max_new_tokens = OSL with ignore_eos, temperature 0. A wave admits and prefills every request, then decodes OSL - 1
steps at the full batch; the next wave starts when every request of the previous one has returned.

parse: the per-DP-rank "Decode batch" lines of the server log (server started with --decode-log-interval N): per
interval the running requests, the graph flag and the generation throughput. The decode step time of a rank is
running / throughput; per running-request count it reports the median over ranks and intervals (the first interval
of each wave also contains that wave's prefill; the median discards it) and the per-GPU output throughput.

    python 49_decode_wave.py run --url http://NODE:30000 --prompts lcb_exec_eval_t48_x1.json --per-rank 1024 \\
        --ranks 16 --osl 48 --waves 2 --batch 512 --tokenizer MODEL_DIR
    python 49_decode_wave.py parse server.log [--per-rank 1024 4096]
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time


async def _call(session, url, ids, osl):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": osl, "ignore_eos": True, "temperature": 0.0},
            "stream": False}
    async with session.post(url + "/generate", json=body) as r:
        r.raise_for_status()
        out = await r.json()
    return sum(int(o["meta_info"]["completion_tokens"]) for o in out)


async def run(a):
    import aiohttp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = json.load(open(a.prompts))
    prompts = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["conversations"][0]["value"]}],
                                       add_generation_prompt=True, tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]
        prompts.append(tok(text, add_special_tokens=False).input_ids)
    C = a.per_rank * a.ranks
    B = min(a.batch, C)
    reqs = [prompts[i % len(prompts)] for i in range(C)]
    plen = sorted(len(p) for p in prompts)
    print(f"[wave] C={C} ({a.per_rank}/rank x {a.ranks}) batch {B} -> {-(-C // B)} calls in flight, osl {a.osl}, "
          f"prompt tokens with template min {plen[0]} median {plen[len(plen) // 2]} max {plen[-1]}", flush=True)
    timeout = aiohttp.ClientTimeout(total=a.timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0, force_close=True)) as s:
        for w in range(a.waves):
            t0 = time.perf_counter()
            got = await asyncio.gather(*[_call(s, a.url, reqs[i:i + B], a.osl) for i in range(0, C, B)])
            dt = time.perf_counter() - t0
            n = sum(got)
            print(f"[wave] {w}: {C} requests, {n} output tokens in {dt:.2f} s (wave wall time; includes prefill) "
                  f"{'OK' if n == C * a.osl else 'SHORT'}", flush=True)


LINE = re.compile(r"DP(\d+).*Decode batch\. #running-req: (\d+), #token: (\d+), token usage: ([0-9.]+), "
                  r"cuda graph: (\w+), gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)")


def parse(a):
    per = {}
    for ln in open(a.log, errors="replace"):
        m = LINE.search(ln)
        if not m:
            continue
        dp, run_, _ntok, usage, graph, thr, _q = m.groups()
        run_, thr = int(run_), float(thr)
        if thr <= 0:
            continue
        per.setdefault(run_, []).append((int(dp), graph == "True", thr, float(usage)))
    keys = sorted(per) if not a.per_rank else [k for k in a.per_rank if k in per]
    for k in keys:
        v = per[k]
        steps = [k / thr * 1e3 for (_dp, _g, thr, _u) in v]
        thr = [t for (_dp, _g, t, _u) in v]
        graphs = {g for (_dp, g, _t, _u) in v}
        q = statistics.quantiles(steps, n=4) if len(steps) >= 4 else [min(steps), statistics.median(steps), max(steps)]
        print(f"running/rank {k}: {len(v)} intervals over {len({d for (d, *_r) in v})} ranks | decode step median "
              f"{statistics.median(steps):.2f} ms (IQR {q[0]:.2f}-{q[-1]:.2f}) | output tok/s per GPU median "
              f"{statistics.median(thr):.0f} | cuda graph {sorted(graphs)} | KV usage max {max(u for *_x, u in v):.2f}")
    if not keys:
        print("no Decode batch lines at the requested running counts; counts seen: "
              f"{sorted(per)[:20]}{' ...' if len(per) > 20 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--per-rank", type=int, required=True)
    r.add_argument("--ranks", type=int, required=True)
    r.add_argument("--osl", type=int, default=48)
    r.add_argument("--waves", type=int, default=2)
    r.add_argument("--batch", type=int, default=512)
    r.add_argument("--timeout", type=float, default=1800)
    p = sub.add_parser("parse")
    p.add_argument("log")
    p.add_argument("--per-rank", type=int, nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        parse(a)


if __name__ == "__main__":
    sys.exit(main())
