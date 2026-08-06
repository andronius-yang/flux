#!/usr/bin/env python3
"""Partial fetcher for the "Patterns behind Chaos" MoE expert-selection traces.

Downloads ONLY the requested model/bench/subject subtrees of the gated HF
dataset core12345/MoE_expert_selection_trace (the full dataset is ~199 GB)
into --traces-root, mirroring the repo-relative layout:

    <traces_root>/Qwen/Qwen3-235B-A22B-FP8/<bench>/<subject>/<N>.json

and writes a pool.manifest.json per subject dir: the sorted per-file sha256
list plus a combined `pool_sha` — the content fingerprint that the `trace`
matrix family folds into matrix identity (see gen_trace_routing.py). File
numbering in the dataset is NOT 0-based (e.g. Qwen mmlu/college_mathematics
starts at 1096), so enumeration always comes from the downloaded tree, never
from range().

Needs `huggingface_hub` and an HF token that has accepted the dataset's gated
terms (both present in the andrewy-fast conda env; token in
~/.cache/huggingface/token). Idempotent: present files are kept
(snapshot_download resumes), the manifest is always recomputed from disk.
"""

import argparse
import hashlib
import json
import os
import time

REPO_ID = "core12345/MoE_expert_selection_trace"

# model aliases (family params use the short name) -> repo path prefix
MODEL_PREFIXES = {
    "Qwen3-235B": "Qwen/Qwen3-235B-A22B-FP8",
    "DeepSeek-R1": "cognitivecomputations/DeepSeek-R1-AWQ",
    "Llama4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Kimi-K2": "moonshotai/Kimi-K2-Thinking",
}

DEFAULT_TRACES_ROOT = "${PSCRATCH}/workspace/andrewy/moe_traces"


def subject_dir(traces_root, model, bench, subject):
    return os.path.join(traces_root, MODEL_PREFIXES[model], bench, subject)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_pool_manifest(sdir, model, bench, subject, revision=None):
    """Recompute the manifest from the files on disk. Returns the manifest."""
    names = sorted(
        (n for n in os.listdir(sdir) if n.endswith(".json") and n != "pool.manifest.json"),
        key=lambda n: (len(n), n),  # numeric-ish order for readability
    )
    if not names:
        raise SystemExit(f"no trace .json files in {sdir}")
    files = []
    for n in names:
        p = os.path.join(sdir, n)
        files.append({"name": n, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    pool_sha = hashlib.sha256(
        "\n".join(f"{f['name']} {f['sha256']}" for f in files).encode()
    ).hexdigest()
    manifest = {
        "repo": REPO_ID,
        "revision": revision,
        "model": model,
        "model_prefix": MODEL_PREFIXES[model],
        "bench": bench,
        "subject": subject,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "pool_sha": pool_sha,
        "files": files,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mpath = os.path.join(sdir, "pool.manifest.json")
    with open(mpath + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.rename(mpath + ".tmp", mpath)
    return manifest


def fetch_subject(traces_root, model, bench, subject):
    from huggingface_hub import HfApi, snapshot_download

    prefix = f"{MODEL_PREFIXES[model]}/{bench}/{subject}"
    revision = HfApi().dataset_info(REPO_ID).sha
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{prefix}/*.json"],
        local_dir=traces_root,
    )
    sdir = subject_dir(traces_root, model, bench, subject)
    manifest = write_pool_manifest(sdir, model, bench, subject, revision=revision)
    print(
        f"{prefix}: {manifest['n_files']} files,"
        f" {manifest['total_bytes'] / (1 << 20):.1f} MiB, pool_sha {manifest['pool_sha'][:12]}"
    )
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--traces-root",
        default=os.path.expandvars(DEFAULT_TRACES_ROOT),
        help="platform traces_root (perlmutter.yaml); repo layout is mirrored under it",
    )
    ap.add_argument("--model", default="Qwen3-235B", choices=sorted(MODEL_PREFIXES))
    ap.add_argument(
        "--pool",
        action="append",
        required=True,
        help="bench/subject to fetch, e.g. mmlu/college_mathematics (repeatable)",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="recompute pool.manifest.json from files already on disk, no download",
    )
    args = ap.parse_args()
    if "$" in args.traces_root:
        raise SystemExit(f"unresolved traces root: {args.traces_root}")

    for pool in args.pool:
        bench, _, subject = pool.partition("/")
        if not subject:
            raise SystemExit(f"--pool must be bench/subject, got: {pool}")
        if args.manifest_only:
            sdir = subject_dir(args.traces_root, args.model, bench, subject)
            m = write_pool_manifest(sdir, args.model, bench, subject)
            print(f"{pool}: {m['n_files']} files, pool_sha {m['pool_sha'][:12]}")
        else:
            fetch_subject(args.traces_root, args.model, bench, subject)


if __name__ == "__main__":
    main()
