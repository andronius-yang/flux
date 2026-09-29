"""48_lcb_workload.py [--parquet P] [--out DIR] [--repeat N]

LiveCodeBench code-execution serving workload (handoff 48): the 479 problems of
livecodebench/execution-v2 (the problem set behind the paper's routing traces, "Patterns behind
Chaos", arXiv 2510.05497), split by contest date into a HISTORY half (older contests: the traffic
the placement is calibrated on) and an EVAL half (newer contests: the traffic that is benchmarked),
so the placement predicts future requests of the same dataset from past ones, as in the paper
(placement from earlier demand, evaluation on later demand) without ever seeing the benchmark's
own tokens.

Prompt: the official LiveCodeBench code-execution direct-output prompt (two worked examples, then
the problem's function and input). The traces' own prompt wording is not published with the trace
files; this is the benchmark's standard form. The chat template is applied by the client
(--apply-chat-template) identically for calibration and benchmark.

Output: ShareGPT-format JSON (conversations: human prompt, gpt reference answer), which
`python -m sglang.bench_serving --dataset-name sharegpt --dataset-path ...` reads; `--repeat N`
writes N copies of each split (servers then run with --disable-radix-cache on every arm, so a
repeated prompt is prefilled again rather than served from the prefix cache).
"""
import argparse
import json
import os

import pandas as pd

TEMPLATE = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
[/PYTHON]
[ANSWER]
assert repeatNumber(number = 17) == 17
[/ANSWER]

[PYTHON]
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
[/PYTHON]
[ANSWER]
assert addCharacterA(string = "x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[ANSWER]
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.expandvars("$PSCRATCH/workspace/andrewy/caches/hf/lcb")
    ap.add_argument("--parquet", default=os.path.join(root, "execution-v2.parquet"))
    ap.add_argument("--out", default=root)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet).sort_values(["contest_date", "question_id", "id"]).reset_index(drop=True)
    half = len(df) // 2
    splits = {"history": df.iloc[:half], "eval": df.iloc[half:]}
    for name, part in splits.items():
        rows = []
        for _, r in part.iterrows():
            prompt = TEMPLATE.format(code=r["code"].strip(), input=r["input"].strip())
            answer = f"assert {r['input'].strip()} == {r['output'].strip()}\n[/ANSWER]"
            rows.append({"id": r["id"], "conversations": [{"from": "human", "value": prompt},
                                                           {"from": "gpt", "value": answer}]})
        out = os.path.join(a.out, f"lcb_exec_{name}_x{a.repeat}.json")
        with open(out, "w") as f:
            json.dump(rows * a.repeat, f)
        d0, d1 = part["contest_date"].min(), part["contest_date"].max()
        print(f"{name}: {len(part)} problems ({d0:%Y-%m-%d} .. {d1:%Y-%m-%d}), x{a.repeat} -> {out}")


if __name__ == "__main__":
    main()
