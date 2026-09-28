#!/usr/bin/env bash
# oss_audit.sh [repo_dir]: audit the LoPEP open-source tree before publication (CLAUDE.md "LoPEP release rules").
# Default target: $PSCRATCH/workspace/andrewy/lopep_release. Exit 1 on any finding.
#   1. provenance: no Claude/Anthropic/co-authorship, no usernames, no site paths, no Slurm accounts, no old names
#   2. no latency / throughput data anywhere (files or numbers with time/throughput units), no results/ directory
#   3. history: a single commit, authored by "LoPEP authors"
set -uo pipefail
R="${1:-$PSCRATCH/workspace/andrewy/lopep_release}"
cd "$R" || { echo "no such repo: $R"; exit 2; }
fail=0
hit() { echo "FAIL [$1]"; shift; printf '   %s\n' "$@" | head -12; fail=1; }

# 1. provenance strings (case-insensitive) in tracked text files and in the git history
PROV='claude|anthropic|co-authored-by|yufeid|andrewy|changchen|pscratch|/global/homes|/global/u1|libra ?x|moe_ep|moe-ep|m[0-9]{4}_g\b|--account m[0-9]|SLURM_ACCOUNT:-m'
out=$(git grep -n -I -i -E "$PROV" -- . 2>/dev/null | grep -v -E "^(LICENSE|NOTICE):" ); [ -n "$out" ] && hit "provenance strings in tracked files" "$out"
out=$(git log --all --format='%H %an <%ae> %cn <%ce>%n%B' | grep -i -E 'claude|anthropic|co-authored|yufeid|andrewy' ); [ -n "$out" ] && hit "provenance strings in git history/authors" "$out"

# 2. latency / throughput data
[ -d results ] && hit "results/ directory present" "$(ls results | head)"
out=$(git ls-files | grep -E '\.(csv|png|pdf|jsonl)$' ); [ -n "$out" ] && hit "data files tracked" "$out"
# numbers with time / throughput / speedup units anywhere in text files (docs, comments, scripts)
NUM='[0-9]+(\.[0-9]+)? ?(ms|us|µs|ns|tok/s|tokens/s|GB/s|MB/s|×|x faster|x slower|% (faster|slower|less|more|latency))\b'
out=$(git grep -n -I -E "$NUM" -- . ':!data/traces/' 2>/dev/null | grep -v -E 'asm volatile|globaltimer' ); [ -n "$out" ] && hit "latency/throughput numbers in text" "$out"

# 3. history shape
n=$(git rev-list --all --count); [ "$n" != "1" ] && hit "history is not a single commit" "$n commits"
a=$(git log -1 --format='%an <%ae>'); [ "$a" != "LoPEP authors <lopep@users.noreply.github.com>" ] && hit "unexpected author" "$a"

# 4. README must say where the measurements live
grep -q -i "measurements\|latency" README.md && grep -q -i "paper\|arxiv" README.md || hit "README lacks the pointer to the published measurements" "add: numbers are in the paper (Figure 9) / artifact"

[ $fail = 0 ] && echo "AUDIT PASS: $R ($(git ls-files | wc -l) files, 1 commit by $a)"
exit $fail
