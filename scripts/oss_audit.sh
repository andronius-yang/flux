#!/usr/bin/env bash
# oss_audit.sh [repo_dir]: audit the Zepp open-source repository before publication (CLAUDE.md "Zepp release rules").
# Default target: $PSCRATCH/workspace/andrewy/zepp. Exit 1 on any finding. Checks every branch tip and every commit.
#   1. provenance: no Claude/Anthropic/co-authorship, no usernames, no site paths, no Slurm accounts, no old names
#   2. no latency / throughput data anywhere (files or numbers with time/throughput units), no results/ directory
#   3. history: one release root shared by both branches, no refs besides the branches / origin / v* tags, one author identity
set -uo pipefail
R="${1:-$PSCRATCH/workspace/andrewy/zepp}"
cd "$R" || { echo "no such repo: $R"; exit 2; }
AUTHOR="Andrew Yang <androniusyang@gmail.com>"
fail=0
hit() { echo "FAIL [$1]"; shift; printf '   %s\n' "$@" | head -12; fail=1; }
TIPS=$(git for-each-ref --format="%(refname:short)" refs/heads | tr "\n" " " | sed "s/ $//")
ALL=$(git rev-list --all)

# 1. provenance strings (case-insensitive), in every branch tip and in every commit of the history
PROV='claude|anthropic|co-authored-by|yufeid|andrewy|changchen|pscratch|/global/homes|/global/u1|m[0-9]{4}_g\b|--account m[0-9]|SLURM_ACCOUNT:-m'
OLD='lopep|moe_ep|moe-ep|libra ?x'   # old names: no file is exempt (NOTICE included)
for b in $TIPS; do
  out=$(git grep -n -I -i -E "$PROV" "$b" -- . 2>/dev/null | grep -v -E "^$b:(LICENSE|NOTICE):"); [ -n "$out" ] && hit "provenance strings in $b" "$out"
  out=$(git grep -n -I -i -E "$OLD" "$b" -- . 2>/dev/null); [ -n "$out" ] && hit "old names in $b" "$out"
  out=$(git ls-tree -r --name-only "$b" | grep -i -E "$OLD|claude"); [ -n "$out" ] && hit "old names / claude in paths of $b" "$out"
done
out=$(git grep -l -a -i -E 'claude|anthropic' $ALL -- . 2>/dev/null); [ -n "$out" ] && hit "claude/anthropic in a file of some commit" "$out"
out=$(git grep -l -I -i -E "$OLD" $ALL -- . 2>/dev/null); [ -n "$out" ] && hit "old names in a file of some commit" "$out"
out=$(git log --all --format='%H %an <%ae> %cn <%ce>%n%B' | grep -i -E "claude|anthropic|co-authored|yufeid|andrewy|$OLD"); [ -n "$out" ] && hit "provenance strings in git history/authors" "$out"

# 2. latency / throughput data, in every branch tip
NUM='[0-9]+(\.[0-9]+)? ?(ms|us|µs|ns|tok/s|tokens/s|GB/s|MB/s|×|x faster|x slower|% (faster|slower|less|more|latency))\b'
for b in $TIPS; do
  out=$(git ls-tree -r --name-only "$b" | grep -E '^results/|\.(csv|png|pdf|jsonl)$'); [ -n "$out" ] && hit "data files / results/ in $b" "$out"
  out=$(git grep -n -I -E "$NUM" "$b" -- . ':!data/traces/' 2>/dev/null | grep -v -E 'asm volatile|globaltimer'); [ -n "$out" ] && hit "latency/throughput numbers in $b" "$out"
  git show "$b:README.md" 2>/dev/null | grep -q -i "measurements\|latency" && git show "$b:README.md" | grep -q -i "paper\|arxiv" \
    || hit "README of $b lacks the pointer to the published measurements" "add: numbers are in the paper"
done

# 3. history shape and identity
[ "$TIPS" = "main sglang-dev" ] || hit "branches are not exactly main + sglang-dev" "$TIPS"
other=$(git for-each-ref --format='%(refname)' | grep -v -E '^refs/(heads|remotes/origin)/(main|sglang-dev|HEAD)$|^refs/tags/v[0-9]'); [ -n "$other" ] && hit "refs besides the two branches (and origin, v* tags)" "$other"
# published 2026-10-05, re-released 2026-10-07 as fresh single commits: history grows by ordinary commits; both branches must keep
# the release root, and sglang-dev must contain main's first commit
root=$(git rev-list --max-parents=0 main 2>/dev/null)
[ "$(echo $root | wc -w)" = "1" ] || hit "main does not have exactly one root commit" "$root"
git merge-base --is-ancestor "$root" sglang-dev 2>/dev/null || hit "sglang-dev does not descend from the release root" "$root"
out=$(git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | grep -v -x -F "$AUTHOR"); [ -n "$out" ] && hit "unexpected author/committer" "$out"
[ -n "$(git remote)" ] && echo "note: remotes configured: $(git remote -v | head -2 | tr '\n' ' ')"

[ $fail = 0 ] && echo "AUDIT PASS: $R (branches: $TIPS; $(echo $ALL | wc -w) commits; every commit by $AUTHOR)"
exit $fail
