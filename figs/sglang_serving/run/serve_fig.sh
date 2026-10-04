#!/bin/bash
# serve_fig.sh <jobid>: one dataset of the SGLang serving figure (handoff 55), on an allocation the caller holds.
# Configuration comes from the environment (run/fig_<dataset>.sh sets it):
#   DS                dataset tag (report: logs/sglang/jfig_${DS}_report.txt)
#   MODEL             model dir;  OURS_TREE / STOCK_TREE  lopep trees (launch scripts + env; ours also its binary)
#   CAL_D / CAL_P     decode / prefill calibrations (ours)
#   DEC_ARMS          decode arm order, e.g. "o1 s1 s2 o2" (o* = ours, s* = stock; the first arm must be ours: its pool
#                     becomes every later arm's --max-total-tokens)
#   DEC_SMAX DEC_MAXRR DEC_CTX DEC_CAP   decode server args; DEC_CAP = --max-total-tokens of the first (ours) arm
#   DEC_PROMPTS       decode prompt file;  DEC_PTS_A (WAVES_A waves) and DEC_PT_B (WAVES_B waves; skipped when the
#                     pool is below DEC_KV_NEED tokens per GPU)
#   TOK_ARMS          decode arms that also run the greedy token check
#   PRE_SMAX PRE_REP PRE_MAXRR PRE_CONC PRE_ORDER   per prefill point (space-separated, same length); PRE_ORDER items
#                     "so" (stock then ours) or "os";  PRE_PIN  prefill --max-total-tokens
#   PRE_XARGS         extra server args for BOTH prefill arms (front end only, e.g. --tokenizer-worker-num 8)
#   PRE_REQUIRE_FILL  1 = when the first arm of a point fails to start or its median prefill chunk is < 0.9 x SMAX,
#                     skip the point's remaining arms (the caller re-runs the pair with other settings)
#   PRE_CPU_SAMPLE    1 = sample the busiest processes on node 0 and the bench client during each prefill bench
# Stock = SGLang with its CUDA graphs off; ours = the lopep defaults of OURS_TREE (+ LOPEP_DEVICE_META=1, which undoes
# lib49's OURS_ENV pin). Everything else is identical between arms.
J=$1; TAG=jfig_${DS}${SUF:-}; R=$PSCRATCH/workspace/andrewy/logs/sglang/${TAG}_report.txt
source $PSCRATCH/workspace/andrewy/logs/sglang/lib49.sh
A=$PSCRATCH/workspace/andrewy; OT=$A/$OURS_TREE; ST=$A/$STOCK_TREE
# 16 MB decode fallback (user, 10-02): when the pool cannot hold DEC_PT_B with DEC_PROMPTS, run that point with shorter
# prompts instead of skipping it (decided once from the first ours arm's pool; stock is pinned to it, so every arm takes
# the same branch). Prompt + 48 output tokens per request: t48 ~105, t16 ~73, t8 ~65.
case $DS in
  235b16n) : ${DEC_PROMPTS_FB:=$HF_HOME/lcb/lcb_exec_eval_t16_x1.json}; : ${DEC_KV_NEED_FB:=$((2048 * 73))} ;;
  30b16n_40g) : ${DEC_PROMPTS_FB:=$HF_HOME/lcb/lcb_exec_eval_t8_x1.json}; : ${DEC_KV_NEED_FB:=$((4096 * 65))} ;;
esac
log "serve_fig $DS: job $J, $NN nodes ($RANKS GPUs), node0 $NODE, model $MODEL, ours $(md5sum $OT/python/lopep/lib/liblopep_cuda.so | cut -c1-12) $(git -C $OT log --oneline -1 | cut -c1-8)"
heap() { bash -lc "source $OT/env/perlmutter_sglang.sh >/dev/null 2>&1 && python -m lopep.heap --config $1/lopep_config.json --headroom 1.5" 2>/dev/null | tail -1; }
HEAP_D=$(heap $CAL_D); HEAP_P=$(heap $CAL_P)
[[ "$HEAP_D" =~ ^[0-9]+G$ && "$HEAP_P" =~ ^[0-9]+G$ ]] || { log "heap size failed: '$HEAP_D' '$HEAP_P'"; exit 1; }
log "heaps: decode $HEAP_D prefill $HEAP_P; calibs $CAL_D $CAL_P"
ours_env() { T28=$OT; OURS_XENV="LOPEP_TIMING=0 NVSHMEM_SYMMETRIC_SIZE=$1 PYTHONPATH=$OT/integrations/sglang:$OT/python LOPEP_DEVICE_META=1"; }
XD="--decode-log-interval 8 --enable-dp-lm-head"

PHASES=${PHASES:-decode prefill}   # supplement runs: PHASES=prefill (with SUF and PRE_* overrides)
# ---- decode ----
PIN=""
[[ " $PHASES " == *" decode "* ]] || DEC_ARMS=""
for lab in $DEC_ARMS; do
  name=d30_$lab
  if [[ $lab == o* ]]; then
    ours_env $HEAP_D
    serve ours $name $MODEL $DEC_SMAX $DEC_MAXRR $DEC_CTX $CAL_D $XD --max-total-tokens ${PIN:-$DEC_CAP} || continue
  else
    [ -n "$PIN" ] || { log "$name: no pin from an ours arm yet; skipped"; continue; }
    T28=$ST; serve baseline $name $MODEL $DEC_SMAX $DEC_MAXRR $DEC_CTX - $XD --max-total-tokens $PIN || continue
  fi
  p=$(pool $name); log "$name: pool $p tokens per GPU"
  [ -z "$PIN" ] && { PIN=$p; log "decode pin $PIN (every later arm)"; }
  WAVES=$WAVES_A waves $name $MODEL $DEC_PROMPTS "$DEC_PTS_A"
  if [ -n "$DEC_PT_B" ]; then
    if [ "${p:-0}" -ge "$DEC_KV_NEED" ]; then WAVES=$WAVES_B waves $name $MODEL $DEC_PROMPTS "$DEC_PT_B"
    elif [ -n "${DEC_PROMPTS_FB:-}" ] && [ "${p:-0}" -ge "${DEC_KV_NEED_FB:-0}" ]; then
      log "$name: point $DEC_PT_B FALLBACK prompts $(basename $DEC_PROMPTS_FB): pool $p < $DEC_KV_NEED, needs $DEC_KV_NEED_FB"
      WAVES=$WAVES_B waves $name $MODEL $DEC_PROMPTS_FB "$DEC_PT_B"
    else log "$name: point $DEC_PT_B INFEASIBLE: pool $p < $DEC_KV_NEED tokens per GPU (fallback needs ${DEC_KV_NEED_FB:-n/a})"; fi
  fi
  case " $TOK_ARMS " in *" $lab "*)
    timeout 900 $PY -m lopep_sglang.check_tokens run --url http://$NODE:$PORT --out $L/tok_${TAG}_$lab.json --max-new 64 --parallel 16 >> $R 2>&1
    log "$name: token check written";; esac
done

# ---- prefill ----
set -- ${PRE_CALS:-}; cals=("$@")   # optional per-point prefill calibration ("-" = CAL_P)
set -- $PRE_REP; reps=("$@"); set -- $PRE_MAXRR; mrrs=("$@"); set -- $PRE_CONC; concs=("$@"); set -- $PRE_ORDER; ords=("$@")
# the point index must not be "i": lib49's serve() overwrites a global i in its health-check loop
[[ " $PHASES " == *" prefill "* ]] || PRE_SMAX=""
pidx=0
for smax in $PRE_SMAX; do
  rep=${reps[$pidx]}; mrr=${mrrs[$pidx]}; conc=${concs[$pidx]}; ord=${ords[$pidx]}; calp=${cals[$pidx]:--}; pidx=$((pidx + 1))
  [ "$calp" = - ] && calp=$CAL_P; hp=$HEAP_P; [ "$calp" != "$CAL_P" ] && hp=$(heap $calp)
  f=$HF_HOME/lcb/lcb_exec_eval_x${rep}.json; np=$(python3 -c "import json;print(len(json.load(open('$f'))))")
  for k in $(echo $ord | fold -w1); do
    if [ $k = o ]; then
      name=p30_kf_s$smax; ours_env $hp; log "$name: calibration $(basename $calp) heap $hp"
      serve ours $name $MODEL $smax $mrr 2048 $calp --max-total-tokens $PRE_PIN ${PRE_XARGS:-} || { [ "${PRE_REQUIRE_FILL:-0}" = 1 ] && { log "$name: FILL FAIL (server did not start); rest of point $smax skipped"; break; }; continue; }
    else
      name=p30_baseA_s$smax; T28=$ST
      serve baseline $name $MODEL $smax $mrr 2048 - --max-total-tokens $PRE_PIN ${PRE_XARGS:-} || { [ "${PRE_REQUIRE_FILL:-0}" = 1 ] && { log "$name: FILL FAIL (server did not start); rest of point $smax skipped"; break; }; continue; }
    fi
    log "$name: pool $(pool $name) tokens per GPU (pinned $PRE_PIN)${PRE_XARGS:+ front end: $PRE_XARGS}"
    csp=""; [ "${PRE_CPU_SAMPLE:-0}" = 1 ] && { $(dirname "$0")/cpu_sample.sh $NODE $L/cpu_${TAG}_$name.log & csp=$!; }
    bench $name lcbp $rep $np $((conc * RANKS))
    [ -n "$csp" ] && kill $csp 2>/dev/null
    med=$(grep -o 'Prefill batch.*#new-token: [0-9]*' $L/server_${TAG}_$name.log | grep -o '[0-9]*$' | sort -n | awk '{a[NR]=$1} END {print (NR ? a[int((NR+1)/2)] : 0)}')
    log "$name: prefill chunk tokens per GPU median $med (SMAX $smax, np $np, conc $((conc * RANKS)))"
    tail_timing $name 4
    if [ "${PRE_REQUIRE_FILL:-0}" = 1 ] && [ $((med * 10)) -lt $((smax * 9)) ]; then
      log "$name: FILL FAIL median $med < 0.9 x $smax; rest of point $smax skipped"; break; fi
  done
done
clean
# token agreement: every token-checked arm against the first stock arm
first=""; for lab in $TOK_ARMS; do [[ $lab == s* ]] && { first=$lab; break; }; done
for lab in $TOK_ARMS; do [ "$lab" = "$first" ] && continue
  [ -f $L/tok_${TAG}_$first.json ] && [ -f $L/tok_${TAG}_$lab.json ] && \
    log "tokens $first vs $lab: $($PY -m lopep_sglang.check_tokens compare $L/tok_${TAG}_$first.json $L/tok_${TAG}_$lab.json 2>&1 | tail -2 | tr '\n' ' ')"
done
log "serve_fig $DS DONE"
