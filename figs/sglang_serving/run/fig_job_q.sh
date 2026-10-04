#!/bin/bash
# fig_job.sh <dataset>: one job of the SGLang serving figure campaign (handoff 55): allocate (regular QOS, salloc
# --no-shell), optional quick gate, serve_fig.sh under the stall watchdog, copy the small artifacts and a manifest into
# figs/sglang_serving/data/<dataset>/, release. Datasets:
#   30b4n       4 nodes,  80 GB, Qwen3-30B,  60 min: decode ours/stock/stock/ours, prefill once per arm
#   235b16n    16 nodes,  80 GB, Qwen3-235B, 75 min: 235B-shape gate, decode ours/stock, prefill once per arm
#   30b16n_40g 16 nodes,  40 GB, Qwen3-30B,  30 min (fallback): decode ours/stock with 16-token prompts, prefill once
# Events: logs/p55/fig.log. Raw server / wave / bench logs stay in logs/sglang (log-root rule).
DS=$1
A=$PSCRATCH/workspace/andrewy; L=$A/logs/p55; S=$A/logs/sglang; EV=$L/fig.log
HERE=$(cd "$(dirname "$0")/.." && pwd); OUT=$HERE/data/$DS
M30=$A/models/Qwen3-30B-A3B; M235=$A/models/Qwen3-235B-A22B; LCB=$A/caches/hf/lcb
export OURS_TREE=lopep_p10f STOCK_TREE=lopep_d1rt DS
case $DS in
  30b4n)
    NN=4; CONS="gpu&hbm80g"; WT=60; GATE=0
    export MODEL=$M30 CAL_D=$S/calib_30b4n_lcbtd_overlap_s1 CAL_P=$S/calib_30b4n_lcbp_overlap_s1
    export DEC_ARMS="o1 s1 s2 o2" DEC_SMAX=4096 DEC_MAXRR=4096 DEC_CTX=256 DEC_CAP=480000
    export DEC_PROMPTS=$LCB/lcb_exec_eval_t48_x1.json DEC_PTS_A="256 1024" WAVES_A=2 DEC_PT_B=4096 WAVES_B=1
    export DEC_KV_NEED=$((4096 * 105)) TOK_ARMS="o1 s1 s2"
    export PRE_SMAX="256 1024 4096" PRE_REP="4 16 64" PRE_MAXRR="32 32 64" PRE_CONC="16 16 32" PRE_ORDER="so so so" PRE_PIN=50000 ;;
  235b16n)
    NN=16; CONS="gpu&hbm80g"; WT=75; GATE=1
    export MODEL=$M235 CAL_D=$S/calib_235b16n_lcbp_overlap_s1 CAL_P=$S/calib_235b16n_lcbp_overlap_s1
    export DEC_ARMS="o1 s1" DEC_SMAX=2048 DEC_MAXRR=2048 DEC_CTX=256 DEC_CAP=240000
    export DEC_PROMPTS=$LCB/lcb_exec_eval_t48_x1.json DEC_PTS_A="128 512" WAVES_A=2 DEC_PT_B=2048 WAVES_B=1
    export DEC_KV_NEED=$((2048 * 105)) TOK_ARMS="o1 s1"
    export PRE_SMAX="128 512 2048" PRE_REP="8 32 128" PRE_MAXRR="32 32 32" PRE_CONC="16 16 16" PRE_ORDER="so os so" PRE_PIN=50000 ;;
  30b16n_40g)
    NN=16; CONS="gpu&hbm40g"; WT=30; GATE=0
    export MODEL=$M30 CAL_D=$S/calib_30b16n_lcbtd_overlap_s1 CAL_P=$S/calib_30b16n_lcbp_overlap_s1
    export DEC_ARMS="o1 s1" DEC_SMAX=4096 DEC_MAXRR=4096 DEC_CTX=256 DEC_CAP=400000
    export DEC_PROMPTS=$LCB/lcb_exec_eval_t16_x1.json DEC_PTS_A="256 1024" WAVES_A=1 DEC_PT_B=4096 WAVES_B=1
    export DEC_KV_NEED=$((4096 * 73)) TOK_ARMS=""
    export PRE_SMAX="256 1024 4096" PRE_REP="16 64 128" PRE_MAXRR="32 32 64" PRE_CONC="16 16 32" PRE_ORDER="so so so" PRE_PIN=50000 ;;
  *) echo "unknown dataset $DS"; exit 2 ;;
esac
for c in $CAL_D $CAL_P; do [ -f $c/lopep_config.json ] || { echo "$(date +%T) $DS: missing calibration $c" >> $EV; exit 1; }; done
[ -f $DEC_PROMPTS ] || { echo "$(date +%T) $DS: missing $DEC_PROMPTS" >> $EV; exit 1; }
[ -n "${DRY:-}" ] && { echo "$DS: NN=$NN $CONS ${WT} min, config OK"; exit 0; }

salloc --no-shell -A m5350_g -q ${QOS:-regular} -C "$CONS" -N $NN --gpus-per-node=4 -t $WT -J p55_$DS > $L/salloc_p55_$DS.log 2>&1 < /dev/null
J=$(grep -o "Granted job allocation [0-9]*" $L/salloc_p55_$DS.log | awk '{print $4}')
[ -n "$J" ] || { echo "$(date +%T) $DS: salloc failed: $(tail -1 $L/salloc_p55_$DS.log)" >> $EV; exit 1; }
until [ "$(squeue -j $J -h -o %T)" = RUNNING ]; do sleep 10; done
T0=$(date +%s); echo "$(date +%T) $DS: job $J running on $(squeue -j $J -h -o %N)" >> $EV
P50=$A/logs/p50
if [ $GATE = 1 ]; then
  # plan-10 code has never run the 235B shape: one flip gate (2 layers); abort on a hang or a real numerical error
  TREE=$A/$OURS_TREE TO=360 $P50/hang_probe3.sh $J fig_${DS}_gate "LOPEP_CHECK_DETAIL=1" 330 --shape qwen3-235b --layers 2 \
    --smax 512 --ref 1 --swap 1 --caps-scale 2.0 --token-mode cycle --steps 12 --skew-flip 1 --flip-every 2 --graphs 1
  g=$P50/hang_fig_${DS}_gate.log; mx=$(grep -o "max_err [0-9.]*" $g | awk '{if ($2 > m) m = $2} END {print m + 0}')
  rc=$(grep "fig_${DS}_gate rc=" $P50/hang_probe.log | tail -1 | grep -o "rc=[0-9]*")
  echo "$(date +%T) $DS gate: $(grep -h '\[check\] PASS\|\[check\] FAIL' $g | head -1 | cut -c1-90) | max err $mx | $rc" >> $EV
  if [ "$rc" = "rc=124" ] || ! awk -v m=$mx 'BEGIN {exit !(m < 0.02)}'; then
    echo "$(date +%T) $DS: gate hung or exceeded max err 0.02: serving NOT run" >> $EV; scancel $J; exit 1; fi
fi
$HERE/run/serve_fig.sh $J &
DP=$!; $P50/stall_watch2.sh $J $S/jfig_${DS}_report.txt $DP > $L/stall_watch_$DS.log 2>&1 &
wait $DP
TAG=jfig_$DS; R=$S/${TAG}_report.txt
echo "$(date +%T) $DS decode: $(awk '/arm |d30_[a-z0-9]*: pool/{} /decode step median/{match($0,/running\/rank [0-9]+/); r=substr($0,RSTART+13,RLENGTH-13); match($0,/median [0-9.]+ ms/); printf "%s %s | ", r, substr($0,RSTART+7,RLENGTH-10)} /INFEASIBLE|tracebacks [1-9]|SERVER FAILED/{printf "!! %s | ", $0}' $R)" >> $EV
echo "$(date +%T) $DS prefill: $(grep -h 'Input token throughput' $R | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-400)" >> $EV
# copy the small artifacts and the manifest into the research tree
mkdir -p $OUT
cp $R $OUT/ 2>/dev/null; cp $S/wave_${TAG}_*.log $S/b_${TAG}_*.out $S/pbench_${TAG}_*.jsonl $S/tok_${TAG}_*.json $OUT/ 2>/dev/null
[ $GATE = 1 ] && cp $P50/hang_fig_${DS}_gate.log $OUT/ 2>/dev/null
{
  echo "dataset $DS"; echo "job $J nodes $NN constraint $CONS walltime ${WT}m nodelist $(squeue -j $J -h -o %N 2>/dev/null)"
  echo "start $(date -d @$T0 '+%F %T') end $(date '+%F %T')"
  echo "ours tree $OURS_TREE $(git -C $A/$OURS_TREE log --oneline -1) liblopep md5 $(md5sum $A/$OURS_TREE/python/lopep/lib/liblopep_cuda.so | cut -c1-32)"
  echo "stock tree $STOCK_TREE (launch scripts only; SGLang graphs off)"
  echo "sglang $(git -C $A/sglang log --oneline -1) local changes: $(git -C $A/sglang diff --stat | tail -1)"
  echo "lib49 md5 $(md5sum $S/lib49.sh | cut -c1-32) bench1 md5 $(md5sum $S/bench1.sh | cut -c1-32)"
  for c in $CAL_D $CAL_P; do echo "calib $c config md5 $(md5sum $c/lopep_config.json | cut -c1-32)"; done
  env | grep -E "^(MODEL|DEC_|PRE_|WAVES_|TOK_ARMS|CAL_)" | sort
} > $OUT/manifest.txt
scancel $J; echo "$(date +%T) $DS: scancel $J; artifacts in $OUT" >> $EV
