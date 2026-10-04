#!/bin/bash
# rerun_a.sh: dataset A (30b4n) re-measurement, user directive 10-03: new decode 1 MB numbers and correct (chunks full)
# 16 MB prefill numbers, on ONE 4n 80 GB interactive allocation, released at the end. Reports jfig_30b4n_r2*; a later
# report replaces the earlier values of the same point in build_figure_src.py.
#   1. decode 1 MB (256 running per GPU): ABBA ours/stock/stock/ours, 4 waves per arm (A had 2), on a 1 MB-sized server
#      = the plan-10 sizing (MAXRR 1024 per GPU, KV pool pinned to 143794 tokens per GPU, as in R7/R8 where 4n 1 MiB
#      measured 1.04x on 40 GB); A served 1/4/16 MB from one server sized for 16 MB (MAXRR 4096, pool 479759).
#   2. prefill 16 MB (SMAX 4096, lcbp4k calibration): A's ours chunks were half full because the front end, not the
#      server, capped the request rate (~266 req/s, scheduler queue empty, ~3.5 s of each request outside the scheduler).
#      Pair 1: SGLang's multi-tokenizer front end (--tokenizer-worker-num 8) for BOTH arms, 64 per GPU, ours first; stock
#      only runs if ours' median chunk is >= 0.9 x 4096. Pair 2 (only if pair 1 fails): 128 per GPU (MAXRR 256), with the
#      multi-tokenizer front end if it started, else the plain one. KV pinned 100000 per GPU for both arms (A: 50000).
A=$PSCRATCH/workspace/andrewy; L=$A/logs/p55; S=$A/logs/sglang; EV=$L/fig.log; HERE=$(cd "$(dirname "$0")" && pwd)
salloc --no-shell -A m5350_g -q interactive -C "gpu&hbm80g" -N 4 --gpus-per-node=4 -t 60 -J p55_30b4n_r2 > $L/salloc_p55_30b4n_r2.log 2>&1 < /dev/null
J=$(grep -o "Granted job allocation [0-9]*" $L/salloc_p55_30b4n_r2.log | awk '{print $4}')
[ -n "$J" ] || { echo "$(date +%T) 30b4n_r2: salloc failed: $(tail -1 $L/salloc_p55_30b4n_r2.log)" >> $EV; exit 1; }
until [ "$(squeue -j $J -h -o %T)" = RUNNING ]; do sleep 10; done
echo "$(date +%T) 30b4n_r2: job $J running on $(squeue -j $J -h -o %N)" >> $EV

# 1. decode 1 MB
env SUF=_r2d PHASES=decode DEC_ARMS_O="o1 s1 s2 o2" DEC_MAXRR_O=1024 DEC_CAP_O=143794 DEC_PTS_A_O=256 WAVES_A_O=4 \
  DEC_PT_B_O= TOK_ARMS_O= $HERE/fig_run_on.sh 30b4n $J

# 2. prefill 16 MB
P16="PHASES=prefill PRE_SMAX_O=4096 PRE_REP_O=64 PRE_ORDER_O=os PRE_CALS_O=$S/calib_30b4n_lcbp4k_overlap_s1 PRE_PIN_O=100000 PRE_CPU_SAMPLE=1"
env $P16 SUF=_r2p PRE_MAXRR_O=128 PRE_CONC_O=64 PRE_XARGS="--tokenizer-worker-num 8" PRE_REQUIRE_FILL=1 HEALTH_TRIES=48 \
  $HERE/fig_run_on.sh 30b4n $J
R=$S/jfig_30b4n_r2p_report.txt
if grep -q "p30_kf_s4096: FILL FAIL (server did not start)" $R; then X=""
elif grep -q "p30_kf_s4096: FILL FAIL median" $R; then X="--tokenizer-worker-num 8"
else X=SKIP; fi
if [ "$X" != SKIP ]; then
  echo "$(date +%T) 30b4n_r2: 16 MB prefill pair 1 failed ($(grep -h 'FILL FAIL' $R | cut -c10-120)); pair 2 at 128 per GPU, front end '${X:-plain}'" >> $EV
  env $P16 SUF=_r2q PRE_MAXRR_O=256 PRE_CONC_O=128 PRE_XARGS="$X" PRE_REQUIRE_FILL=0 $HERE/fig_run_on.sh 30b4n $J
fi
scancel $J; echo "$(date +%T) 30b4n_r2: scancel $J" >> $EV
