#!/bin/bash
# cpu_sample.sh <node> <out>: every 15 s until killed, the busiest processes on the server's node 0 (ssh; top's second
# frame = %CPU over 2 s) and the bench client on the driver's side (lifetime average) -- front-end bottleneck diagnosis.
while true; do
  { echo "== $(date +%T)"
    timeout 12 ssh -o BatchMode=yes -o StrictHostKeyChecking=no $1 "top -b -d 2 -n 2 -w 200 -o %CPU | awk '/^top -/{n++} n==2' | sed -n 7,18p" 2>&1
    ps -eo pid,pcpu,rss,args | grep "[b]ench_serving" | cut -c1-160; } >> $2
  sleep 15
done
