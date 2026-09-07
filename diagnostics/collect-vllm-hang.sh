#!/usr/bin/env bash
# collect-vllm-hang.sh - one-shot evidence collection for a frozen vLLM cluster (TP=2).
#
# Run this WHILE the processes are hung: they stay in that state indefinitely, so
# there is no rush, and everything below is read-only.
#
# It captures, on both ranks:
#   1. GPU counters (utilisation, SM clocks, power) and which process holds the memory
#   2. the worker PIDs and each rank's process state
#   3. py-spy native dump of each worker's Python threads
#   4. a /proc/<pid>/task scan, twice, 5 s apart -> which thread is actually burning CPU
#   5. pystack --native-all -> stacks of ALL threads, including non-Python ones
#      (py-spy alone does NOT show the spinning NCCL progress thread)
#   6. vLLM logs from both ranks, container config, RDMA/RoCE error counters
#
# Then it prints a summary: the spinning thread per rank, its native stack, and the
# crash signature from the head log.
#
# Requirements: docker on both hosts, passwordless ssh to the worker host, and
# py-spy inside the container. pystack is installed on the fly if missing.
#
# Usage:
#   WORKER=<worker-host> ./collect-vllm-hang.sh [output-dir]
#
# Optional environment:
#   CTR=<container name>   default: vllm_node
#   IFACE=<nic name>       NIC used by NCCL; if set, ethtool stats are collected

set -uo pipefail
WORKER="${WORKER:?set WORKER to the worker node hostname or address}"
CTR="${CTR:-vllm_node}"
IFACE="${IFACE:-}"
OUT="${1:-./vllm-hang-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
echo "=> output: $OUT"

# Scripts are piped over stdin, so nothing has to be quote-escaped.
in_head() { docker exec -i "$CTR" bash -s 2>&1; }
in_work() { timeout 300 ssh -o BatchMode=yes "$WORKER" "docker exec -i $CTR bash -s" 2>&1; }
on_work() { timeout 120 ssh -o BatchMode=yes "$WORKER" bash -s 2>&1; }

# --- 1. GPU counters ---
Q="utilization.gpu,utilization.memory,clocks.current.sm,power.draw,temperature.gpu"
{ echo "== head (rank 0) =="
  nvidia-smi --query-gpu=$Q --format=csv 2>&1
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
  echo "== worker (rank 1) =="
  on_work <<EOF
nvidia-smi --query-gpu=$Q --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
EOF
} > "$OUT/1-gpu.txt"; echo "  [1/6] gpu"

# --- 2. worker PIDs and process state ---
# A rank that already exited leaves a zombie; py-spy and pystack cannot attach to it,
# so record the state and skip those dumps rather than emitting a raw attach error.
PID0=$(in_head <<'EOF' | tr -d ' \r'
ps -eo pid,comm | awk '/VLLM::Worker_TP/{print $1; exit}'
EOF
)
PID1=$(in_work <<'EOF' | tr -d ' \r'
ps -eo pid,comm | awk '/VLLM::Worker_TP/{print $1; exit}'
EOF
)
ST0=$(in_head <<EOF
[ -d /proc/$PID0 ] && sed -e 's/.*) //' -e 's/ .*//' /proc/$PID0/stat || echo gone
EOF
)
ST1=$(in_work <<EOF
[ -d /proc/$PID1 ] && sed -e 's/.*) //' -e 's/ .*//' /proc/$PID1/stat || echo gone
EOF
)
ST0=$(echo "$ST0" | tr -d ' \r'); ST1=$(echo "$ST1" | tr -d ' \r')
{ echo "rank0 pid=$PID0 state=$ST0"; echo "rank1 pid=$PID1 state=$ST1"; } | tee "$OUT/2-pids.txt"
# R = running, S = sleeping, Z = zombie (rank already exited), gone = no such process
attachable() { case "$1" in Z|gone|"") return 1 ;; *) return 0 ;; esac; }

# --- 3. py-spy native dump ---
if attachable "$ST0"; then
  in_head > "$OUT/3-pyspy-rank0.txt" <<EOF
py-spy dump --pid $PID0 --native
EOF
else
  echo "rank 0 process state=$ST0 - cannot attach (expected once that rank has exited)" > "$OUT/3-pyspy-rank0.txt"
fi
if attachable "$ST1"; then
  in_work > "$OUT/3-pyspy-rank1.txt" <<EOF
py-spy dump --pid $PID1 --native
EOF
else
  echo "rank 1 process state=$ST1 - cannot attach (expected once that rank has exited)" > "$OUT/3-pyspy-rank1.txt"
fi
echo "  [3/6] py-spy"

# --- 4. /proc scan, twice, 5 s apart: which thread spins? ---
# schedstat field 1 is cumulative runtime in nanoseconds: a spinning thread accumulates
# ~5e9 ns over a 5 s window (one saturated core) while everything else stays flat.
scan_body() { cat <<EOF
for p in 1 2; do
  echo "--- pass \$p ---"
  for t in /proc/$1/task/*; do
    echo "\${t##*/} \$(cat \$t/comm) syscall=\$(cut -d' ' -f1 \$t/syscall) wchan=\$(cat \$t/wchan) ns=\$(cut -d' ' -f1 \$t/schedstat)"
  done
  sleep 5
done
EOF
}
attachable "$ST0" && scan_body "$PID0" | in_head > "$OUT/4-proc-rank0.txt"
attachable "$ST1" && scan_body "$PID1" | in_work > "$OUT/4-proc-rank1.txt"
echo "  [4/6] /proc"

# --- 5. pystack: every thread, including non-Python ones ---
if attachable "$ST0"; then
  in_head > "$OUT/5-pystack-rank0.txt" <<EOF
command -v pystack >/dev/null || pip install -q pystack 2>/dev/null
pystack remote $PID0 --native-all
EOF
fi
if attachable "$ST1"; then
  in_work > "$OUT/5-pystack-rank1.txt" <<EOF
command -v pystack >/dev/null || pip install -q pystack 2>/dev/null
pystack remote $PID1 --native-all
EOF
fi
echo "  [5/6] pystack"

# --- 6. logs, container config, RDMA counters ---
docker logs "$CTR" > "$OUT/6-log-head.txt" 2>&1
timeout 120 ssh -o BatchMode=yes "$WORKER" "docker logs $CTR" > "$OUT/6-log-worker.txt" 2>&1
docker inspect "$CTR" > "$OUT/6-inspect.json" 2>&1
roce_body() { cat <<EOF
for h in /sys/class/infiniband/*/ports/1; do
  echo "-- \$h"
  for c in packet_seq_err out_of_sequence local_ack_timeout_err rnr_nak_retry_err req_cqe_error; do
    [ -f \$h/hw_counters/\$c ] && echo "  \$c=\$(cat \$h/hw_counters/\$c)"
  done
  for c in port_rcv_errors link_error_recovery; do
    [ -f \$h/counters/\$c ] && echo "  \$c=\$(cat \$h/counters/\$c)"
  done
done
if [ -n "$IFACE" ]; then
  ethtool -S "$IFACE" 2>/dev/null | grep -iE "discard|pause|crc|error" | head -12
  ip link show "$IFACE" 2>/dev/null | grep -o "mtu [0-9]*"
fi
EOF
}
{ echo "== head =="; roce_body | bash; echo "== worker =="; roce_body | on_work; } > "$OUT/6-roce.txt"
echo "  [6/6] logs + roce"

# --- summary ---
{ echo "### threads burning CPU (delta over 5 s)"
  for r in 0 1; do
    python3 - "$OUT/4-proc-rank$r.txt" "$r" <<'PYEOF'
import re, sys
try:
    t = open(sys.argv[1]).read(); p1, p2 = t.split('--- pass 2 ---')
except Exception as e:
    print(f"rank {sys.argv[2]}: not collected ({e})"); raise SystemExit
f = lambda s: {m[0]: (m[1], m[2], m[3], int(m[4])) for m in
               re.findall(r'^(\d+) (\S+) syscall=(\S*) wchan=(\S*) ns=(\d+)', s, re.M)}
a, b = f(p1), f(p2)
d = sorted(((b[k][3] - a[k][3], k, b[k]) for k in b if k in a), reverse=True)
print(f"rank {sys.argv[2]} ({len(b)} threads):")
for x, k, v in d[:3]:
    if x > 0:
        print(f"   tid={k:6s} {v[0]:18s} cpu_ms={x//1000000:5d} syscall={v[1]:8s} wchan={v[2]}")
PYEOF
  done
  echo; echo "### native stack of the spinning threads"
  for r in 0 1; do
    echo "rank $r:"
    grep -B2 "sched_yield" "$OUT/5-pystack-rank$r.txt" 2>/dev/null \
      | grep -E "Traceback|libnccl|sched_yield" | head -3 | sed 's/^/   /'
  done
  echo; echo "### crash signature (head log)"
  grep -E "sample_tokens timed out|fatal error" "$OUT/6-log-head.txt" | tail -2 | cut -c1-160
  grep -m1 -oE "total_num_scheduled_tokens=[0-9]*" "$OUT/6-log-head.txt"
  grep -m1 -oE "scheduled_spec_decode_tokens=\{[^}]*\}" "$OUT/6-log-head.txt"
  grep -m1 -oE "num_running_reqs=[0-9]*" "$OUT/6-log-head.txt"
} | tee "$OUT/0-SUMMARY.txt"
echo "=> done: $OUT"
