# Server-side diagnostics for the engine hang

`collect-vllm-hang.sh` gathers, in one command and about two minutes, the evidence that
`ISSUE-engine-hang.md` is built on. Run it **while the cluster is hung** — the processes stay
frozen indefinitely, and every command it runs is read-only.

```bash
WORKER=<worker-host> ./collect-vllm-hang.sh [output-dir]

# optional:
#   CTR=<container name>   (default: vllm_node)
#   IFACE=<nic name>       NIC used by NCCL; enables ethtool stats
```

Requirements: `docker` on both hosts, passwordless ssh to the worker, and `py-spy` inside the
container. `pystack` is installed on the fly if it is missing.

## Why not just py-spy

**py-spy only shows threads that Python knows about.** In this hang, the thread that is actually
burning CPU is an NCCL progress thread with no Python frame, so `py-spy dump` reports every thread
as idle and the real culprit stays invisible. Finding it takes two extra steps, both of which this
script does:

1. scan `/proc/<pid>/task/*/schedstat` twice, 5 s apart, and diff the cumulative runtime — the
   spinning thread stands out immediately at ~5×10⁹ ns (one saturated core) while everything else
   sits at zero;
2. dump **all** threads with `pystack remote <pid> --native-all`, which does include non-Python
   threads, and read the native stack of the one found in step 1.

A related trap: `ps` reports `%CPU` averaged over the process lifetime, not instantaneous use. It
made the hung worker look busy when its Python threads were all parked. `schedstat` deltas avoid
that.

## What it collects

| Output | Content |
|---|---|
| `0-SUMMARY.txt` | spinning thread per rank, its native stack, crash signature |
| `1-gpu.txt` | GPU utilisation, SM clocks, power, and which process holds the memory |
| `2-pids.txt` | worker PIDs, plus each rank's process state (`R`, `Z`, gone) |
| `3-pyspy-rank{0,1}.txt` | `py-spy dump --native` per rank |
| `4-proc-rank{0,1}.txt` | `/proc/<pid>/task/*` scan, two passes 5 s apart |
| `5-pystack-rank{0,1}.txt` | `pystack --native-all`: every thread, Python or not |
| `6-log-head.txt`, `6-log-worker.txt` | vLLM logs from both ranks |
| `6-inspect.json` | container configuration |
| `6-roce.txt` | RDMA error counters (both nodes) and, if `IFACE` is set, ethtool stats |

One rank may have exited by the time you collect (its process shows up as a zombie). The script
detects that and records the state instead of failing on an attach error.

## What a hang looks like

Reference values from four incidents, for comparison:

```
GPU, both nodes ...... 96 % utilisation, 0 % memory utilisation, ~18-20 W, SM clocks high
                       (SMs busy, no memory traffic, no throttling => spin-wait kernel)
rank 0 main thread ... cuCtxSynchronize -> cudaDeviceSynchronize (never returns)
                       or, if the rank already died, a zombie and an idle GPU
rank 1 main thread ... idle in the RPC loop (zmq poll), waiting for the next command
spinning thread ...... one per live rank, ~5.4 s of CPU per 5 s, syscall=running, wchan=0
                       native stack: libstdc++ -> libnccl -> __sched_yield
NCCL watchdogs ....... pt_nccl_watchdg / pt_nccl_heartbt asleep in pthread_cond_wait,
                       so no NCCL timeout is ever raised
head log ............. TimeoutError: RPC call to sample_tokens timed out
                       num_running_reqs=1, total_num_scheduled_tokens=6,
                       scheduled_spec_decode_tokens=[-1,-1,-1,-1,-1], KV cache ~0.5-1 %
RDMA counters ........ all zero, before and after; only PFC pause frames move
```

The hang always lands on a speculative decode step (1 token + 5 draft tokens) with a single
request in flight and no memory pressure. Time to trigger has ranged from 6 requests / 9 minutes
to 1022 requests / 4 h 12 across incidents, so endurance is not the factor.

## Cleaning up afterwards

A hung worker never releases on its own — it keeps spinning and holding its GPU memory even after
the head rank has died. Stop **both** nodes before relaunching, otherwise a fresh server started on
the head will sit forever waiting for the stale rank 1. In this setup that is
`./launch-cluster.sh stop` from the serving repo, which acts on the head and every worker.
