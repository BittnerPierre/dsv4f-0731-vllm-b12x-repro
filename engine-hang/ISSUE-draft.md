# DeepSeek-V4-Flash-0731 on two Sparks (TP=2, dspark): engine dies mid-decode with `RPC call to sample_tokens timed out`, one thread per rank spinning in libnccl

**DRAFT — not posted.** Identified anomaly, reproduced twice with a stdlib script on 2026-09-06
(once in 6 requests on a fresh server, once in one loop of a full-run replay on a server that had
served 1 000+ requests for 4 hours), then **not reproducible any more**: ~13 hours of replays,
fuzzing and two complete runs of the real benchmark on one server instance without a hang.
Workaround: remove `--speculative-config` (dspark) — 0 hangs in 8 full benchmark runs and
~35 isolated long generations, at half the decode speed (27 vs 57 tok/s).

This is a different defect from #376 (fixed by the 2026-09-03 recipe): there the server
stayed up and misbehaved; here the engine dies.

## Environment

| | |
|---|---|
| Recipe | `recipes/deepseek-v4-flash-0731.yaml` (2026-09-03 version), `./run-recipe.sh -d` |
| Image / vLLM | `vllm-node-b12x`, vLLM `0.1.dev20482+g83cb22a0e.d20260903`, mod `instanttensor-hybrid-draft-loader` |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731`, `deepseek_v4_fp8`, experts fp4, KV `fp8_ds_mla` |
| Cluster | TP=2, head node (rank 0) + worker node (rank 1), RoCE (2 HCAs), collectives via PYNCCL only (custom AR / SymmMem / FlashInfer AR unavailable on SM121) |
| Spec decode | `method=dspark`, 5 speculative tokens, `draft_sample_method=probabilistic` |
| Driver / CUDA / NCCL | 580.159.03 / 13.0 / `libnccl.so.2.31.2` loaded (packages say 2.28.3) |
| Client | OpenAI-compatible chat completions, temperature 1.0, top_p 0.95, tools |

## Symptom

With a **single request in flight**, in the middle of decoding, the engine stops producing
tokens. About 300 s later the head logs:

```
TimeoutError: RPC call to sample_tokens timed out.
EngineCore encountered a fatal error.  ->  EngineDeadError  ->  HTTP 500, APIServer shuts down
```

The container stays "Up". The two worker processes stay alive, frozen: relaunching the
recipe in daemon mode then blocks forever on the zombie rank 1 (`launch-cluster.sh stop`
first).

Seen 6 times in two days (4 under an agentic benchmark, 2 with the script below), never
without speculative decoding. After the hang the rank-1 worker never frees itself, even once
the head is gone (incident 4: rank 0 zombie, rank 1 orphan still spinning with 100 GB of VRAM).

## Reproduce

```
git clone https://github.com/BittnerPierre/dsv4f-0731-vllm-b12x-repro
cd dsv4f-0731-vllm-b12x-repro/engine-hang
python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --replay-all --order deps --nonce 2 --metrics --loops 12
```

Stdlib only, no API key. It replays the 47 recorded requests of one real agent run in their
recorded dependency order (a request starts once every request that had finished before it
in the recording has finished here): the first three turns alone and in sequence, then the
search agents concurrently, then the writers. Before each send it rewrites everything after
the second message with a random marker (`--nonce 2`): the shared prefix (system prompt +
task) stays in the prefix cache, the rest — tool results, several thousand tokens — is a real
prefill, as in the agent. It checks `/v1/models` after each loop and exits 1 (REPRODUCED) as
soon as the server stops answering, printing one line per request of the failing loop.

2026-09-06 11:13 UTC, server up for 4 h 12 (1 022 requests served):

```
  loop 01/12  wall=   680s  generated=15432  slowest=#46 324s  errors=1  server=DOWN
      #46 after 324s: HTTP 500: EngineCore encountered an issue ...
REPRODUCED
```

#46 is a search agent's second turn (11 messages, ~16k prompt tokens of which ~14k fresh
tool results, long generated summary), alone in flight after its 4-second sibling finished.
Earlier the same day, `--requests 1,2 --repeat 60` (turn #01 then the "agenda" turn #02,
fresh tail) had killed a fresh server at its first #02 (05:41 UTC).

Controls on the same build, all CLEAN:
- the same turns replayed **verbatim** (served from the prefix cache): 10/10;
- single turns replayed with fresh tails, outside the run sequence: 400/400;
- the full run with compressed start times, where turn #02 shares the batch with the
  searches instead of running alone: 12 loops;
- same recipe **without** `--speculative-config`: script 6/6 and the full benchmark 8/8.

## What the frozen processes show (server side, four incidents, same picture)

Collected on the frozen workers with `collect-vllm-hang.sh` (py-spy, pystack, /proc scan,
nvidia-smi), between 30 min and 2 h 30 after each hang; full details in the four reports
linked below.

- rank 0 (head): main thread in `cuCtxSynchronize` (reached from the model runner's
  `shutdown()` after the RPC timeout), still there hours later in incidents 2 and 3. In
  incident 4 this process had exited (zombie) and its GPU was released.
- rank 1 (worker): Python main thread idle in the RPC receive loop (`zmq_poll`); nothing
  logged since service started. Still alive in every incident, including after the head was
  gone, holding ~100 GB of GPU memory.
- **on each surviving rank exactly one thread runs at 100 % of a core, in userspace, with no
  system calls; its native stack is `libstdc++ → libnccl.so.2.31.2 → __sched_yield`.**
- GPUs of the surviving ranks: 96 % SM utilization, 0 % memory-controller utilization,
  17-20 W, SM clocks at 2.4-2.5 GHz (max 3.0), stable for hours.
- scheduler dump at the timeout, every time: `num_running_reqs=1`, `kv_cache_usage≈0.5-1 %`,
  `total_num_scheduled_tokens=6` (1 + 5 speculative), request in decode after a 10-12k
  prompt, 147 to 1 187 tokens generated. (`[-1]*5` in `scheduled_spec_decode_tokens` is what
  upstream vLLM's scheduler emits as placeholders for draft tokens produced on device after
  scheduling — `Scheduler.update_draft_token_ids_in_output` fills them in later — so the value
  itself is not abnormal.)
- torch's NCCL watchdog and heartbeat threads (`pt_nccl_watchdg`, `pt_nccl_heartbt`) stay
  asleep in `pthread_cond_wait` throughout; no NCCL error or timeout is ever raised. For
  reference, vLLM's tensor-parallel all-reduce on this setup goes through its own
  `PyNcclCommunicator` (`vllm/distributed/device_communicators/pynccl.py`, direct
  `ncclAllReduce` on a communicator it owns), not through `torch.distributed`'s
  ProcessGroupNCCL that those threads monitor. No NCCL or torch debug variable was set for
  any of the incidents; process-group settings are the defaults.

Ruled out by measurement: KV pressure (≤ 1.1 %), load (1 request in flight), endurance (hang
after 6 requests once, after 1 022 another time), the RoCE link (111 Gb/s in RDMA tests, all
HCA error counters at zero across four incidents), GPU co-tenants (incident 4 had nothing else
on the GPUs), thermal state (hangs 1 min and 4 h after start).

Server-side reports: incident 1 <https://gist.github.com/BittnerPierre/0cfa5b1cf0348c45296af97aed421cf8>,
incident 2 <https://gist.github.com/BittnerPierre/ed529bb6d6756cd1763431e5933f2f18>,
incident 3 (script, fresh server) <https://gist.github.com/BittnerPierre/08a1b6b9a7d8a323c85672f82917903b>,
incident 4 (script, 4-hour-old server, no other process on the GPUs) <https://gist.github.com/BittnerPierre/f4b2bd6fa51a0d494509da9c4d86f5bc>.
In incident 4 the head log shows the batch going from 2 running requests to 1 ten seconds
before generation stops, with speculative acceptance at its maximum (6.00).
Client-side timeline: `timeline.md`.

## Ask

We are not in a position to diagnose this further. The reproduction is one command and takes
5 to 15 minutes per attempt on our cluster; the frozen processes stay inspectable afterwards.
Tell us what to run or collect (debug variables, builds, recipes) and we will report back.
