# Engine hang with speculative decoding — client-side timeline (2026-09-05)

Build: recipe of 2026-09-03 (`vllm-node-b12x`, vLLM `0.1.dev20482+g83cb22a0e.d20260903`),
DeepSeek-V4-Flash-0731, TP=2 over two Sparks, spec decode `dspark` / 5 tokens.
Server-side analysis (rank 1 silent, `RPC call to sample_tokens timed out`):
https://gist.github.com/BittnerPierre/0cfa5b1cf0348c45296af97aed421cf8

## What the agent traces show (local times, UTC+2)

Every killed run died on the **turn right after a knowledge-base tool call**, with a
single request in flight. The request stayed open until the server itself gave up
(~300 s — the worker RPC watchdog), then HTTP 500 and the port closed.

| Run | Turn before the kill | Killed turn started | Open for | Server log |
|---|---|---|---|---|
| jj5-concept-1 | get_knowledge_entries + fetch_vector_store_name | 08:55:15 | 326.6 s | 07:00:40 UTC: 10 077 prompt tok, 435 generated |
| jj5-concept-3 | upload_files_to_vectorstore (51.7 s) | 09:59:35 | 249.6 s | not analysed (shell SIGHUP suspected) |
| jj5b-concept-1 | upload_files_to_vectorstore (16.1 s) | 10:11:53 | 323.9 s | 08:17:16 UTC: 11 962 prompt tok, 1 187 generated (gist) |
| jj5s-capex-5 | get_knowledge_entries + fetch_vector_store_name | 16:34:54 | 307.8 s | log requested |

Same turns in a healthy run (jj5s-capex-1): 5.9 s and 22.1 s. So the killed
requests did NOT run slowly for five minutes: at ~57 tok/s the 435 / 1 187 generated
tokens represent 10-25 s of work. The engine froze early in the decode and the
watchdog fired ~300 s after the request started.

## What is special about these turns

- Prompt > 8k tokens (10-12k: system prompt + syllabus + the tool result), and the
  tool result is **new every run** (vector-store name, listing), so it is a real
  prefill of several thousand tokens after a cached shared prefix.
- Followed by a **long decode** (agenda: 1-2.5k tokens), **alone in the batch**.
- The other long decodes of the workflow (7 chapter writers, 1k tokens each) run
  7 in parallel; the other single-request turns (planner, outline) have small
  prompts and ~600 tokens out. Neither ever froze in ~30 runs today.

Counts for the day: 4 freezes in ~30 runs with `dspark` (≈60 such turns), 0 in
8 runs + 14 isolated long generations without it.

## Why the first isolated replay did not reproduce

`repro_c_engine_hang.py --repeat 10` (verbatim body) came back CLEAN with dspark on:
after the first iteration the whole prompt is served from the prefix cache, so only
one real "fresh prefill + long decode alone" trial was made. `--nonce tail` (now the
default) rewrites the last tool message with a random marker at every repetition,
which restores the agent's shape; ~30 repetitions cover the observed ~7 % per-turn
rate with high probability.

## Next

1. `python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --repeat 30`
   with dspark on (expected REPRODUCED), then the same with dspark off (expected CLEAN).
2. If still CLEAN: replay #01 then #02 as a pair, or capture the exact killed body
   through a recording proxy.

## Reconciliation with the server-side deep dive (crash of 14:40 UTC)

Server report: https://gist.github.com/BittnerPierre/ed529bb6d6756cd1763431e5933f2f18

- Hang start 14:35:16 UTC = 16:35:16 local. The killed turn (jj5s-capex-5, turn after
  get_knowledge_entries + fetch_vector_store_name) started 16:34:54: the engine froze
  **22 s into the request**, 10 181 prompt tokens computed, 147 generated. The client then
  waited for the 5-min RPC watchdog — same shape as the three other kills.
- Both ranks have their NCCL progress thread spinning in `sched_yield`, GPUs at 96 % SM /
  0 % memory, torch NCCL watchdogs asleep. vLLM's TP all-reduce goes through its own
  `PyNcclCommunicator` (direct `ncclAllReduce` via ctypes, own communicator), not through
  `torch.distributed`'s ProcessGroupNCCL — so the torch watchdog, heartbeat and flight
  recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`) cannot see these collectives. To log the last
  collective entered by each rank, use NCCL's own tracing (`NCCL_DEBUG=INFO`,
  `NCCL_DEBUG_SUBSYS=COLL`), not the torch flight recorder.
- The "bench never ran on nospec" gap in the server report is closed by this side: the full
  bench ran 8/8 on the same recipe without `--speculative-config` (plus 14 isolated long
  generations), 0 hangs; with dspark 9/10 then a hang, 4 hangs in ~30 runs over the day.

## 2026-09-06 07:41 — reproduced by the script, first attempt

`python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --requests 1,2 --metrics --repeat 60`
(dspark recipe, fresh server started 07:40 local, warm-up passed):

```
  #01 req#01     1.9s  generated=34   finish=tool_calls  server=up  spec-accept= 90% (30 draft tok)
  #02   313.3s  ERROR HTTP 500: EngineCore encountered an issue ...  server=DOWN
REPRODUCED
```

Request #02 (the agenda turn, 11 messages, tool result rewritten with a random marker)
was sent at 07:41:32 local (05:41:32 UTC) and came back 500 at 07:46:45 — the same ~300 s
RPC watchdog. The verbatim replay of the same request (`--nonce none`, 10 times, 2026-09-05
15:10) had stayed CLEAN: the only difference is that the tail of the prompt is new, i.e. a
real prefill after the cached prefix, exactly like the agent. Log: `2026-09-06-repro-c-run1-REPRODUCED.log`.
Server-side: frozen processes left in place for inspection (05:41-05:46 UTC window).

### Server-side confirmation of the script-induced hang (incident 3)

https://gist.github.com/BittnerPierre/08a1b6b9a7d8a323c85672f82917903b — identical signature
(both ranks spinning in `libnccl → __sched_yield`, rank 0 stuck in `cuCtxSynchronize`, GPUs
96 % / 0 % / ~19 W, `sample_tokens` RPC timeout at 05:46:45 UTC, one request, 1+5 spec tokens).
Hang onset 05:42:01 UTC, i.e. ~29 s after request #02 was sent (~650 tokens at 26 tok/s).
RoCE error counters unchanged across the night's RDMA tests and the hang: transport ruled out.

About the `scheduled_spec_decode_tokens=[-1,-1,-1,-1,-1]` seen in every dump: in upstream vLLM
the scheduler emits `-1` placeholders for draft tokens that are produced on device after
scheduling (`update_draft_token_ids_in_output` fills them later; `num_invalid_spec_tokens` is
only set when drafts get trimmed/padded). A dump taken at scheduling time therefore shows
`[-1]*5` with `num_invalid_spec_tokens=null` in normal operation — not a leaked value.

## 2026-09-06, steady-state replays on one instance (started 09:04): all CLEAN

| Command | Repetitions | Fresh part of the prompt per repetition | Result |
|---|---|---|---|
| `--requests 1,2 --nonce tail` | 60 | last tool message only (~1 kB) | CLEAN |
| `--requests 1,2 --nonce tail` | 240 | idem | CLEAN |
| `--requests 2 --nonce 3` | 100 | tool result + rest (~23 kB, ~6k tokens) after a cached ~2.5k prefix | CLEAN |

~370 single-request speculative decodes after a partial prefix hit, zero hangs, 47-58 tok/s,
spec acceptance 56-91 %. Taken with the five hangs, the position matters more than the shape:
3 of 5 hangs hit the **first long speculative decode after a server start** (3 of 8 starts),
2 of 5 hit steady state (2 in ~430 exposures, < 1 %). `/reset_prefix_cache` is not exposed on
this build (404), so the cold state can only be recreated by restarting the cluster.

## 2026-09-06 13:13 — reproduced again, with the full-run replay in dependency order

Same instance as all the CLEAN replays above (started 09:04, ~1 100 requests served).
`repro_c_engine_hang.py --replay-all --order deps --nonce 2 --metrics --loops 12`: a request
starts only when every request that had finished before it in the recording has finished
here (plus the agent's phases: #01, #02, #03 alone and in order). Loop 1:

```
  loop 01/12  wall=   680s  generated=15432  slowest=#46 324s  errors=1  server=DOWN
      #46 after 324s: HTTP 500: EngineCore encountered an issue ...
REPRODUCED
```

#46 is a search agent's second turn: 11 messages, 64 KB (~16k tokens), 8 vector_search tool
results (the fresh part, after a ~2k cached prefix), long generated summary; alone in flight
at that point of the replay (its sibling #47, 4 s, had finished). Sent ~13:13:28 local
(11:13:28 UTC), 500 at 13:18:52 (11:18:52 UTC).

Two compressed-offset loops runs (12 loops, `--time-scale 0.08`) had stayed CLEAN: there #02
overlapped the searches instead of running alone. Together with the 370 CLEAN isolated
replays, the reproducing shape is now: **a request alone in flight, several thousand fresh
prompt tokens after a cached prefix, then a long speculative decode — reached through the
run's real sequence**, whichever agent turn it is (knowledge-preparation #02 in the agent,
search turn #46 here).

## 2026-09-06 19:43 — same full-run command on a fresh server: CLEAN (12 loops)

`--replay-all --order deps --nonce 2 --metrics --loops 12` right after a restart and warm-up:
12 loops, 350-383 s each, ~15k tokens generated per loop, spec acceptance 72-73 %, no hang.
Score of the script so far: 2 hangs in 5 attempts (05:41 UTC first pair; 11:13 UTC loop 1),
0 in 400 isolated turns, 0 in 24 compressed/deps loops besides the hit. The agent benchmark:
4 hangs in ~30 runs. Reproducible, not deterministic.

## 2026-09-07 07:52-08:36 — the full agentic benchmark again, through a recording proxy: no hang

Same vLLM instance (started 2026-09-06 19:43, dspark on), same code and configuration as the
runs that hung on 2026-09-05; every request went through a stdlib proxy that saved it verbatim.
10 runs (5 concept + 5 finance), 373 chat completions, 173k generated tokens, all HTTP 200,
longest request 82 s. Finance A A A A A, coverage 100 % (90-100); concept coverage 75 % (56-88).
Recording exported as `requests-campaign/` (kit format, real offsets and durations).

Tally for this instance after ~13 h: 12 concept-run loops, 40 replays of the killed prompts,
86 fuzz candidates, 10 real benchmark runs — 0 hangs. The same benchmark hung 4 times in ~30
runs on 2026-09-05 across four other instances.

## Status on 2026-09-07 (facts only)

Between the last hang (2026-09-06 11:18 UTC) and the current instance (started 17:43 UTC
after `launch-cluster.sh stop`), the cluster went through RDMA/RoCE bench tests and a clean
stop of both nodes. Everything sent to this instance since then has completed: replays,
fuzzing, the four killed prompts, and two real benchmark campaigns (the second in progress at
the time of writing). Whether the clean stop, the orphaned worker processes of earlier
instances, or the network tests changed anything is not established; we record the sequence.
