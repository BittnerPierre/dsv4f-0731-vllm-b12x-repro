# DeepSeek-V4-Flash-0731 on the 2026-08-15 b12x vLLM build: model ignored "answer with the filename only" — regression kit

> **Status (2026-09-04): fixed.** Broken on the 2026-08-15 `deepseek-v4-flash-0731` recipe (`dev/infernal-invocation`), clean on the 2026-09-03 recipe (`dev/jovian-judgement`, `--attention-backend B12X`). The kit now serves as a regression check. Full story in `ISSUE.md`, posted as [eugr/spark-vllm-docker#376](https://github.com/eugr/spark-vllm-docker/issues/376).

## What this is

47 real HTTP requests to `/v1/chat/completions`, recorded from an agent that
searches documents and writes notes. Each request is a normal chat body:
a system prompt, a user message, two tool definitions (`vector_search`,
`write_file`), and — in later turns — the tool results. Nothing else.

11 of the 47 requests are **final-answer turns**: the agent has just written
its note with `write_file`, and the system prompt says the reply must be
**exactly the filename, nothing else** (e.g. `capex_trends.txt`).

Two scripts, one per symptom:
- `repro_a_filename_only.py` sends one of those final-answer requests and checks: *is the reply just a filename?*
- `repro_b_latency_growth.py` sends the same request several times and prints how long each one took.

Both start by checking that the server is reachable and serves the model, then send two short
probes ("reply pong", "reply with this filename") that must pass — if they fail, the server is
simply down and the scripts stop with exit code 2 instead of claiming anything.

## What we see

On the affected 2026-08-15 `vllm-node-b12x` build serving
`deepseek-ai/DeepSeek-V4-Flash-0731`, the reply was never just the filename.
It was 100-2700 characters like:

```
apple_microsoft_rd_expenses_..._disclosures.txt.txt

Actually, let me correct that. The filename should be exactly as written...
Wait, I need to reconsider... I keep adding commentary. The rule says
"Do not include any other text."
```

and the same request got slower each time it was sent: 3 s, then 20 s, then
timeout after 240 s. Short prompts sent in between stay fast and correct.

On the standard `vllm-node` image, same weights, same requests: 11/11 replies
are exactly the filename, every call takes 5-37 s, and repeating a request
takes 1.7 s each time.

## Run it

```
python3 repro_a_filename_only.py --list                                  # the 11 final-answer requests, numbered
python3 repro_a_filename_only.py --base-url http://HOST:8000/v1          # send request #11, check the reply (~30 s)
python3 repro_a_filename_only.py --base-url http://HOST:8000/v1 --all    # all 11, one at a time
python3 repro_b_latency_growth.py --base-url http://HOST:8000/v1         # request #11 five times, latency series
python3 repro_b_latency_growth.py --base-url http://HOST:8000/v1 --replay-all   # all 47 with the recorded timing
```
`--request N` picks another recorded request (numbers from `--list`); `--model NAME` replays against another
served model. Exit code: 0 clean, 1 reproduced, 2 could not run.

Output of `repro_a` on the affected build:
```
  #11   20.5s  WRONG 2096 chars: 'apple_microsoft_..._disclosures.txt.txt\n\nActually, let me correct that...'
REPRODUCED: 1/1 replies were not just the filename [11]
```
On a healthy build:
```
  #11    5.8s  ok    apple_microsoft_rd_expenses_dividends_workforce_disclosures_fy2025_misc_disclosures.txt
clean: 1/1 replies were exactly the filename
```
Output of `repro_b` on the affected build: `latency series: 9.1s -> 240.0s` (timeout) — on a healthy build:
`latency series: 1.7s -> 1.7s -> 1.8s`.

Python 3.10+, standard library only, no API key (`Authorization: Bearer dummy`).
`requests/` holds the 47 request bodies as readable JSON, one file per request
(`requests/11.json` is the default one), plus `requests/SYSTEM_PROMPT.txt` — the
system message they share, where the "return only the name of the file ...
Do not include any other text" rule lives — and `requests/index.json` /
`requests/README.md` listing them. Temp paths were replaced by
`/tmp/agent-workdir`; the documents quoted in the tool results are synthetic
financial notes generated for a benchmark (freely shareable).

## What was tested

| Weights | Build | Filename-only replies | Latency |
|---|---|---|---|
| V4-Flash-0731 | `vllm-node-b12x`, DSpark speculative on | 0 / 11 | degrades to 480-810 s, timeouts |
| V4-Flash-0731 | `vllm-node-b12x`, speculative off | 0 / 7 | — |
| V4-Flash-0731 | `vllm-node-b12x`, `--max-num-seqs 1` | fails | — |
| V4-Flash-0731 | standard `vllm-node`, speculative off | **11 / 11** | flat, 5-37 s |
| V4-Flash (previous checkpoint) | standard `vllm-node` | 11 / 11 | flat |
| Qwen3.6-35B-A3B | standard `vllm-node` | 10 / 11 | flat |
| V4-Flash-0731 | cloud provider | 7 / 7 (live agent run, twice) | — |
| **V4-Flash-0731** | **`vllm-node-b12x`, 2026-09-03 recipe (`dev/jovian-judgement`, `B12X`)** | **11 / 11** | **flat, 2.6 s x5** |

The controls isolate the regression to the `dev/infernal-invocation` b12x build
path for this model rather than the weights, prompts, speculative decoding by
itself, or concurrency. It is fixed on `dev/jovian-judgement`.

Issue text: `ISSUE.md`.

## Second defect (2026-09-05 → 07): engine dies mid-decode with speculative decoding on

Different symptom, different recipe (the fixed 2026-09-03 one, `dspark` speculative decoding on,
TP=2 over two Sparks): the engine dies during a single-request decode with
`RPC call to sample_tokens timed out`, one thread per rank left spinning in `libnccl`.
Six occurrences in two days, then none: the kit keeps everything needed to try again.

- `ISSUE-engine-hang.md` — the consolidated report (facts, reproduction, what the frozen
  processes show, server-side gists), not posted yet.
- `repro_c_engine_hang.py` — replays recorded agent turns or a whole recorded run
  (`--replay-all --order deps`) with fresh prompt tails (`--nonce`), checks the server after
  each reply, exits 1 when it dies. `--requests-dir` selects the request set.
- `repro_c_fuzz.py` — replays seed-described mutations of a run (subset, scheduling, batch
  transitions, nonce position, greedy) on one server until it dies; `--seed` replays a hit.
- Request sets: `requests/` (finance run, 2026-08-23, broken build), `requests-concept/`
  (concept run, 2026-09-05, fixed build, real timings), `requests-killed/` (the four prompts
  killed under the benchmark, exported from tracing), `requests-campaign/` (10 full benchmark
  runs recorded verbatim on 2026-09-07, no hang).
- `results/2026-09-05-dspark-engine-hang/` — client-side timeline of every occurrence and every
  attempt, with logs.

Score of the reproduction so far: 2 hangs out of 5 script attempts on 2026-09-06 morning, then
0 in ~13 hours of replays, fuzzing and two real benchmark campaigns on one server instance.
- `diagnostics/` — server-side collection: `collect-vllm-hang.sh` gathers, in one command,
  the GPU counters, per-thread `/proc` scan, py-spy and `pystack --native-all` dumps of both
  ranks, the vLLM logs and the RDMA error counters. Its README explains why py-spy alone misses
  the spinning NCCL thread, and lists the reference values a hang produces.
