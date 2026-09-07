# Engine hang with speculative decoding — DRAFT, not reproducible at the moment

Status (2026-09-07): an identified anomaly that we are still trying to reproduce. It is
**not** the defect of the published issue #376 (root of this repository); it appears on the
recipe that fixed #376 (2026-09-03, `vllm-node-b12x`, DeepSeek-V4-Flash-0731, TP=2 over two
Sparks, `dspark` speculative decoding on).

- Seen 6 times on 2026-09-05/06: 4 under an agentic benchmark, 2 with the script below
  (the second on a server that had served 1 000+ requests for 4 hours).
- Since then, on one server instance: ~13 hours of replays, fuzzing, the four killed prompts
  replayed 40 times, and two complete runs of the real benchmark — no hang.
- The report is a draft (`ISSUE-draft.md`); it will be posted only when the hang can be
  shown again. Server-side investigations of the frozen processes are linked from it.

## Files

- `ISSUE-draft.md` — the report: environment, symptom, reproduction attempts, what the frozen
  processes show (with the server-side reports), ask. Facts only.
- `timeline.md` — client-side sequence of every occurrence and every attempt.
- `repro_c_engine_hang.py` — replays recorded agent turns, or a whole recorded run in its
  dependency order (`--replay-all --order deps`), with fresh prompt tails (`--nonce`), checks
  the server after each reply and exits 1 when it dies. `--requests-dir` selects the data set.
- `repro_c_fuzz.py` — replays seed-described mutations of a run on one server until it dies;
  `--seed N` replays one candidate.
- `common.py` — shared helpers (copy of the root one; this directory runs on its own).
- Data sets: `requests-finance/` (a finance run recorded on 2026-08-23; its turns #02 and #46
  hung under the script), `requests-concept/` (a concept run recorded on 2026-09-05 on the
  fixed build, real timings), `requests-killed/` (the four prompts killed under the benchmark,
  exported from tracing with tools and sampling re-attached).
- `results/` — the two script runs that produced a hang.

## Run

```
python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --replay-all --order deps --nonce 2 --metrics --loops 12
python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --requests-dir requests-concept --replay-all --order deps --nonce 2 --metrics --loops 12
python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --requests-dir requests-killed --requests 1,2,3,4 --nonce 2 --repeat 40
python3 repro_c_fuzz.py --base-url http://spark1:8000/v1 --requests-dir requests-concept
```

Stdlib only, no API key. Exit 0 clean, 1 reproduced (server dead), 2 cannot run.
