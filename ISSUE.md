# DeepSeek-V4-Flash-0731 on vllm-node-b12x (2026-08-15 recipe): tool-call replies ignored "filename only" and latency grew with uptime — confirmed FIXED by the 2026-09-03 recipe (dev/jovian-judgement). Repro kit attached.

I had prepared this as a bug report; yesterday's recipe update fixes it. Posting it anyway as a confirmation with a
2-minute reproduction kit, since the commits on `dev/jovian-judgement` describe warmup crashes while what we hit on
the previous ref was silent: wrong outputs and slowdowns, no crash, no error.

## History

| When | Build | Result |
|---|---|---|
| July | `DeepSeek-V4-Flash` (previous checkpoint), standard `vllm-node`, MTP-2 | fine — full agent workflow, dozens of runs |
| Aug 15 recipe, image built Aug 23 | `DeepSeek-V4-Flash-0731`, `vllm-node-b12x`, `EXP_B12X_VLLM_REF=dev/infernal-invocation` (b5f995e73), `B12X_MLA_SPARSE`, DSpark-5 | **broken** — see symptoms below; also broken with DSpark disabled and with `--max-num-seqs 1` |
| Sept 3 | `0731`, standard `vllm-node`, no speculative | fine — this isolated the build: same weights, same requests, clean |
| Sept 3 recipe, tested Sept 4 | `0731`, `vllm-node-b12x`, `EXP_B12X_VLLM_REF=dev/jovian-judgement`, `--attention-backend B12X`, DSpark-5 | **fixed** — kit clean on both symptoms |

## Environment

- 2x DGX Spark (GB10 / SM121), tensor-parallel 2
- Model `deepseek-ai/DeepSeek-V4-Flash-0731`, `--tokenizer-mode/--tool-call-parser/--reasoning-parser deepseek_v4`, `--kv-cache-dtype fp8`, prefix caching on, `max-num-seqs 8`
- Broken: recipe `deepseek-v4-flash-0731.yaml` as of [2026-08-15](https://github.com/eugr/spark-vllm-docker/commit/358bf26e3d7d315450a6159ccdc0b38cbbd6b855) (`dev/infernal-invocation`, vLLM `0.1.dev20133+gb5f995e73.d20260823`, local build of 2026-08-23). That image was overwritten by the rebuild below; its digest was not retained.
- Fixed: same recipe as of [2026-09-03](https://github.com/eugr/spark-vllm-docker/commit/e91bc7632aa841b21c2c46b9da58ba5e4f46f44c) (`dev/jovian-judgement`), vLLM `0.1.dev20482+g83cb22a0e.d20260903`, image `vllm-node-b12x:latest` built 2026-09-03 (image id `21edb7f8046e`, `eugr/spark-vllm-b12x@sha256:16ce6e7efce8bebda9a800fd1432c7dde42d28f901354af00b2f3a57e68e3654`)
- Standard image used for the clean control: `vllm-node:latest` (image id `078a8109a069`, built ~2026-08-26)

## What happened on the broken build

Requests from a tool-using agent: system prompt + 2 tools (`vector_search`, `write_file`) + a few tool results,
10-30k tokens. After writing its note, the model must reply with exactly the filename.

1. **The reply was never just the filename** — 11/11 came back as 100-2700 chars of self-commentary, the model
   narrating its own violation: `capex_reference_data.txt.txt` — "Wait, I need to reconsider... I keep adding
   commentary. The rule says 'Do not include any other text.'" From the first request on a fresh server, in
   thinking and non-thinking modes, at temperature 0.3 and 1.0.
2. **The same request got slower every time**: 3.0 s → 20.5 s (after ~50 large requests) → timeout > 240 s;
   480-810 s per call in live agent runs. Short prompts stayed at 0.5-3 s throughout — no crash, no error.

## On the fixed build (2026-09-04)

```
repro_a_filename_only.py --all      → clean: 11/11 replies were exactly the filename (4-24 s per request)
repro_b_latency_growth.py           → latency series: 2.6s -> 2.6s -> 2.6s -> 2.6s -> 2.6s, clean
```

## Reproduce / regression-check

https://github.com/BittnerPierre/dsv4f-0731-vllm-b12x-repro — stdlib Python, no keys; the 47 recorded request bodies
are readable JSON under `requests/`, the shared system prompt is `requests/SYSTEM_PROMPT.txt`.

```
python3 repro_a_filename_only.py --base-url http://HOST:8000/v1 --all
python3 repro_b_latency_growth.py --base-url http://HOST:8000/v1
```
Exit 0 clean, 1 reproduced, 2 could not run. Each run starts with two short probes that must pass.

## Notes

- Likely fixed by the Sept 3 commits on `dev/jovian-judgement`:
  [`83cb22a0`](https://github.com/local-inference-lab/vllm/commit/83cb22a0e3f7ec4d2fb43f6ead34ba4d4a87a634)
  (B12X sparse MLA per-token cache lengths recomputed as a fresh tensor each build → FULL CUDA-graph replays read
  stale lengths), [`341f198b`](https://github.com/local-inference-lab/vllm/commit/341f198b276e15395dea96b1519873d0258a2663)
  (DSpark decode/prefill split mismatch in the sparse MLA builder),
  [`b60c5e39`](https://github.com/local-inference-lab/vllm/commit/b60c5e397f06ac3d64fb20d719aad8968938e4f6)
  (DSpark decode metadata sizing), and
  [`d662a1b0`](https://github.com/local-inference-lab/vllm/commit/d662a1b0890271915c25439f22247ee22234739a)
  (new `B12X` sparse MLA/DSA backend replacing `B12X_MLA_SPARSE`). Not bisected — the kit makes that cheap if useful.
- Related on the same image era: [eugr/spark-vllm-docker#349](https://github.com/eugr/spark-vllm-docker/issues/349)
  (profiling crash on the 2026-08-15 image) and
  [eugr/spark-vllm-docker#358](https://github.com/eugr/spark-vllm-docker/issues/358)
  (DSpark acceptance collapse ending in empty outputs — different outcome from the silent one here).
