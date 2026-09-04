# Archived results

The affected image (vllm-node-b12x built 2026-08-23) was overwritten by the rebuild, so its behaviour can no longer
be re-run. What is preserved here:

- `2026-08-23-build-broken/recorded-run/` — **captured, not transcribed**: the assistant replies actually returned by the
  affected build during the recorded agent run of 2026-09-02 (one JSON per final-answer request: full reply text,
  recorded latency, verdict), and `all-47-timings.csv` with the latency of every one of the 47 requests
  (note the drift from 3-166 s to 480-810 s as the run progressed). Temp paths genericized.
- `2026-08-23-build-broken/replay-solo-2026-09-02.txt`, `degradation-2026-09-02.txt` — the kit's output against the
  affected build, transcribed from the session log (the scripts were still the first version at the time:
  `FAIL`/`canary` labels instead of `WRONG`/`probe`).
- `2026-08-26-standard-image/solo-2026-09-03.txt` — same requests, same weights, standard `vllm-node` image (transcribed).
- `2026-09-03-build-fixed/` — the current scripts' output against the fixed build on 2026-09-04, verbatim.
