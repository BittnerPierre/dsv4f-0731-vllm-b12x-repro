#!/usr/bin/env python3
"""Symptom C — the engine dies during a long generation (speculative decoding on).

Sends one recorded request whose reply is LONG (request #02 by default: the
agent's "detailed agenda" turn, ~12k prompt tokens, 1-2.5k generated tokens)
several times in a row and checks that the server is still alive after each
repetition. Stdlib only, no API key.

Observed on the 2026-09-03 recipe (vllm-node-b12x, DeepSeek-V4-Flash-0731,
TP=2 over two Sparks, spec decode method=dspark, 5 speculative tokens): with a
single request in flight and 0.6 % KV-cache usage, rank 1 stops answering in
the middle of a speculative decode step (1 + 5 tokens scheduled); the head
hits "RPC call to sample_tokens timed out", EngineCore dies, the request gets
HTTP 500 and the port closes. It hit the same long-generation turn three times
out of five, never a search turn (~200 generated tokens each). The exposure is
per decode step, so the longest generation of the workflow is where it shows.

    python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1                 # request #02, 30 times, fresh tail
    python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --repeat 20
    python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --requests 1,2 --metrics   # agent-like pair
    python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --greedy                   # sampling control
    python3 repro_c_engine_hang.py --base-url http://spark1:8000/v1 --replay-all --loops 10    # 10 full agent runs

Why a nonce: every repetition of an identical body after the first is served
from the prefix cache (no real prefill). The agent never does that: the shared
prefix (system prompt + syllabus) is cached, but the tail — the tool result the
model just received — is new every run, so each turn is a fresh multi-thousand
token prefill followed by a long speculative decode, alone in the batch. That
is the shape of the four turns that froze the engine, so `--nonce tail`
(default) rewrites the last tool/user message with a random marker each time;
`--nonce head` invalidates the whole prefix; `--nonce none` replays verbatim.

Exit 0 = every repetition completed and the server is still up (clean).
Exit 1 = the server stopped answering after a repetition, or a repetition
         errored / timed out (reproduced).
Exit 2 = could not run (server unreachable, wrong model, short probes failing).
"""

import argparse
import concurrent.futures
import copy
import secrets
import sys
import time
import urllib.request

from common import (
    EXIT_CANNOT_RUN,
    EXIT_CLEAN,
    EXIT_REPRODUCED,
    chat,
    describe,
    load_requests,
    preflight,
    short_probes,
)


def with_nonce(body: dict, where: str) -> dict:
    """Copy of `body` with a random marker so the server cannot serve it from cache."""
    if where == "none":
        return body
    if where not in ("head", "tail") and not where.isdigit():
        raise SystemExit(f"--nonce must be tail, head, none or a message index, not {where!r}")
    sent = copy.deepcopy(body)
    marker = f"[replay-id {secrets.token_hex(6)}] "
    msgs = sent["messages"]
    if where == "head":
        target = msgs[0]
    elif where == "tail":  # the last message with textual content (a tool result or the user turn)
        target = next(m for m in reversed(msgs) if isinstance(m.get("content"), str) and m["content"])
    else:  # numeric: first message with textual content at index >= N (else the last one)
        n = int(where)
        target = next(
            (m for m in msgs[n:] if isinstance(m.get("content"), str) and m["content"]),
            next(m for m in reversed(msgs) if isinstance(m.get("content"), str) and m["content"]),
        )
    target["content"] = marker + target["content"]
    return sent


def spec_acceptance(metrics_url: str):
    """(draft tokens, accepted tokens) counters from vLLM's /metrics, or None."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    totals = {}
    for line in text.splitlines():
        for key in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
            if line.startswith("vllm:" + key):
                totals[key] = totals.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    if len(totals) < 2:
        return None
    return (
        totals["spec_decode_num_draft_tokens_total"],
        totals["spec_decode_num_accepted_tokens_total"],
    )


def server_alive(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def recorded_dependencies(requests: list[dict]) -> dict[int, set[int]]:
    """n -> requests that had finished before n started in the recording.

    The agent's own phases are enforced on top (knowledge preparation #01 -> #02 -> planning
    #03 -> everything else), because the recording was made on a build whose replies timed
    out and were retried, which blurs the first turns' timing."""
    ends = {e["n"]: e["offset_s"] + e["recorded_duration_s"] for e in requests}
    deps: dict[int, set[int]] = {}
    for e in requests:
        n = e["n"]
        deps[n] = {j for j, end in ends.items() if j != n and end <= e["offset_s"] + 0.5}
        if n == 2:
            deps[n] = {1}
        elif n == 3:
            deps[n] = {1, 2}
        elif n > 3:
            deps[n] |= {1, 2, 3}
    return deps


def replay_all_loops(args, model: str, requests: list[dict]) -> int:
    """Loop the whole recorded run (47 requests, fresh tails), in recorded order or by dependencies."""
    metrics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    deps = recorded_dependencies(requests) if args.order == "deps" else None
    print(
        f"replaying the full recorded run ({len(requests)} requests) {args.loops} times, "
        f"nonce={args.nonce}, order={args.order}"
        f"{'' if args.order == 'deps' else f', time-scale={args.time_scale}'}\n"
    )
    for loop in range(1, args.loops + 1):
        t0 = time.time()
        results = {}

        inflight: set[int] = set()
        lines: list[str] = []

        def one(e):
            body = dict(e["body"])
            body["model"] = model
            body["stream"] = False
            if args.greedy:
                body["temperature"] = 0.0
                body.pop("top_p", None)
            started = time.time() - t0
            inflight.add(e["n"])
            concurrent_at_start = len(inflight)
            dt, resp, err = chat(args.base_url, with_nonce(body, args.nonce), args.timeout)
            inflight.discard(e["n"])
            usage = (resp.get("usage") or {}) if resp else {}
            done = usage.get("completion_tokens")
            lines.append(
                f"      #{e['n']:02d} start=+{started:5.0f}s inflight={concurrent_at_start} "
                f"prompt={usage.get('prompt_tokens')} generated={done} {dt:6.1f}s"
                f"{'  ERROR ' + err[:60] if err else ''}"
            )
            return e["n"], dt, done, err

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            if deps is None:
                futs = []
                for e in requests:
                    delay = e["offset_s"] * args.time_scale - (time.time() - t0)
                    if delay > 0:
                        time.sleep(delay)
                    futs.append(pool.submit(one, e))
                for f in concurrent.futures.as_completed(futs):
                    n, dt, done, err = f.result()
                    results[n] = (dt, done, err)
            else:
                pending = {e["n"]: e for e in requests}
                running: dict[concurrent.futures.Future, int] = {}
                while pending or running:
                    ready = [n for n in sorted(pending) if deps[n] <= set(results)]
                    for n in ready:
                        running[pool.submit(one, pending.pop(n))] = n
                    if not running:
                        break  # unsatisfiable dependency (should not happen)
                    done_set, _ = concurrent.futures.wait(
                        running, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in done_set:
                        n, dt, done, err = f.result()
                        results[n] = (dt, done, err)
                        running.pop(f)
        wall = time.time() - t0
        alive = server_alive(args.base_url)
        errors = {n: v for n, v in results.items() if v[2]}
        gen = sum(v[1] or 0 for v in results.values())
        slowest = max(results.items(), key=lambda kv: kv[1][0])
        acc = ""
        if args.metrics and alive:
            now = spec_acceptance(metrics_url)
            if now and now[0]:
                acc = f"  spec-accept(total)={now[1] / now[0]:4.0%}"
        print(
            f"  loop {loop:02d}/{args.loops}  wall={wall:6.0f}s  generated={gen}  "
            f"slowest=#{slowest[0]:02d} {slowest[1][0]:.0f}s  errors={len(errors)}  "
            f"server={'up' if alive else 'DOWN'}{acc}",
            flush=True,
        )
        for n, (dt, _, err) in sorted(errors.items()):
            print(f"      #{n:02d} after {dt:.0f}s: {err}")
        if args.verbose or not alive or errors:
            print("\n".join(sorted(lines, key=lambda l: float(l.split("start=+")[1].split("s")[0]))))
        if not alive:
            print(f"\nREPRODUCED: the server stopped answering during loop {loop} (engine dead)")
            return EXIT_REPRODUCED
        if errors:
            print(f"\nREPRODUCED: {len(errors)} request(s) failed in loop {loop} although the server is still up")
            return EXIT_REPRODUCED
    print(f"\nCLEAN: {args.loops} full-run loops completed, server still up")
    return EXIT_CLEAN


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-url",
        required=True,
        help="vLLM OpenAI-compatible endpoint, e.g. http://spark1:8000/v1",
    )
    p.add_argument(
        "--requests",
        default="2",
        metavar="N[,N...]",
        help="recorded request(s) to replay, cycled in order (see repro_a_filename_only.py --list); "
        "default 2. Example: 1,2 replays the tool-call turn then the agenda turn, like the agent",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="temperature 0 instead of the recorded sampling (1.0 / top_p 0.95): if the hang "
        "disappears, suspect the per-rank random draws of the speculative rejection sampler",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="after each repetition, print the server's speculative-decoding acceptance from /metrics",
    )
    p.add_argument("--repeat", type=int, default=30, help="how many times to send it (default 30)")
    p.add_argument(
        "--replay-all",
        action="store_true",
        help="instead of one request: replay all 47 recorded requests of a full agent run with "
        "their recorded start times (concurrency included), and loop that --loops times",
    )
    p.add_argument("--loops", type=int, default=10, help="with --replay-all: number of full-run loops (default 10)")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="with --replay-all: print one line per request in every loop (always printed for a failing loop)",
    )
    p.add_argument(
        "--order",
        choices=["deps", "offsets"],
        default="deps",
        help="with --replay-all: 'deps' (default) starts a request once every request that had "
        "finished before it started in the recording has finished here too — keeps the agent's "
        "phase structure (turn #02 alone in flight, searches concurrent) at this build's speed; "
        "'offsets' uses the recorded start times (see --time-scale)",
    )
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="with --replay-all: multiply the recorded start offsets (recorded on the slow broken "
        "build, 3266 s per run); 0.08 gives the ~260 s a run takes on the fixed build",
    )
    p.add_argument(
        "--nonce",
        default="tail",
        metavar="tail|head|none|N",
        help="where to inject a per-repetition random marker to defeat the prefix cache: tail = last "
        "message (default), head = first message, none = verbatim, N = first message with text at "
        "index >= N (everything before N stays cached, everything after is a real prefill)",
    )
    p.add_argument("--model", default=None, help="replay against another served model name")
    p.add_argument(
        "--requests-dir",
        default=None,
        help="directory of recorded requests: requests-finance/ (default; finance run recorded on the "
        "broken build, its turns #02 and #46 hung under the script), requests-concept/ (concept run "
        "recorded on the fixed build), requests-killed/ (the four generations killed under the benchmark)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="seconds to wait for one repetition (default 900; a healthy reply takes 1-3 min)",
    )
    args = p.parse_args()
    if args.repeat < 1:
        sys.exit(EXIT_CANNOT_RUN)

    from pathlib import Path
    recorded_model, requests = load_requests(Path(args.requests_dir) if args.requests_dir else Path(__file__).parent / "requests-finance")
    model = args.model or recorded_model
    preflight(args.base_url, model)
    if not short_probes(args.base_url, model):
        print("CANNOT RUN: the short probes fail, the server is broken independently of this test")
        return EXIT_CANNOT_RUN

    if args.replay_all:
        return replay_all_loops(args, model, requests)

    wanted = [int(x) for x in args.requests.split(",") if x.strip()]
    entries = []
    for n in wanted:
        entry = next((e for e in requests if e["n"] == n), None)
        if entry is None:
            print(f"CANNOT RUN: no recorded request #{n}")
            return EXIT_CANNOT_RUN
        entries.append(entry)
    for entry in entries:
        print(f"replaying {describe(entry)}  (recorded {entry['recorded_duration_s']:.0f}s)")
    print(
        f"{args.repeat} repetitions, nonce={args.nonce}, "
        f"sampling={'greedy' if args.greedy else 'recorded'}\n"
    )
    metrics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    last_acc = spec_acceptance(metrics_url) if args.metrics else None

    for i in range(1, args.repeat + 1):
        entry = entries[(i - 1) % len(entries)]
        body = dict(entry["body"])
        body["model"] = model
        body["stream"] = False
        if args.greedy:
            body["temperature"] = 0.0
            body.pop("top_p", None)
        sent = with_nonce(body, args.nonce)
        dt, resp, err = chat(args.base_url, sent, args.timeout)
        alive = server_alive(args.base_url)
        if resp is not None:
            usage = resp.get("usage") or {}
            done = usage.get("completion_tokens")
            rate = f"{done / dt:5.1f} tok/s" if done else ""
            finish = resp["choices"][0].get("finish_reason")
            acc = ""
            if args.metrics and alive:
                now = spec_acceptance(metrics_url)
                if now and last_acc:
                    d_draft = now[0] - last_acc[0]
                    d_acc = now[1] - last_acc[1]
                    if d_draft:
                        acc = f"  spec-accept={d_acc / d_draft:4.0%} ({d_draft} draft tok)"
                last_acc = now or last_acc
            print(
                f"  #{i:02d} req#{entry['n']:02d}  {dt:6.1f}s  generated={done}  {rate}  finish={finish}"
                f"  server={'up' if alive else 'DOWN'}{acc}"
            )
        else:
            print(f"  #{i:02d}  {dt:6.1f}s  ERROR {err}  server={'up' if alive else 'DOWN'}")
        if not alive:
            print(
                f"\nREPRODUCED: the server stopped answering during/after repetition {i} "
                f"(engine dead; check the head log for 'RPC call to sample_tokens timed out')"
            )
            return EXIT_REPRODUCED
        if resp is None:
            print(f"\nREPRODUCED: repetition {i} failed although the server is still up ({err})")
            return EXIT_REPRODUCED

    print(f"\nCLEAN: {args.repeat} repetitions completed, server still up")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
