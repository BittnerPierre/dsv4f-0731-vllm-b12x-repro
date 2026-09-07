#!/usr/bin/env python3
"""Search for the engine hang by replaying mutated versions of the recorded run, on ONE server,
until it dies. Every candidate is fully described by a seed, so a hit can be replayed.

    python3 repro_c_fuzz.py --base-url http://spark1:8000/v1                # until a hang (or Ctrl-C)
    python3 repro_c_fuzz.py --base-url http://spark1:8000/v1 --candidates 40
    python3 repro_c_fuzz.py --base-url http://spark1:8000/v1 --seed 1234     # replay one candidate

A candidate = a subset of the 47 recorded requests (first three turns always kept), a scheduling mode
(recorded dependencies / dependencies with a concurrency cap / pairs "short then long" started
together so that the batch shrinks to one), a nonce position (2, 3, tail, head), optional
greedy sampling, and 1-3 loops. Exit 1 = REPRODUCED (the seed and the candidate are printed),
0 = all candidates clean, 2 = cannot run. Stdlib only.
"""

import argparse
import concurrent.futures
import random
import sys
import time

from common import EXIT_CANNOT_RUN, EXIT_CLEAN, EXIT_REPRODUCED, chat, load_requests, preflight, short_probes
from repro_c_engine_hang import recorded_dependencies, server_alive, spec_acceptance, with_nonce


def make_candidate(rng: random.Random, requests: list[dict], deps: dict[int, set[int]]) -> dict:
    ns = [e["n"] for e in requests]
    mode = rng.choice(["deps", "deps", "capped", "pairs"])
    if mode == "pairs":
        # short first-turn requests (2 messages) paired with long later turns (>= 8 messages)
        short = [n for n in ns if len(next(e for e in requests if e["n"] == n)["body"]["messages"]) == 2]
        long_ = [n for n in ns if len(next(e for e in requests if e["n"] == n)["body"]["messages"]) >= 8]
        k = rng.randint(3, 8)
        pairs = [(rng.choice(short), rng.choice(long_)) for _ in range(k)]
        return {"mode": mode, "pairs": pairs, "nonce": rng.choice(["2", "3", "tail", "head"]),
                "greedy": rng.random() < 0.15, "loops": rng.randint(1, 3)}
    # random subset; the first three turns are always kept, ordering only honours the
    # dependencies among the kept requests
    keep = set(rng.sample(ns, rng.randint(6, len(ns)))) | {1, 2, 3}
    return {"mode": mode, "subset": sorted(keep), "cap": rng.randint(1, 4) if mode == "capped" else 16,
            "nonce": rng.choice(["2", "2", "3", "tail", "head"]), "greedy": rng.random() < 0.15,
            "loops": rng.randint(1, 3)}


def send(args, model, e, nonce, greedy, t0, inflight, lines):
    body = dict(e["body"])
    body["model"] = model
    body["stream"] = False
    if greedy:
        body["temperature"] = 0.0
        body.pop("top_p", None)
    started = time.time() - t0
    inflight.add(e["n"])
    k = len(inflight)
    dt, resp, err = chat(args.base_url, with_nonce(body, nonce), args.timeout)
    inflight.discard(e["n"])
    usage = (resp.get("usage") or {}) if resp else {}
    lines.append(f"      #{e['n']:02d} start=+{started:5.0f}s inflight={k} prompt={usage.get('prompt_tokens')} "
                 f"generated={usage.get('completion_tokens')} {dt:6.1f}s{'  ERROR ' + err[:60] if err else ''}")
    return e["n"], err


def run_candidate(args, model, requests, deps, cand) -> tuple[bool, list[str]]:
    """Returns (server still alive and no error, per-request lines)."""
    by_n = {e["n"]: e for e in requests}
    lines: list[str] = []
    for loop in range(cand["loops"]):
        t0 = time.time()
        inflight: set[int] = set()
        errors = 0
        if cand["mode"] == "pairs":
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for a, b in cand["pairs"]:
                    fa = pool.submit(send, args, model, by_n[a], cand["nonce"], cand["greedy"], t0, inflight, lines)
                    fb = pool.submit(send, args, model, by_n[b], cand["nonce"], cand["greedy"], t0, inflight, lines)
                    errors += sum(1 for f in (fa, fb) if f.result()[1])
                    if not server_alive(args.base_url):
                        return False, lines
        else:
            pending = {n: by_n[n] for n in cand["subset"]}
            done: set[int] = set()
            running: dict = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=cand["cap"]) as pool:
                while pending or running:
                    ready = [n for n in sorted(pending) if (deps[n] & set(cand["subset"])) <= done]
                    for n in ready[: max(0, cand["cap"] - len(running))]:
                        running[pool.submit(send, args, model, pending.pop(n), cand["nonce"], cand["greedy"], t0, inflight, lines)] = n
                    if not running:
                        break
                    fin, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in fin:
                        n, err = f.result()
                        done.add(n)
                        errors += bool(err)
                        running.pop(f)
        if errors or not server_alive(args.base_url):
            return False, lines
    return True, lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--candidates", type=int, default=0, help="stop after N candidates (default: until a hang)")
    p.add_argument("--seed", type=int, default=None, help="replay exactly one candidate")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--requests-dir",
        default=None,
        help="directory of recorded requests: requests-finance/ (default; finance run recorded on the "
        "broken build, its turns #02 and #46 hung under the script), requests-concept/ (concept run "
        "recorded on the fixed build), requests-killed/ (the four generations killed under the benchmark)",
    )
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()

    from pathlib import Path
    recorded_model, requests = load_requests(Path(args.requests_dir) if args.requests_dir else Path(__file__).parent / "requests-finance")
    model = args.model or recorded_model
    preflight(args.base_url, model)
    if not short_probes(args.base_url, model):
        return EXIT_CANNOT_RUN
    deps = recorded_dependencies(requests)
    metrics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/metrics"

    seeds = [args.seed] if args.seed is not None else iter(lambda: random.randrange(1, 10**9), None)
    i = 0
    for seed in seeds:
        i += 1
        if args.candidates and i > args.candidates:
            break
        cand = make_candidate(random.Random(seed), requests, deps)
        desc = {k: v for k, v in cand.items() if k != "subset"}
        if "subset" in cand:
            desc["n_requests"] = len(cand["subset"])
        print(f"[{i:03d}] seed={seed} {desc}", flush=True)
        t = time.time()
        ok, lines = run_candidate(args, model, requests, deps, cand)
        acc = spec_acceptance(metrics_url)
        acc_s = f" spec-accept(total)={acc[1] / acc[0]:.0%}" if acc and acc[0] else ""
        print(f"      -> {'clean' if ok else 'FAILED'} in {time.time() - t:.0f}s{acc_s}", flush=True)
        if not ok:
            print("\n".join(sorted(lines, key=lambda l: float(l.split('start=+')[1].split('s')[0]))))
            if not server_alive(args.base_url):
                print(f"\nREPRODUCED: server dead after candidate seed={seed}: {cand}")
                return EXIT_REPRODUCED
            print(f"\nREPRODUCED: request error, server still up, candidate seed={seed}: {cand}")
            return EXIT_REPRODUCED
    print("\nCLEAN: no hang")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
