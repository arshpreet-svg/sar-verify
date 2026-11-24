import argparse
import copy
import csv
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sarverify import fastsim
from sarverify.config import RunConfig

TUNE_SEEDS = list(range(1001, 1061))
EVAL_SEEDS = list(range(2001, 2201))
FD_COSTS = [10, 20, 40, 80, 160, 320]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
POLICIES = ["b0", "b1", "b2", "ours"]


def make_cfg(seed, policy, fd, tau=0.5):
    cfg = RunConfig()
    cfg.seed = seed
    cfg.policy = policy
    cfg.threshold = tau
    cfg.costs.false_dispatch = fd
    return cfg


PASSES = 1


def run_set(seeds, policy, fd, truth, tau=0.5):
    return [fastsim.run_one(make_cfg(s, policy, fd, tau), truth, max_passes=PASSES) for s in seeds]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def tune_tau(fd, truth):
    best = None
    for tau in TAUS:
        rows = run_set(TUNE_SEEDS, "b1", fd, truth, tau)
        c = mean(r["cost"] for r in rows)
        if best is None or c < best[1]:
            best = (tau, c)
    return best[0]


def boot_ci(values, n=2000, seed=0):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    m = len(values)
    means = sorted(mean(values[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return mean(values), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def metrics_of(rows):
    out = {}
    out["recall"] = boot_ci([r["recall"] for r in rows], seed=1)
    out["wasted"] = boot_ci([r["wasted_dispatches"] for r in rows], seed=2)
    out["mean_ttc"] = boot_ci([r["mean_time_to_confirm"] for r in rows if r["mean_time_to_confirm"] is not None], seed=3)
    out["first_ttc"] = boot_ci([r["time_to_first_confirm"] for r in rows if r["time_to_first_confirm"] is not None], seed=4)
    out["mission_time"] = boot_ci([r["mission_time"] for r in rows], seed=5)
    out["cost"] = boot_ci([r["cost"] for r in rows], seed=6)
    out["looks"] = boot_ci([r["scout_looks"] for r in rows], seed=7)
    out["asks"] = boot_ci([r["operator_requests"] for r in rows], seed=8)
    return out


def paired_diff(a_rows, b_rows, key):
    d = [a[key] - b[key] for a, b in zip(a_rows, b_rows)]
    return boot_ci(d, seed=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS))
    args = ap.parse_args()
    global PASSES
    PASSES = args.passes
    os.makedirs(args.out, exist_ok=True)
    truth = fastsim.load_scout_truth()
    seeds = EVAL_SEEDS[: args.eval_seeds]
    table = []
    raw = []
    taus = {}
    for fd in FD_COSTS:
        taus[fd] = tune_tau(fd, truth)
        sets = {}
        for pol in POLICIES:
            sets[pol] = run_set(seeds, pol, fd, truth, taus[fd])
            for r in sets[pol]:
                raw.append({"fd": fd, "policy": pol, "tau": taus[fd] if pol == "b1" else "", **{k: r[k] for k in ("seed", "mission_time", "victims_total", "victims_confirmed", "recall", "time_to_first_confirm", "mean_time_to_confirm", "scout_looks", "operator_requests", "dispatches", "wasted_dispatches", "cost")}})
        for pol in POLICIES:
            m = metrics_of(sets[pol])
            row = {"fd": fd, "policy": pol, "tau": taus[fd] if pol == "b1" else None, "n": len(seeds)}
            for k, (v, lo, hi) in m.items():
                row[k] = v
                row[k + "_lo"] = lo
                row[k + "_hi"] = hi
            table.append(row)
        for pol in ("b0", "b1", "b2"):
            dm, lo, hi = paired_diff(sets["ours"], sets[pol], "cost")
            table.append({"fd": fd, "policy": f"ours_minus_{pol}", "cost": dm, "cost_lo": lo, "cost_hi": hi, "n": len(seeds)})
        print(f"fd={fd} done tau_b1={taus[fd]}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"eval_seeds": [seeds[0], seeds[-1], len(seeds)], "tune_seeds": [TUNE_SEEDS[0], TUNE_SEEDS[-1], len(TUNE_SEEDS)], "b1_tau": taus, "rows": table}, f, indent=2)
    keys = sorted({k for r in table for k in r})
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
