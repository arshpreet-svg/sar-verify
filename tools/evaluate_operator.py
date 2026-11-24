import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from sarverify import fastsim
from sarverify.config import RunConfig
from evaluate import EVAL_SEEDS, TAUS, TUNE_SEEDS, boot_ci, mean

ASK_COSTS = [4, 10, 20, 40]
DELAYS = [6, 30, 60, 120]
POLICIES = ["b0", "b1", "b2", "ours"]


def cfg_for(seed, pol, ask, delay, tau=0.5):
    c = RunConfig()
    c.seed, c.policy, c.threshold = seed, pol, tau
    c.costs.ask = ask
    c.operator.delay_mean = delay
    c.operator.delay_sd = delay / 3.0
    return c


def runs(seeds, pol, ask, delay, truth, passes, tau=0.5):
    return [fastsim.run_one(cfg_for(s, pol, ask, delay, tau), truth, max_passes=passes, serial_operator=True) for s in seeds]


def main():
    truth = fastsim.load_scout_truth()
    passes = 2
    out = []
    for ask in ASK_COSTS:
        for delay in DELAYS:
            best = min(TAUS, key=lambda t: mean(r["cost"] for r in runs(TUNE_SEEDS, "b1", ask, delay, truth, passes, t)))
            sets = {p: runs(EVAL_SEEDS, p, ask, delay, truth, passes, best) for p in POLICIES}
            for p in POLICIES:
                row = {"ask_cost": ask, "op_delay_s": delay, "policy": p, "tau": best if p == "b1" else ""}
                for k, key in (("cost", "cost"), ("recall", "recall"), ("wasted", "wasted_dispatches"), ("looks", "scout_looks"), ("asks", "operator_requests"), ("mission_time", "mission_time")):
                    v, lo, hi = boot_ci([r[key] for r in sets[p]], seed=3)
                    row[k], row[k + "_lo"], row[k + "_hi"] = v, lo, hi
                v, lo, hi = boot_ci([r["mean_time_to_confirm"] for r in sets[p] if r["mean_time_to_confirm"] is not None], seed=4)
                row["ttc"], row["ttc_lo"], row["ttc_hi"] = v, lo, hi
                out.append(row)
            for p in ("b0", "b1", "b2"):
                d = [a["cost"] - b["cost"] for a, b in zip(sets["ours"], sets[p])]
                v, lo, hi = boot_ci(d, seed=11)
                out.append({"ask_cost": ask, "op_delay_s": delay, "policy": f"ours_minus_{p}", "cost": v, "cost_lo": lo, "cost_hi": hi})
            print("ask", ask, "delay", delay, "done", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "operator_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    keys = sorted({k for r in out for k in r})
    with open(os.path.join(ROOT, "results", "operator_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
