import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2

from sarverify import scene as scene_mod
from sarverify.config import RunConfig
from sarverify.detector import CONF, PersonDetector, look_flag
from sarverify.worldfile import write_world

WEBOTS = "/Applications/Webots.app/Contents/MacOS/webots"
FOV = 1.0
W, H = 640, 480
BEARINGS = 3


def empty_points(scene, rng, n):
    pts = []
    objs = [(o["x"], o["y"], max(o["sx"], o["sy"]) / 2 + 6) for o in scene["obstacles"] if o["kind"] != "flood"]
    objs += [(t["x"], t["y"], 8.0) for t in scene["victims"] + scene["decoys"]]
    half = 38.0
    tries = 0
    while len(pts) < n and tries < 2000:
        tries += 1
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if all(math.hypot(x - ox, y - oy) > r for ox, oy, r in objs):
            pts.append((x, y))
    return pts


def build_plan(scene, rng, n_empty, flight):
    items = [("victim", v["x"], v["y"], v["occlusion"], v.get("pose", "lying")) for v in scene["victims"]]
    items += [("decoy", d["x"], d["y"], 0.0, d["kind"]) for d in scene["decoys"]]
    items += [("empty", x, y, 0.0, "none") for x, y in empty_points(scene, rng, n_empty)]
    alt, d = flight.scout_alt, flight.scout_standoff
    pitch = math.atan2(alt - 0.6, d)
    plan = []
    for kind, tx, ty, occ, sub in items:
        for k in range(BEARINGS):
            b = rng.uniform(0, 2 * math.pi)
            sg = 0.02 * 25.0 + 0.1
            ex, ey = tx + rng.gauss(0, sg), ty + rng.gauss(0, sg)
            cx, cy = ex + d * math.cos(b), ey + d * math.sin(b)
            plan.append({"kind": kind, "sub": sub, "occ": occ, "tx": tx, "ty": ty, "ex": ex, "ey": ey,
                         "x": cx, "y": cy, "alt": alt, "yaw": math.atan2(ey - cy, ex - cx), "pitch": pitch})
    for i, e in enumerate(plan):
        e["idx"] = i
    return plan


def render_seed(seed, n_empty, cfg):
    cdir = os.path.join(ROOT, "runs", "calib")
    sdir = os.path.join(cdir, f"scout_seed_{seed}")
    if os.path.exists(os.path.join(sdir, "plan.json")) and os.path.isdir(os.path.join(sdir, "img")):
        return json.load(open(os.path.join(sdir, "plan.json"))), sdir
    scene = scene_mod.generate(seed, cfg.scene, cfg.flight)
    rng = random.Random(seed * 131 + 5)
    plan = build_plan(scene, rng, n_empty, cfg.flight)
    for stale in ("plan.json", "img"):
        p = os.path.join(cdir, stale)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "plan.json"), "w") as f:
        json.dump(plan, f)
    world = os.path.join(ROOT, "worlds", "calib.wbt")
    write_world(scene, cfg, world, calib=True)
    py = os.path.join(ROOT, ".venv", "bin", "python")
    with open(os.path.join(ROOT, "controllers", "calibrator", "runtime.ini"), "w") as f:
        f.write(f"[python]\nCOMMAND = {py}\n")
    subprocess.call([WEBOTS, "--mode=fast", "--batch", "--minimize", "--stdout", "--stderr", world], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir)
    shutil.move(os.path.join(cdir, "img"), os.path.join(sdir, "img"))
    shutil.move(os.path.join(cdir, "plan.json"), os.path.join(sdir, "plan.json"))
    return plan, sdir


def eval_seed(seed, plan, sdir, det, keep_dir):
    rows = []
    for e in plan:
        img = cv2.imread(os.path.join(sdir, "img", f"{e['idx']:05d}.png"))
        if img is None:
            continue
        res = look_flag(det, img, (e["x"], e["y"], e["alt"]), e["yaw"], e["pitch"], (e["ex"], e["ey"]), FOV)
        rows.append({"seed": seed, "kind": e["kind"], "sub": e["sub"], "occ": round(e["occ"], 3), "flag": int(res["flag"]), "score": round(res["score"], 4), "n_dets": len(res["dets"])})
        if keep_dir and e["idx"] % 11 == 0:
            for x1, y1, x2, y2, sc in res["dets"]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if res["target_px"]:
                cv2.circle(img, (int(res["target_px"][0]), int(res["target_px"][1])), 16, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(keep_dir, f"s{seed}_{e['kind']}_{e['sub']}_{e['idx']}.png"), img)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(rows, flight):
    def rate(sel):
        n = len(sel)
        k = sum(r["flag"] for r in sel)
        lo, hi = wilson(k, n)
        return {"n": n, "k": k, "rate": round(k / n, 4) if n else 0.0, "ci95": [round(lo, 3), round(hi, 3)]}

    vic = [r for r in rows if r["kind"] == "victim"]
    out = {
        "source": "measured_webots_render",
        "detector": "sar_person_yolov8n_v2_best_int8.onnx",
        "conf": CONF,
        "geometry": {"alt_m": flight.scout_alt, "standoff_m": flight.scout_standoff, "pitch_down_rad": round(math.atan2(flight.scout_alt - 0.6, flight.scout_standoff), 3)},
        "victim_all": rate(vic),
        "victim_standing": rate([r for r in vic if r["sub"] == "standing"]),
        "victim_lying_clear": rate([r for r in vic if r["sub"] == "lying" and r["occ"] == 0]),
        "victim_lying_occluded": rate([r for r in vic if r["sub"] == "lying" and r["occ"] > 0]),
        "decoy": rate([r for r in rows if r["kind"] == "decoy"]),
        "empty": rate([r for r in rows if r["kind"] == "empty"]),
    }
    def smooth(r):
        return round((r["k"] + 1) / (r["n"] + 2), 4)

    out["p_hit"] = smooth(out["victim_all"])
    out["p_false_decoy"] = smooth(out["decoy"])
    out["p_false_empty"] = smooth(out["empty"])
    out["note"] = "p_* are Laplace-smoothed (k+1)/(n+2) so no rate is exactly 0 or 1; raw rates and Wilson 95% intervals are in each block"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 116)))
    ap.add_argument("--empty", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "scout_detector.json"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "scout_detector_calibration.csv"))
    ap.add_argument("--keep", default="")
    args = ap.parse_args()
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    cfg = RunConfig()
    det = PersonDetector()
    allrows = []
    for s in args.seeds:
        plan, sdir = render_seed(s, args.empty, cfg)
        allrows += eval_seed(s, plan, sdir, det, args.keep)
        print(f"seed {s}: {len(plan)} images", flush=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        wr.writeheader()
        wr.writerows(allrows)
    summ = summarize(allrows, cfg.flight)
    summ["seeds"] = args.seeds
    with open(args.out, "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
