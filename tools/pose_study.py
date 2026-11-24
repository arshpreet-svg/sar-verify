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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2

from calibrate_detector import W, H, FOV, WEBOTS, empty_points
from sarverify import scene as scene_mod
from sarverify.cinema import project
from sarverify.config import RunConfig
from sarverify.detector import PersonDetector, detect_scaled
from sarverify.worldfile import write_world

ELS = [90, 60, 40, 25]
RANGES = [15, 30, 50]


def make_plan(scene, rng, pose):
    items = [("victim", v["x"], v["y"], v["occlusion"]) for v in scene["victims"]]
    items += [("decoy", d["x"], d["y"], 0.0) for d in scene["decoys"]]
    items += [("empty", x, y, 0.0) for x, y in empty_points(scene, rng, 6)]
    plan = []
    for kind, tx, ty, occ in items:
        for el in ELS:
            for R in RANGES:
                if kind != "victim" and R != 30:
                    continue
                b = rng.uniform(0, 2 * math.pi)
                s = 0.02 * R + 0.1
                ax, ay = tx + rng.gauss(0, s), ty + rng.gauss(0, s)
                e = math.radians(el)
                cam = (ax + R * math.cos(b) * math.cos(e), ay + R * math.sin(b) * math.cos(e), R * math.sin(e))
                yaw = math.atan2(-math.sin(b), -math.cos(b)) if el < 89 else rng.uniform(0, 2 * math.pi)
                plan.append({"kind": kind, "tx": tx, "ty": ty, "occ": occ, "el": el, "R": R, "x": cam[0], "y": cam[1], "alt": cam[2],
                             "yaw": yaw, "pitch": e, "aim": (ax, ay, 0.5)})
    for i, p in enumerate(plan):
        p["idx"] = i
    return plan


def match(e, dets, pose):
    cam = (e["x"], e["y"], e["alt"])
    tz = 0.3 if pose == "lying" else 0.9
    r = project((e["tx"], e["ty"], tz), cam, e["aim"], W, H, FOV)
    if r is None:
        return 0.0
    f = (W / 2) / math.tan(FOV / 2)
    rad = 3.0 * f / r[2]
    best = 0.0
    for x1, y1, x2, y2, sc in dets:
        if math.hypot((x1 + x2) / 2 - r[0], (y1 + y2) / 2 - r[1]) <= rad:
            best = max(best, sc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "pose_study.csv"))
    args = ap.parse_args()
    det = PersonDetector()
    rows = []
    cdir = os.path.join(ROOT, "runs", "calib")
    for pose in ("lying", "standing"):
        for seed in args.seeds:
            cfg = RunConfig(seed=seed)
            scene = scene_mod.generate(seed, cfg.scene, cfg.flight)
            rng = random.Random(seed * 17 + 3)
            plan = make_plan(scene, rng, pose)
            shutil.rmtree(cdir, ignore_errors=True)
            os.makedirs(cdir)
            json.dump(plan, open(os.path.join(cdir, "plan.json"), "w"))
            world = os.path.join(ROOT, "worlds", "calib.wbt")
            write_world(scene, cfg, world, calib=True, pose=pose)
            subprocess.call([WEBOTS, "--mode=fast", "--batch", "--minimize", "--stdout", "--stderr", world], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for e in plan:
                img = cv2.imread(os.path.join(cdir, "img", f"{e['idx']:05d}.png"))
                if img is None:
                    continue
                for ppm in (0, 25):
                    dets = detect_scaled(det, img, e["R"], FOV, ppm, conf=0.05)
                    rows.append({"pose": pose, "seed": seed, "kind": e["kind"], "occ": e["occ"], "el": e["el"], "R": e["R"], "ppm": ppm, "best": round(match(e, dets, pose), 4)})
            print(pose, seed, len(plan), flush=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
