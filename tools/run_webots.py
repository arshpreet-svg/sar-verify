import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sarverify import scene as scene_mod
from sarverify.config import RunConfig
from sarverify.worldfile import write_world

WEBOTS = "/Applications/Webots.app/Contents/MacOS/webots"
CONTROLLERS = ["mothership", "scout", "rover", "commander", "referee"]


def write_runtime_ini():
    py = os.path.join(ROOT, ".venv", "bin", "python")
    for name in CONTROLLERS:
        with open(os.path.join(ROOT, "controllers", name, "runtime.ini"), "w") as f:
            f.write(f"[python]\nCOMMAND = {py}\n")


def build_config(args):
    cfg = RunConfig(seed=args.seed, policy=args.policy)
    cfg.threshold = args.threshold
    cfg.operator.mode = args.operator
    cfg.headless = args.headless
    cfg.out_dir = os.path.abspath(args.out) if args.out else os.path.join(ROOT, "runs", f"{args.policy}_seed{args.seed}")
    cfg.record_video = os.path.abspath(args.record) if args.record else ""
    cfg.snapshot_dir = os.path.abspath(args.snapshots) if args.snapshots else ""
    if args.time_limit:
        cfg.flight.time_limit = args.time_limit
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=["ours", "b0", "b1", "b2"], default="ours")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--operator", choices=["auto", "manual"], default="auto")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--world-only", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--snapshots", default="")
    args = ap.parse_args()

    cfg = build_config(args)
    cfg.save()
    scene = scene_mod.generate(cfg.seed, cfg.scene, cfg.flight)
    world = os.path.join(ROOT, "worlds", "sar_verify.wbt")
    write_world(scene, cfg, world)
    write_runtime_ini()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)
    print(f"world: {world}")
    print(f"config: seed={cfg.seed} policy={cfg.policy} out={cfg.out_dir}")
    if args.world_only:
        return
    cmd = [WEBOTS]
    if args.headless:
        cmd += ["--mode=fast", "--no-rendering", "--batch", "--minimize", "--stdout", "--stderr"]
    else:
        cmd += ["--mode=fast" if args.fast else "--mode=realtime", "--stdout", "--stderr"]
    cmd.append(world)
    sys.exit(subprocess.call(cmd))


main()
