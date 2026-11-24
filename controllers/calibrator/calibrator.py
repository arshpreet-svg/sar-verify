import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from controller import Supervisor

from sarverify.worldfile import look_orientation


def main():
    sup = Supervisor()
    step = int(sup.getBasicTimeStep())
    plan = json.load(open(os.path.join(ROOT, "runs", "calib", "plan.json")))
    outdir = os.path.join(ROOT, "runs", "calib", "img")
    os.makedirs(outdir, exist_ok=True)
    cam = sup.getDevice("cam")
    cam.enable(step)
    me = sup.getSelf()
    tr, rot = me.getField("translation"), me.getField("rotation")
    for _ in range(5):
        sup.step(step)
    for e in plan:
        tr.setSFVec3f([e["x"], e["y"], e["alt"]])
        rot.setSFRotation(list(look_orientation(e["yaw"], e.get("pitch", math.pi / 2))))
        for _ in range(3):
            sup.step(step)
        cam.saveImage(os.path.join(outdir, f"{e['idx']:05d}.png"), 100)
    sup.simulationQuit(0)


main()
