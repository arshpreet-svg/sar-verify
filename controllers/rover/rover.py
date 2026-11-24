import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from controller import Robot

from sarverify.comms import Radio
from sarverify.config import RunConfig
from sarverify.flight import clamp, wrap

WHEEL_R = 0.11
TRACK = 0.40
MAX_WHEEL = 12.0
ARRIVE_TOL = 1.2


def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep())
    cfg = RunConfig.load()
    radio = Radio(robot, "rover", step)
    gps = robot.getDevice("gps")
    gps.enable(step)
    compass = robot.getDevice("compass")
    compass.enable(step)
    wheels = {
        "fl": robot.getDevice("front left wheel"),
        "bl": robot.getDevice("back left wheel"),
        "fr": robot.getDevice("front right wheel"),
        "br": robot.getDevice("back right wheel"),
    }
    for w in wheels.values():
        w.setPosition(float("inf"))
        w.setVelocity(0.0)
    cam = robot.getDevice("camera")
    if cam is not None:
        cam.enable(step * 8)
    speed = cfg.flight.rover_speed
    path = []
    idx = 0
    job = None
    last_pose = -10.0

    def drive(v, w):
        left = clamp((v - w * TRACK / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)
        right = clamp((v + w * TRACK / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)
        wheels["fl"].setVelocity(left)
        wheels["bl"].setVelocity(left)
        wheels["fr"].setVelocity(right)
        wheels["br"].setVelocity(right)

    while robot.step(step) != -1:
        t = robot.getTime()
        for msg in radio.poll():
            if msg["type"] == "goto":
                job = msg
                path = [tuple(p) for p in msg["path"]]
                idx = 1 if len(path) > 1 else 0
        x, y, _ = gps.getValues()
        c = compass.getValues()
        yaw = math.atan2(c[0], c[1])
        if job is not None and path:
            goal = path[-1]
            if math.hypot(goal[0] - x, goal[1] - y) < ARRIVE_TOL:
                drive(0.0, 0.0)
                radio.send("referee", "rover_arrived", cand_id=job["cand_id"])
                radio.send("commander", "rover_arrived", cand_id=job["cand_id"])
                job = None
                path = []
            else:
                while idx < len(path) - 1 and math.hypot(path[idx][0] - x, path[idx][1] - y) < 1.5:
                    idx += 1
                tx, ty = path[idx]
                err = wrap(math.atan2(ty - y, tx - x) - yaw)
                w = clamp(2.2 * err, -1.8, 1.8)
                v = speed * max(0.0, math.cos(err)) if abs(err) < 1.0 else 0.0
                drive(v, w)
        else:
            drive(0.0, 0.0)
        if t - last_pose > 0.5:
            radio.send("commander", "pose", who="rover", x=x, y=y, z=0.0)
            last_pose = t


main()
