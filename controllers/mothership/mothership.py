import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from controller import Robot

from sarverify.comms import Radio
from sarverify.config import RunConfig
from sarverify.flight import Quad
from sarverify.scene import sweep_lanes


def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep())
    cfg = RunConfig.load()
    quad = Quad(robot, step, max_speed=2.6)
    radio = Radio(robot, "mothership", step)
    lanes = sweep_lanes(cfg.scene.size, cfg.flight.swath, cfg.flight.sweep_alt)
    alt = cfg.flight.sweep_alt
    state = "takeoff"
    idx = 0
    last_pose = -10.0
    quad.target_alt = alt
    quad.hold()
    while robot.step(step) != -1:
        t = robot.getTime()
        for msg in radio.poll():
            if msg["type"] == "resweep" and state == "done":
                lanes = lanes[::-1]
                idx = 0
                state = "sweep"
        if t > 2.0:
            if state == "takeoff":
                if quad.position()[2] > alt - 2.0:
                    state = "sweep"
            elif state == "sweep":
                if quad.target_xy != lanes[idx]:
                    quad.goto(lanes[idx][0], lanes[idx][1], alt)
                elif quad.distance_to_target() < 3.0:
                    idx += 1
                    if idx >= len(lanes):
                        state = "done"
                        quad.hold()
                        radio.send("commander", "sweep_done")
                        radio.send("referee", "sweep_done")
        if t - last_pose > 1.0:
            x, y, z = quad.position()
            radio.send("commander", "pose", who="mothership", x=x, y=y, z=z)
            last_pose = t
        quad.update()


main()
