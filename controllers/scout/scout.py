import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from controller import Robot

import cv2

from sarverify.comms import Radio
from sarverify.config import RunConfig
from sarverify.detector import PersonDetector, look_flag, webots_image
from sarverify.flight import Quad

TRANSIT_ALT = 15.0


def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep())
    cfg = RunConfig.load()
    quad = Quad(robot, step, max_speed=2.6)
    radio = Radio(robot, "scout", step)
    look_alt = cfg.flight.scout_alt
    standoff = cfg.flight.scout_standoff
    look_pitch = math.atan2(look_alt - 0.6, standoff)
    dwell = cfg.flight.scout_dwell
    det = PersonDetector()
    view_dir = os.path.join(cfg.out_dir, "scout_view")
    os.makedirs(view_dir, exist_ok=True)
    n_looks = 0
    stand = None
    state = "takeoff"
    job = None
    home = None
    dwell_start = 0.0
    last_pose = -10.0
    quad.target_alt = TRANSIT_ALT
    quad.hold()
    while robot.step(step) != -1:
        t = robot.getTime()
        for msg in radio.poll():
            if msg["type"] == "goto":
                job = msg
                state = "transit"
                cx, cy = msg["pos"][0], msg["pos"][1]
                sx, sy, _ = quad.position()
                d = math.hypot(sx - cx, sy - cy)
                ux, uy = ((sx - cx) / d, (sy - cy) / d) if d > 1.0 else (0.0, -1.0)
                stand = (cx + ux * standoff, cy + uy * standoff)
                quad.yaw_target = math.atan2(cy - stand[1], cx - stand[0])
                quad.goto(stand[0], stand[1], TRANSIT_ALT)
            elif msg["type"] == "home" and home is not None:
                job = None
                state = "returning"
                quad.goto(home[0], home[1], TRANSIT_ALT)
        if t > 2.0:
            if state == "takeoff":
                if quad.position()[2] > TRANSIT_ALT - 2.0:
                    x, y, _ = quad.position()
                    home = (x, y)
                    quad.hold()
                    state = "idle"
                    radio.send("commander", "scout_ready")
            elif state == "transit":
                if quad.distance_to_target() < 2.0:
                    quad.target_alt = look_alt
                    quad.cam_pitch_cmd = look_pitch
                    state = "descend"
            elif state == "descend":
                if quad.at_target(tol_xy=1.5, tol_alt=0.8, max_speed=0.6):
                    dwell_start = t
                    state = "dwell"
            elif state == "dwell":
                if t - dwell_start >= dwell:
                    img = webots_image(quad.camera)
                    pos = quad.position()
                    yaw = quad.yaw()
                    res = look_flag(det, img, pos, yaw, quad.cam_pitch_cmd, job["pos"], quad.camera.getFov())
                    n_looks += 1
                    path = os.path.join(view_dir, f"look_{n_looks:03d}.jpg")
                    cv2.imwrite(path, img)
                    radio.send(
                        "referee", "look", cand_id=job["cand_id"], flag=bool(res["flag"]), score=res["score"],
                        dets=[list(d_) for d_ in res["dets"]], img=path, cam=list(pos), yaw=yaw,
                        pitch=quad.cam_pitch_cmd, target_px=list(res["target_px"]) if res["target_px"] else None,
                    )
                    quad.cam_pitch_cmd = 1.5
                    quad.target_alt = TRANSIT_ALT
                    state = "climb"
            elif state == "climb":
                if quad.position()[2] > TRANSIT_ALT - 2.0:
                    state = "idle"
            elif state == "returning":
                if quad.distance_to_target() < 2.0:
                    state = "idle"
        if t - last_pose > 1.0:
            x, y, z = quad.position()
            radio.send("commander", "pose", who="scout", x=x, y=y, z=z)
            last_pose = t
        quad.update()


main()
