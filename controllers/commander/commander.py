import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from controller import Robot

from sarverify import scene as scene_mod
from sarverify.belief import posterior, prior_given_flag
from sarverify.candidate import (
    ASK, CONFIRMED, DISPATCH, IGNORE, REJECTED, ROVER_GOING, SCOUT, SCOUTING,
    WAIT_OPERATOR, WAIT_ROVER, WAIT_SCOUT, WASTED, Candidate,
)
from sarverify.comms import Radio
from sarverify.config import RunConfig
from sarverify.operator import OperatorModel
from sarverify.planner import GridMap, plan, path_length
from sarverify.policy import make_policy
from sarverify.sensor import SensorModel

ASSOC_RADIUS = 6.0
TERMINAL = (CONFIRMED, REJECTED, WASTED)
MAX_PASSES = 2


class Commander:
    def __init__(self, robot, step, cfg):
        self.cfg = cfg
        self.t = 0.0
        self.radio = Radio(robot, "commander", step)
        self.sensor = SensorModel.from_csv(cfg.sensor_csv).marginal(cfg.scene.occluded_fraction)
        self.policy = make_policy(cfg, self.sensor)
        self.op = OperatorModel(cfg.operator)
        sc = scene_mod.generate(cfg.seed, cfg.scene, cfg.flight)
        self.gmap = GridMap(sc["half"], 1.0, sc["rover_obstacles"], 1.5)
        self.cands = {}
        self.next_id = 0
        self.poses = {}
        self.scout_ready = False
        self.scout_busy = None
        self.rover_busy = None
        self.sweep_done = False
        self.idle_sent = False
        self.passes = 1

    def pose(self, who):
        p = self.poses.get(who)
        return (p[0], p[1]) if p else None

    def publish(self, c):
        self.radio.send("referee", "cand_update", id=c.id, pos=list(c.pos), belief=c.belief, status=c.status, looks=c.looks)

    def new_candidate(self, pos, alt):
        p_hit, p_false = self.sensor.likelihoods(alt)
        b0 = prior_given_flag(self.cfg.scene.n_victims, self.cfg.scene.n_decoys, p_hit, p_false)
        c = Candidate(self.next_id, tuple(pos), b0, self.t)
        self.next_id += 1
        self.cands[c.id] = c
        c.log(self.t, "created")
        self.act(c)
        return c

    def act(self, c):
        action = self.policy.decide(c)
        c.log(self.t, action)
        if action == IGNORE:
            c.status = REJECTED
        elif action == SCOUT:
            c.status = WAIT_SCOUT
        elif action == ASK:
            c.status = WAIT_OPERATOR
            self.radio.send("referee", "op_request", cand_id=c.id, pos=list(c.pos), belief=c.belief, looked=c.looks > 0)
        elif action == DISPATCH:
            c.status = WAIT_ROVER
            if not self.policy.uses_gate:
                c.asked = True
                c.op_confirm = True
        self.publish(c)

    def on_detection(self, msg):
        pos = msg["pos"]
        for c in self.cands.values():
            if math.hypot(c.pos[0] - pos[0], c.pos[1] - pos[1]) < ASSOC_RADIUS:
                c.sightings += 1
                return
        self.new_candidate(pos, msg["alt"])

    def on_scout_obs(self, msg):
        c = self.cands[msg["cand_id"]]
        p_hit, p_false = self.sensor.likelihoods(msg["alt"], scout=True)
        c.belief = posterior(c.belief, msg["flag"], p_hit, p_false)
        c.looks += 1
        c.log(self.t, "look_flag" if msg["flag"] else "look_clear")
        self.scout_busy = None
        self.act(c)

    def on_op_response(self, msg):
        c = self.cands[msg["cand_id"]]
        c.asked = True
        c.op_confirm = bool(msg["confirm"])
        looked = c.looks > 0
        c.belief = self.op.posterior_yes(c.belief, looked) if c.op_confirm else self.op.posterior_no(c.belief, looked)
        c.log(self.t, "op_yes" if c.op_confirm else "op_no")
        self.act(c)

    def on_rover_report(self, msg):
        c = self.cands[msg["cand_id"]]
        c.status = CONFIRMED if msg["person_found"] else WASTED
        c.belief = 1.0 if msg["person_found"] else 0.0
        c.log(self.t, "confirmed" if msg["person_found"] else "wasted")
        self.rover_busy = None
        self.publish(c)

    def schedule(self):
        if self.scout_ready and self.scout_busy is None:
            waiting = [c for c in self.cands.values() if c.status == WAIT_SCOUT]
            if waiting:
                sp = self.pose("scout") or (0.0, 0.0)
                c = min(waiting, key=lambda k: math.hypot(k.pos[0] - sp[0], k.pos[1] - sp[1]))
                c.status = SCOUTING
                self.scout_busy = c.id
                self.radio.send("scout", "goto", cand_id=c.id, pos=list(c.pos))
                self.publish(c)
        if self.rover_busy is None:
            waiting = [c for c in self.cands.values() if c.status == WAIT_ROVER]
            if waiting:
                c = max(waiting, key=lambda k: k.belief)
                rp = self.pose("rover") or (-42.0, -47.0)
                path = plan(self.gmap, rp, c.pos)
                if path is None:
                    c.status = REJECTED
                    c.log(self.t, "no_path")
                    self.publish(c)
                    return
                c.status = ROVER_GOING
                self.rover_busy = c.id
                self.radio.send("rover", "goto", cand_id=c.id, path=[list(p) for p in path])
                self.radio.send("referee", "dispatch", cand_id=c.id, pos=list(c.pos), plan_m=path_length(path))
                self.publish(c)

    def step(self, t):
        self.t = t
        for msg in self.radio.poll():
            k = msg["type"]
            if k == "pose":
                self.poses[msg["who"]] = (msg["x"], msg["y"], msg["z"])
            elif k == "detection":
                self.on_detection(msg)
            elif k == "scout_ready":
                self.scout_ready = True
            elif k == "scout_obs":
                self.on_scout_obs(msg)
            elif k == "op_response":
                self.on_op_response(msg)
            elif k == "rover_report":
                self.on_rover_report(msg)
            elif k == "sweep_done":
                self.sweep_done = True
        self.schedule()
        if self.sweep_done and not self.idle_sent:
            if all(c.status in TERMINAL for c in self.cands.values()):
                confirmed = sum(1 for c in self.cands.values() if c.status == CONFIRMED)
                if confirmed < self.cfg.scene.n_victims and self.passes < MAX_PASSES and t < self.cfg.flight.time_limit - 360.0:
                    self.passes += 1
                    self.sweep_done = False
                    self.radio.send("mothership", "resweep")
                    self.radio.send("referee", "resweep", passes=self.passes)
                else:
                    self.radio.send("referee", "mission_idle")
                    self.idle_sent = True


def main():
    robot = Robot()
    step = int(robot.getBasicTimeStep())
    cmd = Commander(robot, step, RunConfig.load())
    while robot.step(step) != -1:
        cmd.step(robot.getTime())


main()
