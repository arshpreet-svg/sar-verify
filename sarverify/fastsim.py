import copy
import json
import math
import os
import random

from . import scene as scene_mod
from .belief import posterior, prior_given_flag
from .candidate import (
    ASK, CONFIRMED, DISPATCH, IGNORE, REJECTED, ROVER_GOING, SCOUT, SCOUTING,
    WAIT_OPERATOR, WAIT_ROVER, WAIT_SCOUT, WASTED, Candidate,
)
from .operator import OperatorModel
from .planner import GridMap, path_length, plan
from .policy import make_policy
from .sensor import SensorModel

ASSOC_RADIUS = 6.0
MATCH_RADIUS = 4.0
RESAMPLE_GAP = 40.0
TERMINAL = (CONFIRMED, REJECTED, WASTED)
DT = 0.5
MOTHER_SPEED = 1.75
SCOUT_SPEED = 1.9
TAKEOFF_S = 8.0
CLIMB_S = 8.0
DESCEND_S = 8.0
ROVER_EFFECTIVE_SPEED = 0.46
ROVER_TRIP_FACTOR = 1.47
ROVER_START = (-42.0, -47.0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_scout_truth(path=None):
    path = path or os.path.join(ROOT, "data", "scout_detector.json")
    with open(path) as f:
        d = json.load(f)
    return {
        "standing": d["victim_standing"]["rate"],
        "lying_clear": d["victim_lying_clear"]["rate"],
        "lying_occluded": d["victim_lying_occluded"]["rate"],
        "decoy": d["decoy"]["rate"],
        "empty": d["empty"]["rate"],
        "decoy_k": d["decoy"]["k"],
        "decoy_n": d["decoy"]["n"],
    }


def scout_flag_prob(truth, tg):
    if tg is None:
        return truth["empty"]
    if tg["kind"] == "decoy":
        return truth["decoy"]
    if tg["pose"] == "standing":
        return truth["standing"]
    return truth["lying_occluded"] if tg["occ"] > 0 else truth["lying_clear"]


def lane_path(lanes, start):
    pts = [start] + list(lanes)
    return pts


def mothership_samples(lanes, sweep_alt, fov, speed, t0):
    pts = lane_path(lanes, (lanes[0][0], lanes[0][1]))
    out = []
    t = t0
    for a, b in zip(pts, pts[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / (speed * DT)))
        for k in range(n):
            f = k / n
            out.append((t + f * d / speed, a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        t += d / speed
    out.append((t, pts[-1][0], pts[-1][1]))
    return out, t


class Mission:
    def __init__(self, cfg, scout_truth, max_passes=1, serial_operator=False):
        self.max_passes = max_passes
        self.serial_operator = serial_operator
        self.op_free_at = 0.0
        self.cfg = cfg
        self.rng = random.Random(cfg.seed * 31 + 7)
        self.scene = scene_mod.generate(cfg.seed, cfg.scene, cfg.flight)
        self.sensor_full = SensorModel.from_csv(cfg.sensor_csv)
        self.sensor = self.sensor_full.marginal(cfg.scene.occluded_fraction)
        self.policy = make_policy(cfg, self.sensor)
        self.op = OperatorModel(cfg.operator)
        self.truth = scout_truth
        self.gmap = GridMap(self.scene["half"], 1.0, self.scene["rover_obstacles"], 1.5)
        self.targets = []
        for v in self.scene["victims"]:
            self.targets.append({"kind": "victim", "id": v["id"], "x": v["x"], "y": v["y"], "occ": v["occlusion"], "pose": v.get("pose", "lying"), "found": False})
        for d in self.scene["decoys"]:
            self.targets.append({"kind": "decoy", "id": d["id"], "x": d["x"], "y": d["y"], "occ": 0.0, "pose": "none", "found": False})
        self.cands = {}
        self.next_id = 0
        self.t = 0.0
        self.m = {"scout_looks": 0, "operator_requests": 0, "dispatches": 0, "wasted_dispatches": 0, "rover_distance_total": 0.0, "rover_distance_wasted": 0.0, "confirm_times": [], "detections_true": 0, "detections_false": 0}
        self.scout = {"pos": (self.scene["start"]["scout"][0], self.scene["start"]["scout"][1]), "busy": None, "until": 0.0, "dest": None, "ready_at": TAKEOFF_S}
        self.rover = {"pos": ROVER_START, "busy": None, "path": None, "arrive": 0.0, "dist": 0.0}
        self.op_queue = []

    def nearest_target(self, x, y, radius):
        best, bd = None, radius
        for tg in self.targets:
            d = math.hypot(tg["x"] - x, tg["y"] - y)
            if d <= bd:
                best, bd = tg, d
        return best

    def new_candidate(self, pos, alt):
        p_hit, p_false = self.sensor.likelihoods(alt)
        b0 = prior_given_flag(self.cfg.scene.n_victims, self.cfg.scene.n_decoys, p_hit, p_false)
        c = Candidate(self.next_id, tuple(pos), b0, self.t)
        self.next_id += 1
        self.cands[c.id] = c
        self.act(c)

    def act(self, c):
        action = self.policy.decide(c)
        if action == IGNORE:
            c.status = REJECTED
        elif action == SCOUT:
            c.status = WAIT_SCOUT
        elif action == ASK:
            c.status = WAIT_OPERATOR
            tg = self.nearest_target(c.pos[0], c.pos[1], MATCH_RADIUS)
            person = bool(tg and tg["kind"] == "victim")
            self.m["operator_requests"] += 1
            self.op_queue.append({"id": c.id, "person": person, "looked": c.looks > 0, "due": self.op_due()})
        elif action == DISPATCH:
            c.status = WAIT_ROVER
            if not self.policy.uses_gate:
                c.asked = True
                c.op_confirm = True

    def op_due(self):
        start = max(self.t, self.op_free_at) if self.serial_operator else self.t
        due = start + self.op.delay(self.rng)
        if self.serial_operator:
            self.op_free_at = due
        return due

    def on_detection(self, pos, alt):
        for c in self.cands.values():
            if math.hypot(c.pos[0] - pos[0], c.pos[1] - pos[1]) < ASSOC_RADIUS:
                c.sightings += 1
                return
        self.new_candidate(pos, alt)

    def schedule(self):
        s = self.scout
        if s["busy"] is None and self.t >= s["ready_at"]:
            waiting = [c for c in self.cands.values() if c.status == WAIT_SCOUT]
            if waiting:
                c = min(waiting, key=lambda k: math.hypot(k.pos[0] - s["pos"][0], k.pos[1] - s["pos"][1]))
                c.status = SCOUTING
                d = math.hypot(c.pos[0] - s["pos"][0], c.pos[1] - s["pos"][1])
                s["busy"] = c.id
                s["until"] = self.t + max(0.0, d - self.cfg.flight.scout_standoff) / SCOUT_SPEED + DESCEND_S + self.cfg.flight.scout_dwell
                s["dest"] = c.pos
        r = self.rover
        if r["busy"] is None:
            waiting = [c for c in self.cands.values() if c.status == WAIT_ROVER]
            if waiting:
                c = max(waiting, key=lambda k: k.belief)
                path = plan(self.gmap, r["pos"], c.pos)
                if path is None:
                    c.status = REJECTED
                    return
                c.status = ROVER_GOING
                r["busy"] = c.id
                plan_m = path_length(path)
                r["dist"] = plan_m * ROVER_TRIP_FACTOR
                r["arrive"] = self.t + plan_m / ROVER_EFFECTIVE_SPEED
                self.m["dispatches"] += 1

    def advance(self):
        s = self.scout
        if s["busy"] is not None and self.t >= s["until"]:
            c = self.cands[s["busy"]]
            tg = self.nearest_target(c.pos[0], c.pos[1], MATCH_RADIUS)
            flag = self.rng.random() < scout_flag_prob(self.truth, tg)
            self.m["scout_looks"] += 1
            p_hit, p_false = self.sensor.likelihoods(self.cfg.flight.scout_alt, scout=True)
            c.belief = posterior(c.belief, flag, p_hit, p_false)
            c.looks += 1
            s["pos"] = c.pos
            s["ready_at"] = self.t + CLIMB_S
            s["busy"] = None
            self.act(c)
        rest = []
        for item in self.op_queue:
            if item["due"] <= self.t:
                c = self.cands[item["id"]]
                confirm = self.op.respond(self.rng, item["person"], item["looked"])
                c.asked = True
                c.op_confirm = confirm
                c.belief = self.op.posterior_yes(c.belief, item["looked"]) if confirm else self.op.posterior_no(c.belief, item["looked"])
                self.act(c)
            else:
                rest.append(item)
        self.op_queue = rest
        r = self.rover
        if r["busy"] is not None and self.t >= r["arrive"]:
            c = self.cands[r["busy"]]
            tg = self.nearest_target(c.pos[0], c.pos[1], MATCH_RADIUS)
            person = bool(tg and tg["kind"] == "victim")
            self.m["rover_distance_total"] += r["dist"]
            if person and not tg["found"]:
                tg["found"] = True
                self.m["confirm_times"].append(self.t)
            if not person:
                self.m["wasted_dispatches"] += 1
                self.m["rover_distance_wasted"] += r["dist"]
            c.status = CONFIRMED if person else WASTED
            c.belief = 1.0 if person else 0.0
            r["pos"] = c.pos
            r["busy"] = None

    def run(self):
        cfg = self.cfg
        z = cfg.flight.sweep_alt
        foot = self.sensor_full.footprint_radius(z, cfg.flight.fov) * 0.8
        samples, sweep_end = mothership_samples(self.scene["lanes"], z, cfg.flight.fov, MOTHER_SPEED, TAKEOFF_S + 6.0)
        inside, last_sample = {}, {}
        si = 0
        sweep_done = False
        passes = 1
        lanes = list(self.scene["lanes"])
        end_hold = None
        limit = cfg.flight.time_limit
        while self.t < limit:
            while si < len(samples) and samples[si][0] <= self.t:
                _, x, y = samples[si]
                si += 1
                for i, tg in enumerate(self.targets):
                    ins = math.hypot(tg["x"] - x, tg["y"] - y) <= foot
                    was = inside.get(i, False)
                    inside[i] = ins
                    if ins and not was and self.t - last_sample.get(i, -1e9) > RESAMPLE_GAP:
                        last_sample[i] = self.t
                        if self.sensor_full.observe(self.rng, z, tg["kind"], tg["occ"]):
                            sg = self.sensor_full.position_sigma(z)
                            pos = (tg["x"] + self.rng.gauss(0, sg), tg["y"] + self.rng.gauss(0, sg))
                            self.m["detections_true" if tg["kind"] == "victim" else "detections_false"] += 1
                            self.on_detection(pos, z)
            if not sweep_done and self.t >= sweep_end:
                sweep_done = True
            self.advance()
            self.schedule()
            if sweep_done and all(c.status in TERMINAL for c in self.cands.values()):
                confirmed = sum(1 for c in self.cands.values() if c.status == CONFIRMED)
                if confirmed < cfg.scene.n_victims and passes < self.max_passes:
                    lanes = lanes[::-1]
                    more, nend = mothership_samples(lanes, z, cfg.flight.fov, MOTHER_SPEED, self.t)
                    if nend < limit - 60.0:
                        samples, si, sweep_end, sweep_done, passes, end_hold = more, 0, nend, False, passes + 1, None
                        self.passes = passes
                        self.t += DT
                        continue
                if end_hold is None:
                    end_hold = self.t + 2.0
                if self.t >= end_hold:
                    break
            self.t += DT
        return self.summary("complete" if self.t < limit else "time_limit")

    def summary(self, reason):
        n_v = len(self.scene["victims"])
        found = sum(1 for tg in self.targets if tg["kind"] == "victim" and tg["found"])
        times = sorted(self.m["confirm_times"])
        asks = self.m["operator_requests"]
        out = {
            "seed": self.cfg.seed,
            "policy": self.cfg.policy,
            "reason": reason,
            "mission_time": round(self.t, 1),
            "victims_total": n_v,
            "passes": getattr(self, "passes", 1),
            "victims_confirmed": found,
            "recall": found / n_v if n_v else 0.0,
            "time_to_first_confirm": times[0] if times else None,
            "mean_time_to_confirm": sum(times) / len(times) if times else None,
        }
        for k, v in self.m.items():
            if k != "confirm_times":
                out[k] = v
        c = self.cfg.costs
        out["cost"] = (
            c.miss * (n_v - found)
            + c.false_dispatch * self.m["wasted_dispatches"]
            + c.travel * self.m["dispatches"]
            + c.look * self.m["scout_looks"]
            + c.ask * asks
        )
        return out


def run_one(cfg, scout_truth, max_passes=1, serial_operator=False):
    return Mission(cfg, scout_truth, max_passes, serial_operator).run()
