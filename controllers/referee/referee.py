import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from controller import Supervisor

from sarverify import scene as scene_mod
from sarverify.comms import Radio
from sarverify import overlay
from sarverify.cinema import Director, pose
from sarverify.config import RunConfig
from sarverify.operator import OperatorModel
from sarverify.sensor import SensorModel

STATUS_COLOR = {
    "new": (1.0, 0.85, 0.0),
    "wait_scout": (1.0, 0.85, 0.0),
    "scouting": (0.2, 0.7, 1.0),
    "wait_operator": (0.75, 0.55, 1.0),
    "wait_rover": (1.0, 0.55, 0.0),
    "rover_going": (1.0, 0.55, 0.0),
    "confirmed": (0.1, 0.9, 0.2),
    "rejected": (0.6, 0.6, 0.6),
    "wasted": (0.95, 0.15, 0.15),
}
OVERVIEW = (0.0, -92.0, 88.0)
FOLLOW_OFFSET = (0.0, -22.0, 20.0)
CAM_PERIOD = 32
HFOV = 1.15
RESAMPLE_GAP = 40.0
MATCH_RADIUS = 4.0
ARRIVE_RADIUS = 1.6


class Referee:
    def __init__(self):
        self.sup = Supervisor()
        self.step = int(self.sup.getBasicTimeStep())
        self.cfg = RunConfig.load()
        self.rng = random.Random(self.cfg.seed * 31 + 7)
        self.sensor = SensorModel.from_csv(self.cfg.sensor_csv)
        self.op = OperatorModel(self.cfg.operator)
        self.radio = Radio(self.sup, "referee", self.step)
        self.scene = scene_mod.generate(self.cfg.seed, self.cfg.scene, self.cfg.flight)
        self.targets = []
        for v in self.scene["victims"]:
            self.targets.append({"kind": "victim", "id": v["id"], "x": v["x"], "y": v["y"], "occ": v["occlusion"], "found": False})
        for d in self.scene["decoys"]:
            self.targets.append({"kind": "decoy", "id": d["id"], "x": d["x"], "y": d["y"], "occ": 0.0, "found": False})
        self.nodes = {n: self.sup.getFromDef(n.upper()) for n in ("mothership", "scout", "rover")}
        self.inside = {}
        self.last_sample = {}
        self.t = 0.0
        self.sweep_done = False
        self.mission_idle = False
        self.cands = {}
        self.markers = {}
        self.op_queue = []
        self.manual_prompt = None
        self.dispatch = None
        self.events = []
        self.metrics = {
            "scout_looks": 0, "operator_requests": 0, "dispatches": 0, "wasted_dispatches": 0,
            "rover_distance_total": 0.0, "rover_distance_wasted": 0.0, "scout_distance": 0.0,
            "mothership_distance": 0.0, "confirm_times": [], "detections_true": 0, "detections_false": 0,
        }
        self.prev = {}
        self.view = self.sup.getFromDef("VIEW")
        self.view_pos = list(OVERVIEW)
        self.snap_times = [2.0, 40.0, 100.0, 180.0, 260.0, 340.0] if self.cfg.snapshot_dir else []
        if self.cfg.snapshot_dir:
            os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        self.finished = False
        self.recording = False
        self.kb = None
        if self.cfg.operator.mode == "manual":
            self.kb = self.sup.getKeyboard()
            self.kb.enable(self.step)
        self.frame_dir = ""
        self.frame_idx = 0
        self.tick = 0
        self.stride_count = 0
        self.cine = None
        self.me = self.sup.getSelf()
        self.director_ = Director((0.0, -85.0, 48.0), (0.0, 0.0, 0.0))
        self.cur_cam = (0.0, -85.0, 48.0)
        self.cur_tgt = (0.0, 0.0, 0.0)
        self.next_cam = (self.cur_cam, self.cur_tgt)
        self.phases = {}
        self.last_look = None
        self.boxes = [{"x": o["x"], "y": o["y"], "sx": o["sx"], "sy": o["sy"], "yaw": o["yaw"], "h": o["h"]} for o in self.scene["obstacles"] if o["kind"] == "building"]
        self.banner = None
        self.reveal = None
        self.ended_at = None
        if self.cfg.record_video and not self.cfg.headless:
            self.frame_dir = os.path.splitext(self.cfg.record_video)[0] + "_frames"
            os.makedirs(self.frame_dir, exist_ok=True)
            self.cine = self.sup.getDevice("cine")
            self.cine.enable(self.step * CAM_PERIOD)

    def say(self, phase, text):
        self.phases[phase] = (text, self.t)

    def pos(self, who):
        v = self.nodes[who].getPosition()
        return v[0], v[1], v[2]

    def log(self, kind, **kw):
        e = {"t": round(self.t, 2), "event": kind}
        e.update(kw)
        self.events.append(e)

    def nearest_target(self, x, y, radius):
        best = None
        bd = radius
        for tg in self.targets:
            d = math.hypot(tg["x"] - x, tg["y"] - y)
            if d <= bd:
                best, bd = tg, d
        return best

    def sense_mothership(self):
        x, y, z = self.pos("mothership")
        if z < self.cfg.flight.sweep_alt - 4.0:
            return
        r = self.sensor.footprint_radius(z, self.cfg.flight.fov) * 0.8
        for i, tg in enumerate(self.targets):
            inside = math.hypot(tg["x"] - x, tg["y"] - y) <= r
            was = self.inside.get(i, False)
            self.inside[i] = inside
            if inside and not was and self.t - self.last_sample.get(i, -1e9) > RESAMPLE_GAP:
                self.last_sample[i] = self.t
                flag = self.sensor.observe(self.rng, z, tg["kind"], tg["occ"])
                self.log("mothership_obs", target=f"{tg['kind']}_{tg['id']}", flag=flag, alt=round(z, 1))
                if flag:
                    self.say("PERCEIVE", f"Mothership AI flags something that may be a person (seen from {z:.0f} m up)")
                    sg = self.sensor.position_sigma(z)
                    px = tg["x"] + self.rng.gauss(0, sg)
                    py = tg["y"] + self.rng.gauss(0, sg)
                    if tg["kind"] == "victim":
                        self.metrics["detections_true"] += 1
                    else:
                        self.metrics["detections_false"] += 1
                    self.radio.send("commander", "detection", pos=[px, py], alt=z, src="mothership")

    def handle_look(self, msg):
        x, y, z = self.pos("scout")
        c = self.cands.get(msg["cand_id"])
        tg = self.nearest_target(c["pos"][0], c["pos"][1], MATCH_RADIUS) if c else None
        truth = tg["kind"] if tg else "empty"
        flag = bool(msg["flag"])
        self.metrics["scout_looks"] += 1
        self.log("scout_look", cand=msg["cand_id"], truth=truth, flag=flag, score=round(msg["score"], 3), n_dets=len(msg["dets"]), alt=round(z, 1))
        self.last_look = {"img": msg["img"], "dets": msg["dets"], "flag": flag, "score": msg["score"], "t": self.t, "cand": msg["cand_id"], "target_px": msg.get("target_px")}
        self.say("PERCEIVE", f"Scout camera + AI detector on #{msg['cand_id']}: " + (f"PERSON detected (confidence {msg['score']:.2f})" if flag else "no person detected"))
        self.radio.send("commander", "scout_obs", cand_id=msg["cand_id"], flag=flag, alt=z)

    def handle_op_request(self, msg):
        tg = self.nearest_target(msg["pos"][0], msg["pos"][1], MATCH_RADIUS)
        is_person = bool(tg and tg["kind"] == "victim")
        self.metrics["operator_requests"] += 1
        if self.cfg.operator.mode == "manual":
            self.op_queue.append({"cand_id": msg["cand_id"], "belief": msg["belief"], "person": is_person, "looked": msg.get("looked", True), "due": None})
        else:
            self.op_queue.append({"cand_id": msg["cand_id"], "belief": msg["belief"], "person": is_person, "looked": msg.get("looked", True), "due": self.t + self.op.delay(self.rng)})
        self.log("op_request", cand=msg["cand_id"], belief=round(msg["belief"], 3))

    def process_operator(self):
        if self.cfg.operator.mode == "manual":
            if self.manual_prompt is None and self.op_queue:
                self.manual_prompt = self.op_queue.pop(0)
            if self.manual_prompt is not None:
                key = self.kb.getKey()
                answer = None
                if key in (ord("Y"), ord("y")):
                    answer = True
                elif key in (ord("N"), ord("n")):
                    answer = False
                if answer is not None:
                    self.reply_operator(self.manual_prompt, answer)
                    self.manual_prompt = None
            return
        remaining = []
        for item in self.op_queue:
            if item["due"] <= self.t:
                self.reply_operator(item, self.op.respond(self.rng, item["person"], item["looked"]))
            else:
                remaining.append(item)
        self.op_queue = remaining

    def reply_operator(self, item, confirm):
        self.log("op_response", cand=item["cand_id"], confirm=confirm, truth=item["person"])
        self.say("ACT", f"Human operator reviews #{item['cand_id']}: " + ("YES, a person" if confirm else "NO, not a person"))
        self.radio.send("commander", "op_response", cand_id=item["cand_id"], confirm=confirm)

    def handle_dispatch(self, msg):
        self.dispatch = {"cand_id": msg["cand_id"], "pos": msg["pos"], "t0": self.t, "dist": 0.0}
        self.metrics["dispatches"] += 1
        self.log("dispatch", cand=msg["cand_id"], plan_m=round(msg["plan_m"], 1))
        self.say("ACT", f"Rover dispatched to #{msg['cand_id']}, planned path {msg['plan_m']:.0f} m")

    def handle_arrival(self, msg):
        if self.dispatch is None or self.dispatch["cand_id"] != msg["cand_id"]:
            return
        tg = self.nearest_target(self.dispatch["pos"][0], self.dispatch["pos"][1], MATCH_RADIUS)
        person = bool(tg and tg["kind"] == "victim")
        d = self.dispatch["dist"]
        if person and not tg["found"]:
            tg["found"] = True
            self.metrics["confirm_times"].append(self.t)
        if not person:
            self.metrics["wasted_dispatches"] += 1
            self.metrics["rover_distance_wasted"] += d
        self.log("rover_report", cand=msg["cand_id"], person=person, trip_m=round(d, 1))
        self.reveal = (tuple(self.dispatch["pos"]), self.t + 5.0)
        if person:
            self.banner = (f"VICTIM FOUND  #{msg['cand_id']}", (20, 170, 60), self.t + 5.0)
            self.say("ACT", f"Rover reached #{msg['cand_id']}: person found, rescue team notified")
        else:
            self.banner = (f"FALSE ALARM  #{msg['cand_id']}  - nobody there", (200, 40, 40), self.t + 5.0)
            self.say("ACT", f"Rover reached #{msg['cand_id']}: false alarm, wasted trip")
        self.radio.send("commander", "rover_report", cand_id=msg["cand_id"], person_found=person)
        self.dispatch = None

    def update_marker(self, u):
        cid = u["id"]
        prev = self.cands.get(cid)
        if prev is None or prev["status"] != u["status"]:
            self.decide_caption(u)
        self.cands[cid] = u
        color = STATUS_COLOR.get(u["status"], (1, 1, 1))
        old = self.markers.get(cid)
        if old is not None:
            if old[1] == u["status"]:
                return
            old[0].remove()
        s = f'Marker {{ translation {u["pos"][0]:.2f} {u["pos"][1]:.2f} 0 color {color[0]} {color[1]} {color[2]} }}'
        children = self.sup.getRoot().getField("children")
        children.importMFNodeFromString(-1, s)
        node = children.getMFNode(children.getCount() - 1)
        self.markers[cid] = (node, u["status"])

    def decide_caption(self, u):
        i, b, st = u["id"], u["belief"], u["status"]
        text = {
            "wait_scout": f"#{i} belief {b:.2f}: unsure, send SCOUT for a closer look",
            "scouting": f"#{i} belief {b:.2f}: scout is flying out to verify",
            "wait_operator": f"#{i} belief {b:.2f}: ask the HUMAN OPERATOR before moving the rover",
            "wait_rover": f"#{i} belief {b:.2f}: operator agrees, send the ROVER",
            "rover_going": f"#{i} belief {b:.2f}: rover is on its way",
            "rejected": f"#{i} belief {b:.2f}: too unlikely, REJECT and save a trip",
        }.get(st)
        if text:
            self.say("DECIDE", text)
        if st == "rejected":
            self.banner = (f"#{i} REJECTED - wasted trip avoided", (110, 110, 120), self.t + 3.0)
        if st == "scouting":
            self.say("ACT", f"Scout drone flying to #{i} to look from 10 m")

    def track_distance(self):
        for who in ("mothership", "scout", "rover"):
            p = self.pos(who)
            if who in self.prev:
                d = math.hypot(p[0] - self.prev[who][0], p[1] - self.prev[who][1])
                if who == "rover":
                    self.metrics["rover_distance_total"] += d
                    if self.dispatch is not None:
                        self.dispatch["dist"] += d
                else:
                    self.metrics[who + "_distance"] += d
            self.prev[who] = p

    def found_count(self):
        return sum(1 for tg in self.targets if tg["kind"] == "victim" and tg["found"])

    def hud(self):
        n_v = len(self.scene["victims"])
        lines = [
            f"SAR-Verify   t = {self.t:6.1f} s   victims found {self.found_count()}/{n_v}   wasted trips {self.metrics['wasted_dispatches']}",
        ]
        for name in ("PERCEIVE", "DECIDE", "ACT"):
            text = self.phases.get(name, ("", 0))[0]
            lines.append(f"{name}: {text}")
        if self.manual_prompt is not None:
            lines.append(f">> OPERATOR: candidate {self.manual_prompt['cand_id']} belief {self.manual_prompt['belief']:.2f}  press Y = person, N = not")
        for i in range(8):
            text = lines[i] if i < len(lines) else ""
            self.sup.setLabel(i, text, 0.02, 0.02 + 0.035 * i, 0.06, 0xFFFFFF, 0.0, "Arial")

    def scouting_cand(self):
        sp = self.pos("scout")
        for c in self.cands.values():
            if c["status"] in ("scouting", "wait_operator") and math.hypot(sp[0] - c["pos"][0], sp[1] - c["pos"][1]) < 25:
                return c
        return None

    def choose_shot(self):
        dt = self.step / 1000.0
        dr = self.director_
        m, sc, r = self.pos("mothership"), self.pos("scout"), self.pos("rover")
        wide = (0.0, -95.0, 55.0), (0.0, -5.0, 0.0)
        if self.ended_at is not None:
            return "end", wide[0], wide[1]
        if self.t < 8.0:
            return "intro", wide[0], wide[1]
        if self.dispatch is not None:
            tp = self.dispatch["pos"]
            if math.hypot(r[0] - tp[0], r[1] - tp[1]) < 16:
                cam, tgt = dr.pick("arrive", dr.orbit((tp[0], tp[1]), 8.0, 9.0, -2.3, 0.7), self.boxes)
                return "arrive", cam, tgt
            cam, tgt = dr.pick("rover", dr.chase("rover", r, dt, 11.0, 14.0, 3.0, 0.0), self.boxes)
            return "rover", cam, tgt
        if self.reveal is not None and self.t < self.reveal[1]:
            tp = self.reveal[0]
            cam, tgt = dr.pick("reveal", dr.orbit((tp[0], tp[1]), 6.0, 8.0, -2.0, 0.7), self.boxes)
            return "reveal", cam, tgt
        c = self.scouting_cand()
        if c is not None:
            cam, tgt = dr.pick("scout_look", dr.frame_two(sc, (c["pos"][0], c["pos"][1], 0.5), -2.2), self.boxes)
            return "scout_look", cam, tgt
        if any(u["status"] in ("scouting",) for u in self.cands.values()):
            cam, tgt = dr.pick("scout_fly", dr.chase("scout", sc, dt, 14.0, 5.0, 4.0, sc[2] - 3.0), self.boxes)
            return "scout_fly", cam, tgt
        cam, tgt = dr.pick("sweep", dr.chase("mothership", m, dt, 20.0, 6.0, 12.0, 2.0), [])
        return "sweep", cam, tgt

    def is_action(self):
        if self.t < 8.0 or self.ended_at is not None:
            return True
        if self.dispatch is not None or (self.reveal is not None and self.t < self.reveal[1]):
            return True
        return any(u["status"] in ("scouting", "wait_operator") for u in self.cands.values())

    def director(self):
        if self.view is None or self.cfg.headless:
            return
        name, cam, tgt = self.choose_shot()
        cam, tgt, _ = self.director_.update(name, cam, tgt, self.step / 1000.0)
        o = pose(cam, tgt)
        self.view.getField("position").setSFVec3f(list(cam))
        self.view.getField("orientation").setSFRotation(list(o))
        if self.cine is not None:
            self.me.getField("translation").setSFVec3f(list(cam))
            self.me.getField("rotation").setSFRotation(list(o))
        self.next_cam = (cam, tgt)

    def capture(self):
        if self.cine is None or self.tick % CAM_PERIOD != 0 or self.tick == 0:
            return
        action = self.is_action()
        stride = 1 if action else 4
        self.stride_count += 1
        if self.stride_count % stride != 0:
            return
        tmp = os.path.join(self.frame_dir, "_raw.jpg")
        self.cine.saveImage(tmp, 92)
        if self.banner is not None and self.t > self.banner[2]:
            self.banner = None
        vehicles = {"MOTHERSHIP": self.pos("mothership"), "SCOUT": self.pos("scout"), "ROVER": self.pos("rover")}
        names = {}
        for d in self.scene["decoys"]:
            names[d["id"]] = d["kind"]
        poses = {v["id"]: v.get("pose", "lying") for v in self.scene["victims"]}
        truth = [{"kind": tg["kind"], "x": tg["x"], "y": tg["y"], "name": names.get(tg["id"], "") if tg["kind"] == "decoy" else "", "pose": poses.get(tg["id"], "") if tg["kind"] == "victim" else ""} for tg in self.targets]
        state = {
            "cam": self.cur_cam, "target": self.cur_tgt, "hfov": HFOV, "t": self.t,
            "speed": CAM_PERIOD * self.step / 1000.0 * stride * 25.0,
            "found": self.found_count(), "n_victims": len(self.scene["victims"]), "wasted": self.metrics["wasted_dispatches"],
            "phases": {k: (v[0], self.t - v[1]) for k, v in self.phases.items()},
            "cands": [self.cands[k] for k in sorted(self.cands)],
            "vehicles": vehicles, "truth": truth,
            "banner": (self.banner[0], self.banner[1]) if self.banner else None,
            "inset": self.last_look if self.last_look and self.t - self.last_look["t"] < 9.0 else None,
        }
        overlay.render(tmp, os.path.join(self.frame_dir, f"f{self.frame_idx:05d}.jpg"), state)
        self.frame_idx += 1

    def finish(self, reason):
        if self.finished:
            return
        self.finished = True
        n_v = len(self.scene["victims"])
        found = sum(1 for tg in self.targets if tg["kind"] == "victim" and tg["found"])
        times = sorted(self.metrics["confirm_times"])
        summary = {
            "seed": self.cfg.seed,
            "policy": self.cfg.policy,
            "reason": reason,
            "mission_time": round(self.t, 1),
            "victims_total": n_v,
            "victims_confirmed": found,
            "recall": found / n_v if n_v else 0.0,
            "time_to_first_confirm": round(times[0], 1) if times else None,
            "mean_time_to_confirm": round(sum(times) / len(times), 1) if times else None,
            "sensor_source": self.sensor.sources,
        }
        for k, v in self.metrics.items():
            if k != "confirm_times":
                summary[k] = round(v, 2) if isinstance(v, float) else v
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        with open(os.path.join(self.cfg.out_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(self.cfg.out_dir, "events.jsonl"), "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        print("RESULT " + json.dumps(summary), flush=True)
        self.sup.setLabel(30, f"MISSION COMPLETE ({reason}): {found}/{n_v} victims, {self.metrics['wasted_dispatches']} wasted dispatches", 0.42, 0.9, 0.07, 0x007a2e, 0.0, "Arial")
        self.banner = (f"MISSION COMPLETE  {found}/{n_v} victims found, {self.metrics['wasted_dispatches']} wasted trips", (20, 120, 200), self.t + 99.0)
        if self.cfg.headless:
            self.sup.simulationQuit(0)
        elif self.cine is not None:
            self.ended_at = self.t
        else:
            self.sup.simulationSetMode(self.sup.SIMULATION_MODE_PAUSE)

    def run(self):
        end_hold = None
        while self.sup.step(self.step) != -1:
            self.t = self.sup.getTime()
            self.tick += 1
            self.capture()
            if self.ended_at is not None:
                self.director()
                self.cur_cam, self.cur_tgt = self.next_cam
                if self.t >= self.ended_at + 7.0:
                    self.sup.simulationQuit(0)
                    break
                continue
            for msg in self.radio.poll():
                k = msg["type"]
                if k == "look":
                    self.handle_look(msg)
                elif k == "op_request":
                    self.handle_op_request(msg)
                elif k == "dispatch":
                    self.handle_dispatch(msg)
                elif k == "rover_arrived":
                    self.handle_arrival(msg)
                elif k == "cand_update":
                    self.update_marker(msg)
                elif k == "sweep_done":
                    self.sweep_done = True
                    self.log("sweep_done")
                elif k == "resweep":
                    self.log("resweep", passes=msg["passes"])
                    self.say("DECIDE", f"Victims still unaccounted for: mothership flies a second sweep (pass {msg['passes']})")
                elif k == "mission_idle":
                    self.mission_idle = True
            if self.t > 3.0:
                self.sense_mothership()
            self.process_operator()
            self.director()
            self.cur_cam, self.cur_tgt = self.next_cam
            self.track_distance()
            if int(self.t * 125) % 8 == 0:
                self.hud()
            if int(self.t * 125) % 625 == 0:
                self.log(
                    "track",
                    m=[round(v, 1) for v in self.pos("mothership")],
                    s=[round(v, 1) for v in self.pos("scout")],
                    r=[round(v, 1) for v in self.pos("rover")],
                )
            while self.snap_times and self.t >= self.snap_times[0]:
                self.sup.exportImage(os.path.join(self.cfg.snapshot_dir, f"t{int(self.snap_times.pop(0)):04d}.png"), 90)
            if self.mission_idle and end_hold is None:
                end_hold = self.t + 2.0
            if end_hold is not None and self.t >= end_hold:
                self.finish("complete")
                if self.ended_at is None:
                    break
            elif self.t >= self.cfg.flight.time_limit:
                self.finish("time_limit")
                if self.ended_at is None:
                    break


Referee().run()
