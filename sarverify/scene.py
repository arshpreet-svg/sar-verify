import math
import random


def _overlaps(x, y, rects, margin):
    for r in rects:
        if abs(x - r["x"]) < r["sx"] / 2 + margin and abs(y - r["y"]) < r["sy"] / 2 + margin:
            return True
    return False


def sweep_lanes(size, swath, alt):
    half = size / 2.0
    lanes = max(1, int(math.ceil(size / swath)))
    pitch = size / lanes
    pts = []
    for i in range(lanes):
        x = -half + pitch * (i + 0.5)
        if i % 2 == 0:
            pts.append((x, -half + 4.0))
            pts.append((x, half - 4.0))
        else:
            pts.append((x, half - 4.0))
            pts.append((x, -half + 4.0))
    return pts


def generate(seed, sp, flight):
    rng = random.Random(seed * 7919 + 17)
    half = sp.size / 2.0
    start = {"mothership": (-half + 3.0, -half + 3.0), "scout": (-half + 3.0, -half + 8.0), "rover": (-half + 8.0, -half + 3.0)}
    keepout = [
        {"x": start["mothership"][0], "y": start["mothership"][1], "sx": 4, "sy": 4},
        {"x": start["scout"][0], "y": start["scout"][1], "sx": 4, "sy": 4},
        {"x": start["rover"][0], "y": start["rover"][1], "sx": 4, "sy": 4},
        {"x": -half + 6, "y": -half + 6, "sx": 20, "sy": 20},
    ]
    obstacles = []
    flood = None
    if sp.flood:
        fw = rng.uniform(16, 24)
        fh = rng.uniform(14, 22)
        flood = {
            "x": rng.uniform(-half * 0.3, half * 0.5),
            "y": rng.uniform(-half * 0.2, half * 0.5),
            "sx": fw,
            "sy": fh,
        }
        obstacles.append({"kind": "flood", "x": flood["x"], "y": flood["y"], "sx": fw, "sy": fh, "h": 0.0})

    def place(kind, count, smin, smax, hmin, hmax):
        placed = 0
        tries = 0
        while placed < count and tries < 500:
            tries += 1
            sx = rng.uniform(smin, smax)
            sy = rng.uniform(smin, smax)
            x = rng.uniform(-half + 6, half - 6)
            y = rng.uniform(-half + 6, half - 6)
            if _overlaps(x, y, keepout + obstacles, max(sx, sy) / 2 + 3.0):
                continue
            h = rng.uniform(hmin, hmax)
            obstacles.append({"kind": kind, "x": x, "y": y, "sx": sx, "sy": sy, "h": h, "yaw": rng.uniform(0, math.pi)})
            placed += 1

    place("building", sp.n_buildings, 6.0, 10.0, 3.0, 6.0)
    place("rubble", sp.n_rubble, 2.5, 5.0, 0.6, 1.4)

    solid = [o for o in obstacles]
    taken = []

    def free_spot():
        for _ in range(1000):
            x = rng.uniform(-half + 5, half - 5)
            y = rng.uniform(-half + 5, half - 5)
            if _overlaps(x, y, keepout + solid, 2.5):
                continue
            if any(math.hypot(x - tx, y - ty) < 7.0 for tx, ty in taken):
                continue
            taken.append((x, y))
            return x, y
        raise RuntimeError("no free spot")

    victims = []
    for i in range(sp.n_victims):
        x, y = free_spot()
        occ = rng.uniform(0.5, 1.0) if rng.random() < sp.occluded_fraction else 0.0
        victims.append({"id": i, "x": x, "y": y, "yaw": rng.uniform(0, 2 * math.pi), "occlusion": occ})
    prng = random.Random(seed * 911 + 3)
    for v in victims:
        if prng.random() < sp.standing_fraction:
            v["pose"] = "standing"
            v["occlusion"] = 0.0
        else:
            v["pose"] = "lying"
    decoys = []
    for i in range(sp.n_decoys):
        x, y = free_spot()
        decoys.append({"id": i, "x": x, "y": y, "yaw": rng.uniform(0, 2 * math.pi), "kind": rng.choice(["tarp", "log", "bag"])})

    rover_obstacles = [
        {"x": o["x"], "y": o["y"], "sx": o["sx"], "sy": o["sy"]}
        for o in obstacles
    ]
    return {
        "seed": seed,
        "half": half,
        "start": start,
        "obstacles": obstacles,
        "rover_obstacles": rover_obstacles,
        "victims": victims,
        "decoys": decoys,
        "lanes": sweep_lanes(sp.size, flight.swath, flight.sweep_alt),
    }
