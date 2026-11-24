import math

from .worldfile import look_orientation


def look_at(cam, target):
    dx, dy, dz = target[0] - cam[0], target[1] - cam[1], target[2] - cam[2]
    yaw = math.atan2(dy, dx)
    pitch_down = math.atan2(-dz, math.hypot(dx, dy))
    return yaw, pitch_down


def pose(cam, target):
    yaw, pitch = look_at(cam, target)
    return look_orientation(yaw, pitch)


def project(point, cam, target, width, height, hfov):
    yaw, pitch = look_at(cam, target)
    return project_pose(point, cam, yaw, pitch, width, height, hfov)


def project_pose(point, cam, yaw, pitch, width, height, hfov):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = (cy * cp, sy * cp, -sp)
    left = (-sy, cy, 0.0)
    up = (cy * sp, sy * sp, cp)
    d = (point[0] - cam[0], point[1] - cam[1], point[2] - cam[2])
    depth = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2]
    if depth < 0.5:
        return None
    f = (width / 2.0) / math.tan(hfov / 2.0)
    lx = d[0] * left[0] + d[1] * left[1] + d[2] * left[2]
    ux = d[0] * up[0] + d[1] * up[1] + d[2] * up[2]
    px = width / 2.0 - lx / depth * f
    py = height / 2.0 - ux / depth * f
    return px, py, depth


def blocked(cam, tgt, boxes, samples=36, tail=0.12):
    for i in range(1, samples):
        t = i / samples
        if t > 1.0 - tail:
            break
        x = cam[0] + (tgt[0] - cam[0]) * t
        y = cam[1] + (tgt[1] - cam[1]) * t
        z = cam[2] + (tgt[2] - cam[2]) * t
        for b in boxes:
            dx, dy = x - b["x"], y - b["y"]
            c, s = math.cos(-b["yaw"]), math.sin(-b["yaw"])
            lx, ly = dx * c - dy * s, dx * s + dy * c
            if abs(lx) <= b["sx"] / 2 + 0.4 and abs(ly) <= b["sy"] / 2 + 0.4 and z <= b["h"] + 0.8:
                return True
    return False


OFFSETS = (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi)


class Shot:
    def __init__(self, name, cam, target, cut):
        self.name = name
        self.cam = cam
        self.target = target
        self.cut = cut


class Director:
    def __init__(self, overview_cam, overview_target):
        self.cam = list(overview_cam)
        self.target = list(overview_target)
        self.name = "intro"
        self.heading = {}
        self.last = {}
        self.offsets = {}

    def heading_of(self, who, p, dt):
        prev = self.last.get(who)
        self.last[who] = p
        h = self.heading.get(who, (0.0, 1.0))
        if prev is not None and dt > 0:
            vx, vy = (p[0] - prev[0]) / dt, (p[1] - prev[1]) / dt
            n = math.hypot(vx, vy)
            if n > 0.3:
                a = 0.04
                hx = h[0] + a * (vx / n - h[0])
                hy = h[1] + a * (vy / n - h[1])
                m = math.hypot(hx, hy) or 1.0
                h = (hx / m, hy / m)
        self.heading[who] = h
        return h

    def chase(self, who, p, dt, back, up, ahead, ahead_z):
        h = self.heading_of(who, p, dt)
        return lambda off: self.chase_h(p, h, back, up, ahead, ahead_z, off)

    def chase_h(self, p, h, back, up, ahead, ahead_z, off):
        c, s = math.cos(off), math.sin(off)
        bx, by = h[0] * c - h[1] * s, h[0] * s + h[1] * c
        cam = (p[0] - bx * back, p[1] - by * back, p[2] + up)
        tgt = (p[0] + h[0] * ahead, p[1] + h[1] * ahead, ahead_z)
        return cam, tgt

    def pick(self, key, build, boxes):
        order = list(OFFSETS)
        last = self.offsets.get(key)
        if last is not None and last in order:
            order.remove(last)
            order.insert(0, last)
        first = None
        for off in order:
            cam, tgt = build(off)
            if first is None:
                first = (off, cam, tgt)
            if not blocked(cam, tgt, boxes):
                self.offsets[key] = off
                return cam, tgt
        self.offsets[key] = first[0]
        return first[1], first[2]

    def orbit(self, c, dist, height, bearing, tz):
        return lambda off: ((c[0] + math.cos(bearing + off) * dist, c[1] + math.sin(bearing + off) * dist, height), (c[0], c[1], tz))

    def frame_two(self, a, b, bearing, min_dist=11.0, elev=0.4):
        return lambda off: self._frame_two(a, b, bearing + off, min_dist, elev)

    def _frame_two(self, a, b, bearing, min_dist, elev):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
        sep = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        dist = min_dist + 1.25 * sep
        cam = (mid[0] + math.cos(bearing) * dist * math.cos(elev), mid[1] + math.sin(bearing) * dist * math.cos(elev), mid[2] + dist * math.sin(elev))
        return cam, mid

    def update(self, name, cam, tgt, dt):
        cut = name != self.name
        if cut:
            self.name = name
            self.cam = list(cam)
            self.target = list(tgt)
        else:
            a = 1.0 - math.exp(-dt / 0.35)
            for i in range(3):
                self.cam[i] += a * (cam[i] - self.cam[i])
                self.target[i] += a * (tgt[i] - self.target[i])
        return tuple(self.cam), tuple(self.target), cut
