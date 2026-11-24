import heapq
import math


class GridMap:
    def __init__(self, half, res, rects, inflate):
        self.half = half
        self.res = res
        self.n = int(round(2 * half / res))
        self.blocked = [[False] * self.n for _ in range(self.n)]
        for r in rects:
            self._mark(r, inflate)

    def to_cell(self, x, y):
        i = int((x + self.half) / self.res)
        j = int((y + self.half) / self.res)
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def to_world(self, i, j):
        return (i + 0.5) * self.res - self.half, (j + 0.5) * self.res - self.half

    def _mark(self, r, inflate):
        x0 = r["x"] - r["sx"] / 2 - inflate
        x1 = r["x"] + r["sx"] / 2 + inflate
        y0 = r["y"] - r["sy"] / 2 - inflate
        y1 = r["y"] + r["sy"] / 2 + inflate
        i0, j0 = self.to_cell(x0, y0)
        i1, j1 = self.to_cell(x1, y1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.blocked[i][j] = True

    def free(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n and not self.blocked[i][j]

    def nearest_free(self, i, j):
        if self.free(i, j):
            return i, j
        for rad in range(1, self.n):
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    if max(abs(di), abs(dj)) == rad and self.free(i + di, j + dj):
                        return i + di, j + dj
        return i, j

    def line_free(self, a, b):
        (x0, y0), (x1, y1) = a, b
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / (self.res * 0.5)))
        for s in range(steps + 1):
            t = s / steps
            i, j = self.to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            if self.blocked[i][j]:
                return False
        return True


NEIGHBOURS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def astar(gmap, start, goal):
    s = gmap.nearest_free(*gmap.to_cell(*start))
    g = gmap.nearest_free(*gmap.to_cell(*goal))
    openq = [(0.0, s)]
    cost = {s: 0.0}
    prev = {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for di, dj, w in NEIGHBOURS:
            nb = (cur[0] + di, cur[1] + dj)
            if not gmap.free(*nb):
                continue
            if di and dj and not (gmap.free(cur[0] + di, cur[1]) and gmap.free(cur[0], cur[1] + dj)):
                continue
            nc = cost[cur] + w
            if nc < cost.get(nb, 1e18):
                cost[nb] = nc
                prev[nb] = cur
                h = math.hypot(g[0] - nb[0], g[1] - nb[1])
                heapq.heappush(openq, (nc + h, nb))
    if g not in cost:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells.reverse()
    return [gmap.to_world(i, j) for i, j in cells]


def smooth(gmap, path):
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    anchor = 0
    k = 2
    while k < len(path):
        if not gmap.line_free(path[anchor], path[k]):
            out.append(path[k - 1])
            anchor = k - 1
        k += 1
    out.append(path[-1])
    return out


def plan(gmap, start, goal):
    raw = astar(gmap, start, goal)
    if raw is None:
        return None
    path = smooth(gmap, raw)
    path[0] = tuple(start)
    path[-1] = tuple(goal)
    return path


def path_length(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
