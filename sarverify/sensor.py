import csv
import json
import math
import os


class SensorModel:
    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["altitude_m"])
        self.alt = [r["altitude_m"] for r in rows]
        self.cols = {
            k: [r[k] for r in rows]
            for k in ("p_detect", "p_false_decoy", "p_false_empty")
        }
        self.sources = sorted({r.get("source", "") for r in rows})
        self.hit_scale = 1.0
        self.scout = None

    def marginal(self, occluded_fraction, mean_occlusion=0.75):
        out = SensorModel([
            {"altitude_m": a, "p_detect": d, "p_false_decoy": f, "p_false_empty": e, "source": ",".join(self.sources)}
            for a, d, f, e in zip(self.alt, self.cols["p_detect"], self.cols["p_false_decoy"], self.cols["p_false_empty"])
        ])
        out.sources = list(self.sources)
        out.hit_scale = 1.0 - 0.5 * occluded_fraction * mean_occlusion
        out.scout = self.scout
        return out

    @classmethod
    def from_csv(cls, path):
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "altitude_m": float(r["altitude_m"]),
                        "p_detect": float(r["p_detect"]),
                        "p_false_decoy": float(r["p_false_decoy"]),
                        "p_false_empty": float(r["p_false_empty"]),
                        "source": r.get("source", ""),
                    }
                )
        m = cls(rows)
        sp = os.path.join(os.path.dirname(os.path.abspath(path)), "scout_detector.json")
        if os.path.exists(sp):
            with open(sp) as f:
                m.scout = json.load(f)
            m.sources = sorted(set(m.sources) | {"measured_scout_detector"})
        return m

    def _interp(self, key, alt):
        xs, ys = self.alt, self.cols[key]
        if alt <= xs[0]:
            return ys[0]
        if alt >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= alt <= xs[i + 1]:
                t = (alt - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    def p_detect(self, alt, occlusion=0.0):
        return self._interp("p_detect", alt) * (1.0 - 0.5 * occlusion)

    def p_false_decoy(self, alt):
        return self._interp("p_false_decoy", alt)

    def p_false_empty(self, alt):
        return self._interp("p_false_empty", alt)

    def p_flag(self, alt, truth, occlusion=0.0):
        if truth == "victim":
            return self.p_detect(alt, occlusion)
        if truth == "decoy":
            return self.p_false_decoy(alt)
        return self.p_false_empty(alt)

    def observe(self, rng, alt, truth, occlusion=0.0):
        return rng.random() < self.p_flag(alt, truth, occlusion)

    def likelihoods(self, alt, scout=False):
        if scout and self.scout is not None:
            return self.scout["p_hit"], self.scout["p_false_decoy"]
        return self.p_detect(alt) * self.hit_scale, self.p_false_decoy(alt)

    @staticmethod
    def footprint_radius(alt, fov):
        return alt * math.tan(fov / 2.0)

    @staticmethod
    def position_sigma(alt):
        return 0.02 * alt + 0.1
