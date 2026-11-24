from .belief import posterior, flag_probability
from .candidate import IGNORE, SCOUT, ASK, DISPATCH
from .operator import OperatorModel


class OptimalPolicy:
    name = "ours"
    uses_gate = True

    def __init__(self, sensor, costs, operator_params, scout_alt, grid=401):
        self.sensor = sensor
        self.costs = costs
        self.op = OperatorModel(operator_params)
        self.scout_alt = scout_alt
        self.n = grid
        self.p_hit, self.p_false = sensor.likelihoods(scout_alt, scout=True)
        self.K = costs.max_looks
        self.V = []
        self._solve()

    def _interp(self, table, b):
        if b <= 0.0:
            return table[0]
        if b >= 1.0:
            return table[-1]
        x = b * (self.n - 1)
        i = int(x)
        t = x - i
        return table[i] + t * (table[i + 1] - table[i])

    def ignore_cost(self, b):
        return self.costs.miss * b

    def dispatch_cost(self, b):
        return self.costs.travel + (1.0 - b) * self.costs.false_dispatch

    def after_yes_cost(self, b):
        return min(self.ignore_cost(b), self.dispatch_cost(b))

    def ask_cost(self, b, looked):
        c = self.costs
        p_yes, p_no = self.op.answer_probabilities(b, looked)
        total = c.ask
        if p_yes > 0:
            total += p_yes * self.after_yes_cost(self.op.posterior_yes(b, looked))
        if p_no > 0:
            total += p_no * self.ignore_cost(self.op.posterior_no(b, looked))
        return total

    def look_cost(self, b, k):
        c = self.costs
        pf = flag_probability(b, self.p_hit, self.p_false)
        total = c.look
        if pf > 0:
            total += pf * self._interp(self.V[k - 1], posterior(b, True, self.p_hit, self.p_false))
        if pf < 1:
            total += (1.0 - pf) * self._interp(self.V[k - 1], posterior(b, False, self.p_hit, self.p_false))
        return total

    def _solve(self):
        grid = [i / (self.n - 1) for i in range(self.n)]
        for k in range(self.K + 1):
            level = []
            for b in grid:
                best = min(self.ignore_cost(b), self.ask_cost(b, k < self.K))
                if k > 0:
                    best = min(best, self.look_cost(b, k))
                level.append(best)
            self.V.append(level)

    def values(self, b, k):
        out = {IGNORE: self.ignore_cost(b), ASK: self.ask_cost(b, k < self.K)}
        if k > 0:
            out[SCOUT] = self.look_cost(b, k)
        return out

    def decide(self, cand):
        if cand.asked:
            if cand.op_confirm and self.dispatch_cost(cand.belief) < self.ignore_cost(cand.belief):
                return DISPATCH
            return IGNORE
        k = max(0, self.K - cand.looks)
        vals = self.values(cand.belief, k)
        return min(vals, key=lambda a: vals[a])


class TrustFirst:
    name = "b0"
    uses_gate = False

    def __init__(self, *args, **kwargs):
        pass

    def decide(self, cand):
        return DISPATCH


class Threshold:
    name = "b1"
    uses_gate = True

    def __init__(self, tau):
        self.tau = tau

    def decide(self, cand):
        if cand.asked:
            return DISPATCH if cand.op_confirm else IGNORE
        return ASK if cand.belief >= self.tau else IGNORE


class AlwaysVerify:
    name = "b2"
    uses_gate = True

    def __init__(self, looks=1):
        self.looks = looks

    def decide(self, cand):
        if cand.asked:
            return DISPATCH if cand.op_confirm else IGNORE
        if cand.looks < self.looks:
            return SCOUT
        return ASK


def make_policy(cfg, sensor):
    if cfg.policy == "ours":
        return OptimalPolicy(sensor, cfg.costs, cfg.operator, cfg.flight.scout_alt)
    if cfg.policy == "b0":
        return TrustFirst()
    if cfg.policy == "b1":
        return Threshold(cfg.threshold)
    if cfg.policy == "b2":
        return AlwaysVerify()
    raise ValueError(cfg.policy)
