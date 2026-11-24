import math


class OperatorModel:
    def __init__(self, params):
        self.p = params

    def delay(self, rng):
        mu = self.p.delay_mean
        sd = self.p.delay_sd
        if mu <= 0:
            return 0.0
        var = sd * sd
        sigma2 = math.log(1.0 + var / (mu * mu))
        m = math.log(mu) - 0.5 * sigma2
        return rng.lognormvariate(m, math.sqrt(sigma2))

    def rates(self, looked):
        if looked:
            return self.p.sensitivity, self.p.specificity
        return self.p.sensitivity_far, self.p.specificity_far

    def respond(self, rng, truth_is_person, looked=True):
        sens, spec = self.rates(looked)
        if truth_is_person:
            return rng.random() < sens
        return rng.random() < (1.0 - spec)

    def answer_probabilities(self, b, looked=True):
        sens, spec = self.rates(looked)
        p_yes = b * sens + (1.0 - b) * (1.0 - spec)
        return p_yes, 1.0 - p_yes

    def posterior_yes(self, b, looked=True):
        sens, _ = self.rates(looked)
        p_yes, _ = self.answer_probabilities(b, looked)
        if p_yes <= 0.0:
            return b
        return b * sens / p_yes

    def posterior_no(self, b, looked=True):
        sens, _ = self.rates(looked)
        _, p_no = self.answer_probabilities(b, looked)
        if p_no <= 0.0:
            return b
        return b * (1.0 - sens) / p_no
