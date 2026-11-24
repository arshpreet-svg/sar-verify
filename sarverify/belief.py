def posterior(b, flag, p_hit, p_false):
    if flag:
        num = b * p_hit
        den = num + (1.0 - b) * p_false
    else:
        num = b * (1.0 - p_hit)
        den = num + (1.0 - b) * (1.0 - p_false)
    if den <= 0.0:
        return b
    return num / den


def prior_given_flag(n_victims, n_decoys, p_hit, p_false):
    num = n_victims * p_hit
    den = num + n_decoys * p_false
    if den <= 0.0:
        return 0.5
    return num / den


def flag_probability(b, p_hit, p_false):
    return b * p_hit + (1.0 - b) * p_false
