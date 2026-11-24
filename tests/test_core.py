import random

import pytest

from sarverify import scene as scene_mod
from sarverify.belief import posterior, prior_given_flag
from sarverify.candidate import ASK, DISPATCH, IGNORE, SCOUT, Candidate
from sarverify.config import Costs, FlightParams, OperatorParams, RunConfig, SceneParams
from sarverify.operator import OperatorModel
from sarverify.planner import GridMap, path_length, plan
from sarverify.policy import AlwaysVerify, OptimalPolicy, Threshold, TrustFirst
from sarverify.sensor import SensorModel
from sarverify.worldfile import build_world


@pytest.fixture(scope="module")
def sensor():
    return SensorModel.from_csv(RunConfig().sensor_csv).marginal(0.3)


def test_posterior_bounds_and_direction():
    assert posterior(0.5, True, 0.9, 0.1) > 0.5
    assert posterior(0.5, False, 0.9, 0.1) < 0.5
    assert posterior(0.0, True, 0.9, 0.1) == 0.0
    assert posterior(1.0, False, 0.9, 0.1) == 1.0


def test_posterior_matches_bayes():
    b = posterior(0.3, True, 0.8, 0.2)
    assert b == pytest.approx(0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.2))


def test_prior_given_flag():
    assert prior_given_flag(4, 6, 0.7, 0.28) == pytest.approx(2.8 / (2.8 + 1.68))
    assert prior_given_flag(0, 0, 0.7, 0.28) == 0.5


def test_sensor_interpolates_and_degrades_with_altitude(sensor):
    assert sensor.p_detect(10.0) > sensor.p_detect(25.0) > sensor.p_detect(40.0)
    assert sensor.p_false_decoy(10.0) < sensor.p_false_decoy(25.0)
    assert sensor.p_detect(20.0) == pytest.approx(0.5 * (sensor.p_detect(15.0) + sensor.p_detect(25.0)))
    assert sensor.p_detect(10.0, occlusion=1.0) == pytest.approx(0.5 * sensor.p_detect(10.0))


def test_sensor_sampling_rate(sensor):
    rng = random.Random(0)
    n = 20000
    hits = sum(sensor.observe(rng, 25.0, "victim") for _ in range(n))
    assert hits / n == pytest.approx(sensor.p_detect(25.0), abs=0.01)


def test_operator_posteriors_consistent():
    op = OperatorModel(OperatorParams())
    for looked in (True, False):
        b = 0.4
        p_yes, p_no = op.answer_probabilities(b, looked)
        assert p_yes + p_no == pytest.approx(1.0)
        recon = p_yes * op.posterior_yes(b, looked) + p_no * op.posterior_no(b, looked)
        assert recon == pytest.approx(b)


def test_operator_closeup_is_more_reliable():
    op = OperatorModel(OperatorParams())
    assert op.posterior_yes(0.5, True) > op.posterior_yes(0.5, False)
    assert op.posterior_no(0.5, True) < op.posterior_no(0.5, False)


def test_operator_delay_positive_with_right_mean():
    op = OperatorModel(OperatorParams(delay_mean=6.0, delay_sd=2.0))
    rng = random.Random(1)
    xs = [op.delay(rng) for _ in range(20000)]
    assert min(xs) > 0
    assert sum(xs) / len(xs) == pytest.approx(6.0, abs=0.15)


def test_optimal_policy_structure(sensor):
    cfg = RunConfig()
    pol = OptimalPolicy(sensor, cfg.costs, cfg.operator, 10.0)
    assert pol.decide(Candidate(0, (0, 0), 0.02, 0)) == IGNORE
    assert pol.decide(Candidate(0, (0, 0), 0.5, 0)) == SCOUT


def test_policy_gate_after_operator(sensor):
    cfg = RunConfig()
    pol = OptimalPolicy(sensor, cfg.costs, cfg.operator, 10.0)
    c = Candidate(0, (0, 0), 0.9, 0, looks=1, asked=True, op_confirm=False)
    assert pol.decide(c) == IGNORE
    c.op_confirm = True
    assert pol.decide(c) == DISPATCH


def test_policy_never_dispatches_without_operator(sensor):
    cfg = RunConfig()
    pol = OptimalPolicy(sensor, cfg.costs, cfg.operator, 10.0)
    for i in range(0, 101):
        for looks in range(0, 3):
            c = Candidate(0, (0, 0), i / 100.0, 0, looks=looks)
            assert pol.decide(c) != DISPATCH


def test_higher_false_dispatch_cost_reduces_dispatching(sensor):
    cfg = RunConfig()
    lo = OptimalPolicy(sensor, Costs(false_dispatch=5.0), cfg.operator, 10.0)
    hi = OptimalPolicy(sensor, Costs(false_dispatch=400.0), cfg.operator, 10.0)
    c_lo = Candidate(0, (0, 0), 0.6, 0, looks=2, asked=True, op_confirm=True)
    c_hi = Candidate(0, (0, 0), 0.6, 0, looks=2, asked=True, op_confirm=True)
    assert lo.decide(c_lo) == DISPATCH
    assert hi.decide(c_hi) == IGNORE


def test_value_never_exceeds_ignoring(sensor):
    cfg = RunConfig()
    pol = OptimalPolicy(sensor, cfg.costs, cfg.operator, 10.0)
    for k in range(len(pol.V)):
        for i in range(0, 401, 10):
            assert 0.0 <= pol.V[k][i] <= cfg.costs.miss * (i / 400.0) + 1e-9


def test_baselines():
    c = Candidate(0, (0, 0), 0.6, 0)
    assert TrustFirst().decide(c) == DISPATCH
    assert Threshold(0.5).decide(c) == ASK
    assert Threshold(0.7).decide(c) == IGNORE
    assert AlwaysVerify().decide(c) == SCOUT
    c.looks = 1
    assert AlwaysVerify().decide(c) == ASK


def test_scene_deterministic_and_valid():
    sp, fl = SceneParams(), FlightParams()
    a = scene_mod.generate(5, sp, fl)
    b = scene_mod.generate(5, sp, fl)
    assert a == b
    assert a != scene_mod.generate(6, sp, fl)
    assert len(a["victims"]) == sp.n_victims
    assert len(a["decoys"]) == sp.n_decoys
    half = a["half"]
    for t in a["victims"] + a["decoys"]:
        assert -half < t["x"] < half and -half < t["y"] < half


def test_planner_reaches_every_target_around_obstacles():
    sp, fl = SceneParams(), FlightParams()
    for seed in range(1, 6):
        sc = scene_mod.generate(seed, sp, fl)
        g = GridMap(sc["half"], 1.0, sc["rover_obstacles"], 1.5)
        for v in sc["victims"]:
            p = plan(g, sc["start"]["rover"], (v["x"], v["y"]))
            assert p is not None
            assert path_length(p) >= ((v["x"] - sc["start"]["rover"][0]) ** 2 + (v["y"] - sc["start"]["rover"][1]) ** 2) ** 0.5 - 1e-6
            for a, b in zip(p, p[1:]):
                assert g.line_free(a, b)


def test_world_file_contains_all_entities():
    cfg = RunConfig()
    sc = scene_mod.generate(1, cfg.scene, cfg.flight)
    w = build_world(sc, cfg)
    for name in ("MOTHERSHIP", "SCOUT", "ROVER", "COMMANDER", "REFEREE", "VIEW"):
        assert f"DEF {name} " in w
    assert w.count("DEF VICTIM_") == cfg.scene.n_victims
    assert w.count("DEF DECOY_") == cfg.scene.n_decoys


def test_config_roundtrip(tmp_path):
    cfg = RunConfig(seed=9, policy="b1")
    p = tmp_path / "c.json"
    cfg.save(str(p))
    assert RunConfig.load(str(p)) == cfg


def test_projection_center_and_side():
    from sarverify.cinema import project
    cam, tgt = (0.0, -20.0, 10.0), (0.0, 0.0, 0.0)
    cx, cy, depth = project(tgt, cam, tgt, 1280, 720, 0.95)
    assert abs(cx - 640) < 1 and abs(cy - 360) < 1 and depth > 20
    left = project((-5.0, 0.0, 0.0), cam, tgt, 1280, 720, 0.95)
    right = project((5.0, 0.0, 0.0), cam, tgt, 1280, 720, 0.95)
    assert left[0] < 640 < right[0]
    assert project((0.0, -40.0, 10.0), cam, tgt, 1280, 720, 0.95) is None


def test_blocked_by_building_and_clear_above():
    from sarverify.cinema import blocked
    boxes = [{"x": 0.0, "y": 0.0, "sx": 10.0, "sy": 10.0, "yaw": 0.0, "h": 6.0}]
    assert blocked((-20.0, 0.0, 2.0), (20.0, 0.0, 1.0), boxes)
    assert not blocked((-20.0, 0.0, 14.0), (20.0, 0.0, 12.0), boxes)
    assert not blocked((-20.0, 20.0, 2.0), (20.0, 20.0, 2.0), boxes)


def test_victim_poses_stable_and_standing_not_occluded():
    from sarverify.config import RunConfig
    from sarverify import scene as sm
    cfg = RunConfig()
    a = sm.generate(5, cfg.scene, cfg.flight)
    b = sm.generate(5, cfg.scene, cfg.flight)
    assert [v["pose"] for v in a["victims"]] == [v["pose"] for v in b["victims"]]
    assert all(v["occlusion"] == 0.0 for v in a["victims"] if v["pose"] == "standing")
    poses = {v["pose"] for s in range(1, 30) for v in sm.generate(s, cfg.scene, cfg.flight)["victims"]}
    assert poses == {"standing", "lying"}


def test_fastsim_resweep_does_not_reduce_recall():
    from sarverify import fastsim
    from sarverify.config import RunConfig

    truth = fastsim.load_scout_truth()
    one = two = 0.0
    for s in range(3001, 3021):
        c = RunConfig()
        c.seed = s
        one += fastsim.run_one(c, truth, max_passes=1)["recall"]
        two += fastsim.run_one(c, truth, max_passes=2)["recall"]
    assert two >= one
