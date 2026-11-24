# SAR-Verify (Webots simulation)

A drone-and-rover search and rescue team that decides which aerial detections are worth acting on.
Physical AI Hackathon, Track 5 (Disaster Management).

The simulator is Webots R2025a. The decision logic lives in `sarverify/` (pure Python, no Webots import), so the same code runs in the 3D sim, in the batch evaluation, and on a Raspberry Pi 5.

## Perceive, decide, act

| Stage | What happens | Where |
|---|---|---|
| Perceive | The mothership sweeps at 25 m. A detection is sampled from an altitude-dependent sensor model. Each flag becomes a candidate with a belief. | `controllers/referee`, `sarverify/sensor.py` |
| Decide | A dynamic-programming policy picks ignore, scout look, ask operator, or dispatch rover, trading time, battery and false-dispatch cost against the cost of a miss. | `sarverify/policy.py` |
| Act | The scout descends to 10 m and re-observes. The operator confirms or rejects (hard gate). The rover plans an A* path around obstacles and drives to the point. | `controllers/scout`, `rover`, `commander` |

Baselines: `b0` trust the first detection, `b1` fixed confidence threshold, `b2` always verify, `ours` optimal policy.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Webots R2025a must be installed at `/Applications/Webots.app`.

## Run

```
.venv/bin/python tools/run_webots.py --seed 1 --policy ours
.venv/bin/python tools/run_webots.py --seed 1 --policy b0 --fast
.venv/bin/python tools/run_webots.py --seed 1 --policy ours --operator manual
.venv/bin/python tools/run_webots.py --seed 1 --policy ours --headless
.venv/bin/python tools/run_webots.py --seed 1 --policy ours --fast --record out.mp4
.venv/bin/python tools/run_webots.py --seed 1 --policy ours --fast --snapshots runs/snaps
```

`--operator manual` lets you play the operator: press Y (person) or N (not a person) in the Webots window when prompted.
Each run writes `metrics.json`, `events.jsonl` and `scene.json` under `runs/<policy>_seed<N>/`.

```
.venv/bin/python -m pytest
```

## Layout

```
sarverify/     config, sensor model, belief, policy, operator, planner, scene, world writer, flight, comms
controllers/   mothership, scout, rover, commander (policy host), referee (supervisor: ground truth, metrics, HUD)
protos/        Marker.proto (candidate markers drawn in the 3D view)
data/          sensor_curves.csv
tools/         run_webots.py
tests/         unit tests for the core library
```

Marker colours: yellow new or waiting, blue scouting, purple awaiting operator, orange rover assigned, green confirmed, grey rejected, red wasted trip.

## Scout detector (real model in the loop)

The scout drone runs the team's trained YOLOv8n person detector (`models/sar_person_yolov8n_v2_best_int8.onnx`, ONNX Runtime, CPU) on its own rendered camera image. The scout looks obliquely from 14 m standoff at 12 m altitude, projects the candidate into the image and flags a person if a detection falls within 3 m of it. The detector is not run on the mothership: mothership detections are still sampled from `data/sensor_curves.csv` (assumed).

Measured with `tools/calibrate_detector.py` on rendered images (15 seeds, scout look geometry, confidence 0.25):

| target | detected | 95% interval |
|---|---|---|
| standing person | 78/84 | 0.85 to 0.97 |
| lying person, clear | 10/51 | 0.11 to 0.33 |
| lying person, partly covered | 18/45 | 0.27 to 0.55 |
| decoy flagged as person | 2/270 | 0.00 to 0.03 |
| empty ground flagged | 0/450 | 0.00 to 0.01 |

`data/scout_detector.json` feeds these (Laplace-smoothed) into the belief update and the planner. Straight-down views were much worse for every pose, which is why the scout looks obliquely (`tools/pose_study.py`).

Run `tools/calibrate_detector.py` to regenerate. The numbers describe the detector on Webots renders, not on real aerial footage.

## Honest limits

- The detector was trained on real aerial footage (VisDrone) and is tested here only on synthetic renders. Sim-to-real behaviour is unmeasured.

- `data/sensor_curves.csv` holds **assumed** values, labelled `assumed` in the file and in every `metrics.json`. They are placeholders until measured on real aerial images and a small altitude test with the real drone. Swapping in measured values needs no code change.
- Detection is a parametric sensor model sampled by the referee from ground truth and the drone's actual altitude. It is not a neural detector running on rendered images.
- The planner uses marginal detection likelihoods averaged over the known occlusion mix, so its beliefs stay calibrated when some victims are partly hidden.
- The operator is modelled with delay and error. Accuracy is higher after a close scout look than from the overview. Scout evidence and operator evidence are treated as independent, which is a simplification.
- The policy treats each candidate independently. It does not model contention for the single scout and rover.
- The rover plans on a known obstacle map and follows it with pure pursuit. It does not use lidar or Nav2.
- The scout is a generic quad, not the Pluto drone.
- Statistics need many seeds. A single run, such as `--seed 1`, shows that the system works, not that one policy beats another.

## Batch evaluation (fast simulator)

`sarverify/fastsim.py` is a pure-Python mission model that reuses the policy, belief, operator, scene and A* code from the Webots run. Its speeds were set from one Webots run (seed 1), so timings are approximate. Scout looks are sampled from the rates measured with the trained detector on rendered images (`data/scout_detector.json`). Mothership detections are still sampled from the assumed `data/sensor_curves.csv`.

```
.venv/bin/python tools/evaluate.py --passes 2 --out results
.venv/bin/python tools/evaluate.py --passes 1 --out results/single_pass
.venv/bin/python tools/evaluate_operator.py
.venv/bin/python tools/plot_results.py
```

200 evaluation seeds (2001-2200), B1 threshold tuned on separate seeds (1001-1060), 95% bootstrap intervals, false-dispatch cost swept 10 to 320 (`results/tradeoff.png`).

Mothership re-sweep: if victims are unaccounted for after the first pass and time remains, the mothership flies the lanes again in reverse (one extra pass). Mean recall at false-dispatch cost 40: 0.68 with one pass, 0.79 with two (ours and B2).

Findings, stated as measured:
- The optimal policy matches B2 (always verify) exactly whenever asking the operator is cheap relative to a miss. It beats B0 (trust first) clearly once a false dispatch costs 80 or more, and loses to B0 below that.
- Operator scarcity sweep (`results/operator_sweep.csv`, operator answers one request at a time): the policy separates from B2 only when an operator question costs as much as a false dispatch (40), where it saves about 18 to 33 cost units per mission. When asking is expensive and false dispatches are cheap, B0 is cheaper than ours.
- Operator delay lowers recall and slows confirmation for every policy through the time limit, but does not change which policy wins.
