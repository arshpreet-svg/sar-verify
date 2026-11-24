import json
import os
from dataclasses import dataclass, field, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CONFIG_PATH = os.path.join(ROOT, "run_config.json")
SENSOR_CSV = os.path.join(ROOT, "data", "sensor_curves.csv")


@dataclass
class Costs:
    miss: float = 100.0
    false_dispatch: float = 40.0
    travel: float = 5.0
    look: float = 8.0
    ask: float = 4.0
    max_looks: int = 2


@dataclass
class OperatorParams:
    sensitivity: float = 0.95
    specificity: float = 0.92
    sensitivity_far: float = 0.80
    specificity_far: float = 0.60
    delay_mean: float = 6.0
    delay_sd: float = 2.0
    mode: str = "auto"


@dataclass
class SceneParams:
    size: float = 100.0
    n_victims: int = 4
    n_decoys: int = 6
    n_buildings: int = 5
    n_rubble: int = 8
    occluded_fraction: float = 0.3
    standing_fraction: float = 0.5
    flood: bool = True


@dataclass
class FlightParams:
    sweep_alt: float = 25.0
    scout_alt: float = 12.0
    scout_standoff: float = 14.0
    fov: float = 1.0
    swath: float = 24.0
    scout_dwell: float = 3.0
    rover_speed: float = 1.6
    time_limit: float = 900.0


@dataclass
class RunConfig:
    seed: int = 1
    policy: str = "ours"
    threshold: float = 0.5
    sensor_csv: str = SENSOR_CSV
    costs: Costs = field(default_factory=Costs)
    operator: OperatorParams = field(default_factory=OperatorParams)
    scene: SceneParams = field(default_factory=SceneParams)
    flight: FlightParams = field(default_factory=FlightParams)
    out_dir: str = os.path.join(ROOT, "runs", "latest")
    headless: bool = False
    record_video: str = ""
    snapshot_dir: str = ""

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    def save(self, path=RUN_CONFIG_PATH):
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d):
        return cls(
            seed=d["seed"],
            policy=d["policy"],
            threshold=d["threshold"],
            sensor_csv=d["sensor_csv"],
            costs=Costs(**d["costs"]),
            operator=OperatorParams(**d["operator"]),
            scene=SceneParams(**d["scene"]),
            flight=FlightParams(**d["flight"]),
            out_dir=d["out_dir"],
            headless=d["headless"],
            record_video=d["record_video"],
            snapshot_dir=d.get("snapshot_dir", ""),
        )

    @classmethod
    def load(cls, path=RUN_CONFIG_PATH):
        with open(path) as f:
            return cls.from_dict(json.load(f))
