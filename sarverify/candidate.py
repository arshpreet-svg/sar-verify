from dataclasses import dataclass, field
from typing import List, Optional, Tuple

IGNORE = "ignore"
SCOUT = "scout"
ASK = "ask"
DISPATCH = "dispatch"

NEW = "new"
WAIT_SCOUT = "wait_scout"
SCOUTING = "scouting"
WAIT_OPERATOR = "wait_operator"
WAIT_ROVER = "wait_rover"
ROVER_GOING = "rover_going"
CONFIRMED = "confirmed"
REJECTED = "rejected"
WASTED = "wasted"


@dataclass
class Candidate:
    id: int
    pos: Tuple[float, float]
    belief: float
    first_seen: float
    looks: int = 0
    asked: bool = False
    op_confirm: Optional[bool] = None
    status: str = NEW
    sightings: int = 1
    history: List[Tuple[float, str, float]] = field(default_factory=list)

    def log(self, t, what):
        self.history.append((t, what, self.belief))
