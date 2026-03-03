"""Curriculum schedule for Sisyphus training.

Manages progressive difficulty: slope angle, rock mass, and infinite mode.
"""

from dataclasses import dataclass


@dataclass
class CurriculumParams:
    phase: str
    slope_deg: float
    rock_mass: float
    infinite: bool
    alive_bonus: float = 0.0
    upright_coef: float = 0.0
    forward_push_coef: float = 5.0


# Default schedule matching the training plan.
# alive_bonus / upright_coef provide posture scaffolding that decays to zero,
# so the agent bootstraps standing quickly but still discovers its own gait.
SCHEDULE = [
    {"phase": "I",   "slope": 0.0,  "mass": 20.0,  "end_step": 5_000_000,
     "infinite": False, "alive_bonus": 2.0, "upright_coef": 1.0, "forward_push_coef": 5.0},
    {"phase": "II",  "slope": 5.0,  "mass": 35.0,  "end_step": 15_000_000,
     "infinite": False, "alive_bonus": 1.0, "upright_coef": 0.3, "forward_push_coef": 5.0},
    {"phase": "III", "slope": 10.0, "mass": 50.0,  "end_step": 30_000_000,
     "infinite": False, "alive_bonus": 0.0, "upright_coef": 0.2, "forward_push_coef": 5.0},
    {"phase": "IV",  "slope": 15.0, "mass": 40.0,  "end_step": 50_000_000,
     "infinite": True,  "alive_bonus": 0.0, "upright_coef": 0.2, "forward_push_coef": 5.0},
]


class CurriculumManager:
    def __init__(self, schedule=None):
        self.schedule = schedule or SCHEDULE
        self._current_phase_idx = 0

    def _params_from_entry(self, entry: dict) -> CurriculumParams:
        return CurriculumParams(
            phase=entry["phase"],
            slope_deg=entry["slope"],
            rock_mass=entry["mass"],
            infinite=entry["infinite"],
            alive_bonus=entry.get("alive_bonus", 0.0),
            upright_coef=entry.get("upright_coef", 0.0),
            forward_push_coef=entry.get("forward_push_coef", 5.0),
        )

    def get_params(self, total_steps: int) -> CurriculumParams:
        """Return curriculum parameters for the given total step count."""
        for entry in self.schedule:
            if total_steps < entry["end_step"]:
                return self._params_from_entry(entry)
        # Past all phases — stay on last
        return self._params_from_entry(self.schedule[-1])

    def check_transition(self, total_steps: int) -> tuple[bool, CurriculumParams]:
        """Check if a phase transition occurred. Returns (changed, new_params)."""
        params = self.get_params(total_steps)
        idx = next(
            (i for i, e in enumerate(self.schedule) if total_steps < e["end_step"]),
            len(self.schedule) - 1,
        )
        changed = idx != self._current_phase_idx
        self._current_phase_idx = idx
        return changed, params
