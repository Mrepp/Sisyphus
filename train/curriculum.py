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


# Default schedule matching the training plan
SCHEDULE = [
    {"phase": "I",   "slope": 0.0,  "mass": 5.0,  "end_step": 5_000_000,  "infinite": False},
    {"phase": "II",  "slope": 5.0,  "mass": 8.0,  "end_step": 15_000_000, "infinite": False},
    {"phase": "III", "slope": 10.0, "mass": 12.0, "end_step": 30_000_000, "infinite": False},
    {"phase": "IV",  "slope": 15.0, "mass": 10.0, "end_step": 50_000_000, "infinite": True},
]


class CurriculumManager:
    def __init__(self, schedule=None):
        self.schedule = schedule or SCHEDULE
        self._current_phase_idx = 0

    def get_params(self, total_steps: int) -> CurriculumParams:
        """Return curriculum parameters for the given total step count."""
        for entry in self.schedule:
            if total_steps < entry["end_step"]:
                return CurriculumParams(
                    phase=entry["phase"],
                    slope_deg=entry["slope"],
                    rock_mass=entry["mass"],
                    infinite=entry["infinite"],
                )
        # Past all phases — stay on last
        last = self.schedule[-1]
        return CurriculumParams(
            phase=last["phase"],
            slope_deg=last["slope"],
            rock_mass=last["mass"],
            infinite=last["infinite"],
        )

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
