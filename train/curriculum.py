"""Curriculum schedule for Sisyphus training.

Manages progressive difficulty: slope angle, rock mass, and infinite mode.
Phase promotion is performance-gated using rolling episode metrics.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumParams:
    phase: str
    slope_deg: float
    rock_mass: float
    infinite: bool
    alive_bonus: float = 0.0
    upright_coef: float = 0.0
    forward_push_coef: float = 5.0
    walk_only_mode: bool = False


# Default schedule matching the training plan.
# Phase promotion is metric-gated: rolling mean promotion_score must exceed
# promotion_threshold, after at least min_steps in the phase.
SCHEDULE = [
    # Phase I: Posture scaffolding + approach/push learning (rock always present).
    {"phase": "I",   "slope": 0.0,  "mass": 20.0,
     "infinite": False, "alive_bonus": 0.3, "upright_coef": 1.5,
     "forward_push_coef": 5.0,
     "promotion_metric": "promotion_score", "promotion_threshold": 3.0,
     "min_steps": 3_000_000},
    # Phase II: Add slope. Reduce scaffolding.
    {"phase": "II",  "slope": 5.0,  "mass": 35.0,
     "infinite": False, "alive_bonus": 0.3, "upright_coef": 2.0,
     "forward_push_coef": 5.0,
     "promotion_metric": "promotion_score", "promotion_threshold": 4.0,
     "min_steps": 5_000_000},
    # Phase III: Steeper, heavier. Minimal scaffolding.
    {"phase": "III", "slope": 10.0, "mass": 50.0,
     "infinite": False, "alive_bonus": 0.0, "upright_coef": 1.0,
     "forward_push_coef": 5.0,
     "promotion_metric": "promotion_score", "promotion_threshold": 4.0,
     "min_steps": 5_000_000},
    # Phase IV: Full difficulty. Infinite mode (terminal phase).
    {"phase": "IV",  "slope": 15.0, "mass": 50.0,
     "infinite": True,  "alive_bonus": 0.0, "upright_coef": 0.5,
     "forward_push_coef": 5.0,
     "promotion_metric": None, "promotion_threshold": None,
     "min_steps": None},
]


class CurriculumManager:
    def __init__(self, schedule=None, fixed_phase: str | None = None,
                 initial_timestep: int = 0):
        self.schedule = schedule or SCHEDULE
        self._current_phase_idx = 0
        self._fixed_phase = fixed_phase
        self._phase_start_step = initial_timestep  # total steps when current phase started
        self._promotion_scores = deque(maxlen=50)  # rolling window of promotion scores

        if fixed_phase is not None:
            # Set initial index to match fixed phase
            self._current_phase_idx = next(
                i for i, e in enumerate(self.schedule) if e["phase"] == fixed_phase
            )

    def _params_from_entry(self, entry: dict, walk_only_override: bool = False) -> CurriculumParams:
        return CurriculumParams(
            phase=entry["phase"],
            slope_deg=entry["slope"],
            rock_mass=entry["mass"],
            infinite=entry["infinite"],
            alive_bonus=entry.get("alive_bonus", 0.0),
            upright_coef=entry.get("upright_coef", 0.0),
            forward_push_coef=entry.get("forward_push_coef", 5.0),
            walk_only_mode=walk_only_override,
        )

    @property
    def current_phase_idx(self) -> int:
        return self._current_phase_idx

    @property
    def current_entry(self) -> dict:
        if self._fixed_phase is not None:
            return next(e for e in self.schedule if e["phase"] == self._fixed_phase)
        return self.schedule[self._current_phase_idx]

    def get_params(self, total_steps: int) -> CurriculumParams:
        """Return curriculum parameters for the current phase."""
        entry = self.current_entry

        # Check if we're in the walk-only stage of Phase I
        walk_only = False
        walk_only_steps = entry.get("walk_only_steps", 0)
        if walk_only_steps > 0:
            steps_in_phase = total_steps - self._phase_start_step
            if steps_in_phase < walk_only_steps:
                walk_only = True

        return self._params_from_entry(entry, walk_only_override=walk_only)

    def add_promotion_score(self, score: float):
        """Record an episode's promotion score for the rolling window."""
        self._promotion_scores.append(score)

    def get_rolling_promotion_score(self) -> float | None:
        """Return the rolling mean promotion score, or None if not enough data."""
        if len(self._promotion_scores) < 10:
            return None
        return float(np.mean(self._promotion_scores))

    def check_promotion(self, total_steps: int) -> bool:
        """Check if the current phase should be promoted based on metrics.

        Returns True if promotion criteria are met.
        """
        if self._fixed_phase is not None:
            return False

        entry = self.schedule[self._current_phase_idx]

        # Terminal phase — no promotion
        if entry.get("promotion_metric") is None:
            return False

        steps_in_phase = total_steps - self._phase_start_step
        min_steps = entry.get("min_steps", 0)

        # Not enough steps yet
        if steps_in_phase < min_steps:
            return False

        # Check metric threshold
        rolling_score = self.get_rolling_promotion_score()
        if rolling_score is None:
            return False

        threshold = entry.get("promotion_threshold", float("inf"))
        return rolling_score >= threshold

    def promote(self, total_steps: int) -> CurriculumParams:
        """Advance to the next phase. Returns new params."""
        if self._current_phase_idx < len(self.schedule) - 1:
            self._current_phase_idx += 1
            self._phase_start_step = total_steps
            self._promotion_scores.clear()
        return self.get_params(total_steps)

    def check_transition(self, total_steps: int) -> tuple[bool, CurriculumParams]:
        """Check if a phase transition should occur. Returns (changed, new_params).

        This method checks metric-gated promotion and walk-only mode transitions.
        """
        if self._fixed_phase is not None:
            return False, self.get_params(total_steps)

        if self.check_promotion(total_steps):
            params = self.promote(total_steps)
            return True, params

        return False, self.get_params(total_steps)
