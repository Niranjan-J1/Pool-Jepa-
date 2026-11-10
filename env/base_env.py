from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict

import numpy as np


class Action(TypedDict):
    left_pwm: float    # -1.0 to 1.0
    right_pwm: float   # -1.0 to 1.0
    fire: bool
    fire_force: float  # 0.0 to 1.0


class Observation(TypedDict):
    frame: np.ndarray              # HxWx3 uint8 (sim: 240x432, see sim_env.RENDER_*)
    robot_pose: tuple[float, float, float]  # (x, y, theta)
    ball_positions: list[tuple[float, float]]  # [(x, y), ...] in metres, id order (0 = cue)
    ball_settled: bool
    # The keys below are present only in multiball mode (SimEnv(n_object_balls=...)); they are
    # absent in legacy mode so existing consumers see an unchanged observation.
    ball_positions_cm: NotRequired[dict[int, tuple[float, float]]]  # id -> (x, y) cm; absent if potted
    potted: NotRequired[list[int]]        # object-ball ids, in the order potted
    scratch: NotRequired[bool]            # True if the cue ball has been potted (then respawned)
    n_balls_on_table: NotRequired[int]    # cue + object balls currently on the table


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply an action and return the resulting observation."""
        ...

    @abstractmethod
    def render(self) -> np.ndarray:
        """Return the current frame as an HxWx3 uint8 numpy array."""
        ...
