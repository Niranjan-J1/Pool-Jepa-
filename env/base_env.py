from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np


class Action(TypedDict):
    left_pwm: float    # -1.0 to 1.0
    right_pwm: float   # -1.0 to 1.0
    fire: bool
    fire_force: float  # 0.0 to 1.0


class Observation(TypedDict):
    frame: np.ndarray              # 128x128x3 uint8
    robot_pose: tuple[float, float, float]  # (x, y, theta)
    ball_positions: list[tuple[float, float]]  # [(x, y), ...]
    ball_settled: bool


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
        """Return the current frame as a 128x128x3 uint8 numpy array."""
        ...
