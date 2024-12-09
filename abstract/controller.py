# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any

import numpy as np

from utils.mj_pin_wrapper.abstract.robot import AbstractRobotWrapper


class ControllerAbstract(object):
    def __init__(
        self,
        robot: AbstractRobotWrapper,
        **kwargs,
    ) -> None:
        self.robot = robot
        self.diverged = False

    def get_torques(
        self,
        q: np.array,
        v: np.array,
        robot_data: Any,
        **kwargs,
    ) -> dict[float]:
        """
        Returns torques as a map from joint names to torque values.
        """
        return {}

    def reset(self) -> None:
        """
        Reset controller
        """
        pass
