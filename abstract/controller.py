import numpy as np
from mujoco._structs import MjData

from .robot import RobotWrapperAbstract

class ControllerAbstract(object):
    def __init__(self,
                 robot: RobotWrapperAbstract,
                 **kwargs,
                 ) -> None:
        self.robot = robot
        self.diverged = False
        
    def get_torques(self,
                    q:np.array,
                    v:np.array,
                    robot_data:MjData,
                    ) -> dict[float] :
        return {}