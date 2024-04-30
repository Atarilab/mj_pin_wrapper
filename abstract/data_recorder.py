import numpy as np
from mujoco._structs import MjData

class DataRecorderAbstract(object):
    def __init__(self,
                 record_dir:str="") -> None:
        self.record_dir = record_dir
        
    def record(self,
               q:np.array,
               v:np.array,
               robot_data:MjData,
               ) -> None:
        pass