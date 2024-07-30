# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import numpy as np
import pinocchio as pin
import mujoco
from typing import Any, Tuple
from numpy.typing import NDArray

from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper

######################################################################
#####
#####                   MJPinRobotWrapper      
#####
######################################################################

class MJPinQuadRobotWrapper(object):
    """
    Rrobot wrapper class that merges MuJoCo robot
    model to Pinocchio robot model.
    The robot is torque controled.
    
    Both URDF and XML description files should correspond to the 
    same physical model with the same description.
    Follow these steps for a custom model:
    https://github.com/machines-in-motion/mujoco_utils/tree/main?tab=readme-ov-file#converting-urdf-to-mujoco-xml-format
    """
    def __init__(self,
                 path_urdf: str,
                 path_xml_mj: str,
                 path_package_dir: str | list[str] | None = None,
                 **kwargs,
                 ) -> None:

        self.path_urdf = path_urdf
        self.path_xml_mj = path_xml_mj
        self.path_package_dir = path_package_dir
        
        self.mj : MJQuadRobotWrapper = MJQuadRobotWrapper(self.path_xml_mj, **kwargs)
        self.pin : PinQuadRobotWrapper = PinQuadRobotWrapper(self.path_urdf, self.path_package_dir, **kwargs)
                
        self.collided = False # True if robot is in collision
        
        self._check_models()
        self.reset()
        
    def _check_models(self) -> None:
        """
        Checks if Pinocchio and MuJoco models are coherent.
        """
        # Same robot description
        assert self.pin.nq == self.mj.nq and\
               self.pin.nv == self.mj.nv and\
               self.pin.nu == self.mj.nu and\
               self.pin.ne == self.mj.ne,\
               "Pinocchio and MuJoCo have different model descriptions."
    
    def is_collision(self,
                     exclude_end_effectors: bool = True) -> bool:
        """
        Return True if some robot geometries are in contact
        with the environment.
        
        Args:
            - exclude_end_effectors (bool): exclude contacts of the end-effectors.
        """
        n_eeff_contact = 0
        if exclude_end_effectors:
            eeff_contact = self.get_mj_eeff_contact_with_floor()
            n_eeff_contact = sum([int(contact) for contact in eeff_contact.values()])
        
        n_contact = len(self.mj_data.contact)
        
        is_collision, self.collided = False, False
        if n_eeff_contact != n_contact:
            is_collision, self.collided = True, True
        
        return is_collision
    
    # def step(self) -> None:
    #     """
    #     MuJoCo simulation step.
    #     Update pinocchio data.
    #     """
    #     mujoco.mj_step(self.mj.model, self.mj.data)
    #     self.mj.contact_updated = False
    #     q, v = self.get_state()
    #     self.pin.update(q, v)
              
    def reset(self, q: NDArray[np.float64] = None, v: NDArray[np.float64] = None) -> None:
        """
        Reset robot state and simulation state.

        Args:
            - q (np.ndarray): Initial state.
            - v (np.ndarray): Initial velocities.
        """
        if q is None:
            # Set to initial position
            q = self.mj.q0
        if v is None:
            v = np.zeros(self.mj.nv)
            
        # Reset data
        self.mj.data = mujoco.MjData(self.mj.model)
        pin.computeAllTerms(self.pin.model, self.pin.data, q, v)
        self.mj.update(q, v)
        self.pin.update(q, v)
        
    def get_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns MuJoCo state.
        """
        return self.mj.get_state()
    
    def is_collision(self,
                     exclude_end_effectors: bool = True,
                     only_base: bool = False,
                     self_collision : bool = False) -> bool:
        """
        Collision status in MuJoCo.
        """
        self.collided = self.mj.is_collision(exclude_end_effectors, only_base, self_collision)
        return self.collided
    
    def foot_contacts(self) -> dict[str, int]:
        """
        Returns foot contacts with static geometries as a dict.

        Returns:
            dict[str, int]: {foot name : geom in contact}
        """
        return self.mj.foot_contacts()