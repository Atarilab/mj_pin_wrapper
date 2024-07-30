# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import numpy as np
import pinocchio as pin
from typing import List, Tuple, Union
from numpy.typing import NDArray

from mj_pin_wrapper.abstract.robot import AbstractQuadRobotWrapper

class PinQuadRobotWrapper(AbstractQuadRobotWrapper):
    """
    Pinocchio quadruped robot wrapper.
    """
    # Default optionals
    DEFAULT_ROTOR_INERTIA = 0.
    DEFAULT_GEAR_RATIO = 1.
    DEFAULT_TORQUE_LIMIT = 100.
    DEFAULT_VEL_LIMIT = 100.
    
    def __init__(self,
                 path_urdf: str,
                 path_package_dir: Union[str, List[str], None] = None,
                 floating_base: bool = True,
                 load_geometry: bool = False,
                 **kwargs,
                 ) -> None:
        
        self.path_urdf = path_urdf
        self.path_package_dir = path_package_dir
        self.floating_base = floating_base

        # Optional args
        optional_args = {
            "rotor_inertia": PinQuadRobotWrapper.DEFAULT_ROTOR_INERTIA,
            "gear_ratio": PinQuadRobotWrapper.DEFAULT_GEAR_RATIO,
            "torque_limit": PinQuadRobotWrapper.DEFAULT_TORQUE_LIMIT,
            "vel_limit": PinQuadRobotWrapper.DEFAULT_VEL_LIMIT,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        # Init pinocchio robot
        self.pin_robot = pin.RobotWrapper.BuildFromURDF(
            filename=self.path_urdf,
            package_dirs=self.path_package_dir,
            root_joint=pin.JointModelFreeFlyer() if self.floating_base else pin.JointModelFixed(),
            verbose=False,
            meshLoader=pin.GeometryModel() if load_geometry else None
        )
        self.model = self.pin_robot.model
        self.data = self.pin_robot.data
        
        self.geom_model = self.pin_robot.visual_model if load_geometry else None
        self.geom_data = pin.GeometryData(self.geom_model) if load_geometry else None
        
        super().__init__(model = self.pin_robot.model, data = self.pin_robot.data)

        self.set_pin_rotor_params(self.rotor_inertia, self.gear_ratio)
        self.set_pin_limits(self.torque_limit, self.vel_limit)

        # Set robot configuration dimensions
        # configuration vector
        self.nq = self.model.nq
        # velocity vector space
        self.nv = self.model.nv
        # Number of actuators
        self.nu = len(self.joint_names)
        # Number of end effectors
        self.ne = len(self.eeff_idx)
        
        self._is_description_valid()
        
        self.q, self.v = np.zeros((self.nq)), np.zeros((self.nv))
        
    def _init_frame_map(self) -> dict[str, int]:
        """
        Init frame name to id map.
        """
        pin_n_frames = len(self.model.frames)
        pin_frame_name2id = {
            self.model.frames[i].name : i
            for i in range(pin_n_frames)
            if self.model.frames[i].parent >= 0  # Exclude invalid parent frames
        }
        return pin_frame_name2id

    def _init_joint_map(self) -> dict[str, int]:
        """
        Init joint name to id map.
        """
        pin_n_joints = len(self.model.joints)
        pin_joint_name2id = {
            self.model.names[i] : i
            for i in range(pin_n_joints)
            if (# Only 1 DoF joints (no root joint)
                self.model.joints[i].nq == 1 and 
                 # No universe joint or ill-defined joints
                self.model.joints[i].id <= pin_n_joints)
        }
        return pin_joint_name2id
    
    def _init_actuator_map(self) -> dict[str, int]:
        # Get joint id to name map sorted by index
        joint_id2name = {
            i : name
            for name, i
            in self.joint_name2id.items() 
        }
        # Map to joints
        joint_name2act_id = {
            name : i 
            for i, name in enumerate(joint_id2name.values())
        }
        return joint_name2act_id
    
    def _init_geom_id(self) -> List[int]:
        # Pinochio geometries parent frame id
        collision_model = self.pin_robot.collision_model
        geom_idx = []
        for geom in collision_model.geometryObjects:
            geom_idx.append(geom.parentFrame)
        return geom_idx
    
    def get_frame_position_world(self, frame_name : str) -> NDArray[np.float64]:
        """
        Get frame position in base frame.

        Args:
            frame_name (str): The name of the frame.

        Returns:
            NDArray[np.float64]: The position of the frame in the base frame.
        """
        frame_position = np.empty((3), np.float64)
        if self.model.existFrame(frame_name):
            frame_id = self.frame_name2id[frame_name]
            frame_position = self.data.oMf[frame_id].translation
        return frame_position
    
    def get_joint_position_world(self, joint_name : str) -> NDArray[np.float64]:
        """
        Get joint position in world frame.

        Args:
            joint_name (str): The name of the frame.

        Returns:
            NDArray[np.float64]: The position of the joint in the world frame.
        """
        joint_position = np.empty((4, 3), np.float64)
        if self.model.existJointName(joint_name):
            joint_id = self.joint_name2id[joint_name]
            joint_position = self.data.oMi[joint_id].translation
        return joint_position
        
    def update(self, q: NDArray[np.float64], v: NDArray[np.float64] = None) -> None:
        """
        Update pinocchio data with new state.

        Args:
            q (NDArray[np.float64]): Joint configuration.
            v (NDArray[np.float64], optional): Joint velocities. Defaults to zero velocities.
        """
        if v is None:
            v = np.zeros(self.model.nv, dtype=np.float64)

        self.q, self.v = q, v
        pin.framesForwardKinematics(self.model, self.data, q)

        self.contact_updated = False
    
    def get_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return state in (x, y, z, qx, qy, qz, qw) format.
        
        Returns:
            q (NDArray[np.float64]): Joint position
            v (NDArray[np.float64], optional): Joint velocities
        """
        return self.q, self.v
    
    def set_pin_rotor_params(self,
                             rotor_inertia: float = 0.,
                             gear_ratio: float = 1.) -> None:
        """
        Set Pinocchio rotor parameters for all the actuators.

        Args:
            rotor_inertia (float): Rotor intertia (kg.m^2)
            gear_ratio (float): Gear ratio
        """
        offset = 6 if self.floating_base else 0
        self.model.rotorInertia[offset:] = rotor_inertia
        self.model.rotorGearRatio[offset:] = gear_ratio
        
    def set_pin_limits(self,
                       torque_limit: float = 100.,
                       vel_limits: float = 100.) -> None:
        """
        Set Pinocchio limits for all the actuators.

        Args:
            torque_limit (float): torque limit (N.m)
            vel_limits (float): velocity limit (rad.s-1)
        """
        offset = 6 if self.floating_base else 0
        self.model.effortLimit[offset:] = torque_limit
        self.model.velocityLimit[offset:] = vel_limits
        
    def update_contacts(self) -> bool:
        """
        Check for collisions in the geometric model.
        """
        if self.geom_model is None:
            raise ValueError("Geometric model not loaded. Cannot check for collisions.")
        
        if not self.contact_updated:
            self.contacts = []
            pin.computeCollisions(self.robot_model.model, self.data, self.collision_model, self.collision_data)

            for collision_pair, result in zip(self.collision_model.geometryObjects, self.collision_data.activeCollisionPairs):
                if result:
                    geom1_id = collision_pair.first
                    geom2_id = collision_pair.second
                    # geom1_name = self.collision_model.geometryObjects[geom1_id].name
                    # geom2_name = self.collision_model.geometryObjects[geom2_id].name
                    self.contacts.append((geom1_id, geom2_id))
                    
            self.contact_updated = True
            
    def info(self) -> None:
        print("--- Description file:")
        print(self.path_urdf)
        super().info()