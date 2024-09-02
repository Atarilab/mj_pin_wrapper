# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import numpy as np
import pinocchio as pin
from typing import List, Tuple
from numpy.typing import NDArray

from .utils import transform_points

######
######
###### AbstractRobotWrapper
######
######

class AbstractRobotWrapper(object):
    """
    Abstract robot wrapper class.
    """    
    def __init__(self,
                 **kwargs,
                 ) -> None:

        # joint name to id map
        self.joint_name2id = self._init_joint_map()
        self.joint_names  = list(self.joint_name2id.keys())
        
        # joint name to id map
        self.joint_name2act_id = self._init_actuator_map()
        
        # frame name to id
        self.frame_name2id = self._init_frame_map()
        self.frame_names = list(self.frame_name2id.keys())
        
        # end effectors geom id
        self.eeff_name2id = self._init_eeff_map()
        self.eeff_idx = list(self.eeff_name2id.values())

        # All robot geometries id
        self.geom_idx = self._init_geom_id()
        
        # static geometries id
        self.static_geom_id = self._init_static_geom_id()
        
        # Set robot configuration dimensions
        # configuration vector
        self.nq = 0
        # velocity vector space
        self.nv = 0
        # Number of actuators
        self.nu = 0
        # Number of end effectors
        self.ne = 0
        
        # Flag that checks if contact pairs have already been computed in the current step
        self.collided = False
        self.contact_updated = False
        self.contacts = []
        self.base_geom_id = []
        
    def _init_frame_map(self) -> dict[str, int]:
        """
        Init frame name to id map.
        """
        return {}
    
    def _init_joint_map(self) -> dict[str, int]:
        """
        Init joint name to id map.
        """
        return {}
    
    def _init_actuator_map(self) -> dict[str, id]:
        """
        Init actuator name to joint id.
        """
        return {}
      
    def _init_eeff_map(self) -> dict[str, id]:
        """
        Init end effector name to id map.
        """
        return {}
       
    def _init_geom_id(self) -> List[int]:
        """
        Returns all robot geometries indices.
        """
        return []
    
    def _init_static_geom_id(self) -> List[int]:
        """
        Returns all static geometries of the model.
        """
        return []
    
    def get_frame_position_world(self, frame_name : str) -> NDArray[np.float64]:
        """
        Get frame position in world frame.

        Args:
            frame_name (str): The name of the frame.

        Returns:
            NDArray[np.float64]: The position of the frame in the base frame.
        """
        pass
    
    def get_joint_position_world(self, joint_name : str) -> NDArray[np.float64]:
        """
        Get joint position in world frame.

        Args:
            joint_name (str): The name of the frame.

        Returns:
            NDArray[np.float64]: The position of the joint in the world frame.
        """
        pass
        
    def get_frames_position_world(self, frame_names : List[str]) -> NDArray[np.float64]:
        """
        Get all frames position in world frame.

        Args:
            frame_names (List[str]): List of frame names.

        Returns:
            NDArray[np.float64]: Positions of the frames in the base frame.
        """
        frame_positions = np.array([self.get_frame_position_world(frame_name) for frame_name in frame_names])
        return frame_positions
    
    def send_joint_torques(self, joint_torque_map : dict[str, float]) -> None:
        """
        Send joint torques.

        Args:
            joint_torque_map (dict): dict {joint_name : torque value}
        """
        pass
    
    def get_joints_position_world(self, joint_names : List[str]) -> NDArray[np.float64]:
        """
        Get all joints position in world frame.

        Args:
            joint_names (List[str]): List of joint names.

        Returns:
            NDArray[np.float64]: Positions of the joints in the base frame.
        """
        joint_positions = np.array([self.get_joint_position_world(joint_name) for joint_name in joint_names])
        return joint_positions
        
    def update_contacts(self) -> None:
        """
        Update contact pairs (geom1_id, geom2_id).
        """
        pass
    
    def update(self, q: NDArray[np.float64], v: NDArray[np.float64] = None) -> None:
        """
        Update robot with new configuration.

        Args:
            q (NDArray[np.float64]): Joint configuration.
            v (NDArray[np.float64], optional): Joint velocities. Defaults to zero velocities.
        """
        pass
    
    def is_collision(self,
                     exclude_end_effectors: bool = True,
                     only_base: bool = False,
                     self_collision : bool = False) -> bool:
        """
        Return True if some robot geometries are in contact with the environment.
        
        Args:
            - exclude_end_effectors (bool): exclude contacts of the end-effectors.
            - only_base (bool): check only the contacts with the base.
            - self_collision (bool): check for self-collision.
        """
        if not self.contact_updated: self.update_contacts()
        
        is_collision, self.collided = False, False

        if only_base:
            # True if base in contact with one other geometry
            is_contact_base = lambda cnt_pair : (
                cnt_pair[1] in self.base_geom_id
                or
                cnt_pair[0] in self.base_geom_id
            )

            # Filter contacts
            if next(filter(is_contact_base, self.contacts), None):
                is_collision, self.collided = True, True

        elif exclude_end_effectors:
            # True if a geom, different from an end-effector, is in contact
            is_contact_non_eeff = lambda cnt_pair : (
                (cnt_pair[0] in self.static_geoms_id and
                not cnt_pair[1] in self.eeff_idx)
                or
                (cnt_pair[1] in self.static_geoms_id and
                not cnt_pair[0] in self.eeff_idx)
            )
            
            # Filter contacts
            if next(filter(is_contact_non_eeff, self.contacts), None):
                is_collision, self.collided = True, True
        
        # Check for self collision
        if not is_collision and self_collision:
            # True if self collision
            is_self_collision = lambda cnt_pair : (
                (cnt_pair[0] in self.geom_idx and
                cnt_pair[1] in self.geom_idx)
            )
            
            # Filter contacts
            if next(filter(is_self_collision, self.contacts), None):
                is_collision, self.collided = True, True
        
        return is_collision
    
    def transform_points(self, b_T_W, points_w):
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w.shape[0], 1))
        points_w_homogeneous = np.hstack((points_w, ones))
        # Apply the transformation matrix
        points_b_homogeneous = b_T_W @ points_w_homogeneous.T
        # Convert back to 3D coordinates
        points_b = points_b_homogeneous[:3, :].T
        return points_b
    
    def to_base(self, points_w : NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform a set of points in world frame to current base frame.

        Args:
            NDArray[np.float64]: shape [N, 3]

        Returns:
            NDArray[np.float64]: shape [N, 3]
        """
        q, _ = self.get_state()
        b_T_W = pin.XYZQUATToSE3(q).inverse() # transform from base to world
        points_b = transform_points(b_T_W, points_w)
        return points_b
     
    def to_world(self, points_b : NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform a set of points in current base frame to world frame.

        Args:
            NDArray[np.float64]: shape [N, 3]

        Returns:
            NDArray[np.float64]: shape [N, 3]
        """
        q, _ = self.get_state()
        W_T_b = pin.XYZQUATToSE3(q) # transform from world to base
        points_W = transform_points(W_T_b, points_b)
        return points_W
    
    def get_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return state in (x, y, z, qx, qy, qz, qw) format.
        
        Returns:
            q (NDArray[np.float64]): Joint position
            v (NDArray[np.float64], optional): Joint velocities
        """
        pass
    
    @staticmethod
    def _print_dict(d : dict[str, int]) -> None:
        for k, v in d.items(): print(k, ":", v)
    
    def info(self) -> None:
        """
        Prints robot info.
        """
        print("--- Robot description:")
        print("nq:", self.nq, "| nv:", self.nv, "| nu:", self.nu, "| ne:", self.ne)
        print("--- End-effectors bodies")
        self._print_dict(self.eeff_name2id)
        print("--- Joints")
        self._print_dict(self.joint_name2id)
        print("--- Frames")
        self._print_dict(self.frame_name2id)
        print("--- Actuator to joints")
        self._print_dict(self.joint_name2act_id)

######
######s
###### AbstractQuadRobotWrapper
######
######

class AbstractQuadRobotWrapper(AbstractRobotWrapper):
    """
    Abstract quadruped robot wrapper class.
    Maps robot joint description to default description.
    """
    FOOT_NAMES = ["FL", "FR", "RL", "RR"]
    HIP_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
    THIGH_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
    CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    BASE_NAME = "base"
    JOINT_NAMES = HIP_NAMES +\
                  THIGH_NAMES +\
                  CALF_NAMES
    
    DEFAULT_FOOT_SIZE = 0.015 # m
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.foot_size = kwargs.get("foot_size", AbstractQuadRobotWrapper.DEFAULT_FOOT_SIZE)
        
        # foot geom to id
        self.foot_names = AbstractQuadRobotWrapper.FOOT_NAMES
        
        self._init_feet_frames()
        self._update_joint_frames()
        
    def _is_description_valid(self) -> bool:
        """
        Check that quadruped description is valid.
        """
        if (self.nq == 19 and
            self.nv == 18 and
            self.nu == 12 and
            self.ne == 4):
            return True
        
        return False
    
    def _init_feet_frames(self):
        """
        Init feet frames as the lowest frames of the model in world frame.
        Add the corresponding frame names to the frame name to id map.
        """
        self.update(np.zeros(19))

        frame_positions = self.get_frames_position_world(self.frame_names)

        # frames with minimal z
        min_z = np.min(frame_positions[:, -1], axis=0)
        id_min_z = np.where(frame_positions[:, -1] == min_z)[0]

        # Filter frames at the same location
        _, id_unique = np.unique(frame_positions[id_min_z, :], return_index=True, axis=0)
        id_min_z_unique = id_min_z[id_unique]

        # Order FL, FR, RL, RR
        ordered_id = self._order_positions(frame_positions[id_min_z_unique, :])
        self.eeff_idx = id_min_z_unique[ordered_id].tolist()
        
        # Add frames to the map
        for name, id in zip(AbstractQuadRobotWrapper.FOOT_NAMES, self.eeff_idx):
            self.frame_name2id[name] = id

    def _update_joint_frames(self):
        """
        Update the joint frame names according to their placement.
        The new joint names will be:
        HIP_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        THIGH_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        """
        self.update(np.zeros(19))

        joint_positions = self.get_joints_position_world(self.joint_names)
        id_sorted_by_height = np.argsort(np.linalg.norm(joint_positions, axis=-1))

        # Get indices
        hip_id, thigh_id, calf_id = np.split(id_sorted_by_height, 3)
        hip_id_oredered = self._order_positions(joint_positions[hip_id])
        thigh_id_ordered = self._order_positions(joint_positions[thigh_id])
        calf_id_oredered = self._order_positions(joint_positions[calf_id])
        
        hip_id = hip_id[hip_id_oredered].tolist()
        thigh_id = thigh_id[thigh_id_ordered].tolist()
        calf_id = calf_id[calf_id_oredered].tolist()

        # Update joint map
        new_joint_name2id = {}
        new_joint_name2act_id = {}
        new_joint_name = AbstractQuadRobotWrapper.JOINT_NAMES
        new_joint_id = hip_id + thigh_id + calf_id

        for name, id_name in zip(new_joint_name, new_joint_id):
            old_name = self.joint_names[id_name]
            id = self.joint_name2id[old_name]
            new_joint_name2id[name] = id
            new_joint_name2act_id[name] = self.joint_name2act_id[old_name]
            self.joint_names = list(map(lambda x: x.replace(old_name, name), self.joint_names))

        self.joint_name2act_id = new_joint_name2act_id
        self.joint_name2id = new_joint_name2id
        
        
    def foot_contacts(self) -> dict[str, int]:
        """
        Returns foot contacts with static geometries as a dict.

        Returns:
            dict[str, int]: {foot name : geom in contact}
        """
        if not self.contact_updated: self.update_contacts()

        foot_contacts = dict.fromkeys(self.foot_names, -1)

        # Filter contacts
        for cnt_pair in self.contacts:
            if (cnt_pair[0] in self.static_geoms_id and
                cnt_pair[1] in self.mj_geom_eeff_id):
                eeff_name = self.mj_model.geom(cnt_pair[1]).name
                foot_contacts[eeff_name] = cnt_pair[0]
                
            elif (cnt_pair[1] in self.static_geoms_id and
                cnt_pair[0] in self.mj_geom_eeff_id):
                eeff_name = self.mj_model.geom(cnt_pair[0]).name
                foot_contacts[eeff_name] = cnt_pair[1]
         
        return foot_contacts
    
    def get_foot_pos_world(self) -> NDArray[np.float64]:
        """
        Returns foot positions in world frame.
        [FL, FR, RL, RR]
        
        Returns:
            NDArray[np.float64]: shape [4, 3].
        """
        foot_pos_w = self.get_frames_position_world(AbstractQuadRobotWrapper.FOOT_NAMES)
        return foot_pos_w
        
    def get_foot_pos_base(self) -> NDArray[np.float64]:
        """
        Returns foot positions in base frame.
        [FL, FR, RL, RR]
        
        Returns:
            NDArray[np.float64]: shape [4, 3].
        """
        foot_pos_w = self.get_foot_pos_world()
        foot_pos_b = self.to_base(foot_pos_w)
        return foot_pos_b
     
    def get_hip_pos_world(self) -> NDArray[np.float64]:
        """
        Returns hip positions in world frame.
        [FL, FR, RL, RR]
        
        Returns:
            NDArray[np.float64]: shape [4, 3].
        """
        pos_w = self.get_frames_position_world(AbstractQuadRobotWrapper.HIP_NAMES)
        return pos_w
         
    def get_thigh_pos_world(self) -> NDArray[np.float64]:
        """
        Returns thigh positions in world frame.
        [FL, FR, RL, RR]
        
        Returns:
            NDArray[np.float64]: shape [4, 3].
        """
        pos_w = self.get_frames_position_world(AbstractQuadRobotWrapper.THIGH_NAMES)
        return pos_w
     
    def get_calf_pos_world(self) -> NDArray[np.float64]:
        """
        Returns calf positions in world frame.
        [FL, FR, RL, RR]
        
        Returns:
            NDArray[np.float64]: shape [4, 3].
        """
        pos_w = self.get_frames_position_world(AbstractQuadRobotWrapper.CALF_NAMES)
        return pos_w
    
    def _order_positions(self, pos_w : NDArray[np.float64]) -> List[int]:
        """
        Order a set of 4 3D positions as follows : [FL, FR, RL, RR]

        Args:
            pos_w (NDArray[np.float64]): positions to order

        Returns:
            List[int]: indices to order the positions
        """
        # Center frame positions
        ordered_indices = [0] * len(pos_w)
        pos_w -= np.mean(pos_w, axis=0, keepdims=True)
        # Find index in ordered list assuming forward is x > 0 and left is y > 0
        for i, pos in enumerate(pos_w):
            id_oredered = 0
            # FL
            if pos[0] > 0 and pos[1] > 0:
                id_oredered = 0
            # FR
            elif pos[0] > 0 and pos[1] < 0:
                id_oredered = 1
            # RL
            elif pos[0] < 0 and pos[1] > 0:
                id_oredered = 2
            # RR
            elif pos[0] < 0 and pos[1] < 0:
                id_oredered = 3
            
            ordered_indices[id_oredered] = i
        
        return ordered_indices
    
    def info(self) -> None:
        super().info()
        print("--- foot geometries")
        foot_dict = {k : v for k, v in self.frame_name2id.items() if k in self.foot_names}
        self._print_dict(foot_dict)