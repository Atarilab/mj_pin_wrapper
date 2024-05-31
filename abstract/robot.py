# TUM - MIRMI - AIPD
# © Victor DHEDIN, 2024

import numpy as np
import pinocchio as pin
import mujoco
import os
from copy import deepcopy
from typing import Any, Callable, Tuple

######################################################################
#####
#####                   RobotWrapperAbstract      
#####
######################################################################

class RobotWrapperAbstract(object):
    """
    Abstract robot wrapper class that merges MuJoCo robot
    model to Pinocchio robot model.
    The robot is torque controled.
    
    Both URDF and XML description files should correspond to the 
    same physical model with the same description.
    Follow these steps for a custom model:
    https://github.com/machines-in-motion/mujoco_utils/tree/main?tab=readme-ov-file#converting-urdf-to-mujoco-xml-format
    """
    
    # Constants
    PIN_2_MJ_POS = [0,1,2,6,3,4,5]
    MJ_2_PIN_POS = [0,1,2,4,5,6,3]
    MJ_FLOOR_NAME = "floor"
    # Default optionals
    DEFAULT_ROTOR_INERTIA = 0.
    DEFAULT_GEAR_RATIO = 1.
    DEFAULT_JOINT_DAMPING = 0.1
    DEFAULT_FRICTION_LOSS = 0.001
    
    def __init__(self,
                 path_urdf: str,
                 path_xml_mj: str,
                 path_package_dir: str | list[str] | None = None,
                 **kwargs,
                 ) -> None:

        self.path_urdf = path_urdf
        self.path_xml_mj = path_xml_mj
        self.path_package_dir = path_package_dir
        self.collided = False
        
        # Optional args
        optional_args = {
            "rotor_inertia" : RobotWrapperAbstract.DEFAULT_ROTOR_INERTIA,
            "gear_ratio" : RobotWrapperAbstract.DEFAULT_GEAR_RATIO,
            "joint_damping" : RobotWrapperAbstract.DEFAULT_JOINT_DAMPING,
            "friction_loss" : RobotWrapperAbstract.DEFAULT_FRICTION_LOSS,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        # Init MuJoCo model and data
        if os.path.splitext(self.path_xml_mj)[-1] == ".xml": # From xml file
            self.mj_model = mujoco.MjModel.from_xml_path(self.path_xml_mj)
        else: # or from string
            self.mj_model = mujoco.MjModel.from_xml_string(self.path_xml_mj)
        
        # Set model damping and friction loss
        self.mj_model.dof_damping[6:] = RobotWrapperAbstract.DEFAULT_JOINT_DAMPING
        self.mj_model.dof_frictionloss[6:] = RobotWrapperAbstract.DEFAULT_FRICTION_LOSS
        # For torque control
        self.mj_model.actuator_gainprm[:, 0] = 1
        
        self.mj_data = mujoco.MjData(self.mj_model)
        # Set robot to initial configuration (if defined)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        
        # Init pinocchio robot
        self.pin_robot = pin.RobotWrapper.BuildFromURDF(
            filename=self.path_urdf,
            package_dirs=self.path_package_dir,
            root_joint=pin.JointModelFreeFlyer(),
            verbose=False,
            meshLoader=None
        )
        self.pin_model = self.pin_robot.model
        self.pin_data = self.pin_robot.data
        self.set_pin_rotor_params(self.rotor_inertia, self.gear_ratio)
        
        # Robot model parameters from MuJoCo model
        # Number of joints
        self.n_joints = self.mj_model.njnt
        # Number of bodies
        self.n_body = self.mj_model.nbody
        # Number of actuators
        self.n_ctrl = self.mj_model.nu
        
        # MuJoCo end effectors bodies names and indexes 
        self.body_eeff_names = self._get_mj_eeffectors_body()
        self.body_eeff_id = [
            mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                eeff_name
                )
            for eeff_name in self.body_eeff_names
        ]
        
        # MuJoCo enf effectors geometries names and indexes (for contacts)
        self.mj_geom_eeff_names = [
            self.mj_model.geom(i).name
            for i in range(self.mj_model.ngeom)
            if (self.mj_model.geom(i).bodyid in self.body_eeff_id and
                # End effector geom should have a defined name
                self.mj_model.geom(i).name != '')
        ]  
        self.mj_geom_eeff_id = [
            mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                eeff_name
                )
            for eeff_name in self.mj_geom_eeff_names
        ]
        
        # Check that the end effectors have named geometries
        assert  len(self.mj_geom_eeff_names) >=\
                len(self.body_eeff_id),\
                "End effectors geometry not found for all end effectors.\
                 Check that the end effectors geometries have a name defined."
        
        self.n_eeff = len(self.body_eeff_names)
        
        # Check that pinocchio and MuJoCo have the same number of end effectors
        last_child = np.unique([
            lastChild for lastChild in self.pin_data.lastChild
        ]).tolist()

        n_eeff = len(last_child)
        assert n_eeff == self.n_eeff,\
            f"Pinocchio model and MuJoCo models have not \
             the same number of end effectors (pin: {n_eeff}, mj: {self.n_eeff})"

        # Floor MuJoCo id (for contacts)
        self.mj_id_floor = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            RobotWrapperAbstract.MJ_FLOOR_NAME
            )
        # Check that it is defined in the MuJoCo model.
        assert self.mj_id_floor >= 0,\
               "Floor not found in MuJoCo model.\
                Floor geometry should have name 'floor'."

        # MuJoCo joint name to id map
        self.mj_joint_name2id = {
            self.mj_model.joint(i).name : self.mj_model.joint(i).id
            for i in range(self.n_joints)
            # Only 1 DoF joints (no root joint)
            if len(self.mj_model.joint(i).qpos0) == 1
        }
        self.mj_joint_names = list(self.mj_joint_name2id.keys())
        
        # Pinocchio joint name to id map
        pin_n_joints = len(self.pin_model.joints)
        self.pin_joint_name2id = {
            self.pin_model.names[i] : i
            for i in range(pin_n_joints)
            if (# Only 1 DoF joints (no root joint)
                self.pin_model.joints[i].nq == 1 and 
                 # No universe joint or ill-defined joints
                self.pin_model.joints[i].id <= pin_n_joints)
        }
        self.pin_joint_names = list(self.pin_joint_name2id.keys())

        # Check that the joint descriptions are the same
        assert  set(self.pin_joint_names)\
                ==\
                set(self.mj_joint_names),\
                f"MuJoCo and Pinocchio have different joint descriptions\n\
                 pinocchio: {self.pin_joint_name2id}\n\
                 mujoco: {self.mj_joint_name2id}"
        
        # Map from joint name to actuator id in MuJoCo
        # For control
        self.n_act = self.mj_model.nu
        self.mj_joint_name2actuator_id = {
            self.mj_model.joint(
                self.mj_model.actuator(i).trnid[0] # Joint id
            ).name : self.mj_model.actuator(i).id
            for i in range(self.n_act)
        }
        
        # Get all static geometries
        self.static_geoms_id = self._get_all_static_geoms_id()
        
        # Set pin to mj state indices
        self.pin2mjstate_id = RobotWrapperAbstract.PIN_2_MJ_POS + list(range(7, 7 + len(self.pin_joint_names)))
        self.mj2pinstate_id = RobotWrapperAbstract.MJ_2_PIN_POS + list(range(7, 7 + len(self.pin_joint_names)))
        
        self.q0, _ = self.get_pin_state()
        pin.framesForwardKinematics(self.pin_model, self.pin_data, self.q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
    def _get_all_static_geoms_id(self) -> list[int]:
        """
        Returns the id of all static geometries of the model.
        Usefull for collision detection.

        Returns:
            list[int]: List of all static geometries indices.
        """
        # List to store the IDs of static geometries
        static_geoms = []

        # Loop through all geometries in the model
        for geom_id in range(self.mj_model.ngeom):
            # Check if the geometry's body ID is 0 (world body)
            if self.mj_model.geom_bodyid[geom_id] == 0:
                # Get the name of the geometry (if it has one)
                geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                static_geoms.append(geom_id)
                
        return static_geoms
    
    def _get_mj_eeffectors_body(self) ->  list[str]:
        """
        Return name of the end effector bodies as a list.
        
        Returns:
            List[str]: names of the end effector bodies.
        """
        # Body names
        body_names = [
            self.mj_model.body(i).name 
            for i in range(self.n_body)
            ]
        
        # Parent body names
        body_parent_names = [
            self.mj_model.body(
                self.mj_model.body(i).parentid
            ).name
            for i in range(self.n_body)
            ]
        
        # Body end effectors (that have no children)
        body_eeff_names = list(
            set(body_names) - set(body_parent_names)
            )
        
        return body_eeff_names

    def step(self) -> None:
        """
        Mujoco environment step.
        """
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def set_pin_rotor_params(self,
                             rotor_inertia: float = 0.,
                             gear_ratio: float = 1.) -> None:
        """
        Set Pinocchio rotor parameters for the actuators.

        Args:
            rotor_inertia (float): Rotor intertia (kg.m^2)
            gear_ratio (float): Gear ratio
        """
        self.pin_model.rotorInertia[6:] = rotor_inertia
        self.pin_model.rotorGearRatio[6:] = gear_ratio
        self.mj_model.dof_armature[6:] = rotor_inertia
        
    def get_pin_state(self) -> Tuple[np.array, np.array]:
        """
        Return robot state (q, v) in Pinocchio format.
        q = [x, y, z, qx, qy, qz, qw] in world frame
        v = [vx, vy, vz, wx, wy, wz] in base frame

        Returns:
            Tuple[np.array, np.array]: (q, v)
        """
        q_mj = np.take(
            self.mj_data.qpos,
            self.mj2pinstate_id,
            mode="clip",
            )
        v_mj = self.mj_data.qvel
        
        return q_mj, v_mj
    
    def get_mj_state(self) -> Tuple[np.array, np.array]:
        """
        Return robot state (q, v) in MuJoCo format.
        q = [x, y, z, qw, qx, qy, qz] in world frame
        v = [vx, vy, vz, wx, wy, wz] in base frame

        Returns:
            Tuple[np.array, np.array]: (q, v)
        """
        q_mj = self.mj_data.qpos
        v_mj = self.mj_data.qvel
        
        return q_mj, v_mj
    
    def send_mj_joint_torques(self, joint_torque_map:dict) -> None:
        """
        Send joint torques to the robot in simulation.

        Args:
            joint_torque_map (dict): dict {joint_name : torque value}
        """
        torque_ctrl = np.empty((self.n_act,), dtype=np.float32)
        for joint_name, torque_value in joint_torque_map.items():
            torque_ctrl[
                self.mj_joint_name2actuator_id[joint_name]
                ] = torque_value
        
        self.mj_data.ctrl = torque_ctrl
            
    def get_mj_eeff_contact_with_floor(self) -> dict[bool]:
        """
        Computes contacts between end effectors and the floor.
        
        Returns:
            eeff_in_contact_floor (dict): dict {eeff name : contact (as a bool)}
        """

        eeff_in_contact_floor = dict.fromkeys(self.mj_geom_eeff_names, False)

        # Filter contacts
        for cnt in self.mj_data.contact:
            if (cnt.geom[0] in self.static_geoms_id and
                cnt.geom[1] in self.mj_geom_eeff_id):
                eeff_name = self.mj_model.geom(cnt.geom[0]).name
                eeff_in_contact_floor[eeff_name] = True
                
            elif (cnt.geom[1] in self.static_geoms_id and
                cnt.geom[0] in self.mj_geom_eeff_id):
                eeff_name = self.mj_model.geom(cnt.geom[0]).name
                eeff_in_contact_floor[eeff_name] = True
           
        return eeff_in_contact_floor
    
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
       
    def reset(self, q0: Any|np.ndarray = None) -> None:
        """
        Reset robot state and simulation state.

        Args:
            - q0 (np.ndarray): Initial state.
        """
        # Reset mj data
        self.mj_data = mujoco.MjData(self.mj_model)
        
        if not isinstance(q0, np.ndarray) and q0 == None:
            # Set to initial position
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
            q0, _ = self.get_mj_state()
        self.mj_data.qpos = q0
            
        
        # Reset pin data
        q0, _ = self.get_pin_state()
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
    def pin2mj_state(self, q_pin: np.ndarray) -> np.ndarray:
        """
        Convert Pinocchio to MuJoCo state format.

        Args:
            q (np.ndarray): State in pin format.

        Returns:
            np.ndarray: State in mj format.
        """
        q_mj = np.take(
            q_pin,
            self.mj2pinstate_id,
            mode="clip",
            )
        return q_mj
        
    def mj2pin_state(self, q_mj: np.ndarray) -> np.ndarray:
        """
        Convert MuJoCo to Pinocchio state format.

        Args:
            q_mj (np.ndarray): State in mj format.

        Returns:
            np.ndarray: State in pin format.
        """
        q_pin = np.take(
            q_mj,
            self.pin2mjstate_id,
            mode="clip",
            )
        return q_pin

######################################################################
#####
#####                   QuadrupedWrapperAbstract      
#####
######################################################################

class QuadrupedWrapperAbstract(RobotWrapperAbstract):
    HIP_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
    THIGH_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
    CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    BASE_NAME = "base"
    JOINT_NAMES = [BASE_NAME] +\
                  HIP_NAMES +\
                  THIGH_NAMES +\
                  CALF_NAMES
    
    DEFAULT_FOOT_SIZE = 0.015
                      
    def __init__(self,
                 path_urdf: str,
                 path_xml_mj: str,
                 path_mesh_dir: str | None = None,
                 **kwargs,
                 ) -> None:
        super().__init__(path_urdf, path_xml_mj, path_mesh_dir, **kwargs)
        
        # Optional args
        optional_args = {
            "foot_size" : QuadrupedWrapperAbstract.DEFAULT_FOOT_SIZE,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        pin_joint_names_ordered = self._pin_base_hip_thigh_calf_joints_names()
        
        # Map from abstract joint name to pinocchio joint id
        self.joint_name2pin_name = {
            QuadrupedWrapperAbstract.JOINT_NAMES[i]
            :
            pin_joint_names_ordered[i]
            for i in range(len(QuadrupedWrapperAbstract.JOINT_NAMES))
        }
        
        # Feet frame names
        self.pin_feet_frame_name = self._get_pin_lowest_frames()
        
        self.n_pin_frames = len(self.pin_model.frames)

        # Base geometries
        self.mj_base_geom_id = self._get_base_mj_geometry_id()
            
    def get_pin_position_world(self, frame_name_list: list[str]) -> np.array:
        """
        Return frame positions in world frame give a list of
        frame names. If frame is not found, position is nan.
        
        Args:
            frame_name_list (list[str]): list of frame name.

        Returns:
            np.array: positions in world frame
        """
        positions = np.empty((len(frame_name_list), 3))
        
        for i, frame_name in enumerate(frame_name_list):
            id_frame = self.pin_model.getFrameId(frame_name)
            
            if id_frame >= self.n_pin_frames:
                positions[i, :] = np.nan
                continue
            
            positions[i, :] = self.pin_data.oMf[id_frame].translation
        
        return positions
        
    def get_pin_feet_position_world(self) -> np.array:
        """
        Return feet positions in world frame.
        Order is FL, FR, RL, RR.

        Returns:
            np.array: feet positions in world frame
        """
        return self.get_pin_position_world(self.pin_feet_frame_name)
    
    def get_pin_hip_position_world(self) -> np.array:
        """
        Return hip joint positions in world frame.
        Order is FL, FR, RL, RR.

        Returns:
            np.array: feet positions in world frame
        """
        joint_names = [
            self.joint_name2pin_name[name]
            for name in QuadrupedWrapperAbstract.HIP_NAMES
            ]
        return self.get_pin_position_world(joint_names)

    def get_pin_thigh_position_world(self) -> np.array:
        """
        Return thigh joint positions in world frame.
        Order is FL, FR, RL, RR.

        Returns:
            np.array: feet positions in world frame
        """
        joint_names = [
            self.joint_name2pin_name[name]
            for name in QuadrupedWrapperAbstract.THIGH_NAMES
            ]
        return self.get_pin_position_world(joint_names)
    
    def get_pin_calf_position_world(self) -> np.array:
        """
        Return thigh joint positions in world frame.
        Order is FL, FR, RL, RR.

        Returns:
            np.array: feet positions in world frame
        """
        joint_names = [
            self.joint_name2pin_name[name]
            for name in QuadrupedWrapperAbstract.CALF_NAMES
            ]
        return self.get_pin_position_world(joint_names)
    
    def _get_pin_lowest_frames(self) -> list[str]:
        """
        Returns the lowest frames of the pinocchio
        model in the current configuration.
        Can be used to find the frames of the feet!

        Returns:
            list[str]: list of Pinocchio frame name.
        """
        frame_positions = np.empty((len(self.pin_model.frames), 3))
        for i, frame in enumerate(self.pin_model.frames):
            if "universe" in frame.name or "root" in frame.name:
                frame_positions[i, :] = np.inf
                continue
            id = self.pin_model.getFrameId(frame.name)
            frame_positions[i, :] = self.pin_data.oMf[id].translation

        # frames with minimal z
        min_z = np.min(frame_positions[:, -1], axis=0)
        id_min_z = np.where(frame_positions[:, -1] == min_z)[0]

        # Filter frames at the same location
        _, id_unique = np.unique(frame_positions[id_min_z, :], return_index=True, axis=0)
        id_min_z_unique = id_min_z[id_unique]

        # Order FL, FR, RL, RR
        ordered_id_min_z_unique = self._order_from_frame_position(id_min_z_unique.tolist(), use_frames=True)
        
        # Get names of the frames
        pin_eeff_frames_name = [
            self.pin_model.frames[int(id)].name
            for id in ordered_id_min_z_unique
        ]
        
        return pin_eeff_frames_name
    
    def _find_parent_joint_id(self,
                              joint_id_list: list[int] | int
                              ) -> list[int]:
        """
        Find joint id corresponding to the parents of the given joints
        according to pinocchio robot model.

        Args:
            joint_id_list (list[int] | int): joint id

        Returns:
            list[int]: parent joint
        """
        
        if not isinstance(joint_id_list, list):
            try:
                joint_id_list_copy = [int(joint_id_list)]
            except Exception as e:
                print("find_parent_joint_id takes a list or an int as first argument.")
                print(e)
        else:
            joint_id_list_copy = deepcopy(joint_id_list)
            
        parent_joint_id = []

        # Find frames that have parent in given list
        for frame_parent_id in map(
            lambda frame: frame.parent,
            filter(
                lambda frame: frame.parent in joint_id_list_copy,
                self.pin_model.frames[1:]
                )
            ):
            id_list = joint_id_list_copy.index(frame_parent_id)
            joint_id_list_copy.pop(id_list)
            
            parent_id = self.pin_model.parents[frame_parent_id]
            parent_joint_id.append(parent_id)
            
        return parent_joint_id
    
    def _order_from_frame_position(self,
                                   id_list: list[int],
                                   use_frames: bool  = False
                                   ) -> list[int]:
        """
        Order a given joint/frame id list of 4 elements as following:
        0: Front Left
        1: Front Right
        2: Rear Left
        3: Rear Right
        
        if use_frames = True, the given id_list should be indices of frames.

        Args:
            joint_id_list (list[int]): list of joint/frame id in pinocchio
            use_frames (bool): given indices are frame indices (True) otherwise joint indices

        Returns:
            list[int]: ordered list of joint/frame id in pinocchio
        """
        
        assert isinstance(id_list, list),\
            "order_from_frame_position takes a list or an int as first argument."
            
        assert len(id_list) == 4,\
            f"joint_id_list should have 4 elements. {len(id_list)} found."
            
        # Order joints according to their positions: FL, FR, RL, RR
        ordered_joint_id_list = [-1] * len(id_list)
        
        # Get frame position in world frame
        frame_positions = np.empty((len(id_list), 3))
        for i, joint_id in enumerate(id_list):
            if use_frames:
                frame_id = self.pin_model.getFrameId(self.pin_model.frames[joint_id].name)
            else:
                frame_id = self.pin_model.getFrameId(self.pin_model.names[joint_id])
            frame_positions[i, :] = self.pin_data.oMf[frame_id].translation

        # Center frame positions
        frame_positions -= np.mean(frame_positions, axis=0, keepdims=True)

        # Find index in ordered list assuming forward is x > 0 and left is y > 0
        for i, joint_id in enumerate(id_list):
            pos = frame_positions[i, :]
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
            
            ordered_joint_id_list[id_oredered] = joint_id
        
        return ordered_joint_id_list
        
    def _pin_base_hip_thigh_calf_joints_names(self) -> list[str]:
        """
        Finds base, hip, thigh, calf joints name from the pinocchio model data.
        
        
        Returns:
            list[int]: joint indices as a list.
        """
        
        # Calf joints are child joints
        calf_joint_id = np.unique([
            lastChild for lastChild in self.pin_data.lastChild
        ]).tolist()
        
        # Thigh joints
        thigh_joint_id = self._find_parent_joint_id(calf_joint_id)
        thigh_joint_id = self._order_from_frame_position(thigh_joint_id)
        
        # Hip joints
        hip_joint_id = self._find_parent_joint_id(thigh_joint_id)
        hip_joint_id = self._order_from_frame_position(hip_joint_id)
        
        base_joint_id = self._find_parent_joint_id(hip_joint_id)
        base_joint_id = np.unique(base_joint_id).tolist()
        
        assert len(base_joint_id) == 1,\
            f"Several ({len(base_joint_id)}) candidates found as base joint.\
              Check URDF description. {base_joint_id}"
        
        res = base_joint_id + hip_joint_id + thigh_joint_id + calf_joint_id
        res = [self.pin_model.names[id] for id in res]
        return res
    
    def _get_base_mj_geometry_id(self) -> str:
        """
        Return the name of the geometries of the base body, based on the kinematic tree.
        """
        # Parent body id
        body_parent_id = [
            self.mj_model.body(i).parentid[0]
            for i in range(self.n_body)
            ]
        
        # The base should be parent of at least 4 bodies
        count_dict = dict.fromkeys(body_parent_id, 0)
        body_base_id = -1
        for id in body_parent_id:
            count_dict[id] += 1
        
            if count_dict[id] == 4:
                body_base_id = id
                break
        
        assert body_base_id != -1, "Base body name not found in MuJoCo model."

        # Find the geometries of that body
        base_geom_id = []
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[geom_id] == body_base_id:
                # Get the name of the geometry
                base_geom_id.append(geom_id)

        assert len(base_geom_id) > 0, f"No geometries found for body {body_base_id}"

        return base_geom_id

    def is_collision(self,
                     exclude_end_effectors: bool = True,
                     only_base: bool = False) -> bool:
        """
        Return True if some robot geometries are in contact
        with the environment.
        
        Args:
            - exclude_end_effectors (bool): exclude contacts of the end-effectors.
            - only_base (bool): check only the contacts with the base.
        """
        is_collision, self.collided = False, False

        if only_base:
            # True if one of the end effector geometries is in contact with
            # the floor
            is_contact_base_floor = lambda contact : (
                contact.geom[0] == self.mj_id_floor and
                contact.geom[1] in self.mj_base_geom_id
                )

            # Filter contacts
            if next(filter(is_contact_base_floor, self.mj_data.contact), None):
                is_collision, self.collided = True, True
            return is_collision

        n_eeff_contact = 0
        if exclude_end_effectors:
            eeff_contact = self.get_mj_eeff_contact_with_floor()
            n_eeff_contact = sum([int(contact) for contact in eeff_contact.values()])
            n_contact = len(self.mj_data.contact)
            
        if n_eeff_contact != n_contact:
            is_collision, self.collided = True, True
        
        return is_collision