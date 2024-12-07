# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import copy
import numpy as np
import mujoco
import gc

from mujoco import MjSpec
import os
from typing import Tuple
from numpy.typing import NDArray

from utils.mj_pin_wrapper.abstract.robot import AbstractQuadRobotWrapper

######################################################################
#####
#####                   MJQuadRobotWrapper
#####
######################################################################


class MJQuadRobotWrapper(AbstractQuadRobotWrapper):
    """
    MuJoCo quadruped robot wrapper.
    """

    # Constants
    PIN_2_MJ_POS = [0, 1, 2, 6, 3, 4, 5]
    MJ_2_PIN_POS = [0, 1, 2, 4, 5, 6, 3]
    # Default optionals
    DEFAULT_ROTOR_INERTIA = 0.0
    DEFAULT_JOINT_DAMPING = 0.2
    DEFAULT_FRICTION_LOSS = 0.01
    # Noise for randomize state
    POSITION_NOISE = 7e-3
    ORIENTATION_NOISE = 1e-2
    VELOCITIES_NOISE = 2e-2
    ANG_VELOCITIES_NOISE = 2e-2
    JOINT_POSITION_NOISE = 2e-3
    JOINT_VELOCITIES_NOISE = 5e-3

    def __init__(
        self,
        path_xml_mj: str,
        q0: NDArray[np.float64] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            path_xml_mj (str): Path of .xml model file or xml string.
            q0 (NDArray[np.float64], optional): Initial configuration.
        """
        self.path_xml_mj = path_xml_mj

        # Optional args
        optional_args = {
            "rotor_inertia": MJQuadRobotWrapper.DEFAULT_ROTOR_INERTIA,
            "joint_damping": MJQuadRobotWrapper.DEFAULT_JOINT_DAMPING,
            "friction_loss": MJQuadRobotWrapper.DEFAULT_FRICTION_LOSS,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items():
            setattr(self, k, v)

        # https://mujoco.readthedocs.io/en/stable/python.html#pymodeledit
        self.spec = self._init_mj_spec(path_xml_mj)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        # Set pin to mj state indices
        J_ID = list(range(7, self.model.nq))
        self.nv = self.model.nv
        self.xyzw2wxyz_id = MJQuadRobotWrapper.PIN_2_MJ_POS + J_ID
        self.wxyz2xyzw_id = MJQuadRobotWrapper.MJ_2_PIN_POS + J_ID

        # Set robot to initial configuration (if defined).
        # In (x, y, z, qx, qy, qz, qw) format.
        if q0 is None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.q0 = self.get_state()[0]
        else:
            self.q0 = q0
        mujoco.mj_forward(self.model, self.data)
        super().__init__()

        # Set model damping and friction loss
        self.model.dof_damping[6:] = self.joint_damping
        self.model.dof_frictionloss[6:] = self.friction_loss
        self.model.dof_armature[6:] = self.rotor_inertia
        # For torque control
        self.model.actuator_gainprm[:, 0] = 1
        self.model.actuator_biasprm = 0.0
        self.model.actuator_dynprm = 0.0

        # Set robot configuration dimensions
        # configuration vector
        self.nq = self.model.nq
        # velocity vector space
        self.nv = self.model.nv
        # Number of actuators
        self.nu = self.model.nu
        # Number of end effectors
        self.ne = len(self.eeff_idx)

    def _init_feet_frames(self):
        """
        Init feet frames as the lowest frames of the model in world frame.
        Add the corresponding frame names to the frame name to id map.
        """
        self.update(np.zeros(19))

        frame_positions = self.get_frames_position_world(self.frame_names)

        # frames with minimal z and contact type > 0
        frame_positions_cnt = np.array(
            [
                pos
                for (i, pos) in enumerate(frame_positions)
                if self.model.geom_contype[i] > 0
            ]
        )

        min_z = np.min(frame_positions_cnt[:, -1], axis=0)
        id_min_z = np.where(frame_positions[:, -1] == min_z)[0]

        # Filter frames at the same location
        _, id_unique = np.unique(
            frame_positions[id_min_z, :], return_index=True, axis=0
        )
        id_min_z_unique = id_min_z[id_unique]

        # Order FL, FR, RL, RR
        ordered_id = self._order_positions(frame_positions[id_min_z_unique, :])
        self.eeff_idx = id_min_z_unique[ordered_id].tolist()

        # Add frames to the map
        for name, id in zip(AbstractQuadRobotWrapper.FOOT_NAMES, self.eeff_idx):
            self.frame_name2id[name] = id
            self.eeff_name2id[name] = id

    def _init_mj_spec(self, mj_model_str: str) -> MjSpec:
        """
        Init MjSpec for model editing.

        Args:
            mj_model_str (str): File path or xml string.
        """
        spec = mujoco.MjSpec()
        # Init MuJoCo model and data
        if os.path.splitext(mj_model_str)[-1] == ".xml" and os.path.exists(
            mj_model_str
        ):
            # From xml file
            spec.from_file(self.pathmj_model_str_xml_mj)
        else:  # or from string
            spec.from_string(mj_model_str)

        return spec

    def get_mj_spec(self) -> MjSpec:
        """
        Returns a copy of the initial robot mjSpec.
        """
        spec = self._init_mj_spec(self.path_xml_mj)
        return spec

    def recompile_model(self, new_spec: MjSpec) -> None:
        """
        Recompile model with new spec.
        """
        old_path_xml_mj = self.path_xml_mj
        new_spec.compile()
        self.__init__(new_spec.to_xml())
        self.path_xml_mj = old_path_xml_mj

    def _init_frame_map(self) -> dict:

        # Loop through all geometries in the model
        frame_name_to_id = {str(i): i for i in range(self.model.ngeom)}
        return frame_name_to_id

    def _init_static_geom_id(self) -> list[int]:
        """
        Returns the id of all static geometries of the model.
        Usefull for collision detection.

        Returns:
            list[int]: List of all static geometries indices.
        """
        # List to store the IDs of static geometries
        static_geoms = []

        # Loop through all geometries in the model
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            # Check if the geometry's body ID is 0 (world body)
            if body_id == 0:
                # Get the name of the geometry (if it has one)
                static_geoms.append(geom_id)

        return static_geoms

    def _init_eeff_map(self) -> dict[str, int]:
        """
        Init end effector body name to id map.
        """
        # Body names
        body_names = [self.model.body(i).name for i in range(self.model.nbody)]

        # Parent body names
        body_parent_names = [
            self.model.body(self.model.body(i).parentid).name
            for i in range(self.model.nbody)
        ]

        # Body end effectors (that have no children)
        body_eeff_names = list(set(body_names) - set(body_parent_names))

        body_eeff_name2id = {
            eeff_name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, eeff_name
            )
            for eeff_name in body_eeff_names
        }

        return body_eeff_name2id

    def _init_actuator_map(self) -> dict[str, int]:
        """
        Init actuator name to id map.
        """
        joint_name2act_id = {
            self.model.joint(self.model.actuator(i).trnid[0])  # Joint id
            .name: self.model.actuator(i)
            .id
            for i in range(self.model.nu)
        }
        return joint_name2act_id

    def _init_joint_map(self) -> dict[str, int]:
        """
        Init joint name to id map.
        """
        mj_joint_name2id = {
            self.model.joint(i).name: self.model.joint(i).id
            for i in range(self.model.njnt)
            # Only 1 DoF joints (no root joint)
            if len(self.model.joint(i).qpos0) == 1
        }
        return mj_joint_name2id

    def get_frame_position_world(self, frame_name: str) -> np.ndarray:
        if frame_name in self.frame_name2id.keys():
            geom_id = self.frame_name2id[frame_name]
            xpos = self.data.geom_xpos[geom_id]
            return xpos
        else:
            raise ValueError(f"Frame {frame_name} not found in the model.")

    def get_joint_position_world(self, joint_name: str) -> np.ndarray:
        """
        Get the position of a body in the world frame.

        Args:
            body_name (str): The name of the body.

        Returns:
            np.ndarray: The position of the body in the world frame.
        """
        if joint_name in self.joint_names:
            joint_id = self.joint_name2id[joint_name]
            body_id = self.model.joint(joint_id).bodyid
            xpos = self.data.xpos[body_id][0]
            return xpos
        else:
            raise ValueError(f"Frame {joint_name} not found in the model.")

    def send_joint_torques(self, joint_torque_map: dict[str, float]) -> None:
        """
        Send joint torques to the robot in simulation.

        Args:
            joint_torque_map (dict): dict {joint_name : torque value}
        """
        torque_ctrl = np.zeros((self.nu,), dtype=np.float64)
        for joint_name, torque_value in joint_torque_map.items():
            torque_ctrl[self.joint_name2act_id[joint_name]] = torque_value

        self.data.ctrl = torque_ctrl

    def update(
        self, q: NDArray[np.float64] = None, v: NDArray[np.float64] = None
    ) -> None:
        """
        Reset robot state and simulation state.

        Args:
            - q (NDArray[np.float64]): Initial state.
            - v (NDArray[np.float64]): Initial velocities.
        """
        if q is None:
            q = self.q0
        if v is None:
            v = np.zeros(self.model.nv)

        self.data.qpos = self.xyzw2wxyz(q)
        self.data.qvel = v

        mujoco.mj_forward(self.model, self.data)

        self.contact_updated = False

    def reset(self) -> None:
        """
        Reset robot state and simulation state to initial configuration.
        """
        # Reset data
        self.collided = False
        self.data = mujoco.MjData(self.model)
        self.update(self.q0, np.zeros(self.model.nv))

    def reset_randomize(self) -> None:
        """
        Reset robot state and simulation state to
        a randomize pose close to the initial configuration.
        """
        # Reset data
        self.collided = False
        self.data = mujoco.MjData(self.model)

        noise_q, noise_dq = self._get_rand_state_noise()
        q = copy.deepcopy(self.q0)
        v = np.zeros(self.model.nv)

        q += noise_q
        q[3:7] /= np.linalg.norm(q[3:7])  # normalize quaternion
        v += noise_dq

        self.update(q, v)

    def _get_rand_state_noise(self):
        noise_q = np.concatenate(
            (
                np.random.randn(
                    2,
                )
                * MJQuadRobotWrapper.POSITION_NOISE,
                np.abs(
                    np.random.randn(
                        1,
                    )
                    * MJQuadRobotWrapper.POSITION_NOISE
                ),  # positive height
                np.random.randn(
                    4,
                )
                * MJQuadRobotWrapper.ORIENTATION_NOISE,
                np.random.randn(self.nu) * MJQuadRobotWrapper.JOINT_POSITION_NOISE,
            )
        )
        noise_dq = np.concatenate(
            (
                np.random.randn(
                    3,
                )
                * MJQuadRobotWrapper.VELOCITIES_NOISE,
                np.random.randn(
                    3,
                )
                * MJQuadRobotWrapper.ANG_VELOCITIES_NOISE,
                np.random.randn(self.nu) * MJQuadRobotWrapper.JOINT_VELOCITIES_NOISE,
            )
        )

        return noise_q, noise_dq

    def get_state(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return MuJoCo state in (x, y, z, qx, qy, qz, qw) format.

        Returns:
            q (NDArray[np.float64]): Joint position
            v (NDArray[np.float64], optional): Joint velocities
        """
        q = self.wxyz2xyzw(self.data.qpos)
        v = self.data.qvel

        return q, v

    def update_contacts(self) -> None:
        """
        Update contact pairs (geom1_id, geom2_id).
        """
        self.contacts = [cnt.geom.tolist() for cnt in self.data.contact]
        self.contact_updated = True

    def get_foot_contact_forces(self) -> dict[str, np.ndarray]:
        """
        Compute and return the contact forces for each foot in world coordinates.

        Returns:
            dict[str, np.ndarray]: Dictionary with contact forces for each foot,
                                in the format {foot_name: contact_force}.
        """
        foot_forces = {}

        # Iterate over each foot to calculate contact forces
        for foot_name in MJQuadRobotWrapper.FOOT_NAMES:
            foot_geom_id = self.eeff_name2id[foot_name]
            # Sum all forces applied to the foot geometry
            force_torque = np.zeros(6)

            for j, contact in enumerate(self.data.contact):
                # Check if the contact involves the foot geometry
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # Add the contact force contribution
                    mujoco.mj_contactForce(self.model, self.data, j, force_torque)

            # Store the total force for the foot
            foot_forces[foot_name] = force_torque[:3]

        return foot_forces

    def xyzw2wxyz(self, q_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Pinocchio to MuJoCo state format.

        Args:
            q (NDArray[np.float64]): State in pin format.

        Returns:
            NDArray[np.float64]: State in mj format.
        """
        q_wxyz = np.take(
            q_xyzw,
            self.xyzw2wxyz_id,
            mode="clip",
        )
        return q_wxyz

    def wxyz2xyzw(self, q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert MuJoCo to Pinocchio state format.

        Args:
            q_mj (NDArray[np.float64]): State in mj format.

        Returns:
            NDArray[np.float64]: State in pin format.
        """
        q_xyzw = np.take(
            q_wxyz,
            self.wxyz2xyzw_id,
            mode="clip",
        )
        return q_xyzw
