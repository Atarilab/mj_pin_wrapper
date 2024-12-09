# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from datetime import datetime
import os
import time
from typing import Any, Callable

import cv2
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
import mujoco
from mujoco import viewer
import numpy as np


class Simulator(object):
    DEFAULT_SIM_DT = 1.0e-3  # s
    DEFAULT_N_BODY_FEXT = 4
    DEFAULT_START_FEXT = 0.5  # s

    def __init__(
        self,
        robot: MJQuadRobotWrapper,
        controller: ControllerAbstract = None,
        data_recorder: DataRecorderAbstract = None,
        sim_dt: float = 1.0e-3,
    ) -> None:

        self.robot = robot
        self.controller = controller if controller != None else ControllerAbstract(robot)
        self.data_recorder = data_recorder if data_recorder != None else DataRecorderAbstract()

        self.sim_dt = sim_dt
        self.robot.model.opt.timestep = sim_dt

        self._reset()
        self.visual_callback_fn = None
        self.use_viewer = False

    def _reset(self):
        """
        Reset flags and timings.
        """
        self.sim_step = 0
        self.frame_count = 0
        self.simulation_it_time = []
        self.verbose = False
        self.stop_sim = False
        self.external_force_active = False
        self.external_force_duration = 0
        self.external_force_time_remaining = 0
        self.external_force_intensity = 0.0
        self.external_force_period = 0.0
        self.perturbs = None
        self.body_ids = None

    def _record_data(self) -> None:
        """
        Call the data recorder.
        To be inherited.
        """
        self.data_recorder.record(self.q, self.v, mj_data=self.robot.data)

    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        self.q, self.v = self.robot.get_state()

        # Apply external force if active
        self._apply_external_force()

        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(self.q, self.v, robot_data=self.robot.data)

        # Record data if no perturbation applied
        self._record_data()

        # Apply torques
        self.robot.send_joint_torques(torques)

        # Simulation step
        mujoco.mj_step(self.robot.model, self.robot.data)
        self.robot.reset_contacts()
        self.sim_step += 1

    def _simulation_step_with_timings(
        self,
        real_time: bool,
    ) -> None:
        """
        Simulation step with time keeping and timings measurements.
        """

        step_start = time.time()
        self._simulation_step()
        step_duration = time.time() - step_start

        self.simulation_it_time.append(step_duration)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.sim_dt - step_duration
        if real_time and time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def _stop_sim(self) -> bool:
        """
        True if the simulation has to be stopped.

        Returns:
            bool: stop simulation
        """
        if self.stop_on_collision and (self.robot.collided or self.robot.is_collision()):
            if self.verbose:
                print("/!\ Robot collision")
            return True

        if self.stop_sim:
            if self.verbose:
                print("/!\ Simulation stopped")
            return True

        if self.controller.diverged:
            if self.verbose:
                print("/!\ Controller diverged")
            return True

        return False

    def _set_video_params(self, **kwargs):
        """
        Record video of the simulation
        """
        video_save_path = kwargs.get("video_path", "./video/")
        fps = kwargs.get("fps", 30)
        playback_speed = kwargs.get("playback_speed", 1.0)
        # TODO: Bug with 1080p frames
        frame_height = kwargs.get("frame_height", 360)
        frame_width = kwargs.get("frame_width", 640)

        # Video file path
        video_dir, video_file = os.path.split(video_save_path)
        if not video_file:
            now = datetime.now()  # current date and time
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            video_file = f"recording_{date_time}.mp4"
            video_save_path = os.path.join(video_dir, video_file)
        os.makedirs(video_dir, exist_ok=True)

        # Video writer
        renderer = mujoco.Renderer(self.robot.model, height=frame_height, width=frame_width)
        video_writer = cv2.VideoWriter(
            video_save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (renderer.width, renderer.height),
        )

        # Camera settings
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.azimuth, cam.elevation = 1.35, -130, -20
        cam.lookat[0], cam.lookat[1], cam.lookat[2] = 0.0, 0.0, 0.2

        self.frame_count = 0
        self.last_frame_time = 0.0  # Initialize time of the last frame

        return video_writer, renderer, cam

    def _record_frame(self, video_writer, renderer, cam, playback_speed, fps):
        """
        Record a frame based on simulation time and playback speed.
        """
        # Time per frame in simulation, adjusted for playback speed
        frame_time_interval = 1.0 / fps * playback_speed

        # Check if it's time to record the next frame
        current_sim_time = self.robot.data.time
        if current_sim_time - self.last_frame_time >= frame_time_interval:
            renderer.update_scene(self.robot.data, cam)
            image = renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)

            # Update frame time and frame count
            self.last_frame_time = current_sim_time
            self.frame_count += 1

    def _get_random_bodies_random_forces(self, N_perturb: int):
        """
        Sample a random body and a random external force.
        """
        N_perturb = Simulator.DEFAULT_N_BODY_FEXT

        # Set random force direction
        external_force_direction = np.random.randn(N_perturb, 3)
        external_force_direction /= np.linalg.norm(
            external_force_direction, axis=-1, keepdims=True
        )  # Normalize to unit vector
        external_torque_direction = np.random.randn(N_perturb, 3)
        external_torque_direction /= np.linalg.norm(
            external_torque_direction, axis=-1, keepdims=True
        )  # Normalize to unit vector

        # Randomize the force intensity between a range (e.g., [0.5 * intensity, intensity])
        force_intensity = np.random.uniform(0.1, 1.0, size=(N_perturb, 1)) * self.external_force_intensity
        torque_intensity = np.random.uniform(0.1, 1.0, size=(N_perturb, 1)) * self.external_force_intensity / 3.0

        # force and torque vectors
        force = force_intensity * external_force_direction
        force[:, -1] /= 3.0  # Less force on z to avoid robot lift-off
        torque = torque_intensity * external_torque_direction
        perturbs = np.concatenate((force, torque), axis=-1)

        # Pick random bodies
        # Choose a random body to apply the force to
        body_ids = np.random.choice(self.robot.model.nbody, N_perturb)

        return body_ids, perturbs

    def _apply_external_force(self):
        """
        Apply external force to the robot if active.
        """
        if (
            self.external_force_active
            and self.sim_step * self.sim_dt > Simulator.DEFAULT_START_FEXT  # No force applied before 0.5s of simulation
            and self.external_force_time_remaining > 0
        ):

            # Apply force at random timing
            prob_apply_force = self.sim_dt * self.external_force_period
            if np.random.rand() < prob_apply_force or self.external_force_time_remaining < self.external_force_duration:
                if self.perturbs is None:
                    self.body_ids, self.perturbs = self._get_random_bodies_random_forces(Simulator.DEFAULT_N_BODY_FEXT)

                self.robot.data.xfrc_applied[self.body_ids] = self.perturbs

                # Decrease remaining force duration
                self.external_force_time_remaining -= self.sim_dt
        else:
            # Reset external force if duration is over
            if not self.use_viewer:
                self.robot.data.xfrc_applied = 0.0

            self.perturbs = None
            self.body_ids = None
            self.external_force_time_remaining = self.external_force_duration

    def _set_external_force(self, duration: float, intensity: float, period: float):
        """
        Apply a random external force to the robot.

        Args:
        - duration (float): Duration of the external force in seconds.
        - intensity (float): Magnitude of the external force (N).
        """
        self.external_force_active = True
        self.perturbs = None
        self.body_ids = None
        self.external_force_intensity = intensity
        self.external_force_duration = duration
        self.external_force_period = period
        self.external_force_time_remaining = duration

        if self.verbose:
            print(f"Applying external force: Duration = {duration}s, Intensity = {intensity}N, Period = {period}s")

    def run(
        self,
        simulation_time: float = -1.0,
        use_viewer: bool = True,
        real_time: bool = True,
        verbose: bool = True,
        stop_on_collision: bool = False,
        record_video: bool = False,
        video_path: str = "./video/",
        fps: int = 30,
        playback_speed: float = 1.0,
        frame_height: int = 360,
        frame_width: int = 640,
        force_duration: float = 0.0,
        force_period: float = 1.0,
        force_intensity: float = 0.0,
        visual_callback_fn: Callable = None,
    ):
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            - simulation_time (float): Duration of the simulation in seconds.
            Set to -1 for an unlimited simulation (default is -1).
            - use_viewer (bool): Whether to use the MuJoCo viewer. Default is True.
            - real_time (bool): Run simulation in real time (if True). Default is True.
            - verbose (bool): If True, print simulation and timing information. Default is True.
            - stop_on_collision (bool): If True, stop the simulation upon collision. Default is False.
            - record_video (bool): Whether to record video. Default is False.
            - video_path (str): Path where the video will be saved. Default is "./video/".
            - fps (int): Frames per second for video recording. Default is 30.
            - playback_speed (float): Playback speed for the video. Default is 1.0.
            - frame_height (int): Frame height for video recording. Default is 360.
            - frame_width (int): Frame width for video recording. Default is 640.
            - force_duration (float): Duration of the external force in seconds. Default is 0.
            - force_period (float): Apply force every <force_period> on average. Default is 1s.
            - force_intensity (float): Intensity of the external force in Newtons. Default is 0.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
                it creates visual geometries using the mjv_initGeom function.
                See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
                for an example.
        """
        self.verbose = verbose
        self.stop_on_collision = stop_on_collision
        self.visual_callback_fn = visual_callback_fn

        if self.verbose:
            print("-----> Simulation start")

        # Apply external force if requested
        apply_force = force_duration > 0.0 and force_intensity > 0.0
        if apply_force:
            self._set_external_force(force_duration, force_intensity, force_period)

        self.sim_step = 0

        if record_video:
            video_writer, renderer, cam = self._set_video_params(
                video_path=video_path,
                fps=fps,
                playback_speed=playback_speed,
                frame_height=frame_height,
                frame_width=frame_width,
            )

        # With viewer
        if use_viewer:
            with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:

                # Enable wireframe rendering of the entire scene.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

                viewer.sync()
                sim_start_time = time.time()
                while viewer.is_running() and (
                    simulation_time < 0.0 or self.sim_step < simulation_time * (1 / self.sim_dt)
                ):
                    self._simulation_step_with_timings(real_time)
                    self.update_visuals(viewer)
                    viewer.sync()

                    if record_video:
                        self._record_frame(video_writer, renderer, cam, playback_speed, fps)

                    if self._stop_sim():
                        break

        # No viewer
        else:
            sim_start_time = time.time()
            while simulation_time < 0.0 or self.sim_step < simulation_time * (1 / self.sim_dt):
                self._simulation_step_with_timings(real_time)

                if record_video:
                    self._record_frame(video_writer, renderer, cam, playback_speed, fps)

                if self._stop_sim():
                    break

        if self.verbose:
            print(f"-----> Simulation end\n")
            sum_step_time = sum(self.simulation_it_time)
            mean_step_time = sum_step_time / len(self.simulation_it_time)
            total_sim_time = time.time() - sim_start_time
            print(f"--- Total optimization step time: {sum_step_time:.2f} s")
            print(f"--- Mean simulation step time: {mean_step_time*1000:.2f} ms")
            print(f"--- Total simulation time: {total_sim_time:.2f} s")

        # Reset flags
        self._reset()

        if record_video:
            video_writer.release()

    def update_visuals(self, viewer) -> None:
        """
        Update visuals according to visual_callback_fn.

        Args:
            viewer (fn): Running MuJoCo viewer.
        """
        if self.visual_callback_fn != None:
            try:
                self.visual_callback_fn(viewer, self.sim_step, self.q, self.v, self.robot.data)

            except Exception as e:
                if self.verbose:
                    print("Can't update visual geometries.")
                    print(e)

    def vis_trajectory(
        self,
        q_traj: np.ndarray,
        dt_traj: np.ndarray | None = None,
        loop: bool = True,
        record_video: bool = False,
        video_path: str = "./video/",
        fps: int = 30,
        playback_speed: float = 1.0,
        frame_height: int = 360,
        frame_width: int = 640,
    ) -> None:
        """
        Visualize a trajectory in the viewer.

        Args:
            q_traj (np.ndarray): State trajectory ([T, nq])
            dt_traj (np.ndarray): Time intervals ([T, nq])
            loop (bool): Loop the visualization
        """
        if dt_traj is None:
            dt_traj = np.full((len(q_traj)), self.sim_dt)

        assert len(dt_traj) == len(q_traj), "Provided state and time intervals should have the same length."

        if record_video:
            video_writer, renderer, cam = self._set_video_params(
                video_path=video_path,
                fps=fps,
                playback_speed=playback_speed,
                frame_height=frame_height,
                frame_width=frame_width,
            )

        # With viewer
        with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:

            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

            viewer.sync()

            replay = True
            while viewer.is_running() and replay:

                for q, dt in zip(q_traj, dt_traj):
                    self.robot.data.qpos = self.robot.xyzw2wxyz(q)

                    mujoco.mj_kinematics(self.robot.model, self.robot.data)
                    viewer.sync()
                    time.sleep(dt)

                    if record_video:
                        self._record_frame(video_writer, renderer, cam, playback_speed, fps)

                if record_video:
                    video_writer.release()

                replay = loop
                if replay:
                    time.sleep(1)
