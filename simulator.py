# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

import os
from typing import Any, Callable
import cv2
import mujoco
from mujoco import viewer
import time

from datetime import datetime
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract

class Simulator(object):
    DEFAULT_SIM_DT = 1.0e-3 #s
    def __init__(self,
                 robot: MJQuadRobotWrapper,
                 controller: ControllerAbstract = None,
                 data_recorder: DataRecorderAbstract = None,
                 sim_dt : float = 1.0e-3,
                 ) -> None:
        
        self.robot = robot
        self.controller = (controller
                            if controller != None
                            else ControllerAbstract(robot)
                            )
        self.data_recorder = (data_recorder
                              if data_recorder != None
                              else DataRecorderAbstract()
                              )
        
        self.sim_dt = sim_dt
        self.robot.model.opt.timestep = sim_dt

        # Set counters and timings variables
        self._reset()
        self.visual_callback_fn = None
        self.use_viewer = False
        
    def _reset(self) -> None:
        """
        Reset flags and timings.
        """
        self.sim_step = 0
        self.frame_count = 0
        self.simulation_it_time = []
        self.verbose = False
        self.stop_sim = False
        
    def _record_data(self) -> None:
        """
        Call the data recorder.
        To be inherited.
        """
        self.data_recorder.record(self.q,
                                  self.v,
                                  mj_data = self.robot.data)
    
    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        self.q, self.v = self.robot.get_state()

        # Record data
        self._record_data()
        
        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(self.q,
                                              self.v,
                                              robot_data = self.robot.data)
        
        # Apply torques
        self.robot.send_joint_torques(torques)
        
        # Simulation step
        mujoco.mj_step(self.robot.model, self.robot.data)
        self.robot.reset_contacts()
        self.sim_step += 1
        
        # TODO: Add external disturbances
        
    def _simulation_step_with_timings(self,
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
            if self.verbose: print("/!\ Robot collision")
            return True

        if self.stop_sim:
            if self.verbose: print("/!\ Simulation stopped")
            return True
        
        if self.controller.diverged:
            if self.verbose: print("/!\ Controller diverged")
            return True
        
        return False
    
    def _get_video_params(self, **kwargs):
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
            now = datetime.now() # current date and time
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
            (renderer.width, renderer.height)
        )
        
        # Camera settings
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.azimuth, cam.elevation = 1.35, -130, -20
        cam.lookat[0], cam.lookat[1], cam.lookat[2] = 0.0, 0.0, 0.2
        
        self.frame_count = 0
        self.last_frame_time = 0.0  # Initialize time of the last frame
    
        return video_writer, renderer, cam, playback_speed, fps
    
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
        
    def run(self,
            simulation_time: float = -1.,
            use_viewer: bool = True,
            visual_callback_fn: Callable = None,
            **kwargs,
            ) -> None:
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            - simulation_time (float, optional): Simulation time in second.
            Unlimited if -1. Defaults to -1.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
            - viewer (bool, optional): Use viewer. Defaults to True.
            - verbose (bool, optional): Print timing informations.
            - stop_on_collision (bool, optional): Stop the simulation when there is a collision.
        """
        self.verbose = kwargs.get("verbose", True)
        self.stop_on_collision = kwargs.get("stop_on_collision", False)
        real_time = kwargs.get("real_time", use_viewer)
        record_video = kwargs.get("record_video", False)
        self.visual_callback_fn = visual_callback_fn
        
        if self.verbose:
            print("-----> Simulation start")
        
        self.sim_step = 0
        
        if record_video:
            video_writer, renderer, cam, playback_speed, fps = self._get_video_params(**kwargs)
        
        # With viewer
        if use_viewer:
            with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
                
                  # Enable wireframe rendering of the entire scene.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                
                viewer.sync()
                sim_start_time = time.time()
                while (viewer.is_running() and
                       (simulation_time < 0. or
                        self.sim_step < simulation_time * (1 / self.sim_dt))
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
            while (simulation_time < 0. or self.sim_step < simulation_time * (1 / self.sim_dt)):
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