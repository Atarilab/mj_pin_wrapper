import mujoco
from mujoco import viewer
import time

from .abstract.robot import RobotWrapperAbstract
from .abstract.controller import ControllerAbstract
from .abstract.data_recorder import DataRecorderAbstract

class Simulator(object):
    def __init__(self,
                 robot: RobotWrapperAbstract,
                 controller: ControllerAbstract,
                 data_recorder: DataRecorderAbstract = None,
                 ) -> None:
        
        self.robot = robot
        self.controller = controller
        self.data_recorder = (data_recorder
                              if data_recorder != None
                              else DataRecorderAbstract()
                              )
        self.sim_dt = self.robot.mj_model.opt.timestep
        
        self.mj_model = self.robot.mj_model
                
        self.sim_step = 0
        self.simulation_it_time = []
        
    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        q, v = self.robot.get_pin_state()
        
        # Record data
        self.data_recorder.record(q,
                                  v,
                                  self.robot.mj_data)
        
        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(q,
                                              v,
                                              robot_data = self.robot.mj_data)
        # Apply torques
        self.robot.send_mj_joint_torques(torques)

        # MuJoCo sim step
        self.robot.step()
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
            
    def run(self,
            simulation_time: float = -1.,
            viewer: bool = True,
            **kwarg,
            ) -> None:
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            simulation_time (float, optional): Simulation time in second.
            Unlimited if -1. Defaults to -1.
            viewer (bool, optional): Use viewer. Defaults to True.
        """
        
        real_time = kwarg.get("real_time", viewer)
        verbose = kwarg.get("verbose", True)
        
        if verbose:
            print("--- Simulation start")
        
        self.sim_step = 0
        
        # With viewer
        if viewer:
            with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
                sim_start_time = time.time()
                while (viewer.is_running() and
                       (simulation_time < 0 or
                        self.sim_step < simulation_time * (1 / self.sim_dt))
                       ):
                    self._simulation_step_with_timings(real_time)
                    viewer.sync()

        # No viewer    
        else:
            sim_start_time = time.time()
            while (self.sim_step < simulation_time * (1 / self.sim_dt)):
                self._simulation_step_with_timings(real_time)
        
        if verbose:
            print(f"--- Simulation end")
            sum_step_time = sum(self.simulation_it_time)
            mean_step_time = sum_step_time / len(self.simulation_it_time)
            total_sim_time = time.time() - sim_start_time
            print(f"\tTotal optimization step time {sum_step_time:.2f} s")
            print(f"\tMean simulation step time {mean_step_time*1000:.2f} ms")
            print(f"\tTotal simulation time {total_sim_time:.2f} s")

        # TODO: Record video
      