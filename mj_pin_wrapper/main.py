# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

class Go2Config:
    # Name of the robot in robot descriptions repo
    name = "go2"
    # Local mesh dir
    mesh_dir = "assets"
    # Rotor ineretia (optional)
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    # Gear ratio (optional)
    gear_ratio = 6.33


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from sim_env.utils import RobotModelLoader
    from mj_pin_robot import MJPinQuadRobotWrapper
    from simulator import Simulator

    ###### Robot model
    
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )

    ###### Simulator

    simulator = Simulator(robot.mj)
    # Run simulation
    SIM_TIME = 10 #s
    simulator.run(
        simulation_time=SIM_TIME,
        viewer=True,
        real_time=True,
        )
    