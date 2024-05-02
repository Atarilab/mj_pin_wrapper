# MuJoCo Pinocchio Wrapper

[MuJoCo](https://mujoco.org/) + [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/) robot description interface and associated tools for simulation.

---

## Requirements

- Python 3.10
- [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/)
- [MuJoCo](https://mujoco.org/)
- [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py/tree/main)

```bash
python3 -m pip install --upgrade pip pinocchio mujoco robot_descriptions
```

## Wrapper interface

This repo provides a wrapper to interface [MuJoCo](https://mujoco.org/) for the simulation environment and [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/) library for efficient kino-dyamic computations.

#### Pinocchio + MuJoCo interface

Defined in the [`RobotWrapperAbstract`](abstract/robot.py) class. This class provides tools to interface model descriptions from both MuJoCo and Pinocchio. It also checks that both descriptions are similar.

Paths of `urdf` and `mjcf` (as `xml`) description files needs to be provided. One can use [`RobotModelLoader`](env/utils.py) to get the paths automatically given the robot name (now only working with `go2` robot). The model will be loaded in the custom environment defined in [`scene.xml`](sim_env/utils.py).

- `get_pin_state` and `get_mj_state` return the state of robot respectively in Pinocchio and MuJoCo format (different quaternion representation). One can also access to the joint names or end effector names.

- `get_j_eeff_contact_with_floor` returns which end effectors is in contact with the floor in the simulation as a map {end effector name : True/False}


[`QuadrupedWrapperAbstract`](abstract/robot.py) inherits from [`RobotWrapperAbstract`](abstract/robot.py) and provide additional tools for quadruped robots by abstracting the robot description.

- `get_pin_feet_position_world` returns the feet positions in world frame (same for `hip`, `thigh` and `calf`).





#### Simulator

For custom usage of the wrapper, two abstract classes used in the [`Simulator`](simulator.py) need to be inherited.


1. [`ControllerAbstract`](abstract/controller.py)
Implements the **robot controller or policy**. The following method should be inherited. 
    - `get_torques(q, v, robot_data)`
    This method is called every simulation step by the simulator.
    It takes as input the robot position state `q: np.array`, the robot velocity state `v: np.array` and the simulation data `robot_data: MjData`.
    It should return  the torques for each actuator as a map {joint name : $\tau$}. One can find the joint names the `RobotWrapperAbstract`.

    See [`BiConMPC`](mpc_controller/bicon_mpc.py) for exemple of inheritance.


2. [`DataRecorderAbstract`](abstract/controller.py)
Implements a class to **record the data from the simulation**. The following method should be inherited.
    - `record(q, v, robot_data)`
    This method is called every simulation step by the simulator.
    It takes as input the robot position state `q: np.array`, the robot velocity state `v: np.array` and the simulation data `robot_data: MjData`.

One can run the simulation with or without viewer using the `run` method.
