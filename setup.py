from setuptools import setup, find_packages

setup(
    name='mj_pin_wrapper',
    version='0.1',
    packages=find_packages(),  # Automatically finds all packages and subpackages
    install_requires=[
        "pinocchio",
        "mujoco",
        "robot_descriptions"
    ],  # List dependencies here
)