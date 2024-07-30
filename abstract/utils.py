# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any
import pinocchio as pin
import numpy as np
from numpy.typing import NDArray

def transform_points(
    B_T_A : NDArray[np.float64],
    points_A : NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """
    Transform a set of points from frame A to frame B. 

    Args:
        A_T_B (NDArray[np.float64]): SE3 transform of frame A expressed in frame B.
        points_A (NDArray[np.float64]): set of points expressed in frame A. Shape [N, 3]

    Returns:
        points_B (NDArray[np.float64]): set of points expressed in frame B. Shape [N, 3]
    """
    
    if len(points_A.shape) < 2:
        points_A = points_A[np.newaxis, :]
        
    assert points_A.shape[-1] == 3, "Points provided are not 3d points."     
        
    # Add a fourth homogeneous coordinate (1) to each point
    ones = np.ones((points_A.shape[0], 1))
    points_A_homogeneous = np.hstack((points_A, ones))
    # Apply the transformation matrix
    points_B_homogeneous = B_T_A @ points_A_homogeneous.T
    # Convert back to 3D coordinates
    points_B = points_B_homogeneous[:3, :].T
    
    return points_B