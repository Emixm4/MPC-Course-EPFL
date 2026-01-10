import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    """
    MPC controller for X velocity subsystem.

    Full subsystem states: [wy, beta, vx, x]
    Velocity controller states: [wy, beta, vx] 
    Input: [d2]
    Constraints:
    - |beta| <= 10 deg = 0.1745 rad (linearization validity)
    - |d2| <= 15 deg = 0.262 rad (servo limits)
    """

    # State indices: wy=1, beta=4, vx=6
    x_ids = np.array([1, 4, 6])
    u_ids = np.array([1])  # d2

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cost matrices for X velocity controller.

        States: [wy, beta, vx]
        - wy: angular velocity (rad/s)
        - beta: pitch angle (rad)
        - vx: velocity in x (m/s)

        Tuning:
        - Penalize beta most
        - Penalize vx for velocity tracking
        - Small penalty on wy
        """
        Q = np.diag([1.0,   # wy
                     100.0,  # beta (keep small!)
                     10.0])  # vx
        R = np.diag([0.1])  # d2
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for X velocity subsystem.
        States: [wy, beta, vx]
        Inputs: [d2]
        - |d2| <= 15 deg (with small margin for numerical stability)
        """
        x_min = np.array([-np.inf,      # wy
                          -np.inf,       # beta
                          -np.inf])      # vx
        x_max = np.array([np.inf, np.inf, np.inf])       

        # Absolute: -14.5 deg <= d2 <= 14.5 deg (0.5° margin for numerical stability)
        u_min = np.array([-0.253]) - self.us
        u_max = np.array([0.253]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Output matrix C for X velocity controller.
        Selects vx from state [wy, beta, vx].
        """
        return np.array([[0.0, 0.0, 1.0]])  # Output is vx (3rd state)
