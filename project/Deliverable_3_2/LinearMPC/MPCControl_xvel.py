import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    """
    MPC controller for X velocity subsystem.

    Full subsystem states: [wy, beta, vx, x]
    Velocity controller states: [wy, beta, vx] (no x position!)
    Input: [d2]

    Constraints:
    - |beta| <= 10 deg = 0.1745 rad (linearization validity)
    - |d2| <= 15 deg = 0.262 rad (servo limits)
    """

    # State indices: wy=1, beta=4, vx=6 (NO x=9 for velocity controller!)
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
        - Penalize beta most (keep small angles for linearization validity)
        - Penalize vx heavily for velocity tracking
        - Small penalty on wy (it's a derivative term)
        """
        Q = np.diag([1.0,   # wy
                     5.0,   # beta (keep small!)
                     10.0])  # vx (increased for better tracking!)
        R = np.diag([10])  # d2
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for X velocity subsystem.

        States: [wy, beta, vx]
        - |beta| <= 9.9 deg (linearization validity)

        Inputs: [d2]
        - |d2| <= 15 deg (with small margin for numerical stability)
        """
        # State constraints - NO hard constraints (rely on cost function penalties)
        # Hard state constraints cause infeasibility when starting outside bounds
        x_min = np.array([-np.inf,      # wy
                          -0.172788,    # beta >= -9.9 deg = -0.172788 rad
                          -np.inf])      # vx
        x_max = np.array([np.inf,        # wy
                          0.172788,     # beta <= 9.9 deg = 0.172788 rad
                          np.inf])       # vx

        # Input constraints - in delta coordinates with safety margin
        # Absolute: -14.5 deg <= d2 <= 14.5 deg (0.5° margin for numerical stability)
        # Delta: -0.253 - us[d2] <= delta_d2 <= 0.253 - us[d2]
        u_min = np.array([-0.253]) - self.us
        u_max = np.array([0.253]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Output matrix C for X velocity controller.
        Selects vx from state [wy, beta, vx].
        """
        return np.array([[0.0, 0.0, 1.0]])  # Output is vx (3rd state)
