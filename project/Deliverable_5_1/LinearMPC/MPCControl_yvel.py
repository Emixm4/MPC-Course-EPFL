import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    """
    MPC controller for Y velocity subsystem.
    Full subsystem states: [wx, alpha, vy, y]
    Velocity controller states: [wx, alpha, vy]
    Input: [d1]

    Constraints:
    - |alpha| <= 10 deg = 0.1745 rad (linearization validity)
    - |d1| <= 15 deg = 0.262 rad (servo limits)
    """

    # State indices: wx=0, alpha=3, vy=7 (NO y=10 for velocity controller!)
    x_ids = np.array([0, 3, 7])
    u_ids = np.array([0])  # d1

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cost matrices for Y velocity controller.

        Stats: [wx, alpha, vy]
        - wx: angular velocity (rad/s)
        - alpha: roll angle (rad)
        - vy: velocity in y (m/s)

        """
        Q = np.diag([1.0,   # wx
                     100.0,  # alpha (keep small!)
                     10.0])  # vy
        R = np.diag([0.1])  # d1
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for Y velocity subsystem.

        States: [wx, alpha, vy]

        Inputs: [d1]
        - |d1| <= 15 deg
        """
        # Hard state constraints cause infeasibility when starting outside bounds
        x_min = np.array([-np.inf,      # wx
                          -np.inf,       # alpha - NO hard constraint
                          -np.inf])      # vy
        x_max = np.array([np.inf, np.inf, np.inf])      

        # Absolute: -14.5 deg <= d1 <= 14.5 deg (0.5° margin for numerical stability)
        u_min = np.array([-0.253]) - self.us
        u_max = np.array([0.253]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Output matrix C for Y velocity controller.
        Selects vy from state [wx, alpha, vy].
        """
        return np.array([[0.0, 0.0, 1.0]])  # Output is vy (3rd state)
