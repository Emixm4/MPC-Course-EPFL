import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    """
    MPC controller for roll angle subsystem.

    States: [wz, gamma]
    Input: [Pdiff]

    Constraints:
    - No constraints on states (can spin freely)
    - -20 <= Pdiff <= 20 (differential throttle limits)
    """

    # State indices: wz=2, gamma=5
    x_ids = np.array([2, 5])
    u_ids = np.array([3])  # Pdiff

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cost matrices for roll controller.

        States: [wz, gamma]
        - wz: angular velocity about z (rad/s)
        - gamma: roll angle (rad)

        Tuning:
        - Penalize gamma (roll angle) to keep rocket upright
        - Small penalty on wz (derivative term)
        - Small R to allow aggressive control
        """
        Q = np.diag([1.0,    # wz
                     10.0])  # gamma (keep upright)
        R = np.diag([0.1])   # Pdiff
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for roll subsystem.

        States: [wz, gamma]
        - No constraints (can spin freely)

        Inputs: [Pdiff]
        - -20 <= Pdiff <= 20 (percentage)
        """
        # State constraints - in delta coordinates (no limits)
        x_min = np.array([-np.inf,   # wz
                          -np.inf])  # gamma
        x_max = np.array([np.inf,    # wz
                          np.inf])   # gamma

        # Input constraints - in delta coordinates
        # Absolute: -20 <= Pdiff <= 20 (percentage)
        # Delta: -20 - us[Pdiff] <= delta_Pdiff <= 20 - us[Pdiff]
        u_min = np.array([-20.0]) - self.us
        u_max = np.array([20.0]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Output matrix C for roll controller.
        Selects gamma (roll angle) from state [wz, gamma].
        """
        return np.array([[0.0, 1.0]])  # Output is gamma (2nd state)
