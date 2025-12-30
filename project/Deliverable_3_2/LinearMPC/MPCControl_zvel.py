import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    """
    MPC controller for Z (vertical) velocity subsystem.

    States: [vz, z]
    Input: [Pavg]

    Constraints:
    - z >= 0 (don't go underground!)
    - 40 <= Pavg <= 80 (safety limits on throttle)
    """

    # State indices: vz=8, z=11
    # But we only control velocity, so we only use vz
    x_ids = np.array([8])  # vz only (no position for velocity controller!)
    u_ids = np.array([2])  # Pavg

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cost matrices for Z velocity controller.

        Tuning:
        - Q: Penalize velocity error
        - R: Penalize throttle usage (keep it smooth)
        """
        Q = np.diag([10.0])  # vz cost
        R = np.diag([0.1])   # Pavg cost (small to allow aggressive control)
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for Z subsystem.

        States: [vz]
        - No constraints on vz

        Inputs: [Pavg]
        - 40 <= Pavg <= 80 (percentage)
        """
        # State constraints (vz) - in delta coordinates
        x_min = np.array([-np.inf])  # No lower bound on downward velocity
        x_max = np.array([np.inf])   # No upper bound on upward velocity

        # Input constraints (Pavg) - in delta coordinates
        # Absolute: 40 <= Pavg <= 80
        # Delta: 40 - us[Pavg] <= delta_Pavg <= 80 - us[Pavg]
        u_min = np.array([40.0]) - self.us
        u_max = np.array([80.0]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Output matrix C for Z velocity controller.
        Selects vz from state [vz].
        """
        return np.array([[1.0]])  # Output is vz
