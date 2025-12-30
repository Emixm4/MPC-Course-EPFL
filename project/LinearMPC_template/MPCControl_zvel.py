import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # Call parent setup (inherited from base class)
        super()._setup_controller()

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # Call parent get_u (inherited from base class)
        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    def _get_Q(self):
        # [vz] - we want vz -> 0
        return np.array([[100.0]])
    
    def _get_R(self):
        return np.array([[1.0]])
    
    def _get_u_constraints(self):
        # Input constraints: 40% <= Pavg <= 80% for all N steps
        return [
            self.u_var <= 80.0,
            self.u_var >= 40.0
        ]