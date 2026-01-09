import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

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
        # [wy, beta, vx] - we want vx -> 0
        return np.diag([1.0, 10.0, 100.0])
    
    def _get_R(self):
        return np.array([[1.0]])
    
    def _get_u_constraints(self):
        # Input constraints: -15° <= dP <= 15° for all N steps
        u_min = np.deg2rad(-15)
        u_max = np.deg2rad(15)
        return [
            self.u_var <= u_max,
            self.u_var >= u_min
        ]
    
    def _get_x_constraints(self):
        # State constraints: -10° <= beta <= 10° for steps 0 to N-1
        # beta is state index 1
        beta_min = np.deg2rad(-10)
        beta_max = np.deg2rad(10)
        return [
            self.x_var[1, :-1] <= beta_max,
            self.x_var[1, :-1] >= beta_min
        ]