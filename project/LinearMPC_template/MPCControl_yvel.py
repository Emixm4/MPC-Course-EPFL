import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

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
        # [wx, alpha, vy] - we want vy -> 0
        return np.diag([1.0, 10.0, 100.0])
    
    def _get_R(self):
        return np.array([[1.0]])
    
    def _get_u_constraints(self):
        # Input constraints: -15° <= dR <= 15° for all N steps
        u_min = np.deg2rad(-15)
        u_max = np.deg2rad(15)
        return [
            self.u_var <= u_max,
            self.u_var >= u_min
        ]
    
    def _get_x_constraints(self):
        # State constraints: -10° <= alpha <= 10° for steps 0 to N-1
        # alpha is state index 1
        alpha_min = np.deg2rad(-10)
        alpha_max = np.deg2rad(10)
        return [
            self.x_var[1, :-1] <= alpha_max,
            self.x_var[1, :-1] >= alpha_min
        ]
