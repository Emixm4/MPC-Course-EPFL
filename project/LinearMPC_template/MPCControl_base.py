import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # Define variables
        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_param = cp.Parameter((self.nx,), name='x0')
        
        # Get cost matrices (can be overridden in subclasses)
        Q = self._get_Q()
        R = self._get_R()
        Qf = self._get_P()  # Terminal cost from LQR
        
        # Build cost function
        cost = 0
        for i in range(self.N):
            # Stage cost around equilibrium
            x_dev = self.x_var[:, i] - self.xs
            u_dev = self.u_var[:, i] - self.us
            cost += cp.quad_form(x_dev, Q) + cp.quad_form(u_dev, R)
        
        # Terminal cost
        x_dev_N = self.x_var[:, -1] - self.xs
        cost += cp.quad_form(x_dev_N, Qf)
        
        # Build constraints list
        constraints = []
        
        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)
        
        # System dynamics: x[:,k+1] = A*x[:,k] + B*u[:,k]
        constraints.append(self.x_var[:, 1:] == self.A @ self.x_var[:, :-1] + self.B @ self.u_var)
        
        # State constraints (if any)
        x_constraints = self._get_x_constraints()
        if x_constraints:
            constraints.extend(x_constraints)
        
        # Input constraints (if any)
        u_constraints = self._get_u_constraints()
        if u_constraints:
            constraints.extend(u_constraints)
        
        # Terminal constraints (if any)
        terminal_constraints = self._get_terminal_constraints()
        if terminal_constraints:
            constraints.extend(terminal_constraints)
        
        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # Set current state
        self.x0_param.value = x0
        
        # Solve the optimization problem
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        
        # Check solver status
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: MPC solver status: {self.ocp.status}")
            # Fallback to equilibrium input
            u0 = self.us
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
        else:
            # Extract optimal solution
            u0 = self.u_var.value[:, 0]  # First control input
            x_traj = self.x_var.value     # State trajectory
            u_traj = self.u_var.value     # Input trajectory
        
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
    
    def _get_Q(self):
        """State cost matrix (to be overridden in subclasses)"""
        return np.eye(self.nx)
    
    def _get_R(self):
        """Input cost matrix (to be overridden in subclasses)"""
        return np.eye(self.nu)
    
    def _get_P(self):
        """Terminal cost matrix computed using LQR"""
        try:
            from control import dlqr
            K, S, _ = dlqr(self.A, self.B, self._get_Q(), self._get_R())
            return S
        except:
            # Fallback: use Q
            return self._get_Q()
    
    def _get_u_constraints(self):
        """Input constraints (to be overridden in subclasses)
        Returns list of CVXPY constraints on self.u_var
        """
        return []
    
    def _get_x_constraints(self):
        """State constraints (to be overridden in subclasses)
        Returns list of CVXPY constraints on self.x_var
        """
        return []
    
    def _get_terminal_constraints(self):
        """Terminal constraints (to be overridden in subclasses)
        Returns list of CVXPY constraints on self.x_var[:, -1]
        """
        return []
