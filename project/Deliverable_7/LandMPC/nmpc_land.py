import numpy as np
import casadi as ca
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller for rocket landing.

    Uses CasADi with IPOPT solver for full nonlinear optimization.
    No subsystem decomposition - controls all 12 states and 4 inputs simultaneously.
    """

    def __init__(self, rocket, Ts=0.1, H=2.0):
        """
        Initialize nonlinear MPC controller.

        Args:
            rocket: Rocket object with symbolic dynamics
            Ts: Sampling time (default: 0.1s)
            H: Prediction horizon (default: 2.0s)
        """
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)

        # State and input dimensions
        self.nx = 12  # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        self.nu = 4   # [d1, d2, Pavg, d3]

        # Reference target (will be set in get_u)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = np.zeros(self.nu)

        # Setup the optimization problem
        self._setup_controller()

    def _setup_controller(self) -> None:
        """
        Setup nonlinear MPC optimization problem using CasADi.

        Uses direct multiple shooting with Euler integration.
        """
        # Decision variables
        # We stack all states and inputs: [x_0, u_0, x_1, u_1, ..., x_N-1, u_N-1, x_N]
        X = ca.MX.sym('X', self.nx, self.N + 1)  # States over horizon
        U = ca.MX.sym('U', self.nu, self.N)       # Inputs over horizon

        # Parameters (initial state and reference)
        x0_param = ca.MX.sym('x0', self.nx)
        x_ref_param = ca.MX.sym('x_ref', self.nx)
        u_ref_param = ca.MX.sym('u_ref', self.nu)

        # Cost matrices
        # State cost: penalize deviation from reference
        Q = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,          # angular velocities (wx, wy, wz)
            10.0, 10.0, 10.0,       # angles (alpha, beta, gamma)
            1.0, 1.0, 1.0,          # velocities (vx, vy, vz)
            10.0, 10.0, 10.0        # positions (x, y, z)
        ))

        # Input cost: penalize input effort
        R = ca.diag(ca.vertcat(0.1, 0.1, 0.1, 0.1))  # d1, d2, Pavg, d3

        # Terminal cost (same as stage cost for simplicity)
        P = Q

        # Build the cost function
        cost = 0
        for k in range(self.N):
            dx = X[:, k] - x_ref_param
            du = U[:, k] - u_ref_param
            cost += dx.T @ Q @ dx + du.T @ R @ du

        # Terminal cost
        dx_N = X[:, self.N] - x_ref_param
        cost += dx_N.T @ P @ dx_N

        # Build constraints
        g = []  # Constraint vector
        lbg = []  # Lower bounds on constraints
        ubg = []  # Upper bounds on constraints

        # Initial condition constraint
        g.append(X[:, 0] - x0_param)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # Dynamics constraints (using Euler integration)
        for k in range(self.N):
            # Euler: x_{k+1} = x_k + Ts * f(x_k, u_k)
            x_next = X[:, k] + self.Ts * self.f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

        # State constraints
        # z >= 0 (don't go underground)
        # We add this as a soft constraint by allowing violations
        # For simplicity, we enforce it as a hard constraint
        for k in range(self.N + 1):
            g.append(X[11, k])  # z (12th state, index 11)
            lbg.append(0.0)     # z >= 0
            ubg.append(ca.inf)  # No upper bound

        # Concatenate constraints
        g = ca.vertcat(*g)

        # Decision variable bounds
        # State bounds (mostly unbounded except z >= 0 which is in constraints)
        lbx = []
        ubx = []

        for k in range(self.N + 1):
            # States: unbounded except constraints above
            lbx.extend([-ca.inf] * self.nx)
            ubx.extend([ca.inf] * self.nx)

        for k in range(self.N):
            # Input bounds
            lbx.extend([-15.0, -15.0, 40.0, -20.0])  # d1, d2, Pavg, d3
            ubx.extend([15.0, 15.0, 80.0, 20.0])

        # Decision variables vector
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters vector
        p = ca.vertcat(x0_param, x_ref_param, u_ref_param)

        # Create NLP solver
        nlp = {
            'x': w,
            'f': cost,
            'g': g,
            'p': p
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.max_iter': 100,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store problem dimensions for solution extraction
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Store variable shapes
        self.X_shape = (self.nx, self.N + 1)
        self.U_shape = (self.nu, self.N)

        # Initialize solution with zeros
        self.w0 = np.zeros((self.nx * (self.N + 1) + self.nu * self.N, 1))

    def get_u(
        self, t0: float, x0: np.ndarray, x_ref: np.ndarray = None, u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve nonlinear MPC and return control input.

        Args:
            t0: Current time
            x0: Current state (12,)
            x_ref: Reference state (12,) - optional
            u_ref: Reference input (4,) - optional

        Returns:
            u0: Optimal control input (4,)
            x_ol: Predicted state trajectory (12, N+1)
            u_ol: Predicted input trajectory (4, N)
            t_ol: Time vector (N+1,)
        """
        # Set references
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        if u_ref is None:
            u_ref = np.array([0.0, 0.0, 56.67, 0.0])  # Hover input

        # Parameters
        p = np.concatenate([x0, x_ref, u_ref])

        # Solve NLP
        try:
            sol = self.solver(
                x0=self.w0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )

            # Extract solution
            w_opt = sol['x'].full().flatten()

            # Warm start for next iteration
            self.w0 = w_opt.reshape(-1, 1)

            # Parse solution
            X_opt = w_opt[:self.nx * (self.N + 1)].reshape(self.nx, self.N + 1, order='F')
            U_opt = w_opt[self.nx * (self.N + 1):].reshape(self.nu, self.N, order='F')

            # First control input
            u0 = U_opt[:, 0]

            # Trajectories
            x_ol = X_opt
            u_ol = U_opt
            t_ol = np.arange(self.N + 1) * self.Ts + t0

            return u0, x_ol, u_ol, t_ol

        except Exception as e:
            print(f"NMPC solve failed: {e}")
            # Return zero control and trajectories
            u0 = u_ref if u_ref is not None else np.zeros(self.nu)
            x_ol = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_ol = np.tile(u0.reshape(-1, 1), (1, self.N))
            t_ol = np.arange(self.N + 1) * self.Ts + t0

            return u0, x_ol, u_ol, t_ol
