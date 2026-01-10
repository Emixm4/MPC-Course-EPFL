import numpy as np
import casadi as ca
from typing import Tuple
from scipy.linalg import solve_discrete_are


class NmpcCtrl:
    """
    Nonlinear MPC controller for rocket landing.

    Uses CasADi with IPOPT solver for full nonlinear optimization.
    No subsystem decomposition - controls all 12 states and 4 inputs simultaneously.
    """

    def __init__(self, rocket, Ts=0.1, H=2.0, x_ref=None):
        """
        Initialize nonlinear MPC controller.

        Args:
            rocket: Rocket object with symbolic dynamics
            Ts: Sampling time (default: 0.1s)
            H: Prediction horizon (default: 2.0s)
            x_ref: Reference state (default: hover at [0,0,0,...,1,0,3])
        """
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
        self.rocket = rocket

        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)

        # State and input dimensions
        self.nx = 12  # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        self.nu = 4   # [d1, d2, Pavg, Pdiff]

        # Set reference target (trim around it)
        if x_ref is None:
            x_ref = np.array([0.]*9 + [1., 0., 3.])  # Default: hover at (1, 0, 3)
        
        # Compute trim point around reference
        self.xs, self.us = rocket.trim(x_ref)
        print(f"NMPC: Trim point xs = {self.xs}")
        print(f"NMPC: Trim input us = {self.us}")

        # Hint 3: Linearize system and compute terminal cost (done ONCE for efficiency)
        # This provides a better terminal cost than just scaling Q
        self._compute_terminal_cost()

        # Setup the optimization problem
        self._setup_controller()

    def _compute_terminal_cost(self) -> None:
        """
        Compute terminal cost matrix P from linearized system.
        
        Following Hint 3 from project description:
        "A common approximate terminal cost is to linearize your system and 
        compute a terminal cost based on this. To save computation time, you 
        should only compute the linearization and the terminal cost matrix once."
        """
        # Linearize around trim point
        from scipy.signal import cont2discrete
        
        # Get continuous-time linearization
        A_cont, B_cont = self.rocket.linearize(self.xs, self.us)
        
        # Discretize (matches the RK4 integration used in NMPC)
        Ad, Bd, _, _, _ = cont2discrete((A_cont, B_cont, np.zeros((12, 12)), np.zeros((12, 4))), 
                                         self.Ts, method='zoh')
        
        # Stage cost matrices
        Q_np = np.diag([1.0, 1.0, 1.0,          # angular velocities
                        20.0, 20.0, 20.0,       # angles
                        10.0, 10.0, 10.0,       # velocities
                        50.0, 50.0, 100.0])     # positions (z most critical)
        
        R_np = np.diag([0.01, 0.01, 0.01, 0.01])  # inputs
        
        # Solve discrete-time algebraic Riccati equation (DARE)
        # Ad'*P*Ad - P - Ad'*P*Bd*(R + Bd'*P*Bd)^-1*Bd'*P*Ad + Q = 0
        try:
            P_np = solve_discrete_are(Ad, Bd, Q_np, R_np)
            self.P_terminal = P_np
            print(f"NMPC: Terminal cost computed from DARE (condition number: {np.linalg.cond(P_np):.2e})")
        except Exception as e:
            print(f"NMPC: Warning - DARE solve failed ({e}), using P = 10*Q fallback")
            self.P_terminal = 10.0 * Q_np

    def _setup_controller(self) -> None:
        """
        Setup nonlinear MPC optimization problem using CasADi.

        Uses direct multiple shooting with Euler integration.
        """
        # Decision variables
        # We stack all states and inputs: [x_0, u_0, x_1, u_1, ..., x_N-1, u_N-1, x_N]
        X = ca.MX.sym('X', self.nx, self.N + 1)  # States over horizon
        U = ca.MX.sym('U', self.nu, self.N)       # Inputs over horizon

        # Parameters (initial state only - reference is stored in self.xs, self.us)
        x0_param = ca.MX.sym('x0', self.nx)

        # Cost matrices (tuned for landing task)
        # State cost: penalize deviation from reference
        Q = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,          # angular velocities (wx, wy, wz)
            20.0, 20.0, 20.0,       # angles (alpha, beta, gamma) - avoid large tilts
            10.0, 10.0, 10.0,       # velocities (vx, vy, vz) - reach zero velocity
            50.0, 50.0, 100.0       # positions (x, y, z) - z most critical for landing
        ))

        # Input cost: penalize input effort (small penalties to allow aggressive control)
        R = ca.diag(ca.vertcat(0.01, 0.01, 0.01, 0.01))  # d1, d2, Pavg, Pdiff

        # Terminal cost from DARE solution (Hint 3 implementation)
        P = ca.DM(self.P_terminal)

        # Build the cost function
        # Use stored reference (self.xs, self.us) instead of parameters
        x_ref = ca.DM(self.xs)
        u_ref = ca.DM(self.us)
        
        cost = 0
        for k in range(self.N):
            dx = X[:, k] - x_ref
            du = U[:, k] - u_ref
            cost += dx.T @ Q @ dx + du.T @ R @ du

        # Terminal cost
        dx_N = X[:, self.N] - x_ref
        cost += dx_N.T @ P @ dx_N

        # Build constraints
        g = []  # Constraint vector
        lbg = []  # Lower bounds on constraints
        ubg = []  # Upper bounds on constraints

        # Initial condition constraint
        g.append(X[:, 0] - x0_param)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # Dynamics constraints (using RK4 integration for better accuracy)
        for k in range(self.N):
            # RK4 integration: x_{k+1} = x_k + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
            k1 = self.f(X[:, k], U[:, k])
            k2 = self.f(X[:, k] + (self.Ts/2)*k1, U[:, k])
            k3 = self.f(X[:, k] + (self.Ts/2)*k2, U[:, k])
            k4 = self.f(X[:, k] + self.Ts*k3, U[:, k])
            x_next = X[:, k] + (self.Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
            
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

        # State constraints
        # 1. z >= 0 (don't go underground)
        # 2. |beta| <= 80° (avoid singularity at beta = 90°)
        for k in range(self.N + 1):
            # z >= 0
            g.append(X[11, k])  # z (12th state, index 11)
            lbg.append(0.0)     # z >= 0
            ubg.append(ca.inf)  # No upper bound

            # |beta| <= 80° (beta is state index 4)
            g.append(X[4, k])   # beta (5th state, index 4)
            lbg.append(-np.deg2rad(80))  # beta >= -80°
            ubg.append(np.deg2rad(80))   # beta <= 80°

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
            # Input bounds (constrained per project description)
            # d1, d2: ±15° = ±0.26 rad (servo deflection)
            # Pavg: 40-80% (minimum 40% for safety, max 80% for structural limits)
            # Pdiff: ±20% (differential throttle)
            lbx.extend([-0.26, -0.26, 40.0, -20.0])  # d1, d2, Pavg, Pdiff
            ubx.extend([0.26, 0.26, 80.0, 20.0])

        # Decision variables vector
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters vector (only initial state now)
        p = x0_param

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
            'ipopt.max_iter': 500,  # Increased for better convergence
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'ipopt.tol': 1e-5,
            'expand': True  # Expand NLP for faster evaluation
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
        self, t0: float, x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve nonlinear MPC and return control input.

        Args:
            t0: Current time
            x0: Current state (12,)

        Returns:
            u0: Optimal control input (4,)
            x_ol: Predicted state trajectory (12, N+1)
            u_ol: Predicted input trajectory (4, N)
            t_ol: Time vector (N+1,)
        """
        # Parameters (only initial state, reference is stored internally)
        p = x0

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
            # Return hover control and constant trajectories
            u0 = self.us.copy()
            x_ol = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_ol = np.tile(u0.reshape(-1, 1), (1, self.N))
            t_ol = np.arange(self.N + 1) * self.Ts + t0

            return u0, x_ol, u_ol, t_ol
