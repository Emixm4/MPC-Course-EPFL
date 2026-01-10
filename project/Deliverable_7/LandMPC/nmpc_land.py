import numpy as np
import casadi as ca
from typing import Tuple
from scipy.linalg import solve_discrete_are


class NmpcCtrl:
    """
    Nonlinear MPC controller for rocket landing.
    Uses CasADi with IPOPT solver for full nonlinear optimization.
    """

    def __init__(self, rocket, Ts=0.1, H=2.0, x_ref=None):
        """
        Initialize nonlinear MPC controller.
        Args:
            rocket: Rocket object with symbolic dynamics
            Ts: Sampling time 
            H: Prediction horizon 
            x_ref: Reference state 
        """
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
        self.rocket = rocket

        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)

        # State and input dimensions
        self.nx = 12 
        self.nu = 4 

        # Set reference target
        if x_ref is None:
            x_ref = np.array([0.]*9 + [1., 0., 3.]) 
        
        # Compute trim point around reference
        self.xs, self.us = rocket.trim(x_ref)
        print(f"NMPC: Trim point xs = {self.xs}")
        print(f"NMPC: Trim input us = {self.us}")

        self._compute_terminal_cost()

        # Setup the optimization problem
        self._setup_controller()

    def _compute_terminal_cost(self) -> None:
        """
        Compute terminal cost matrix P from linearized system.

        """
        # Linearize around trim point
        from scipy.signal import cont2discrete
        
        # Get continuous-time linearization
        A_cont, B_cont = self.rocket.linearize(self.xs, self.us)
        
        # Discretize (RK4)
        Ad, Bd, _, _, _ = cont2discrete((A_cont, B_cont, np.zeros((12, 12)), np.zeros((12, 4))), 
                                         self.Ts, method='zoh')
        
        # Stage cost matrices
        Q_np = np.diag([1.0, 1.0, 1.0,        
                        20.0, 20.0, 20.0,       # angles
                        10.0, 10.0, 10.0,       # velocities
                        50.0, 50.0, 100.0])     # positions
        
        R_np = np.diag([0.01, 0.01, 0.01, 0.01])  # inputs
        
        #DARE
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
        """
        # Decision variables
        X = ca.MX.sym('X', self.nx, self.N + 1)  # States over horizon
        U = ca.MX.sym('U', self.nu, self.N)       # Inputs over horizon

        x0_param = ca.MX.sym('x0', self.nx)

        # Cost matrices (tuned for landing task)
        Q = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,        
            20.0, 20.0, 20.0,       
            10.0, 10.0, 10.0,       
            50.0, 50.0, 100.0       
        ))

        R = ca.diag(ca.vertcat(0.01, 0.01, 0.01, 0.01)) 

        # Terminal cost from DARE solution
        P = ca.DM(self.P_terminal)

        # Build the cost function
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

        # constraints
        g = []  
        lbg = []  
        ubg = []  

        # Initial condition constraint
        g.append(X[:, 0] - x0_param)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # Dynamics constraints
        for k in range(self.N):
            # RK4 x_{k+1} = x_k + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
            k1 = self.f(X[:, k], U[:, k])
            k2 = self.f(X[:, k] + (self.Ts/2)*k1, U[:, k])
            k3 = self.f(X[:, k] + (self.Ts/2)*k2, U[:, k])
            k4 = self.f(X[:, k] + self.Ts*k3, U[:, k])
            x_next = X[:, k] + (self.Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
            
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

        # State constraints
        for k in range(self.N + 1):
            # z >= 0
            g.append(X[11, k]) 
            lbg.append(0.0)   
            ubg.append(ca.inf)  

            # |beta| <= 80°
            g.append(X[4, k])   
            lbg.append(-np.deg2rad(80))  
            ubg.append(np.deg2rad(80))   

        # Concatenate constraints
        g = ca.vertcat(*g)

        # Decision variable bounds
        lbx = []
        ubx = []

        for k in range(self.N + 1):
            lbx.extend([-ca.inf] * self.nx)
            ubx.extend([ca.inf] * self.nx)

        for k in range(self.N):

            lbx.extend([-0.26, -0.26, 40.0, -20.0])  # d1, d2, Pavg, Pdiff
            ubx.extend([0.26, 0.26, 80.0, 20.0])

        # Decision variables vector
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters vector 
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
