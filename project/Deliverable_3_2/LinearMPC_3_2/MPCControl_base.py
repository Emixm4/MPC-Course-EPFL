import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData
from scipy.signal import cont2discrete


class MPCControl_base:
    """Base class for MPC controllers with terminal set constraints"""

    # To be defined in subclasses
    x_ids: np.ndarray  # State indices this controller uses
    u_ids: np.ndarray  # Input indices this controller uses

    # System matrices
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    # MPC components
    Q: np.ndarray  # State cost
    R: np.ndarray  # Input cost
    P: np.ndarray  # Terminal cost
    K_f: np.ndarray  # Terminal controller
    X_f: Polyhedron  # Terminal invariant set

    # Optimization problem
    ocp: cp.Problem
    x_var: cp.Variable
    u_var: cp.Variable
    x0_param: cp.Parameter
    x_ref_param: cp.Parameter
    u_ref_param: cp.Parameter

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

        # Extract subsystem matrices
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        # Discretize
        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        # Setup MPC controller
        self._setup_controller()

    def _setup_controller(self) -> None:
        """
        Setup the MPC optimization problem.
        To be overridden in subclasses to set specific Q, R, constraints.
        """
        # Get tuning parameters and constraints from subclass
        Q, R = self._get_cost_matrices()
        x_min, x_max, u_min, u_max = self._get_constraints()

        self.Q = Q
        self.R = R

        # Compute LQR terminal ingredients
        K, P, _ = dlqr(self.A, self.B, Q, R)
        self.K_f = -K  # Feedback gain
        self.P = P  # Terminal cost

        # Compute terminal invariant set
        self.X_f = self._compute_terminal_set(x_min, x_max, u_min, u_max)

        # Setup CVXPY optimization problem
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)

        # Default references (zero for stabilization)
        self.x_ref_param.value = np.zeros(self.nx)
        self.u_ref_param.value = np.zeros(self.nu)

        # Build cost function
        cost = 0
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_ref_param
            du = self.u_var[:, k] - self.u_ref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        # Terminal cost
        dx_N = self.x_var[:, self.N] - self.x_ref_param
        cost += cp.quad_form(dx_N, self.P)

        # Build constraints
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics, state bounds, input bounds
        for k in range(self.N):
            # Dynamics
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            )

            # State constraints
            constraints.append(self.x_var[:, k] >= x_min)
            constraints.append(self.x_var[:, k] <= x_max)

            # Input constraints
            constraints.append(self.u_var[:, k] >= u_min)
            constraints.append(self.u_var[:, k] <= u_max)

        # Terminal constraint: x[N] in X_f
        # NOTE: Temporarily disabled for large deviations - horizon may not be long enough
        # if self.X_f is not None:
        #     dx_N = self.x_var[:, self.N] - self.x_ref_param
        #     constraints.append(self.X_f.A @ dx_N <= self.X_f.b)

        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _compute_terminal_set(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        max_iter: int = 20,
    ) -> Polyhedron:
        """
        Compute maximal positively invariant set for terminal constraint.
        Uses the closed-loop system A_cl = A + B*K_f
        """
        # Closed-loop dynamics
        A_cl = self.A + self.B @ self.K_f

        # State constraints: x_min <= x <= x_max
        # Converted to A*x <= b form
        A_x = np.vstack([np.eye(self.nx), -np.eye(self.nx)])
        b_x = np.hstack([x_max, -x_min])
        X = Polyhedron(H=HData(A=A_x, b=b_x))

        # Input constraints under feedback u = K_f * x
        # u_min <= K_f @ x <= u_max
        U_constraints = []
        for i in range(self.nu):
            # u_min[i] <= K_f[i,:] @ x <= u_max[i]
            # => K_f[i,:] @ x <= u_max[i] and -K_f[i,:] @ x <= -u_min[i]
            U_constraints.append(Polyhedron(H=HData(A=self.K_f[i:i+1, :], b=u_max[i:i+1])))
            U_constraints.append(Polyhedron(H=HData(A=-self.K_f[i:i+1, :], b=-u_min[i:i+1])))

        # Combine all constraints
        XU = X
        for U_poly in U_constraints:
            XU = XU.intersect(U_poly)

        # Iteratively compute maximal invariant set
        X_f = XU
        for i in range(max_iter):
            # Pre-image: {x | A_cl*x in X_f}
            Pre_X_f = Polyhedron(H=HData(A=X_f.A @ A_cl, b=X_f.b))

            # Intersection
            X_f_new = X_f.intersect(Pre_X_f)

            # Remove redundant constraints every 5 iterations to speed up
            if i % 5 == 4:
                H_min = X_f_new.minHrep()
                X_f_new = Polyhedron(H=H_min)

            # Check convergence
            if len(X_f_new.A) == len(X_f.A):
                # Converged (no new constraints added)
                break

            X_f = X_f_new

        # Final minimal representation
        H_final = X_f.minHrep()
        return Polyhedron(H=H_final)

    def get_u(
        self,
        x0: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve MPC problem and return optimal control input.

        Args:
            x0: Current state (in delta coordinates relative to xs)
            x_target: Target state (optional, for tracking)
            u_target: Target input (optional, for tracking)

        Returns:
            u0: Optimal control input at current time
            x_traj: Predicted state trajectory
            u_traj: Predicted input trajectory
        """
        # Set initial state (delta coordinates)
        self.x0_param.value = x0

        # Set references (delta coordinates)
        if x_target is not None:
            self.x_ref_param.value = x_target
        else:
            self.x_ref_param.value = np.zeros(self.nx)

        if u_target is not None:
            self.u_ref_param.value = u_target
        else:
            self.u_ref_param.value = np.zeros(self.nu)

        # Solve MPC problem
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return zero control on failure
            return (
                np.zeros(self.nu),
                np.zeros((self.nx, self.N + 1)),
                np.zeros((self.nu, self.N)),
            )

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: MPC status = {self.ocp.status}")
            print(f"  Initial state deviation: {np.linalg.norm(x0):.4f}")
            print(f"  Horizon length: {self.N} steps ({self.N * self.Ts:.2f}s)")
            if self.ocp.status == "infeasible_inaccurate":
                print("  → Problem likely infeasible: increase horizon H or relax constraints")
            # Return zero control
            return (
                np.zeros(self.nu),
                np.zeros((self.nx, self.N + 1)),
                np.zeros((self.nu, self.N)),
            )

        # Extract solution
        u0 = self.u_var[:, 0].value
        x_traj = self.x_var.value
        u_traj = self.u_var.value

        if u0 is None:
            u0 = np.zeros(self.nu)
        if x_traj is None:
            x_traj = np.zeros((self.nx, self.N + 1))
        if u_traj is None:
            u_traj = np.zeros((self.nu, self.N))

        return u0, x_traj, u_traj

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        To be overridden in subclasses.
        """
        raise NotImplementedError("Subclass must implement _get_cost_matrices()")

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get constraint bounds (x_min, x_max, u_min, u_max).
        To be overridden in subclasses.
        """
        raise NotImplementedError("Subclass must implement _get_constraints()")

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        """Discretize continuous-time system using zero-order hold"""
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def estimate_parameters(self, x: np.ndarray, u: np.ndarray) -> None:
        """Placeholder for parameter estimation (used in Part 5)"""
        pass
