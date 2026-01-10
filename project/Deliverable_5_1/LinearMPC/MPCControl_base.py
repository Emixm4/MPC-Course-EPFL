import cvxpy as cp
import numpy as np
from control import dlqr
# from mpt4py import Polyhedron # Not needed for Part 5.1 
# from mpt4py.base import HData  # Not needed for Part 5.1
from scipy.signal import cont2discrete


class MPCControl_base:
    """Base class for MPC controllers for tracking"""

    # To be defined in subclasses
    x_ids: np.ndarray  # State indices this controller uses
    u_ids: np.ndarray  # Input indices this controller uses

    # System matrices
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray  # Output matrix (for steady-state target)
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    ny: int
    Ts: float
    H: float
    N: int

    # MPC components
    Q: np.ndarray  # State cost
    R: np.ndarray  # Input cost
   

    # Optimization problems
    ocp: cp.Problem  # Main MPC controller
    target_ocp: cp.Problem  # Steady-state target solver
    x_var: cp.Variable
    u_var: cp.Variable
    x0_param: cp.Parameter
    x_ref_param: cp.Parameter
    u_ref_param: cp.Parameter

    # Steady-state target solver variables
    xs_var: cp.Variable
    us_var: cp.Variable
    ref_param: cp.Parameter

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

        # Get output matrix C from subclass
        self.C = self._get_output_matrix()
        self.ny = self.C.shape[0]

        # Setup MPC controller and steady-state target solver
        self._setup_controller()
        self._setup_steady_state_target()

    def _setup_controller(self) -> None:
        """
        Setup the MPC optimization problem for tracking.
        """
        # Get tuning parameters and constraints from subclass
        Q, R = self._get_cost_matrices()
        x_min, x_max, u_min, u_max = self._get_constraints()

        self.Q = Q
        self.R = R

        # Setup CVXPY optimization problem
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)

        # Default references (zero for stabilization)
        self.x_ref_param.value = np.zeros(self.nx)
        self.u_ref_param.value = np.zeros(self.nu)

        # Build cost function - track references
        cost = 0
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_ref_param
            du = self.u_var[:, k] - self.u_ref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

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

        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _setup_steady_state_target(self) -> None:
        """
        Sstup steady-state target optimization problem.
        given a reference output, solvs for equilibrium :
        - xs = A*xs + B*us (equilibrium)
        - C*xs = ref (output matches ref)
        - constraints are satisfied
        - Min. input effort: us'*us
        """
        _, _, u_min, u_max = self._get_constraints()

        # Variables for steady-state
        self.xs_var = cp.Variable(self.nx)
        self.us_var = cp.Variable(self.nu)
        self.ref_param = cp.Parameter(self.ny)

        # Default reference
        self.ref_param.value = np.zeros(self.ny)

        # Cost: minimize input effort
        cost = cp.quad_form(self.us_var, np.eye(self.nu))

        constraints = []

        constraints.append(self.xs_var == self.A @ self.xs_var + self.B @ self.us_var)

        # Output matches ref: C*xs = ref
        constraints.append(self.C @ self.xs_var == self.ref_param)

        # Input constraints
        constraints.append(self.us_var >= u_min)
        constraints.append(self.us_var <= u_max)

        # Create target optimization problem
        self.target_ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self,
        x0: np.ndarray,
        ref: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve MPC problem and return optimal control input.

        Args:
            x0: Current state (in delta coordinates relative to xs)
            ref: Reference output (e.g., desired velocity or angle)

        Returns:
            u0: Optimal control input at current time
            x_traj: Predicted state trajectory
            u_traj: Predicted input trajectory
        """
        # Set initial state (delta coordinates)
        self.x0_param.value = x0

        # Compute steady-state target if reference is given
        if ref is not None:
            self.ref_param.value = ref

            # Solve steady-state target problem
            try:
                self.target_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as e:
                print(f"Target solve failed: {e}")
                # Use zero reference on failure
                self.x_ref_param.value = np.zeros(self.nx)
                self.u_ref_param.value = np.zeros(self.nu)
            else:
                if self.target_ocp.status in ["optimal", "optimal_inaccurate"]:
                    # Use computed steady-state targets
                    self.x_ref_param.value = self.xs_var.value
                    self.u_ref_param.value = self.us_var.value
                else:
                    print(f"Warning: Target status = {self.target_ocp.status}")
                    # Use zero reference on failure
                    self.x_ref_param.value = np.zeros(self.nx)
                    self.u_ref_param.value = np.zeros(self.nu)
        else:
            # No reference given - regulate to zero
            self.x_ref_param.value = np.zeros(self.nx)
            self.u_ref_param.value = np.zeros(self.nu)

        # Solve MPC problem
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return zero on failure
            return (
                np.zeros(self.nu),
                np.zeros((self.nx, self.N + 1)),
                np.zeros((self.nu, self.N)),
            )

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: MPC status = {self.ocp.status}")
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

    def _get_output_matrix(self) -> np.ndarray:
        """
        Get output matrix C for steady-state target computation.
        C selects the controlled output from the state vector.
        To be overridden in subclasses.

        For velocity controllers: C selects the velocity state
        For roll controller: C selects the roll angle state
        """
        raise NotImplementedError("Subclass must implement _get_output_matrix()")

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
