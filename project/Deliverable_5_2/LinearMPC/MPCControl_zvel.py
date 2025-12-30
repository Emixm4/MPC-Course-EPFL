import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base
import cvxpy as cp


class MPCControl_zvel(MPCControl_base):
    """
    MPC controller for Z (vertical) velocity subsystem with offset-free tracking.

    Augmented system with disturbance estimation:
    - States: [vz, d] where d is estimated disturbance
    - Input: [Pavg]
    - Dynamics: x+ = Ax + Bu + Bd, with d estimated via Kalman filter

    Constraints:
    - z >= 0 (don't go underground!)
    - 40 <= Pavg <= 80 (safety limits on throttle)
    """

    # State indices: vz=8
    x_ids = np.array([8])  # vz only (no position for velocity controller!)
    u_ids = np.array([2])  # Pavg

    # Augmented state estimator variables
    x_hat: np.ndarray  # State estimate [vz]
    d_hat: np.ndarray  # Disturbance estimate [d]

    # Kalman filter covariance
    P: np.ndarray  # Error covariance matrix

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

    def __init__(self, A: np.ndarray, B: np.ndarray, xs: np.ndarray, us: np.ndarray, Ts: float, H: float):
        """Initialize offset-free MPC with disturbance estimator."""
        # Call parent constructor
        super().__init__(A, B, xs, us, Ts, H)

        # Initialize state and disturbance estimates
        self.x_hat = np.zeros(self.nx)  # Initial state estimate
        self.d_hat = np.zeros(self.nx)  # Initial disturbance estimate

        # Initialize Kalman filter error covariance (augmented: [x; d])
        self.P = np.eye(2 * self.nx) * 1.0

        # Setup augmented MPC with disturbance compensation
        self._setup_offset_free_mpc()

    def _setup_offset_free_mpc(self):
        """
        Setup MPC with disturbance compensation.
        The disturbance is compensated in the dynamics: x+ = Ax + Bu + Bd
        """
        # Get tuning parameters and constraints from subclass
        Q, R = self._get_cost_matrices()
        x_min, x_max, u_min, u_max = self._get_constraints()

        # Setup CVXPY optimization problem (same as base, but we'll add d in get_u)
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)
        self.d_param = cp.Parameter(self.nx)  # Disturbance parameter

        # Default references
        self.x_ref_param.value = np.zeros(self.nx)
        self.u_ref_param.value = np.zeros(self.nu)
        self.d_param.value = np.zeros(self.nx)

        # Build cost function - track references
        cost = 0
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_ref_param
            du = self.u_var[:, k] - self.u_ref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        # NOTE: No terminal cost or terminal set for offset-free tracking (Part 5)

        # Build constraints with disturbance compensation
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics with disturbance: x+ = Ax + Bu + Bd
        for k in range(self.N):
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] +
                self.B @ self.u_var[:, k] + self.B @ self.d_param
            )

            # State constraints
            constraints.append(self.x_var[:, k] >= x_min)
            constraints.append(self.x_var[:, k] <= x_max)

            # Input constraints
            constraints.append(self.u_var[:, k] >= u_min)
            constraints.append(self.u_var[:, k] <= u_max)

        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _kalman_update(self, y_meas: np.ndarray, u_prev: np.ndarray):
        """
        Kalman filter update for augmented system [x; d].

        Augmented dynamics:
        [x+; d+] = [A, B; 0, I][x; d] + [B; 0]u + [w_x; w_d]
        y = [C, 0][x; d] + v

        where w_x, w_d are process noise, v is measurement noise.
        """
        # Augmented system matrices
        nx = self.nx
        A_aug = np.block([[self.A, self.B],
                         [np.zeros((nx, nx)), np.eye(nx)]])
        B_aug = np.block([[self.B],
                         [np.zeros((self.nu, nx)).T]])
        C_aug = np.block([[self.C, np.zeros((self.ny, nx))]])

        # Process and measurement noise covariances (tuning parameters)
        Q_process = np.eye(2 * nx) * 0.01  # Process noise covariance
        Q_process[nx:, nx:] *= 0.001  # Smaller noise on disturbance (it's assumed constant)
        R_meas = np.eye(self.ny) * 0.1  # Measurement noise covariance

        # Prediction step
        x_aug = np.concatenate([self.x_hat, self.d_hat])
        x_aug_pred = A_aug @ x_aug + B_aug @ u_prev
        P_pred = A_aug @ self.P @ A_aug.T + Q_process

        # Update step
        y_pred = C_aug @ x_aug_pred
        innovation = y_meas - y_pred
        S = C_aug @ P_pred @ C_aug.T + R_meas
        K = P_pred @ C_aug.T @ np.linalg.inv(S)

        x_aug_updated = x_aug_pred + K @ innovation
        self.P = (np.eye(2 * nx) - K @ C_aug) @ P_pred

        # Extract updated estimates
        self.x_hat = x_aug_updated[:nx]
        self.d_hat = x_aug_updated[nx:]

    def get_u(self, x0: np.ndarray, ref: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve offset-free MPC with disturbance estimation.

        Args:
            x0: Current measured state (in delta coordinates)
            ref: Reference output

        Returns:
            u0: Optimal control input
            x_traj: Predicted state trajectory
            u_traj: Predicted input trajectory
        """
        # Run Kalman filter to update state and disturbance estimates
        # Note: x0 is the measurement in delta coordinates
        y_meas = self.C @ x0  # Measured output
        u_prev = self.u_ref_param.value if hasattr(self.u_ref_param, 'value') else np.zeros(self.nu)
        self._kalman_update(y_meas, u_prev)

        # Set initial state to estimated state (in delta coordinates)
        self.x0_param.value = self.x_hat

        # Set estimated disturbance
        self.d_param.value = self.d_hat

        # Compute steady-state target if reference is given
        if ref is not None:
            self.ref_param.value = ref

            # Solve steady-state target problem
            try:
                self.target_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as e:
                print(f"Target solve failed: {e}")
                self.x_ref_param.value = np.zeros(self.nx)
                self.u_ref_param.value = np.zeros(self.nu)
            else:
                if self.target_ocp.status in ["optimal", "optimal_inaccurate"]:
                    # Adjust target to account for disturbance
                    # At steady-state: 0 = (A-I)xs + B*us + B*d
                    # So we need: xs = (I-A)^{-1} * B * (us + d)
                    self.x_ref_param.value = self.xs_var.value - self.d_hat
                    self.u_ref_param.value = self.us_var.value
                else:
                    print(f"Warning: Target status = {self.target_ocp.status}")
                    self.x_ref_param.value = np.zeros(self.nx)
                    self.u_ref_param.value = np.zeros(self.nu)
        else:
            # No reference given - regulate to zero accounting for disturbance
            self.x_ref_param.value = -self.d_hat  # Compensate for disturbance
            self.u_ref_param.value = np.zeros(self.nu)

        # Solve MPC problem
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return (
                np.zeros(self.nu),
                np.zeros((self.nx, self.N + 1)),
                np.zeros((self.nu, self.N)),
            )

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: MPC status = {self.ocp.status}")
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

    def estimate_parameters(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Get current disturbance estimate (for logging/debugging).
        This method is called by the wrapper but we don't need to do anything here.
        """
        pass
