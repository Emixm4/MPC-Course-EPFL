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
    d_hat: np.ndarray  # Constant disturbance estimate [d]
    u_prev: np.ndarray  # Previous control input (for Kalman prediction)

    # Kalman filter covariance
    P: np.ndarray  # Error covariance matrix

    # Debug logging
    debug_log: list  # List to store debug information

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
        self.d_hat = np.zeros(self.nu)  # Initial INPUT disturbance estimate (nu dimension)
        self.u_prev = np.zeros(self.nu)  # Previous control input (delta coordinates)

        # Initialize Kalman filter error covariance (augmented: [x; d])
        self.P = np.eye(self.nx + self.nu) * 1.0

        # Initialize debug logging
        self.debug_log = []

        # Setup augmented MPC with disturbance compensation
        self._setup_offset_free_mpc()

        # Override steady-state target to include disturbance
        self._setup_steady_state_target_with_disturbance()

    def _setup_offset_free_mpc(self):
        """
        Setup MPC with input disturbance d.
        Dynamics: x+ = A*x + B*(u + d) where d is a constant input disturbance
        """
        # Get tuning parameters and constraints from subclass
        Q, R = self._get_cost_matrices()
        x_min, x_max, u_min, u_max = self._get_constraints()

        # Setup CVXPY optimization problem
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)
        self.d_param = cp.Parameter(self.nu)  # Input disturbance parameter

        # Default references
        self.x_ref_param.value = np.zeros(self.nx)
        self.u_ref_param.value = np.zeros(self.nu)
        self.d_param.value = np.zeros(self.nu)

        # Build cost function - track references
        cost = 0
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_ref_param
            du = self.u_var[:, k] - self.u_ref_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        # NOTE: No terminal cost or terminal set for offset-free tracking (Part 5)

        # Build constraints with additive disturbance
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics WITH input disturbance: x+ = A*x + B*(u + d)
        for k in range(self.N):
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] +
                self.B @ (self.u_var[:, k] + self.d_param)
            )

            # State constraints
            constraints.append(self.x_var[:, k] >= x_min)
            constraints.append(self.x_var[:, k] <= x_max)

            # Input constraints
            constraints.append(self.u_var[:, k] >= u_min)
            constraints.append(self.u_var[:, k] <= u_max)

        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _setup_steady_state_target_with_disturbance(self):
        """
        Setup steady-state target optimization problem with disturbance compensation.
        Given a reference output, solves for equilibrium (xs, us) such that:
        - xs = A*xs + B*us + B*d (equilibrium with disturbance)
        - C*xs = ref (output matches reference)
        - Constraints are satisfied
        - Minimize input effort: us'*us
        """
        _, _, u_min, u_max = self._get_constraints()

        # Variables for steady-state (reuse from parent class)
        self.xs_var = cp.Variable(self.nx)
        self.us_var = cp.Variable(self.nu)
        self.ref_param = cp.Parameter(self.ny)
        self.d_param_target = cp.Parameter(self.nu)  # INPUT disturbance parameter for target

        # Default reference and disturbance
        self.ref_param.value = np.zeros(self.ny)
        self.d_param_target.value = np.zeros(self.nu)

        # Cost: minimize input effort
        cost = cp.quad_form(self.us_var, np.eye(self.nu))

        # Constraints
        constraints = []

        # Equilibrium WITH input disturbance: xs = A*xs + B*(us + d)
        constraints.append(
            self.xs_var == self.A @ self.xs_var + self.B @ (self.us_var + self.d_param_target)
        )

        # Output matches reference: C*xs = ref
        constraints.append(self.C @ self.xs_var == self.ref_param)

        # Input constraints
        constraints.append(self.us_var >= u_min)
        constraints.append(self.us_var <= u_max)

        # Create target optimization problem (override parent's version)
        self.target_ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _kalman_update(self, y_meas: np.ndarray, u_prev: np.ndarray):
        """
        Kalman filter update for augmented system [x; d] with constant INPUT disturbance.

        Augmented dynamics (disturbance through input):
        [x+; d+] = [A, B; 0, I][x; d] + [B; 0]u + [w_x; w_d]
        y = [C, 0][x; d] + v

        This means: x+ = A*x + B*(u + d) (input disturbance)
                    d+ = d (constant)
        """
        nx = self.nx
        nu = self.nu

        # Augmented system matrices for INPUT disturbance
        A_aug = np.block([[self.A, self.B],  # Disturbance through B
                         [np.zeros((nu, nx)), np.eye(nu)]])
        B_aug = np.block([[self.B],
                         [np.zeros((nu, nu))]])
        C_aug = np.block([[self.C, np.zeros((self.ny, nu))]])  # No disturbance on output

        # Process and measurement noise covariances
        # Tuning strategy:
        # - Lower Q_process[state] = trust model dynamics more
        # - Lower Q_process[disturb] = disturbance should be very constant
        # - Lower R_meas = trust measurements more
        Q_process = np.eye(nx + nu) * 0.001  # Trust model dynamics
        Q_process[nx:, nx:] *= 0.0001  # Disturbance is nearly constant (very low process noise)
        R_meas = np.eye(self.ny) * 0.01  # Trust measurements (lower = faster convergence)

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
        self.P = (np.eye(nx + nu) - K @ C_aug) @ P_pred

        # Extract updated estimates
        self.x_hat = x_aug_updated[:nx]
        self.d_hat = x_aug_updated[nx:]  # Input disturbance (nu x 1)

        # Log debug information
        self.debug_log.append({
            'y_meas': y_meas.copy(),
            'y_pred': y_pred.copy(),
            'innovation': innovation.copy(),
            'x_hat_before': x_aug[:nx].copy(),
            'x_hat_after': self.x_hat.copy(),
            'd_hat_before': x_aug[nx:].copy(),
            'd_hat_after': self.d_hat.copy(),
            'u_prev': u_prev.copy(),
            'K': K.copy()
        })

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
        self._kalman_update(y_meas, self.u_prev)  # Use actual applied control from previous step

        # Set initial state to MEASURED state (not estimated!)
        # The Kalman filter is only used to estimate the INPUT disturbance d_hat
        self.x0_param.value = x0

        # Set estimated INPUT disturbance for MPC and steady-state target solver
        self.d_param.value = self.d_hat
        self.d_param_target.value = self.d_hat

        # Compute steady-state target if reference is given
        if ref is not None:
            self.ref_param.value = ref

            # Solve steady-state target problem (now includes disturbance!)
            try:
                self.target_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as e:
                print(f"Target solve failed: {e}")
                self.x_ref_param.value = np.zeros(self.nx)
                self.u_ref_param.value = np.zeros(self.nu)
            else:
                if self.target_ocp.status in ["optimal", "optimal_inaccurate"]:
                    # Use computed steady-state targets
                    # Target solver accounts for disturbance: xs = A*xs + B*us + B*d
                    self.x_ref_param.value = self.xs_var.value
                    self.u_ref_param.value = self.us_var.value

                    # Debug: Log target solver results every 100 steps
                    if len(self.debug_log) % 100 == 0:
                        xs_computed = self.xs_var.value[0] if self.xs_var.value is not None else 0.0
                        us_computed = self.us_var.value[0] if self.us_var.value is not None else 0.0
                        print(f"[Step {len(self.debug_log)}] Target: xs={xs_computed:.4f}, us={us_computed:.4f}, d={self.d_hat[0]:.4f}, ratio={us_computed/self.d_hat[0] if abs(self.d_hat[0])>1e-6 else 0:.4f}")
                else:
                    print(f"Warning: Target status = {self.target_ocp.status}")
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

        # Store applied control for next Kalman filter iteration
        self.u_prev = u0.copy()

        return u0, x_traj, u_traj

    def estimate_parameters(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Get current disturbance estimate (for logging/debugging).
        This method is called by the wrapper but we don't need to do anything here.
        """
        pass

    def save_debug_log(self, filename: str):
        """Save debug log to CSV file for analysis."""
        import csv

        if not self.debug_log:
            print("No debug data to save")
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'step', 'y_meas', 'y_pred', 'innovation',
                'x_hat_before', 'x_hat_after', 'd_hat_before', 'd_hat_after',
                'u_prev', 'K_x', 'K_d'
            ])

            # Data
            for i, log in enumerate(self.debug_log):
                writer.writerow([
                    i,
                    log['y_meas'][0],
                    log['y_pred'][0],
                    log['innovation'][0],
                    log['x_hat_before'][0],
                    log['x_hat_after'][0],
                    log['d_hat_before'][0],
                    log['d_hat_after'][0],
                    log['u_prev'][0],
                    log['K'][0, 0],  # Kalman gain for state
                    log['K'][1, 0]   # Kalman gain for disturbance
                ])

        print(f"Debug log saved to {filename}")
