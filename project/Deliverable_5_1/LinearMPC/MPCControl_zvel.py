import numpy as np
from LinearMPC.MPCControl_base import MPCControl_base
import cvxpy as cp


class MPCControl_zvel(MPCControl_base):
    """
    MPC controller for z velocity subsystem with offset-free tracking.

    Augmented system with disturbance estimation:
    - States: [vz, d] where d is estimated disturbance
    - Input: [Pavg]
    - Dynamics: x+ = Ax + Bu + Bd, with d estimated via Kalman filter

    Constraints:
    - z >= 0
    - 40 <= Pavg <= 80
    """

    # State indices: vz=8
    x_ids = np.array([8]) 
    u_ids = np.array([2])  # Pavg

    # Augmented state estimator variables
    x_hat: np.ndarray 
    d_hat: np.ndarray 
    u_prev: np.ndarray

    # Kalman filter covariance
    P: np.ndarray  # Error covariance matrix

    def _get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cost matrices for Z velocity controller.
        """
        Q = np.diag([10.0])  # vz cost
        R = np.diag([0.1])   # Pavg cost (small to allow aggressive control)
        return Q, R

    def _get_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constraint bounds for Z subsystem.
        States: [vz]

        Inputs: [Pavg]
        - 40 <= Pavg <= 80 (percentage)
        """

        x_min = np.array([-np.inf])  
        x_max = np.array([np.inf])  

        # Absolute: 40 <= Pavg <= 80
        # Delta: 40 - us[Pavg] <= delta_Pavg <= 80 - us[Pavg]
        u_min = np.array([40.0]) - self.us
        u_max = np.array([80.0]) - self.us

        return x_min, x_max, u_min, u_max

    def _get_output_matrix(self) -> np.ndarray:
        """
        Selects vz from state [vz].
        """
        return np.array([[1.0]]) # vz

    def __init__(self, A: np.ndarray, B: np.ndarray, xs: np.ndarray, us: np.ndarray, Ts: float, H: float):
        """Initialize offset-free MPC with disturbance estimator."""

        super().__init__(A, B, xs, us, Ts, H)

        # Initialize state and disturbance estimates
        self.x_hat = np.zeros(self.nx) 
        self.d_hat = np.zeros(self.nu)  
        self.u_prev = np.zeros(self.nu) 

        # Initialize Kalman filter error covariance
        self.P = np.eye(self.nx + self.nu) * 1.0

        # Setup augmented MPC with disturbance compensation
        self._setup_offset_free_mpc()

        self._setup_steady_state_target_with_disturbance()

    def _setup_offset_free_mpc(self):
        """
        Setup MPC with input disturbance d.
        Dynamics: x+ = A*x + B*(u + d)
        """

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

        constraints = []

        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics ac input disturbance
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

        _, _, u_min, u_max = self._get_constraints()

        self.xs_var = cp.Variable(self.nx)
        self.us_var = cp.Variable(self.nu)
        self.ref_param = cp.Parameter(self.ny)
        self.d_param_target = cp.Parameter(self.nu) 

        # Default reference and disturbance
        self.ref_param.value = np.zeros(self.ny)
        self.d_param_target.value = np.zeros(self.nu)

        # Cost: minimize input effort
        cost = cp.quad_form(self.us_var, np.eye(self.nu))

        constraints = []

        constraints.append(
            self.xs_var == self.A @ self.xs_var + self.B @ (self.us_var + self.d_param_target)
        )

        # Output matches reference
        constraints.append(self.C @ self.xs_var == self.ref_param)

        constraints.append(self.us_var >= u_min)
        constraints.append(self.us_var <= u_max)

        # Create target optimization problem
        self.target_ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _kalman_update(self, y_meas: np.ndarray, u_prev: np.ndarray):
        """
        Kalman filter update for augmented system ac constant input disturbance.

        """
        nx = self.nx
        nu = self.nu

        # Augmented system matrices for INPUT disturbance
        A_aug = np.block([[self.A, self.B], 
                         [np.zeros((nu, nx)), np.eye(nu)]])
        B_aug = np.block([[self.B],
                         [np.zeros((nu, nu))]])
        C_aug = np.block([[self.C, np.zeros((self.ny, nu))]])

       
        # Tuning strategy:
        # Lower Q_process= trust model dynamics more
        # Lower Q_process= disturbance should be very constant
        # Lower R_meas = trust measurements more
        Q_process = np.eye(nx + nu) * 0.001  
        Q_process[nx:, nx:] *= 0.0001  
        R_meas = np.eye(self.ny) * 0.01 

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
        y_meas = self.C @ x0  
        self._kalman_update(y_meas, self.u_prev)

        self.x0_param.value = x0

        # Set estimated input dist for MPC and steady-state target solver
        self.d_param.value = self.d_hat
        self.d_param_target.value = self.d_hat

        # Compute steady-state target
        if ref is not None:
            self.ref_param.value = ref

            try:
                self.target_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as e:
                print(f"Target solve failed: {e}")
                self.x_ref_param.value = np.zeros(self.nx)
                self.u_ref_param.value = np.zeros(self.nu)
            else:
                if self.target_ocp.status in ["optimal", "optimal_inaccurate"]:
                    # Use computed steady-state targets
                    self.x_ref_param.value = self.xs_var.value
                    self.u_ref_param.value = self.us_var.value
                else:
                    print(f"Warning: Target status = {self.target_ocp.status}")
                    self.x_ref_param.value = np.zeros(self.nx)
                    self.u_ref_param.value = np.zeros(self.nu)
        else:
            self.x_ref_param.value = np.zeros(self.nx)
            self.u_ref_param.value = np.zeros(self.nu)

        # Solve MPC
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

        # Solution
        u0 = self.u_var[:, 0].value
        x_traj = self.x_var.value
        u_traj = self.u_var.value

        if u0 is None:
            u0 = np.zeros(self.nu)
        if x_traj is None:
            x_traj = np.zeros((self.nx, self.N + 1))
        if u_traj is None:
            u_traj = np.zeros((self.nu, self.N))

        self.u_prev = u0.copy()

        return u0, x_traj, u_traj

    def estimate_parameters(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Get current disturbance estimate (for logging/debugging).
        """
        pass
