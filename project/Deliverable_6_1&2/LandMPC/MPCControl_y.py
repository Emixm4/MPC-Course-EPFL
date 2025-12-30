import numpy as np
import cvxpy as cp

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    """
    Nominal MPC controller for Y position subsystem.

    States: [wx, alpha, vy, y]
    Input: [d1]

    Uses soft constraints (no hard state constraints) and no terminal constraint.
    """
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    # CVXPY variables
    x_var: cp.Variable
    u_var: cp.Variable
    x0_param: cp.Parameter
    x_target_param: cp.Parameter
    u_target_param: cp.Parameter

    def _setup_controller(self) -> None:
        """
        Setup nominal MPC for y-position tracking with soft constraints.
        """
        # Cost matrices - higher penalty on alpha to keep tilt small
        Q = np.diag([1.0, 100.0, 1.0, 10.0])  # wx, alpha, vy, y - alpha penalty increased
        R = np.diag([1.0])                     # d1 - increased to reduce aggressive control

        # Setup MPC optimization problem
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)

        # Default targets
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        # Build cost function (stage cost only, no terminal cost)
        cost = 0
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_target_param
            du = self.u_var[:, k] - self.u_target_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)

        # Build constraints
        constraints = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Input constraints: -15 <= d1 <= 15 degrees (converted to radians)
        # Add small margin (0.5°) to avoid boundary violations due to numerical precision
        u_min = np.array([np.deg2rad(-14.5)]) - self.us
        u_max = np.array([np.deg2rad(14.5)]) - self.us

        # Note: State constraints on alpha are NOT enforced as hard constraints
        # The cost function penalizes alpha deviations from target (which is zero)
        # This keeps the MPC feasible even from far initial conditions

        # Dynamics and input bounds
        for k in range(self.N):
            # Dynamics
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            )

            # Hard input constraints only
            constraints.append(self.u_var[:, k] >= u_min)
            constraints.append(self.u_var[:, k] <= u_max)

        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve nominal MPC for y-position tracking.

        Args:
            x0: Current state in ABSOLUTE coordinates
            x_target: Target state in delta coordinates
            u_target: Target input in delta coordinates

        Returns:
            u0: Optimal control input in ABSOLUTE coordinates
            x_traj: Predicted state trajectory
            u_traj: Predicted input trajectory
        """
        # Convert initial state from ABSOLUTE to DELTA coordinates
        x0_delta = x0 - self.xs

        # Set initial condition in delta coordinates
        self.x0_param.value = x0_delta

        # Set targets (also convert to DELTA coordinates)
        if x_target is not None:
            self.x_target_param.value = x_target - self.xs
        else:
            self.x_target_param.value = np.zeros(self.nx)

        if u_target is not None:
            self.u_target_param.value = u_target - self.us
        else:
            self.u_target_param.value = np.zeros(self.nu)

        # Solve MPC
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return self.us, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: MPC status = {self.ocp.status}")
            return self.us, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        # Extract solution (in delta coordinates)
        u0_delta = self.u_var[:, 0].value
        x_traj = self.x_var.value
        u_traj = self.u_var.value

        # Convert to absolute coordinates for constraint checking and simulation
        if u0_delta is not None:
            u0 = u0_delta + self.us
        else:
            u0 = self.us  # Return steady-state control if solve failed

        if x_traj is None:
            x_traj = np.zeros((self.nx, self.N + 1))
        if u_traj is None:
            u_traj = np.zeros((self.nu, self.N))

        return u0, x_traj, u_traj
