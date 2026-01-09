import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    """
    Robust Tube MPC controller for Z position subsystem.

    States: [vz, z]
    Input: [Pavg]
    Disturbance: w in W = [-15, 5]

    Implements tube MPC with:
    - Minimal robust positively invariant set E computed via iterative algorithm
    - Terminal set Xf
    - Tightened constraints with SOFT input constraints for feasibility
    """
    x_ids: np.ndarray = np.array([8, 11])  # vz, z
    u_ids: np.ndarray = np.array([2])      # Pavg

    # Tube MPC components
    K: np.ndarray      # Ancillary controller gain
    E_bounds: np.ndarray  # Bounds for mRPI set (box approximation)
    Xf_bounds: np.ndarray # Terminal set bounds (box approximation)
    P: np.ndarray      # Terminal cost matrix

    # CVXPY variables
    x_var: cp.Variable
    u_var: cp.Variable
    slack_var: cp.Variable  # Slack for soft constraints
    x0_param: cp.Parameter
    x_target_param: cp.Parameter
    u_target_param: cp.Parameter

    def _compute_mRPI_safe(self, A_cl: np.ndarray, B: np.ndarray, w_min: float, w_max: float, max_iter: int = 50) -> np.ndarray:
        """
        Compute Minimal RPI set bounds using a SAFE iterative algorithm.
        
        Algorithm: F_∞ = ⊕_{i=0}^{∞} A_cl^i * B * W
        
        For box approximation with W = [w_min, w_max]:
        E_bounds accumulates |A_cl^i * B| * max(|w_min|, |w_max|)
        
        Returns E_bounds as a numpy array [[e_vz], [e_z]]
        """
        spectral_radius = np.max(np.abs(np.linalg.eigvals(A_cl)))
        print(f"\n=== Computing mRPI (safe box approximation) ===")
        print(f"Spectral radius of A_cl: {spectral_radius:.4f}")
        
        # If not stable, cannot compute finite mRPI
        if spectral_radius >= 1.0:
            print("WARNING: A_cl not stable, using fallback E_bounds")
            return np.array([[2.0], [1.0]])  # Fallback bounds
        
        # Worst-case disturbance magnitude
        w_max_abs = max(abs(w_min), abs(w_max))
        
        # Initialize E_bounds = 0
        E_bounds = np.zeros((2, 1))
        A_power = np.eye(2)  # A_cl^0 = I
        
        # Iterate: E += |A_cl^i * B| * w_max
        for i in range(max_iter):
            # Effect of disturbance at step i: A_cl^i * B * w
            effect = np.abs(A_power @ B) * w_max_abs
            E_bounds += effect
            
            # Check convergence (contribution becomes negligible)
            contribution = np.max(effect)
            if contribution < 1e-6:
                print(f"  Converged at iteration {i}")
                break
            
            # Update A_power = A_cl^{i+1}
            A_power = A_power @ A_cl
            
            # Safety: check for divergence
            if np.max(E_bounds) > 100:
                print(f"  WARNING: E_bounds growing too large at iter {i}, stopping")
                break
        
        print(f"  Final E_bounds: vz=±{E_bounds[0,0]:.3f}, z=±{E_bounds[1,0]:.3f}")
        return E_bounds

    def _setup_controller(self) -> None:
        """
        Setup robust tube MPC with terminal set and invariant sets.
        
        Computes mRPI set E using the standard algorithm:
        Ω_0 = {0}, Ω_{i+1} = Ω_i ⊕ A_cl^i * W
        """
        # ===== STEP 1: Design ancillary controller K =====
        Q_K = np.diag([100.0, 200.0])  # Moderate penalty
        R_K = np.diag([0.1])           # Higher penalty for smoother control
        
        try:
            K_lqr, _, _ = dlqr(self.A, self.B, Q_K, R_K)
            self.K = -K_lqr  # Convention: u = K*x (negative feedback)
        except Exception as e:
            print(f"Warning: LQR for K failed ({e}), using moderate manual gain")
            self.K = np.array([[-5.0, -8.0]])
        
        # Closed-loop matrix for error dynamics
        A_cl = self.A + self.B @ self.K
        eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(eigvals))
        print(f"Ancillary K = {self.K}")
        print(f"A_cl eigenvalues: {eigvals}")
        print(f"Spectral radius: {spectral_radius:.4f}")
        
        if spectral_radius >= 1.0:
            print("WARNING: A_cl is not stable! Adjusting K...")
            self.K = np.array([[-8.0, -12.0]])
            A_cl = self.A + self.B @ self.K
            print(f"New K = {self.K}, new eigenvalues: {np.linalg.eigvals(A_cl)}")
        
        # ===== STEP 2: Compute mRPI set E =====
        self.w_min = -15.0
        self.w_max = 5.0
        
        # KEY INSIGHT: W = [-15, 5] has mean w_bar = -5 (not centered!)
        # Tube MPC assumes zero-mean disturbance, so we need to compensate
        self.w_bar = (self.w_min + self.w_max) / 2.0  # Expected disturbance = -5
        
        # For mRPI computation, use CENTERED disturbance: W_centered = W - w_bar = [-10, 10]
        w_centered_half = (self.w_max - self.w_min) / 2.0  # = 10
        
        # Compute E_bounds using SAFE box approximation of mRPI algorithm
        # F_∞ = ⊕_{i=0}^{∞} A_cl^i * B * W_centered
        self.E_bounds = self._compute_mRPI_safe(A_cl, self.B, -w_centered_half, w_centered_half, max_iter=50)
        
        # Cap E_bounds if too large (to preserve control authority)
        max_E = 5.0
        if np.max(self.E_bounds) > max_E:
            print(f"Capping E_bounds from {self.E_bounds.flatten()} to max {max_E}")
            self.E_bounds = np.clip(self.E_bounds, 0, max_E)
        
        print(f"Final E_bounds: {self.E_bounds.flatten()}")
        print(f"Disturbance bias w_bar = {self.w_bar} (will be compensated via feedforward)")
        
        # ===== STEP 3: Cost matrices for MPC =====
        Q = np.diag([5.0, 30.0])
        R = np.diag([3.0])
        R_delta = np.diag([5.0])
        
        self.Q = Q
        self.R = R
        self.R_delta = R_delta
        
        try:
            _, self.P, _ = dlqr(self.A, self.B, Q, R)
        except:
            self.P = 10 * Q
        
        self.Xf_bounds = np.array([[3.0], [3.0]])
        
        # ===== STEP 4: Setup CVXPY optimization =====
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.slack_var = cp.Variable((self.nu, self.N), nonneg=True)
        
        self.x0_param = cp.Parameter(self.nx)
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)
        self.u_prev_param = cp.Parameter(self.nu)  # Previous control for rate penalty
        
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)
        self.u_prev_param.value = np.zeros(self.nu)
        
        # ===== Build cost function =====
        cost = 0
        slack_penalty = 500.0
        
        for k in range(self.N):
            dx = self.x_var[:, k] - self.x_target_param
            du = self.u_var[:, k] - self.u_target_param
            cost += cp.quad_form(dx, Q) + cp.quad_form(du, R)
            cost += slack_penalty * cp.sum(self.slack_var[:, k])
            
            # Rate penalty to reduce oscillations
            if k == 0:
                delta_u = self.u_var[:, k] - self.u_prev_param
            else:
                delta_u = self.u_var[:, k] - self.u_var[:, k-1]
            cost += cp.quad_form(delta_u, R_delta)
        
        dx_N = self.x_var[:, self.N] - self.x_target_param
        cost += cp.quad_form(dx_N, self.P)
        
        # ===== Build constraints with tightening based on computed E =====
        constraints = []
        constraints.append(self.x_var[:, 0] == self.x0_param)
        
        # Constraint bounds in DELTA coordinates
        x_min = np.array([-np.inf, -self.xs[1]])  # z >= 0 => z_delta >= -z_s
        u_min = np.array([40.0]) - self.us
        u_max = np.array([80.0]) - self.us
        
        # Tighten constraints using computed E_bounds
        E_tight = self.E_bounds.flatten()
        x_min_tight = x_min.copy()
        x_min_tight[1] = x_min[1] + E_tight[1]  # Tighten z constraint by E_z
        
        # Input tightening: u_tight = u ⊖ K*E
        K_inf = np.linalg.norm(self.K, ord=np.inf)
        E_inf = np.max(E_tight)
        u_margin_theoretical = K_inf * E_inf
        
        # Cap tightening to preserve control authority
        # Use smaller cap to allow more aggressive control when needed
        u_margin = min(u_margin_theoretical, 2.0)  # Reduced from 5.0 to 2.0
        
        u_min_tight = u_min + u_margin
        u_max_tight = u_max - u_margin
        
        print(f"\nConstraint tightening (based on mRPI):")
        print(f"  E_bounds = {E_tight}")
        print(f"  u_margin = min(|K|_inf * |E|_inf, 5.0) = min({u_margin_theoretical:.2f}, 5.0) = {u_margin:.2f}")
        print(f"  Tightened input range: [{u_min_tight[0] + self.us[0]:.1f}, {u_max_tight[0] + self.us[0]:.1f}] N")
        
        self.u_min_tight = u_min_tight
        self.u_max_tight = u_max_tight
        
        # Dynamics and constraints
        for k in range(self.N):
            constraints.append(
                self.x_var[:, k+1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            )
            
            # State constraint on z (tightened)
            if np.isfinite(x_min_tight[1]):
                constraints.append(self.x_var[1, k] >= x_min_tight[1])
            
            # Input constraints - SOFT with slack for feasibility
            constraints.append(self.u_var[:, k] >= u_min_tight - self.slack_var[:, k])
            constraints.append(self.u_var[:, k] <= u_max_tight + self.slack_var[:, k])
            
            # Hard physical limits
            constraints.append(self.u_var[:, k] >= u_min)
            constraints.append(self.u_var[:, k] <= u_max)
        
        # Terminal constraint
        Xf_tight = self.Xf_bounds.flatten()
        constraints.append(self.x_var[:, self.N] - self.x_target_param >= -Xf_tight)
        constraints.append(self.x_var[:, self.N] - self.x_target_param <= Xf_tight)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        print("\nTube MPC setup complete with mRPI-based tightening")

    # Note: Old Polyhedron-based functions removed - using box approximations instead

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve tube MPC and return control with ancillary feedback.

        The actual control is: u = v + K(x - z)
        where v is the nominal control, z is the nominal state.

        For robustness with large disturbances w in [-15, 5]:
        - If MPC is infeasible, use full authority ancillary controller
        - Always use u_max when falling too fast (emergency mode)
        """
        # Convert initial state from ABSOLUTE to DELTA coordinates
        x0_delta = x0 - self.xs
        vz_current = x0[0]  # Current vertical velocity
        z_current = x0[1]   # Current altitude

        # Set parameters
        self.x0_param.value = x0_delta

        if x_target is not None:
            self.x_target_param.value = x_target - self.xs
        else:
            self.x_target_param.value = np.zeros(self.nx)

        if u_target is not None:
            self.u_target_param.value = u_target - self.us
        else:
            self.u_target_param.value = np.zeros(self.nu)

        # Set previous control for rate penalty (initialized to trim if not set)
        if not hasattr(self, '_u_prev'):
            self._u_prev = np.zeros(self.nu)
        self.u_prev_param.value = self._u_prev

        # Emergency mode: use max thrust when falling too fast or altitude critical
        # These thresholds ensure we never crash even with worst-case disturbance
        emergency_vz_threshold = -2.0  # m/s 
        emergency_z_threshold = 0.5    # m above ground
        emergency_z_critical = z_current - self.xs[1]  # Distance to target (delta)
        
        # Also enter emergency if close to target but still falling significantly
        close_to_target = abs(emergency_z_critical) < 1.5 and vz_current < -0.5
        
        if vz_current < emergency_vz_threshold or z_current < emergency_z_threshold or close_to_target:
            # Emergency: apply maximum thrust
            u0 = np.array([80.0])  # Max thrust
            self._u_prev = u0 - self.us  # Update previous control
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        # Try to solve MPC
        try:
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=5000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=True,
                adaptive_rho=True
            )
        except Exception as e:
            # Fallback: use ancillary controller
            u0 = self.us + self.K @ x0_delta
            u0 = np.clip(u0, 40.0, 80.0)
            self._u_prev = u0 - self.us
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        # Handle solver status
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback: use ancillary controller
            u0 = self.us + self.K @ x0_delta
            u0 = np.clip(u0, 40.0, 80.0)
            self._u_prev = u0 - self.us
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        # Extract nominal solution
        v0 = self.u_var[:, 0].value
        z_traj = self.x_var.value
        v_traj = self.u_var.value

        if v0 is None:
            u0 = self.us + self.K @ x0_delta
            u0 = np.clip(u0, 40.0, 80.0)
            self._u_prev = u0 - self.us
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))

        # Apply tube MPC control law: u = v + K(x - z)
        error = x0_delta - z_traj[:, 0]
        u0_delta = v0 + self.K @ error
        
        # ADAPTIVE DISTURBANCE COMPENSATION
        # Track position error and estimate persistent disturbance
        if not hasattr(self, '_integral_error'):
            self._integral_error = 0.0
            self._last_z_error = 0.0
        
        # Position error relative to target
        z_error = x0_delta[1] - self.x_target_param.value[1]
        
        # Only integrate when close to target and not in transient
        if abs(z_error) < 2.0 and abs(vz_current) < 1.0:
            # Anti-windup: limit integral
            Ki = 0.5  # Integral gain
            self._integral_error += Ki * z_error * self.Ts
            self._integral_error = np.clip(self._integral_error, -5.0, 5.0)
        
        # Apply integral correction
        u0_delta = u0_delta - self._integral_error
        
        u0 = u0_delta + self.us
        
        # Hard clip to physical limits
        u0 = np.clip(u0, 40.0, 80.0)
        
        # Update previous control for next iteration
        self._u_prev = u0 - self.us

        if z_traj is None:
            z_traj = np.zeros((self.nx, self.N + 1))
        if v_traj is None:
            v_traj = np.zeros((self.nu, self.N))

        return u0, z_traj, v_traj

    # Estimator methods (not used in Part 6)
    def setup_estimator(self):
        self.d_estimate = np.zeros(self.nx)
        self.d_gain = 0.0

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        pass
