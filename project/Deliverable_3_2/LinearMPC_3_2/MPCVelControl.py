import numpy as np
from LinearMPC_3_2.MPCControl_xvel import MPCControl_xvel
from LinearMPC_3_2.MPCControl_yvel import MPCControl_yvel
from LinearMPC_3_2.MPCControl_zvel import MPCControl_zvel
from LinearMPC_3_2.MPCControl_roll import MPCControl_roll


class MPCVelControl:
    """
    Wrapper class for all 4 MPC velocity controllers.

    Combines:
    - MPCControl_xvel: Controls [wy, beta, vx] via d2
    - MPCControl_yvel: Controls [wx, alpha, vy] via d1
    - MPCControl_zvel: Controls [vz] via Pavg
    - MPCControl_roll: Controls [wz, gamma] via Pdiff

    Full state vector: x = [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
    Full input vector: u = [d1, d2, Pavg, Pdiff]
    """

    def __init__(
        self,
        mpc_xvel: MPCControl_xvel,
        mpc_yvel: MPCControl_yvel,
        mpc_zvel: MPCControl_zvel,
        mpc_roll: MPCControl_roll,
    ):
        """
        Initialize with 4 MPC controllers.

        Args:
            mpc_xvel: X velocity MPC controller
            mpc_yvel: Y velocity MPC controller
            mpc_zvel: Z velocity MPC controller
            mpc_roll: Roll MPC controller
        """
        self.mpc_xvel = mpc_xvel
        self.mpc_yvel = mpc_yvel
        self.mpc_zvel = mpc_zvel
        self.mpc_roll = mpc_roll

    @staticmethod
    def new_controller(rocket, Ts: float, H: float):
        """
        Create a new MPCVelControl instance with all 4 subsystem controllers.

        Args:
            rocket: Rocket object with linearized dynamics
            Ts: Sampling time (s)
            H: MPC horizon (s)

        Returns:
            MPCVelControl instance
        """
        # Get trim point and linearized dynamics from rocket
        xs, us = rocket.trim()  # Trim state and input
        lti_sys = rocket.linearize_sys(xs, us)  # Linearize at trim point
        A, B = lti_sys.A, lti_sys.B

        # Create all 4 MPC controllers
        mpc_xvel = MPCControl_xvel(A, B, xs, us, Ts, H)
        mpc_yvel = MPCControl_yvel(A, B, xs, us, Ts, H)
        mpc_zvel = MPCControl_zvel(A, B, xs, us, Ts, H)
        mpc_roll = MPCControl_roll(A, B, xs, us, Ts, H)

        return MPCVelControl(mpc_xvel, mpc_yvel, mpc_zvel, mpc_roll)

    def get_u(
        self,
        t: float,
        x: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get control input from all 4 MPC controllers.

        Args:
            t: Current time (not used, but required by simulate_control)
            x: Current full state [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
            x_target: Target full state (optional, for tracking)
            u_target: Target input (optional, not used in velocity control)

        Returns:
            u: Control input [d1, d2, Pavg, Pdiff]
            x_ol: Open-loop state trajectory (12 x N+1)
            u_ol: Open-loop input trajectory (4 x N)
            t_ol: Open-loop time trajectory (N+1,)
        """
        # Initialize outputs
        u = np.zeros(4)
        N = self.mpc_xvel.N  # All controllers have same horizon
        x_ol = np.zeros((12, N + 1))
        u_ol = np.zeros((4, N))
        t_ol = np.linspace(t, t + N * self.mpc_xvel.Ts, N + 1)

        # Safety margin for numerical precision (0.1%)
        eps = 1e-3

        # Extract delta states (x - xs) for each controller
        # Y velocity controller: [wx, alpha, vy] -> d1
        x0_yvel = x[self.mpc_yvel.x_ids] - self.mpc_yvel.xs
        x_target_yvel = None
        if x_target is not None:
            x_target_yvel = x_target[self.mpc_yvel.x_ids] - self.mpc_yvel.xs
        u_yvel, x_traj_yvel, u_traj_yvel = self.mpc_yvel.get_u(x0_yvel, x_target_yvel)
        u[0] = np.clip(u_yvel[0] + self.mpc_yvel.us[0], -0.262 + eps, 0.262 - eps)  # d1

        # X velocity controller: [wy, beta, vx] -> d2
        x0_xvel = x[self.mpc_xvel.x_ids] - self.mpc_xvel.xs
        x_target_xvel = None
        if x_target is not None:
            x_target_xvel = x_target[self.mpc_xvel.x_ids] - self.mpc_xvel.xs
        u_xvel, x_traj_xvel, u_traj_xvel = self.mpc_xvel.get_u(x0_xvel, x_target_xvel)
        u[1] = np.clip(u_xvel[0] + self.mpc_xvel.us[0], -0.262 + eps, 0.262 - eps)  # d2

        # Z velocity controller: [vz] -> Pavg
        x0_zvel = x[self.mpc_zvel.x_ids] - self.mpc_zvel.xs
        x_target_zvel = None
        if x_target is not None:
            x_target_zvel = x_target[self.mpc_zvel.x_ids] - self.mpc_zvel.xs
        u_zvel, x_traj_zvel, u_traj_zvel = self.mpc_zvel.get_u(x0_zvel, x_target_zvel)
        u[2] = np.clip(u_zvel[0] + self.mpc_zvel.us[0], 40.0 + eps, 80.0 - eps)  # Pavg

        # Roll controller: [wz, gamma] -> Pdiff
        x0_roll = x[self.mpc_roll.x_ids] - self.mpc_roll.xs
        x_target_roll = None
        if x_target is not None:
            x_target_roll = x_target[self.mpc_roll.x_ids] - self.mpc_roll.xs
        u_roll, x_traj_roll, u_traj_roll = self.mpc_roll.get_u(x0_roll, x_target_roll)
        u[3] = np.clip(u_roll[0] + self.mpc_roll.us[0], -20.0 + eps, 20.0 - eps)  # Pdiff

        # Assemble open-loop trajectories from all controllers
        # State trajectories (convert from delta back to absolute)
        if x_traj_yvel is not None:
            x_ol[self.mpc_yvel.x_ids, :] = x_traj_yvel + self.mpc_yvel.xs.reshape(-1, 1)
        if x_traj_xvel is not None:
            x_ol[self.mpc_xvel.x_ids, :] = x_traj_xvel + self.mpc_xvel.xs.reshape(-1, 1)
        if x_traj_zvel is not None:
            x_ol[self.mpc_zvel.x_ids, :] = x_traj_zvel + self.mpc_zvel.xs.reshape(-1, 1)
        if x_traj_roll is not None:
            x_ol[self.mpc_roll.x_ids, :] = x_traj_roll + self.mpc_roll.xs.reshape(-1, 1)

        # Input trajectories (convert from delta back to absolute)
        if u_traj_yvel is not None:
            u_ol[0, :] = u_traj_yvel[0, :] + self.mpc_yvel.us[0]
        if u_traj_xvel is not None:
            u_ol[1, :] = u_traj_xvel[0, :] + self.mpc_xvel.us[0]
        if u_traj_zvel is not None:
            u_ol[2, :] = u_traj_zvel[0, :] + self.mpc_zvel.us[0]
        if u_traj_roll is not None:
            u_ol[3, :] = u_traj_roll[0, :] + self.mpc_roll.us[0]

        return u, x_ol, u_ol, t_ol

    def estimate_parameters(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Placeholder for parameter estimation (used in Part 5).

        Args:
            x: State vector
            u: Input vector
        """
        # Forward to all controllers
        self.mpc_xvel.estimate_parameters(x, u)
        self.mpc_yvel.estimate_parameters(x, u)
        self.mpc_zvel.estimate_parameters(x, u)
        self.mpc_roll.estimate_parameters(x, u)
