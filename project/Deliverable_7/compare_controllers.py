"""
Comparison Script: Linear MPC (6.2) vs NMPC (7.1)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

script_dir = Path(__file__).parent
deliverable_7_dir = script_dir  
project_dir = script_dir.parent

# paths for imports
sys.path.insert(0, str(project_dir))  # For src.rocket
sys.path.insert(0, str(deliverable_7_dir))  # NMPC
sys.path.insert(0, str(project_dir / "Deliverable_6_1&2"))  #Linear MPC

from src.rocket import Rocket

import sys
import os
sys.path.insert(0, os.path.join(str(deliverable_7_dir), "LandMPC"))
import nmpc_land as nmpc_d7_module
NmpcCtrl = nmpc_d7_module.NmpcCtrl
sys.path.pop(0)

# Linear MPC from Deliverable 6
from LandMPC.MPCLandControl import MPCLandControl as LinearMPCController


def simulate_controller(rocket, controller, sim_time, H, x0, controller_name):
    """
    Simulate a controller and collect metrics.

    Returns:
        dict with trajectory data and performance metrics
    """
    start_time = time.time()

    # Run simulation
    t_cl, x_cl, u_cl, t_ol, x_ol, u_ol = rocket.simulate_land(
        controller, sim_time, H, x0
    )

    total_sim_time = time.time() - start_time
    
    # Get target state
    xs = controller.xs
    us = controller.us
    
    # Compute metrics
    metrics = {
        'name': controller_name,
        't_cl': t_cl,
        'x_cl': x_cl,
        'u_cl': u_cl,
        't_ol': t_ol,
        'x_ol': x_ol,
        'u_ol': u_ol,
        'xs': xs,
        'us': us,
        'total_sim_time': total_sim_time
    }
    
    # Final state errors
    pos_error = np.linalg.norm(x_cl[9:12, -1] - xs[9:12])
    vel_error = np.linalg.norm(x_cl[6:9, -1] - xs[6:9])
    angle_error = np.abs(x_cl[5, -1] - xs[5])
    
    metrics['final_pos_error'] = pos_error
    metrics['final_vel_error'] = vel_error
    metrics['final_angle_error'] = np.rad2deg(angle_error)
    
    pos_tol = 0.02
    vel_tol = 0.05
    angle_tol = np.deg2rad(1)
    ang_vel_tol = np.deg2rad(2)
    
    pos_err_traj = np.linalg.norm(x_cl[9:12, :] - xs[9:12, np.newaxis], axis=0)
    vel_err_traj = np.linalg.norm(x_cl[6:9, :] - xs[6:9, np.newaxis], axis=0)
    angle_err_traj = np.abs(x_cl[5, :] - xs[5])
    ang_vel_err_traj = np.linalg.norm(x_cl[0:3, :] - xs[0:3, np.newaxis], axis=0)
    
    settled = (pos_err_traj < pos_tol) & (vel_err_traj < vel_tol) & \
              (angle_err_traj < angle_tol) & (ang_vel_err_traj < ang_vel_tol)
    
    settled_indices = np.where(settled)[0]
    if len(settled_indices) > 0:
        metrics['settling_time'] = t_cl[settled_indices[0]]
    else:
        metrics['settling_time'] = sim_time
    
    # Constraint violations
    metrics['min_z'] = np.min(x_cl[11, :])
    metrics['max_beta'] = np.rad2deg(np.max(np.abs(x_cl[4, :])))
    metrics['max_d1'] = np.rad2deg(np.max(np.abs(u_cl[0, :])))
    metrics['max_d2'] = np.rad2deg(np.max(np.abs(u_cl[1, :])))
    metrics['min_Pavg'] = np.min(u_cl[2, :])
    metrics['max_Pavg'] = np.max(u_cl[2, :])
    metrics['max_Pdiff'] = np.max(np.abs(u_cl[3, :]))
    
    # Check constraint
    metrics['ground_satisfied'] = metrics['min_z'] >= 0
    metrics['beta_satisfied'] = metrics['max_beta'] <= 80.0
    metrics['servo_satisfied'] = (metrics['max_d1'] <= 15.0) and (metrics['max_d2'] <= 15.0)
    metrics['throttle_satisfied'] = (metrics['min_Pavg'] >= 40.0) and (metrics['max_Pavg'] <= 80.0)
    metrics['pdiff_satisfied'] = metrics['max_Pdiff'] <= 20.0
    
    # Control effort
    dt = t_cl[1] - t_cl[0]
    metrics['control_effort'] = np.sum(u_cl**2) * dt
    
    # Average computation time per step
    metrics['avg_comp_time'] = total_sim_time / len(t_cl)
   
    return metrics


def plot_comparison(linear_metrics, nmpc_metrics, save_dir=None):
    """
    Generate position and attitude comparison plots.
    """
    # Create figure directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Figure 1: Position Trajectories
    fig1, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig1.suptitle('Position Comparison: Linear MPC vs NMPC', fontsize=14, fontweight='bold')
    
    positions = ['x', 'y', 'z']
    colors_linear = ['blue', 'orange', 'green']
    colors_nmpc = ['red', 'purple', 'brown']
    
    for i, (pos, ax) in enumerate(zip(positions, axs)):
        # Linear MPC
        ax.plot(linear_metrics['t_cl'], linear_metrics['x_cl'][9+i, :], 
                label=f'{pos} (Linear MPC)', color=colors_linear[i], linewidth=2)
        # NMPC
        ax.plot(nmpc_metrics['t_cl'], nmpc_metrics['x_cl'][9+i, :], 
                label=f'{pos} (NMPC)', color=colors_nmpc[i], linewidth=2, linestyle='--')
        # Target
        ax.axhline(y=linear_metrics['xs'][9+i], color='black', linestyle=':', 
                   linewidth=1.5, label=f'{pos} target')
        
        ax.set_ylabel(f'{pos} [m]', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axs[-1].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()

    if save_dir:
        fig1.savefig(save_dir / 'position_comparison.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Angles Comparison
    fig2, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig2.suptitle('Attitude Comparison: Linear MPC vs NMPC', fontsize=14, fontweight='bold')
    
    angles = ['α (pitch)', 'β (yaw)', 'γ (roll)']
    
    for i, (angle, ax) in enumerate(zip(angles, axs)):
        ax.plot(linear_metrics['t_cl'], np.rad2deg(linear_metrics['x_cl'][3+i, :]), 
                label=f'{angle} (Linear MPC)', color=colors_linear[i], linewidth=2)
        ax.plot(nmpc_metrics['t_cl'], np.rad2deg(nmpc_metrics['x_cl'][3+i, :]), 
                label=f'{angle} (NMPC)', color=colors_nmpc[i], linewidth=2, linestyle='--')
        ax.axhline(y=np.rad2deg(linear_metrics['xs'][3+i]), color='black', 
                   linestyle=':', linewidth=1.5, label=f'{angle} target')
        
        ax.set_ylabel(f'{angle} [deg]', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axs[-1].set_xlabel('Time [s]', fontsize=11)
    plt.tight_layout()

    if save_dir:
        fig2.savefig(save_dir / 'attitude_comparison.png', dpi=300, bbox_inches='tight')

    plt.close('all')




def main():
    """
    Main comparison script.
    """
    # Setup
    Ts = 1/20  # 20 Hz
    sim_time = 15  # seconds
    H = 4.0  # horizon time

    rocket_params_path = str(project_dir / "rocket.yaml")

    # Initial state: (3, 2, 10, 30°)
    x0 = np.array([0, 0, 0,  # angular velocities
                   0, 0, np.deg2rad(30),  # angles (alpha, beta, gamma/roll)
                   0, 0, 0,  # linear velocities
                   3, 2, 10])  # positions (x, y, z)

    # Target state: (1, 0, 3, 0°)
    x_ref = np.array([0.]*9 + [1., 0., 3.])

    # Linear MPC
    rocket_linear = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket_linear.mass = 1.7
    rocket_linear.controller_type = 'MPCLandControl'

    linear_ctrl = LinearMPCController().new_controller(rocket_linear, Ts, H, x_ref)

    linear_metrics = simulate_controller(
        rocket_linear, linear_ctrl, sim_time, H, x0, "LINEAR MPC (6.2)"
    )

    # NMPC
    rocket_nmpc = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket_nmpc.mass = 1.7
    rocket_nmpc.controller_type = 'NmpcCtrl'

    nmpc_ctrl = NmpcCtrl(rocket=rocket_nmpc, Ts=Ts, H=H, x_ref=x_ref)

    nmpc_metrics = simulate_controller(
        rocket_nmpc, nmpc_ctrl, sim_time, H, x0, "NMPC (7.1)"
    )

    # Generate plots
    figures_dir = script_dir / "comparison_figures"
    plot_comparison(linear_metrics, nmpc_metrics, save_dir=figures_dir)


if __name__ == "__main__":
    main()
