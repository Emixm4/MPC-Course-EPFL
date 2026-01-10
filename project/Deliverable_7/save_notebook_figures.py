"""
Save all figures from Deliverable 7.1 notebook to files.
This script runs the NMPC simulation and generates all analysis plots.
"""

import sys
import os

# Change to script directory first
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add parent directory to path
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from src.rocket import Rocket
from LandMPC.nmpc_land import NmpcCtrl

def main():
    # Create output directory
    output_dir = "figures_deliverable_7_1"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {output_dir}/")
    
    # Setup rocket and controller
    rocket_params_path = os.path.join(parent_dir, "rocket.yaml")
    Ts = 1/20
    rocket = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket.mass = 1.7
    rocket.controller_type = 'NmpcCtrl'
    
    # Simulation parameters
    sim_time = 15
    x0 = np.array([0, 0, 0,  # angular velocities
                   0, 0, np.deg2rad(30),  # angles (30° roll)
                   0, 0, 0,  # linear velocities
                   3, 2, 10])  # positions
    x_ref = np.array([0.]*9 + [1., 0., 3.])  # Target: x=1, y=0, z=3
    H = 4.0
    
    print("\nInitializing NMPC controller...")
    nmpc = NmpcCtrl(rocket, Ts=Ts, H=H, x_ref=x_ref)
    xs, us = nmpc.xs, nmpc.us
    
    # === FIGURE 1: Open-loop trajectory ===
    print("\nGenerating open-loop trajectory plot...")
    u0, x_ol, u_ol, t_ol = nmpc.get_u(0.0, x0)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Positions
    axs[0].plot(t_ol, x_ol[9, :], label='x', linewidth=2)
    axs[0].plot(t_ol, x_ol[10, :], label='y', linewidth=2)
    axs[0].plot(t_ol, x_ol[11, :], label='z', linewidth=2)
    axs[0].axhline(y=xs[9], color='r', linestyle='--', alpha=0.7, label='x_target')
    axs[0].axhline(y=xs[10], color='g', linestyle='--', alpha=0.7, label='y_target')
    axs[0].axhline(y=xs[11], color='b', linestyle='--', alpha=0.7, label='z_target')
    axs[0].set_ylabel('Position [m]', fontsize=12)
    axs[0].legend(loc='best')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Open-Loop Optimal Trajectory', fontsize=14, fontweight='bold')
    
    # Angles
    axs[1].plot(t_ol, np.rad2deg(x_ol[3, :]), label='alpha', linewidth=2)
    axs[1].plot(t_ol, np.rad2deg(x_ol[4, :]), label='beta', linewidth=2)
    axs[1].plot(t_ol, np.rad2deg(x_ol[5, :]), label='gamma (roll)', linewidth=2)
    axs[1].set_ylabel('Angles [deg]', fontsize=12)
    axs[1].legend(loc='best')
    axs[1].grid(True, alpha=0.3)
    
    # Inputs
    axs[2].plot(t_ol[:-1], u_ol[0, :], label='d1', linewidth=2)
    axs[2].plot(t_ol[:-1], u_ol[1, :], label='d2', linewidth=2)
    axs[2].plot(t_ol[:-1], u_ol[2, :], label='Pavg', linewidth=2)
    axs[2].plot(t_ol[:-1], u_ol[3, :], label='Pdiff', linewidth=2)
    axs[2].set_xlabel('Time [s]', fontsize=12)
    axs[2].set_ylabel('Inputs', fontsize=12)
    axs[2].legend(loc='best')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_open_loop_trajectory.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 01_open_loop_trajectory.png")
    plt.close()
    
    # === Run closed-loop simulation ===
    print("\nRunning closed-loop simulation...")
    t_cl, x_cl, u_cl, t_ol, x_ol, u_ol = rocket.simulate_land(nmpc, sim_time, H, x0)
    
    # === FIGURE 2: Settling time verification ===
    print("Generating settling time analysis plot...")
    
    # Compute errors
    pos_tol = 0.05  # 5 cm
    vel_tol = 0.01  # 1 cm/s
    angle_tol = np.deg2rad(1)  # 1 degree
    ang_vel_tol = np.deg2rad(2)  # 2 deg/s
    
    pos_error = np.linalg.norm(x_cl[9:12, :] - xs[9:12, np.newaxis], axis=0)
    vel_error = np.linalg.norm(x_cl[6:9, :] - xs[6:9, np.newaxis], axis=0)
    angle_error = np.abs(x_cl[5, :] - xs[5])
    ang_vel_error = np.linalg.norm(x_cl[0:3, :] - xs[0:3, np.newaxis], axis=0)
    
    settled = (pos_error < pos_tol) & (vel_error < vel_tol) & \
              (angle_error < angle_tol) & (ang_vel_error < ang_vel_tol)
    
    settled_indices = np.where(settled)[0]
    settling_time = t_cl[settled_indices[0]] if len(settled_indices) > 0 else sim_time
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    axs[0, 0].plot(t_cl, pos_error, linewidth=2, color='blue')
    axs[0, 0].axhline(y=pos_tol, color='r', linestyle='--', linewidth=2, label='Tolerance')
    axs[0, 0].axvline(x=settling_time, color='g', linestyle='--', linewidth=2, 
                      label=f'Settling: {settling_time:.2f}s')
    axs[0, 0].set_xlabel('Time [s]', fontsize=12)
    axs[0, 0].set_ylabel('Position Error [m]', fontsize=12)
    axs[0, 0].set_title('Position Convergence', fontsize=13, fontweight='bold')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(t_cl, vel_error, linewidth=2, color='blue')
    axs[0, 1].axhline(y=vel_tol, color='r', linestyle='--', linewidth=2, label='Tolerance')
    axs[0, 1].axvline(x=settling_time, color='g', linestyle='--', linewidth=2,
                      label=f'Settling: {settling_time:.2f}s')
    axs[0, 1].set_xlabel('Time [s]', fontsize=12)
    axs[0, 1].set_ylabel('Velocity Error [m/s]', fontsize=12)
    axs[0, 1].set_title('Velocity Convergence', fontsize=13, fontweight='bold')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    axs[1, 0].plot(t_cl, np.rad2deg(angle_error), linewidth=2, color='blue')
    axs[1, 0].axhline(y=np.rad2deg(angle_tol), color='r', linestyle='--', linewidth=2, 
                      label='Tolerance')
    axs[1, 0].axvline(x=settling_time, color='g', linestyle='--', linewidth=2,
                      label=f'Settling: {settling_time:.2f}s')
    axs[1, 0].set_xlabel('Time [s]', fontsize=12)
    axs[1, 0].set_ylabel('Roll Error [deg]', fontsize=12)
    axs[1, 0].set_title('Roll Angle Convergence', fontsize=13, fontweight='bold')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    axs[1, 1].plot(t_cl, np.rad2deg(ang_vel_error), linewidth=2, color='blue')
    axs[1, 1].axhline(y=np.rad2deg(ang_vel_tol), color='r', linestyle='--', linewidth=2,
                      label='Tolerance')
    axs[1, 1].axvline(x=settling_time, color='g', linestyle='--', linewidth=2,
                      label=f'Settling: {settling_time:.2f}s')
    axs[1, 1].set_xlabel('Time [s]', fontsize=12)
    axs[1, 1].set_ylabel('Angular Velocity Error [deg/s]', fontsize=12)
    axs[1, 1].set_title('Angular Velocity Convergence', fontsize=13, fontweight='bold')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_settling_time_verification.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 02_settling_time_verification.png")
    plt.close()
    
    # === FIGURE 3: Constraint verification ===
    print("Generating constraint verification plot...")
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # z constraint
    axs[0, 0].plot(t_cl, x_cl[11, :], label='z (altitude)', linewidth=2, color='blue')
    axs[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Ground (z=0)')
    axs[0, 0].fill_between(t_cl, -1, 0, color='red', alpha=0.2, label='Forbidden (z<0)')
    axs[0, 0].set_xlabel('Time [s]', fontsize=12)
    axs[0, 0].set_ylabel('Altitude z [m]', fontsize=12)
    axs[0, 0].set_title('Ground Collision Constraint', fontsize=13, fontweight='bold')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_ylim([-0.5, max(x_cl[11, :]) + 1])
    
    # Beta constraint
    axs[0, 1].plot(t_cl, np.rad2deg(x_cl[4, :]), label='β (pitch)', linewidth=2, color='blue')
    axs[0, 1].axhline(y=80, color='r', linestyle='--', linewidth=2, label='Limit (+80°)')
    axs[0, 1].axhline(y=-80, color='r', linestyle='--', linewidth=2, label='Limit (-80°)')
    axs[0, 1].fill_between(t_cl, 80, 90, color='red', alpha=0.2, label='Forbidden')
    axs[0, 1].fill_between(t_cl, -90, -80, color='red', alpha=0.2)
    axs[0, 1].set_xlabel('Time [s]', fontsize=12)
    axs[0, 1].set_ylabel('Beta β [deg]', fontsize=12)
    axs[0, 1].set_title('Beta Singularity Constraint', fontsize=13, fontweight='bold')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim([-90, 90])
    
    # Servo angles
    axs[1, 0].plot(t_cl[:-1], np.rad2deg(u_cl[0, :]), label='δ1', linewidth=2)
    axs[1, 0].plot(t_cl[:-1], np.rad2deg(u_cl[1, :]), label='δ2', linewidth=2)
    axs[1, 0].axhline(y=15, color='r', linestyle='--', linewidth=2, label='Limit (±15°)')
    axs[1, 0].axhline(y=-15, color='r', linestyle='--', linewidth=2)
    axs[1, 0].fill_between(t_cl[:-1], 15, 20, color='red', alpha=0.2, label='Forbidden')
    axs[1, 0].fill_between(t_cl[:-1], -20, -15, color='red', alpha=0.2)
    axs[1, 0].set_xlabel('Time [s]', fontsize=12)
    axs[1, 0].set_ylabel('Servo Angles [deg]', fontsize=12)
    axs[1, 0].set_title('Servo Deflection Constraints', fontsize=13, fontweight='bold')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_ylim([-20, 20])
    
    # Throttle
    axs[1, 1].plot(t_cl[:-1], u_cl[2, :], label='Pavg', linewidth=2, color='blue')
    axs[1, 1].axhline(y=40, color='r', linestyle='--', linewidth=2, label='Min (40%)')
    axs[1, 1].axhline(y=80, color='r', linestyle='--', linewidth=2, label='Max (80%)')
    axs[1, 1].fill_between(t_cl[:-1], 0, 40, color='red', alpha=0.2, label='Forbidden')
    axs[1, 1].fill_between(t_cl[:-1], 80, 100, color='red', alpha=0.2)
    axs[1, 1].set_xlabel('Time [s]', fontsize=12)
    axs[1, 1].set_ylabel('Average Throttle [%]', fontsize=12)
    axs[1, 1].set_title('Throttle Constraints', fontsize=13, fontweight='bold')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_constraint_verification.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 03_constraint_verification.png")
    plt.close()
    
    # === Print summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Settling time: {settling_time:.2f}s (requirement: ≤ 4.0s)")
    print(f"Final position error: {pos_error[-1]*100:.2f} cm")
    print(f"Final roll angle: {np.rad2deg(x_cl[5, -1]):.2f}°")
    print(f"\nMin altitude: {np.min(x_cl[11, :]):.4f} m (≥ 0 required)")
    print(f"Max |β|: {np.rad2deg(np.max(np.abs(x_cl[4, :]))):.2f}° (≤ 80° required)")
    print(f"Max throttle: {np.max(u_cl[2, :]):.1f}% (≤ 80% required)")
    print(f"Min throttle: {np.min(u_cl[2, :]):.1f}% (≥ 40% required)")
    print("="*60)
    print(f"\nAll figures saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
