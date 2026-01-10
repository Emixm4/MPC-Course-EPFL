"""
Save all figures from Deliverable 7.1 notebook to files.
This script runs the NMPC simulation and generates all analysis plots.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.rocket import Rocket
from LandMPC.nmpc_land import NmpcCtrl

def main():
    # output directory
    output_dir = "figures_deliverable_7_1"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving settling time figure to: {output_dir}/")

    rocket_params_path = os.path.join(parent_dir, "rocket.yaml")
    Ts = 1/20
    rocket = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket.mass = 1.7
    rocket.controller_type = 'NmpcCtrl'

    # Sim para
    sim_time = 15
    x0 = np.array([0, 0, 0,  
                   0, 0, np.deg2rad(30), 
                   0, 0, 0,  
                   3, 2, 10])  
    x_ref = np.array([0.]*9 + [1., 0., 3.])  
    H = 4.0

    nmpc = NmpcCtrl(rocket, Ts=Ts, H=H, x_ref=x_ref)
    xs, us = nmpc.xs, nmpc.us

    print("\n closed-loop sim")
    t_cl, x_cl, _, _, _, _ = rocket.simulate_land(nmpc, sim_time, H, x0)
    
    print("\nsettling time plot")

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
    plt.savefig(f"{output_dir}/settling_time_verification.png", dpi=300, bbox_inches='tight')
    print(f" png saved")
    plt.close()


if __name__ == "__main__":
    main()
