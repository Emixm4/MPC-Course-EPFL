"""
Deliverable 5.1 - Comparison between Part 4 and Part 5.1
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.rocket import Rocket
from PIControl.PIControl import PIControl

part4_dir = os.path.join(parent_dir, "Deliverable_4_1")
sys.path.insert(0, part4_dir)
from LinearMPC_4_1.MPCVelControl import MPCVelControl as MPCVelControl_Part4
sys.path.remove(part4_dir)


part5_dir = os.path.join(parent_dir, "Deliverable_5_1")
sys.path.insert(0, part5_dir)
from LinearMPC.MPCVelControl import MPCVelControl as MPCVelControl_Part5
sys.path.remove(part5_dir) 

# Simulation parameters
Ts = 0.05
sim_time = 15.0
H = 5.0

# Initial conditions
# pos0 = [0, 0, 1], v0 = [5, 5, 10]
x0 = np.array([0, 0, 0, 0, 0, 0, 5, 5, 10, 0, 0, 1])

# vref = [0, 0, 0]
x_target = np.zeros((12,))

rocket_params_path = os.path.join(parent_dir, "rocket.yaml")

print("Deliverable 5.1 - Offset-Free Tracking Comparison")

print("\nSimulating Part 4 Controller")

rocket_part4 = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
mpc_part4 = MPCVelControl_Part4.new_controller(rocket_part4, Ts, H)

rocket_part4.mass = 1.5
rocket_part4.fuel_rate = 0.0

t_cl_4, x_cl_4, u_cl_4, t_ol_4, x_ol_4, u_ol_4, ref_4 = rocket_part4.simulate_control(
    mpc_part4, sim_time, H, x0, x_target=x_target, method='nonlinear'
)

print(f"Part 4 complete: {len(t_cl_4)} timesteps")

print("\nSimulating Part 5.1 Controller")

rocket_part5 = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
mpc_part5 = MPCVelControl_Part5.new_controller(rocket_part5, Ts, H)

rocket_part5.mass = 1.5
rocket_part5.fuel_rate = 0.0

t_cl_5, x_cl_5, u_cl_5, t_ol_5, x_ol_5, u_ol_5, ref_5 = rocket_part5.simulate_control(
    mpc_part5, sim_time, H, x0, x_target=x_target, method='nonlinear'
)

print(f"Part 5.1 complete: {len(t_cl_5)} timesteps")


print("\nExtracting estimates")

# The z-controller stores disturbance estimates
d_hat_history = []
x_hat_history = []

# Re-run Part 5.1 to collect disturbance estimates
rocket_part5_analysis = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
mpc_part5_analysis = MPCVelControl_Part5.new_controller(rocket_part5_analysis, Ts, H)
rocket_part5_analysis.mass = 1.5
rocket_part5_analysis.fuel_rate = 0.0

x_current = x0.copy()
for i in range(int(sim_time / Ts)):
    u, _, _, _ = mpc_part5_analysis.get_u(0, x_current, x_target=x_target)

    # Store constant disturbance estimate d_hat
    d_hat_history.append(mpc_part5_analysis.mpc_zvel.d_hat[0])
    x_hat_history.append(mpc_part5_analysis.mpc_zvel.x_hat[0])

    # Simulate one step
    x_current = rocket_part5_analysis.simulate_step(x_current, Ts, u, method='nonlinear')

d_hat_history = np.array(d_hat_history)
x_hat_history = np.array(x_hat_history)
t_analysis = np.arange(len(d_hat_history)) * Ts


print("\nGenerating plots")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Part 4 vs Part 5.1 Comparison (mass=1.5, fuel_rate=0)',
             fontsize=14, fontweight='bold')

state_names = ['wx', 'wy', 'wz', 'alpha', 'beta', 'gamma', 'vx', 'vy', 'vz', 'x', 'y', 'z']

# Plot 1: Z velocity
ax = axes[0, 0]
ax.plot(t_cl_4, x_cl_4[8, :], 'b-', linewidth=2, label='Part 4 (No offset-free)')
ax.plot(t_cl_5, x_cl_5[8, :], 'r-', linewidth=2, label='Part 5.1 (Offset-free)')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Reference vz=0')
ax.set_xlabel('Time [s]')
ax.set_ylabel('vz [m/s]')
ax.set_title('Z-Velocity Tracking')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Z position
ax = axes[0, 1]
ax.plot(t_cl_4, x_cl_4[11, :], 'b-', linewidth=2, label='Part 4')
ax.plot(t_cl_5, x_cl_5[11, :], 'r-', linewidth=2, label='Part 5.1')
ax.set_xlabel('Time [s]')
ax.set_ylabel('z [m]')
ax.set_title('Z-Position')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Pavg
ax = axes[1, 0]
ax.plot(t_cl_4[:-1], u_cl_4[2, :], 'b-', linewidth=2, label='Part 4')
ax.plot(t_cl_5[:-1], u_cl_5[2, :], 'r-', linewidth=2, label='Part 5.1')
ax.axhline(40, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(80, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Pavg [%]')
ax.set_title('Z-Control Input (Throttle)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: X and Y velocities
ax = axes[1, 1]
ax.plot(t_cl_4, x_cl_4[6, :], 'b-', linewidth=1.5, label='vx (Part 4)')
ax.plot(t_cl_5, x_cl_5[6, :], 'r-', linewidth=1.5, label='vx (Part 5.1)')
ax.plot(t_cl_4, x_cl_4[7, :], 'b--', linewidth=1.5, label='vy (Part 4)')
ax.plot(t_cl_5, x_cl_5[7, :], 'r--', linewidth=1.5, label='vy (Part 5.1)')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.set_title('X and Y Velocities')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: Disturbance Estimate
ax = axes[2, 0]
ax.plot(t_analysis, d_hat_history, 'g-', linewidth=2, label='Estimated d_hat')
ax.set_xlabel('Time [s]')
ax.set_ylabel('d (constant disturbance)')
ax.set_title('Constant Disturbance Estimate (Part 5.1 Kalman Filter)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Steady-state error comparison
ax = axes[2, 1]
# steady-state error last 5 sec
ss_start_idx_4 = int((sim_time - 5.0) / Ts)
ss_start_idx_5 = int((sim_time - 5.0) / Ts)

vz_error_4 = x_cl_4[8, ss_start_idx_4:] 
vz_error_5 = x_cl_5[8, ss_start_idx_5:]

ax.plot(t_cl_4[ss_start_idx_4:], vz_error_4, 'b-', linewidth=2, label='Part 4 error')
ax.plot(t_cl_5[ss_start_idx_5:], vz_error_5, 'r-', linewidth=2, label='Part 5.1 error')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('vz error [m/s]')
ax.set_title('Steady-State Tracking Eror (Last 5s)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'deliverable_5_1_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"      Saved plot: {output_path}")

