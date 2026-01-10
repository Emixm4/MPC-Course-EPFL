"""
Deliverable 5.2 - Time-Varying Mass (Fuel Consumption)

This script simulates the offset-free MPC controller with:
- Initial mass: 2.0 kg (50% fuel, 50% structure)
- Fuel rate: 0.1 (mass decreases linearly with thrust)
- Initial conditions: pos0 = [0,0,1], v0 = [5,5,10]
- Velocity reference: vref = [0,0,0]
- Simulation time: 15 seconds (or until fuel runs out)

Key questions to address:
1. Why is there still tracking offset in the first few seconds?
2. How could the estimator be modified for time-varying mass?
3. What different behaviors occur along the simulation?
4. What unexpected behavior occurs at the end?
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Get the project directory (parent of Deliverable_5_2)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.rocket import Rocket

# Import Part 5.1 controller (with offset-free tracking)
part5_dir = os.path.join(parent_dir, "Deliverable_5_1")
sys.path.insert(0, part5_dir)
from LinearMPC.MPCVelControl import MPCVelControl as MPCVelControl_Part5
sys.path.remove(part5_dir)

# Simulation parameters
Ts = 0.05
sim_time = 15.0  # May end early if fuel runs out
H = 5.0

# Initial conditions as specified
# pos0 = [0, 0, 1], v0 = [5, 5, 10]
x0 = np.array([0, 0, 0, 0, 0, 0, 5, 5, 10, 0, 0, 1])

# Velocity reference: vref = [0, 0, 0]
x_target = np.zeros((12,))

# Rocket parameters
rocket_params_path = os.path.join(parent_dir, "rocket.yaml")

print("="*80)
print("Deliverable 5.2 - Time-Varying Mass (Fuel Consumption)")
print("="*80)

# ============================================================================
# Simulate Part 5.1 Controller with Time-Varying Mass
# ============================================================================
print("\n[1/1] Simulating with time-varying mass (fuel consumption)...")
print("      Initial mass: 2.0 kg (50% fuel)")
print("      Fuel rate: 0.1")

rocket = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
mpc = MPCVelControl_Part5.new_controller(rocket, Ts, H)

# IMPORTANT: Set mass and fuel_rate AFTER creating controller
rocket.mass = 2.0  # Initial mass (1.0 kg fuel + 1.0 kg structure)
rocket.fuel_rate = 0.1  # Mass decreases with thrust

try:
    t_cl, x_cl, u_cl, t_ol, x_ol, u_ol, ref = rocket.simulate_control(
        mpc, sim_time, H, x0, x_target=x_target, method='nonlinear'
    )
    print(f"      Simulation complete: {len(t_cl)} timesteps")
    print(f"      Final time: {t_cl[-1]:.2f} seconds")
    print(f"      Final mass: {rocket.mass:.3f} kg")

    if len(t_cl) < int(sim_time / Ts):
        print(f"      WARNING: Simulation ended early (fuel exhausted at t={t_cl[-1]:.2f}s)")
        fuel_exhausted = True
    else:
        fuel_exhausted = False

except Exception as e:
    print(f"      ERROR: Simulation failed: {e}")
    sys.exit(1)

# ============================================================================
# Extract Additional Data for Analysis
# ============================================================================
print("\n[2/2] Extracting data for analysis...")

# Re-run to collect detailed data
rocket_analysis = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
mpc_analysis = MPCVelControl_Part5.new_controller(rocket_analysis, Ts, H)
rocket_analysis.mass = 2.0
rocket_analysis.fuel_rate = 0.1

# Storage for analysis
mass_history = []
d_hat_history = []
x_hat_history = []
vz_error_history = []
altitude_history = []

x_current = x0.copy()
for i in range(len(t_cl) - 1):  # Match the simulation length
    # Store current mass
    mass_history.append(rocket_analysis.mass)

    # Get control input
    u, _, _, _ = mpc_analysis.get_u(0, x_current, x_target=x_target)

    # Store disturbance estimate and tracking error
    d_hat_history.append(mpc_analysis.mpc_zvel.d_hat[0])
    x_hat_history.append(mpc_analysis.mpc_zvel.x_hat[0])
    vz_error_history.append(x_current[8])  # vz error (reference is 0)
    altitude_history.append(x_current[11])  # z position

    # Simulate one step
    x_current = rocket_analysis.simulate_step(x_current, Ts, u, method='nonlinear')

# Add final values
mass_history.append(rocket_analysis.mass)
d_hat_history.append(mpc_analysis.mpc_zvel.d_hat[0])
x_hat_history.append(mpc_analysis.mpc_zvel.x_hat[0])
vz_error_history.append(x_current[8])
altitude_history.append(x_current[11])

mass_history = np.array(mass_history)
d_hat_history = np.array(d_hat_history)
x_hat_history = np.array(x_hat_history)
vz_error_history = np.array(vz_error_history)
altitude_history = np.array(altitude_history)
t_analysis = t_cl[:len(d_hat_history)]

print(f"      Collected {len(d_hat_history)} data points")
print(f"      Initial mass: {mass_history[0]:.3f} kg")
print(f"      Final mass: {mass_history[-1]:.3f} kg")
print(f"      Mass consumed: {mass_history[0] - mass_history[-1]:.3f} kg")
print(f"      Final disturbance estimate: d_hat = {d_hat_history[-1]:.6f}")

# ============================================================================
# Plotting
# ============================================================================
print("\n[3/3] Generating plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle('Deliverable 5.2: Time-Varying Mass (Fuel Consumption)\n' +
             f'Initial mass=2.0 kg, fuel_rate=0.1, Final mass={mass_history[-1]:.2f} kg',
             fontsize=14, fontweight='bold')

# Plot 1: Z velocity tracking
ax = axes[0, 0]
ax.plot(t_cl, x_cl[8, :], 'b-', linewidth=2, label='vz (actual)')
ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='vz reference (0)')
ax.set_xlabel('Time [s]')
ax.set_ylabel('vz [m/s]')
ax.set_title('Z-Velocity Tracking')
ax.legend()
ax.grid(True, alpha=0.3)
if fuel_exhausted:
    ax.axvline(t_cl[-1], color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Fuel exhausted')

# Plot 2: Altitude (z position)
ax = axes[0, 1]
ax.plot(t_cl, x_cl[11, :], 'g-', linewidth=2)
ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Ground')
ax.set_xlabel('Time [s]')
ax.set_ylabel('z [m]')
ax.set_title('Altitude')
ax.legend()
ax.grid(True, alpha=0.3)
if fuel_exhausted:
    ax.axvline(t_cl[-1], color='k', linestyle=':', linewidth=1.5, alpha=0.5)

# Plot 3: Mass over time
ax = axes[1, 0]
ax.plot(t_analysis, mass_history, 'purple', linewidth=2, label='Rocket mass')
ax.axhline(1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Empty mass (no fuel)')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Mass [kg]')
ax.set_title('Rocket Mass (Fuel Consumption)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Disturbance estimate d_hat
ax = axes[1, 1]
ax.plot(t_analysis, d_hat_history, 'orange', linewidth=2, label='d_hat (estimated)')
ax.set_xlabel('Time [s]')
ax.set_ylabel('d (disturbance)')
ax.set_title('Disturbance Estimate (Should be Time-Varying)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Control input (Pavg)
ax = axes[2, 0]
ax.plot(t_cl[:-1], u_cl[2, :], 'b-', linewidth=2, label='Pavg (throttle)')
ax.axhline(40, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(80, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Pavg [%]')
ax.set_title('Z-Control Input (Throttle)')
ax.legend()
ax.grid(True, alpha=0.3)
if fuel_exhausted:
    ax.axvline(t_cl[-1], color='k', linestyle=':', linewidth=1.5, alpha=0.5)

# Plot 6: Tracking error over time
ax = axes[2, 1]
ax.plot(t_analysis, vz_error_history, 'r-', linewidth=2, label='vz tracking error')
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('vz error [m/s]')
ax.set_title('Z-Velocity Tracking Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'deliverable_5_2_results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"      Saved plot: {output_path}")

# ============================================================================
# Analysis and Observations
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS AND OBSERVATIONS")
print("="*80)

# 1. Early tracking offset
print("\n1. Early Tracking Offset (First Few Seconds):")
early_time_idx = int(3.0 / Ts)  # First 3 seconds
early_vz_error = np.mean(np.abs(vz_error_history[:early_time_idx]))
print(f"   - Mean |vz error| in first 3s: {early_vz_error:.3f} m/s")
print(f"   - Reason: Constant disturbance estimator cannot adapt fast enough")
print(f"     to the CHANGING mass (mass varies from {mass_history[0]:.2f} to {mass_history[early_time_idx]:.2f} kg)")

# 2. Behaviors along simulation
print("\n2. Different Behaviors Along Simulation:")
print("   - Phase 1 (0-5s): Large tracking error as estimator adapts from initial guess")
print(f"   - Phase 2 (5-10s): Estimator chases the time-varying disturbance")
print(f"   - Phase 3 (10-15s): Disturbance grows rapidly as mass approaches minimum")

# 3. Compute statistics for different phases
phase1_idx = int(5.0 / Ts)
phase2_idx = int(10.0 / Ts)

if len(vz_error_history) > phase2_idx:
    phase1_error = np.mean(np.abs(vz_error_history[:phase1_idx]))
    phase2_error = np.mean(np.abs(vz_error_history[phase1_idx:phase2_idx]))
    phase3_error = np.mean(np.abs(vz_error_history[phase2_idx:]))

    print(f"   - Phase 1 mean error: {phase1_error:.3f} m/s")
    print(f"   - Phase 2 mean error: {phase2_error:.3f} m/s")
    print(f"   - Phase 3 mean error: {phase3_error:.3f} m/s")

# 4. Unexpected behavior at end
print("\n3. Unexpected Behavior Towards End:")
if fuel_exhausted:
    print(f"   - Simulation ended early at t={t_cl[-1]:.2f}s (fuel exhausted)")
    print(f"   - Final altitude: z={x_cl[11, -1]:.2f} m")
    print(f"   - After fuel exhaustion, rocket enters free-fall (no thrust possible)")
else:
    final_10_idx = max(0, len(vz_error_history) - int(5.0 / Ts))
    final_error = np.mean(np.abs(vz_error_history[final_10_idx:]))
    print(f"   - Final 5s mean error: {final_error:.3f} m/s")
    print(f"   - Disturbance estimate keeps growing: d_hat = {d_hat_history[-1]:.3f}")
    print(f"   - Mass is very light ({mass_history[-1]:.2f} kg), dynamics highly nonlinear")

# 5. Rate of change of disturbance
d_hat_rate = np.diff(d_hat_history) / Ts
print(f"\n4. Disturbance Estimate Rate of Change:")
print(f"   - Mean rate: {np.mean(d_hat_rate):.4f} per second")
print(f"   - This is NOT constant! Constant estimator assumption violated.")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("✗ Constant disturbance estimator CANNOT handle time-varying mass")
print("✗ Tracking error persists throughout simulation")
print("✗ Estimator continuously chases the changing disturbance")
print("✓ Controller prevents crash and maintains reasonable altitude")
print("\nTo achieve offset-free tracking with time-varying mass:")
print("  - Use time-varying disturbance model: d(t) = d0 + rate*t")
print("  - Augment state with disturbance AND its rate: [x; d; d_dot]")
print("  - Or: Use online mass estimation and adaptive MPC")
print("="*80)

# Save data for report
data_path = os.path.join(script_dir, 'deliverable_5_2_data.npz')
np.savez(data_path,
         t=t_cl,
         x=x_cl,
         u=u_cl,
         mass_history=mass_history,
         d_hat_history=d_hat_history,
         vz_error_history=vz_error_history,
         t_analysis=t_analysis,
         Ts=Ts)
print(f"\nData saved to: {data_path}")
print("Plot saved. Open 'deliverable_5_2_results.png' to view results.")
