"""
Comparison Script: Linear MPC (6.2) vs NMPC (7.1)
Runs both controllers on the same landing scenario and generates detailed comparison plots and metrics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
deliverable_7_dir = script_dir  
project_dir = script_dir.parent

# Add all necessary paths for imports
sys.path.insert(0, str(project_dir))  # For src.rocket
sys.path.insert(0, str(deliverable_7_dir))  # For NMPC from D7
sys.path.insert(0, str(project_dir / "Deliverable_6_1&2"))  # For Linear MPC from D6

# Import rocket
from src.rocket import Rocket

# Import controllers
# NMPC from Deliverable 7 - import BEFORE the Linear MPC to avoid conflicts
import sys
import os
sys.path.insert(0, os.path.join(str(deliverable_7_dir), "LandMPC"))
import nmpc_land as nmpc_d7_module
NmpcCtrl = nmpc_d7_module.NmpcCtrl
sys.path.pop(0)  # Remove to avoid conflicts

# Linear MPC from Deliverable 6
from LandMPC.MPCLandControl import MPCLandControl as LinearMPCController


def simulate_controller(rocket, controller, sim_time, H, x0, controller_name):
    """
    Simulate a controller and collect detailed metrics.
    
    Returns:
        dict with trajectory data and performance metrics
    """
    print(f"\n{'='*70}")
    print(f"SIMULATING {controller_name}")
    print(f"{'='*70}")
    
    # Record start time
    start_time = time.time()
    
    # Run simulation
    t_cl, x_cl, u_cl, t_ol, x_ol, u_ol = rocket.simulate_land(
        controller, sim_time, H, x0
    )
    
    # Total simulation time
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
    
    # Settling time (2cm position, 5cm/s velocity, 1° angle tolerance)
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
    
    # Check constraint satisfaction
    metrics['ground_satisfied'] = metrics['min_z'] >= 0
    metrics['beta_satisfied'] = metrics['max_beta'] <= 80.0
    metrics['servo_satisfied'] = (metrics['max_d1'] <= 15.0) and (metrics['max_d2'] <= 15.0)
    metrics['throttle_satisfied'] = (metrics['min_Pavg'] >= 40.0) and (metrics['max_Pavg'] <= 80.0)
    metrics['pdiff_satisfied'] = metrics['max_Pdiff'] <= 20.0
    
    # Control effort (integral of squared control)
    dt = t_cl[1] - t_cl[0]
    metrics['control_effort'] = np.sum(u_cl**2) * dt
    
    # Average computation time per step
    metrics['avg_comp_time'] = total_sim_time / len(t_cl)
    
    # Print summary
    print(f"\n{controller_name} PERFORMANCE SUMMARY:")
    print(f"  Final Position Error: {pos_error*100:.2f} cm")
    print(f"  Final Velocity Error: {vel_error*100:.2f} cm/s")
    print(f"  Final Angle Error:    {np.rad2deg(angle_error):.3f}°")
    print(f"  Settling Time:        {metrics['settling_time']:.2f} s")
    print(f"  Avg Computation:      {metrics['avg_comp_time']*1000:.2f} ms/step")
    print(f"  Total Sim Time:       {total_sim_time:.2f} s")
    
    print(f"\nCONSTRAINT VERIFICATION:")
    print(f"  Ground (z≥0):         {'PASS ✓' if metrics['ground_satisfied'] else 'FAIL ✗'} (min: {metrics['min_z']:.3f}m)")
    print(f"  Beta (|β|≤80°):       {'PASS ✓' if metrics['beta_satisfied'] else 'FAIL ✗'} (max: {metrics['max_beta']:.2f}°)")
    print(f"  Servos (|δ|≤15°):     {'PASS ✓' if metrics['servo_satisfied'] else 'FAIL ✗'} (max: {max(metrics['max_d1'], metrics['max_d2']):.2f}°)")
    print(f"  Throttle (40-80%):    {'PASS ✓' if metrics['throttle_satisfied'] else 'FAIL ✗'} (range: [{metrics['min_Pavg']:.1f}, {metrics['max_Pavg']:.1f}])")
    print(f"  Pdiff (|Pdiff|≤20):   {'PASS ✓' if metrics['pdiff_satisfied'] else 'FAIL ✗'} (max: {metrics['max_Pdiff']:.2f})")
    
    return metrics


def plot_comparison(linear_metrics, nmpc_metrics, save_dir=None):
    """
    Generate comprehensive comparison plots.
    """
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*70}")
    
    # Create figure directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Figure 1: Position Trajectories
    fig1, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig1.suptitle('Position Comparison: Linear MPC vs NMPC', fontsize=14, fontweight='bold')
    
    positions = ['x', 'y', 'z']
    colors_linear = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2']
    colors_nmpc = ['#d62728', '#9467bd', '#8c564b', '#bcbd22']
    
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
        print(f"  Saved: {save_dir / 'position_comparison.png'}")
    
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
        print(f"  Saved: {save_dir / 'attitude_comparison.png'}")
    
    # Figure 3: Control Inputs
    fig3, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Control Inputs Comparison: Linear MPC vs NMPC', fontsize=14, fontweight='bold')
    
    control_names = ['δ₁ [deg]', 'δ₂ [deg]', 'Pavg [%]', 'Pdiff [%]']
    axs = axs.flatten()
    
    for i, (name, ax) in enumerate(zip(control_names, axs)):
        if i < 2:  # Servo angles - convert to degrees
            ax.plot(linear_metrics['t_cl'][:-1], np.rad2deg(linear_metrics['u_cl'][i, :]), 
                    label=f'{name} (Linear MPC)', color=colors_linear[i], linewidth=2)
            ax.plot(nmpc_metrics['t_cl'][:-1], np.rad2deg(nmpc_metrics['u_cl'][i, :]), 
                    label=f'{name} (NMPC)', color=colors_nmpc[i], linewidth=2, linestyle='--')
            ax.axhline(y=15, color='red', linestyle=':', alpha=0.7, label='Limit')
            ax.axhline(y=-15, color='red', linestyle=':', alpha=0.7)
        else:  # Throttle
            ax.plot(linear_metrics['t_cl'][:-1], linear_metrics['u_cl'][i, :], 
                    label=f'{name} (Linear MPC)', color=colors_linear[i], linewidth=2)
            ax.plot(nmpc_metrics['t_cl'][:-1], nmpc_metrics['u_cl'][i, :], 
                    label=f'{name} (NMPC)', color=colors_nmpc[i], linewidth=2, linestyle='--')
            if i == 2:  # Pavg
                ax.axhline(y=40, color='red', linestyle=':', alpha=0.7, label='Min')
                ax.axhline(y=80, color='red', linestyle=':', alpha=0.7, label='Max')
            else:  # Pdiff
                ax.axhline(y=20, color='red', linestyle=':', alpha=0.7, label='Limit')
                ax.axhline(y=-20, color='red', linestyle=':', alpha=0.7)
        
        ax.set_ylabel(name, fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        fig3.savefig(save_dir / 'control_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_dir / 'control_comparison.png'}")
    
    # Figure 4: Error Evolution
    fig4, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Error Evolution: Linear MPC vs NMPC', fontsize=14, fontweight='bold')
    
    # Position error
    pos_err_linear = np.linalg.norm(
        linear_metrics['x_cl'][9:12, :] - linear_metrics['xs'][9:12, np.newaxis], axis=0
    )
    pos_err_nmpc = np.linalg.norm(
        nmpc_metrics['x_cl'][9:12, :] - nmpc_metrics['xs'][9:12, np.newaxis], axis=0
    )
    
    axs[0, 0].plot(linear_metrics['t_cl'], pos_err_linear*100, 
                   label='Linear MPC', color='blue', linewidth=2)
    axs[0, 0].plot(nmpc_metrics['t_cl'], pos_err_nmpc*100, 
                   label='NMPC', color='red', linewidth=2, linestyle='--')
    axs[0, 0].axhline(y=2, color='green', linestyle=':', label='Tolerance (2cm)')
    axs[0, 0].set_ylabel('Position Error [cm]', fontsize=11)
    axs[0, 0].set_xlabel('Time [s]', fontsize=11)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_title('Position Error')
    
    # Velocity error
    vel_err_linear = np.linalg.norm(
        linear_metrics['x_cl'][6:9, :] - linear_metrics['xs'][6:9, np.newaxis], axis=0
    )
    vel_err_nmpc = np.linalg.norm(
        nmpc_metrics['x_cl'][6:9, :] - nmpc_metrics['xs'][6:9, np.newaxis], axis=0
    )
    
    axs[0, 1].plot(linear_metrics['t_cl'], vel_err_linear*100, 
                   label='Linear MPC', color='blue', linewidth=2)
    axs[0, 1].plot(nmpc_metrics['t_cl'], vel_err_nmpc*100, 
                   label='NMPC', color='red', linewidth=2, linestyle='--')
    axs[0, 1].axhline(y=5, color='green', linestyle=':', label='Tolerance (5cm/s)')
    axs[0, 1].set_ylabel('Velocity Error [cm/s]', fontsize=11)
    axs[0, 1].set_xlabel('Time [s]', fontsize=11)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_title('Velocity Error')
    
    # Roll angle error
    angle_err_linear = np.abs(linear_metrics['x_cl'][5, :] - linear_metrics['xs'][5])
    angle_err_nmpc = np.abs(nmpc_metrics['x_cl'][5, :] - nmpc_metrics['xs'][5])
    
    axs[1, 0].plot(linear_metrics['t_cl'], np.rad2deg(angle_err_linear), 
                   label='Linear MPC', color='blue', linewidth=2)
    axs[1, 0].plot(nmpc_metrics['t_cl'], np.rad2deg(angle_err_nmpc), 
                   label='NMPC', color='red', linewidth=2, linestyle='--')
    axs[1, 0].axhline(y=1, color='green', linestyle=':', label='Tolerance (1°)')
    axs[1, 0].set_ylabel('Roll Error [deg]', fontsize=11)
    axs[1, 0].set_xlabel('Time [s]', fontsize=11)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_title('Roll Angle Error')
    
    # Settling time visualization
    axs[1, 1].bar(['Linear MPC', 'NMPC'], 
                  [linear_metrics['settling_time'], nmpc_metrics['settling_time']],
                  color=['blue', 'red'], alpha=0.7)
    axs[1, 1].axhline(y=4.0, color='green', linestyle=':', linewidth=2, label='Requirement (4s)')
    axs[1, 1].set_ylabel('Settling Time [s]', fontsize=11)
    axs[1, 1].set_title('Settling Time Comparison')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        fig4.savefig(save_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_dir / 'error_comparison.png'}")
    
    # Figure 5: Performance Metrics Bar Chart
    fig5, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig5.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    
    metrics_to_plot = [
        ('final_pos_error', 'Final Position Error [cm]', 100),
        ('final_angle_error', 'Final Angle Error [°]', 1),
        ('settling_time', 'Settling Time [s]', 1),
        ('avg_comp_time', 'Avg Computation Time [ms]', 1000),
        ('control_effort', 'Control Effort', 1),
        ('max_beta', 'Max Beta Angle [°]', 1),
    ]
    
    axs = axs.flatten()
    for i, (metric, label, scale) in enumerate(metrics_to_plot):
        values = [linear_metrics[metric] * scale, nmpc_metrics[metric] * scale]
        bars = axs[i].bar(['Linear MPC', 'NMPC'], values, color=['blue', 'red'], alpha=0.7)
        axs[i].set_ylabel(label, fontsize=10)
        axs[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        fig5.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_dir / 'metrics_comparison.png'}")
    
    plt.show()
    print("\nAll plots generated successfully!")


def generate_report(linear_metrics, nmpc_metrics, save_path=None):
    """
    Generate a text report with detailed comparison.
    """
    report = []
    report.append("="*80)
    report.append("CONTROLLER COMPARISON REPORT: LINEAR MPC (6.2) vs NMPC (7.1)")
    report.append("="*80)
    report.append(f"\nTest Scenario: Landing from (3, 2, 10, 30°) to (1, 0, 3, 0°)")
    report.append(f"Simulation Time: {linear_metrics['t_cl'][-1]:.2f} seconds")
    
    report.append("\n" + "-"*80)
    report.append("FINAL STATE ACCURACY")
    report.append("-"*80)
    report.append(f"{'Metric':<30} {'Linear MPC':>20} {'NMPC':>20}")
    report.append(f"{'-'*30} {'-'*20} {'-'*20}")
    report.append(f"{'Position Error [cm]':<30} {linear_metrics['final_pos_error']*100:>20.3f} {nmpc_metrics['final_pos_error']*100:>20.3f}")
    report.append(f"{'Velocity Error [cm/s]':<30} {linear_metrics['final_vel_error']*100:>20.3f} {nmpc_metrics['final_vel_error']*100:>20.3f}")
    report.append(f"{'Roll Angle Error [°]':<30} {linear_metrics['final_angle_error']:>20.3f} {nmpc_metrics['final_angle_error']:>20.3f}")
    
    report.append("\n" + "-"*80)
    report.append("SETTLING TIME PERFORMANCE")
    report.append("-"*80)
    report.append(f"{'Settling Time [s]':<30} {linear_metrics['settling_time']:>20.2f} {nmpc_metrics['settling_time']:>20.2f}")
    report.append(f"{'Requirement Met (≤4s)?':<30} {('YES ✓' if linear_metrics['settling_time']<=4 else 'NO ✗'):>20} {('YES ✓' if nmpc_metrics['settling_time']<=4 else 'NO ✗'):>20}")
    
    report.append("\n" + "-"*80)
    report.append("CONSTRAINT SATISFACTION")
    report.append("-"*80)
    report.append(f"{'Constraint':<30} {'Linear MPC':>20} {'NMPC':>20}")
    report.append(f"{'-'*30} {'-'*20} {'-'*20}")
    report.append(f"{'Ground (z≥0)':<30} {('PASS ✓' if linear_metrics['ground_satisfied'] else 'FAIL ✗'):>20} {('PASS ✓' if nmpc_metrics['ground_satisfied'] else 'FAIL ✗'):>20}")
    report.append(f"{'  Min altitude [m]':<30} {linear_metrics['min_z']:>20.3f} {nmpc_metrics['min_z']:>20.3f}")
    report.append(f"{'Beta (|β|≤80°)':<30} {('PASS ✓' if linear_metrics['beta_satisfied'] else 'FAIL ✗'):>20} {('PASS ✓' if nmpc_metrics['beta_satisfied'] else 'FAIL ✗'):>20}")
    report.append(f"{'  Max |β| [°]':<30} {linear_metrics['max_beta']:>20.2f} {nmpc_metrics['max_beta']:>20.2f}")
    report.append(f"{'Servos (|δ|≤15°)':<30} {('PASS ✓' if linear_metrics['servo_satisfied'] else 'FAIL ✗'):>20} {('PASS ✓' if nmpc_metrics['servo_satisfied'] else 'FAIL ✗'):>20}")
    report.append(f"{'  Max |δ1| [°]':<30} {linear_metrics['max_d1']:>20.2f} {nmpc_metrics['max_d1']:>20.2f}")
    report.append(f"{'  Max |δ2| [°]':<30} {linear_metrics['max_d2']:>20.2f} {nmpc_metrics['max_d2']:>20.2f}")
    report.append(f"{'Throttle (40-80%)':<30} {('PASS ✓' if linear_metrics['throttle_satisfied'] else 'FAIL ✗'):>20} {('PASS ✓' if nmpc_metrics['throttle_satisfied'] else 'FAIL ✗'):>20}")
    report.append(f"{'  Min Pavg [%]':<30} {linear_metrics['min_Pavg']:>20.2f} {nmpc_metrics['min_Pavg']:>20.2f}")
    report.append(f"{'  Max Pavg [%]':<30} {linear_metrics['max_Pavg']:>20.2f} {nmpc_metrics['max_Pavg']:>20.2f}")
    report.append(f"{'Pdiff (|Pdiff|≤20)':<30} {('PASS ✓' if linear_metrics['pdiff_satisfied'] else 'FAIL ✗'):>20} {('PASS ✓' if nmpc_metrics['pdiff_satisfied'] else 'FAIL ✗'):>20}")
    report.append(f"{'  Max |Pdiff| [%]':<30} {linear_metrics['max_Pdiff']:>20.2f} {nmpc_metrics['max_Pdiff']:>20.2f}")
    
    report.append("\n" + "-"*80)
    report.append("COMPUTATIONAL PERFORMANCE")
    report.append("-"*80)
    report.append(f"{'Metric':<30} {'Linear MPC':>20} {'NMPC':>20}")
    report.append(f"{'-'*30} {'-'*20} {'-'*20}")
    report.append(f"{'Total Sim Time [s]':<30} {linear_metrics['total_sim_time']:>20.2f} {nmpc_metrics['total_sim_time']:>20.2f}")
    report.append(f"{'Avg Time/Step [ms]':<30} {linear_metrics['avg_comp_time']*1000:>20.2f} {nmpc_metrics['avg_comp_time']*1000:>20.2f}")
    report.append(f"{'Control Effort':<30} {linear_metrics['control_effort']:>20.2f} {nmpc_metrics['control_effort']:>20.2f}")
    report.append(f"{'Real-time @ 20Hz?':<30} {('YES ✓' if linear_metrics['avg_comp_time']<0.05 else 'NO ✗'):>20} {('YES ✓' if nmpc_metrics['avg_comp_time']<0.05 else 'NO ✗'):>20}")
    
    report.append("\n" + "="*80)
    report.append("SUMMARY")
    report.append("="*80)
    
    # Winner analysis
    winner_accuracy = "NMPC" if nmpc_metrics['final_pos_error'] < linear_metrics['final_pos_error'] else "Linear MPC"
    winner_speed = "Linear MPC" if linear_metrics['avg_comp_time'] < nmpc_metrics['avg_comp_time'] else "NMPC"
    winner_settling = "Linear MPC" if linear_metrics['settling_time'] < nmpc_metrics['settling_time'] else "NMPC"
    
    report.append(f"\nBest Accuracy:        {winner_accuracy}")
    report.append(f"Best Computational:   {winner_speed}")
    report.append(f"Best Settling Time:   {winner_settling}")
    
    report.append("\nRECOMMENDATIONS:")
    report.append("- Linear MPC: Best for real-time applications, guaranteed stability")
    report.append("- NMPC: Best for accuracy, large maneuvers, coupled dynamics")
    report.append("- Hybrid: Use NMPC for planning, Linear MPC for tracking")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    
    print("\n" + report_text)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")
    
    return report_text


def main():
    """
    Main comparison script.
    """
    print("="*80)
    print("CONTROLLER COMPARISON: LINEAR MPC (6.2) vs NMPC (7.1)")
    print("="*80)
    
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
    
    # ========== TEST LINEAR MPC (6.2) ==========
    print("\n" + "="*80)
    print("INITIALIZING LINEAR MPC (Deliverable 6.2)")
    print("="*80)
    
    rocket_linear = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket_linear.mass = 1.7
    rocket_linear.controller_type = 'MPCLandControl'
    
    linear_ctrl = LinearMPCController().new_controller(rocket_linear, Ts, H, x_ref)
    
    linear_metrics = simulate_controller(
        rocket_linear, linear_ctrl, sim_time, H, x0, "LINEAR MPC (6.2)"
    )
    
    # ========== TEST NMPC (7.1) ==========
    print("\n" + "="*80)
    print("INITIALIZING NMPC (Deliverable 7.1)")
    print("="*80)
    
    rocket_nmpc = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket_nmpc.mass = 1.7
    rocket_nmpc.controller_type = 'NmpcCtrl'
    
    nmpc_ctrl = NmpcCtrl(rocket=rocket_nmpc, Ts=Ts, H=H, x_ref=x_ref)
    
    nmpc_metrics = simulate_controller(
        rocket_nmpc, nmpc_ctrl, sim_time, H, x0, "NMPC (7.1)"
    )
    
    # ========== GENERATE COMPARISON ==========
    figures_dir = script_dir / "comparison_figures"
    report_path = script_dir / "comparison_report.txt"
    
    # Generate report
    generate_report(linear_metrics, nmpc_metrics, save_path=report_path)
    
    # Generate plots
    plot_comparison(linear_metrics, nmpc_metrics, save_dir=figures_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Report saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}/")
    print("\nUse these results in your Deliverable 7 report!")


if __name__ == "__main__":
    main()
