"""
Script de génération des plots pour Deliverable 6.1 & 6.2
MPC Course - EPFL

Ce script génère tous les plots nécessaires pour la documentation :
1. Ensemble mRPI E
2. Ensemble terminal Xf
3. Contraintes d'entrée resserrées
4. Trajectoires closed-loop (random et extreme disturbances)
5. Trajectoire complète 4-subsystems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from LandMPC.MPCControl_z import MPCControl_z
from LandMPC.MPCLandControl import MPCLandControl
from src.rocket import Rocket


def setup_rocket():
    """Initialize rocket and MPC parameters."""
    rocket_params_path = os.path.join(parent_dir, "rocket.yaml")
    
    Ts = 1/20
    rocket = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
    rocket.mass = 1.7
    
    x_ref = np.array([0.]*9 + [1., 0., 3.])
    xs, us = rocket.trim(x_ref)
    A, B = rocket.linearize(xs, us)
    
    return rocket, A, B, xs, us, x_ref, Ts


def plot_mrpi_terminal_constraints(mpc, us, save_path=None):
    """
    Generate plots for:
    - Minimal RPI Set E
    - Terminal Set Xf
    - Input Constraints (Original vs Tightened)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- Plot 1: Minimal RPI Set E ---
    ax1 = axes[0]
    E_bounds = mpc.E_bounds.flatten()
    e_rect = Rectangle((-E_bounds[0], -E_bounds[1]), 
                       2*E_bounds[0], 2*E_bounds[1],
                       linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5)
    ax1.add_patch(e_rect)
    ax1.plot(0, 0, 'ro', markersize=10, label='Origin')
    ax1.set_xlim(-1.5*E_bounds[0], 1.5*E_bounds[0])
    ax1.set_ylim(-1.5*E_bounds[1], 1.5*E_bounds[1])
    ax1.set_xlabel('$v_z$ error (m/s)', fontsize=12)
    ax1.set_ylabel('$z$ error (m)', fontsize=12)
    ax1.set_title('Minimal RPI Set $\\mathcal{E}$\n(Allowed state deviations)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.text(0, 1.3*E_bounds[1], f'$\\mathcal{{E}}$: $\\pm$[{E_bounds[0]:.2f}, {E_bounds[1]:.2f}]',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # --- Plot 2: Terminal Set Xf ---
    ax2 = axes[1]
    Xf_bounds = mpc.Xf_bounds.flatten()
    xf_rect = Rectangle((-Xf_bounds[0], -Xf_bounds[1]), 
                        2*Xf_bounds[0], 2*Xf_bounds[1],
                        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax2.add_patch(xf_rect)
    ax2.plot(0, 0, 'ro', markersize=10, label='Target (equilibrium)')
    ax2.set_xlim(-1.5*Xf_bounds[0], 1.5*Xf_bounds[0])
    ax2.set_ylim(-1.5*Xf_bounds[1], 1.5*Xf_bounds[1])
    ax2.set_xlabel('$v_z$ (m/s)', fontsize=12)
    ax2.set_ylabel('$z$ (m)', fontsize=12)
    ax2.set_title('Terminal Set $\\mathcal{X}_f$\n(Safe termination region)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.text(0, 1.3*Xf_bounds[1], f'$\\mathcal{{X}}_f$: $\\pm$[{Xf_bounds[0]:.1f}, {Xf_bounds[1]:.1f}]',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # --- Plot 3: Input Constraints ---
    ax3 = axes[2]
    u_min_orig, u_max_orig = 40.0, 80.0
    
    K_inf = np.max(np.abs(mpc.K))
    E_inf = np.max(E_bounds)
    u_margin = K_inf * E_inf
    u_min_tight = u_min_orig + min(u_margin, 1.5)
    u_max_tight = u_max_orig - min(u_margin, 1.5)
    
    ax3.barh(['Tightened $\\tilde{\\mathcal{U}}$\n(Nominal $v$)', 
              'Original $\\mathcal{U}$\n(Actual $u$)'], 
             [u_max_tight - u_min_tight, u_max_orig - u_min_orig],
             left=[u_min_tight, u_min_orig],
             color=['coral', 'skyblue'], alpha=0.7, edgecolor='black', linewidth=2)
    
    ax3.axvline(us[2], color='red', linestyle='--', linewidth=2, label=f'Trim: {us[2]:.1f}N')
    ax3.set_xlabel('$P_{avg}$ (N)', fontsize=12)
    ax3.set_title('Input Constraints\n(Original vs Tightened)', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.legend()
    
    ax3.text(u_min_orig, 1.3, f'{u_min_orig:.0f}N', ha='center', fontsize=9)
    ax3.text(u_max_orig, 1.3, f'{u_max_orig:.0f}N', ha='center', fontsize=9)
    ax3.text(u_min_tight, 0.3, f'{u_min_tight:.1f}N', ha='center', fontsize=9)
    ax3.text(u_max_tight, 0.3, f'{u_max_tight:.1f}N', ha='center', fontsize=9)
    ax3.text(70, -0.7, f'Margin: $\\pm${min(u_margin, 1.5):.2f}N\n= $\\|K\\|_\\infty \\cdot \\|E\\|_\\infty$\n= {K_inf:.2f} × {E_inf:.2f}',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print vertices
    print("\n" + "="*60)
    print("VERTICES SUMMARY")
    print("="*60)
    print(f"\nMinimal RPI Set E vertices (box approximation):")
    print(f"  (-{E_bounds[0]:.3f}, -{E_bounds[1]:.3f})")
    print(f"  ( {E_bounds[0]:.3f}, -{E_bounds[1]:.3f})")
    print(f"  ( {E_bounds[0]:.3f},  {E_bounds[1]:.3f})")
    print(f"  (-{E_bounds[0]:.3f},  {E_bounds[1]:.3f})")
    
    print(f"\nTerminal Set Xf vertices:")
    print(f"  (-{Xf_bounds[0]:.1f}, -{Xf_bounds[1]:.1f})")
    print(f"  ( {Xf_bounds[0]:.1f}, -{Xf_bounds[1]:.1f})")
    print(f"  ( {Xf_bounds[0]:.1f},  {Xf_bounds[1]:.1f})")
    print(f"  (-{Xf_bounds[0]:.1f},  {Xf_bounds[1]:.1f})")
    
    print(f"\nTightened Input Constraints Ũ vertices (1D interval):")
    print(f"  [{u_min_tight:.1f}N, {u_max_tight:.1f}N]")
    print("="*60)


def plot_closed_loop_z(rocket, mpc, x0, sim_time, xs, title, w_type='random', save_path=None):
    """
    Generate closed-loop plots for z-subsystem with disturbances.
    """
    t_cl, x_cl, u_cl = rocket.simulate_subsystem(mpc, sim_time, x0, w_type=w_type)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: z trajectory
    ax1 = axes[0, 0]
    ax1.plot(t_cl[:-1], x_cl[11, :-1], 'b-', linewidth=2, label='z(t)')
    ax1.axhline(3.0, color='r', linestyle='--', linewidth=1.5, label='Target z=3m')
    ax1.axhline(0.0, color='k', linestyle='-', linewidth=2, label='Ground')
    ax1.fill_between(t_cl[:-1], 0, x_cl[11, :-1], alpha=0.1)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('z (m)', fontsize=12)
    ax1.set_title('Altitude', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.5, 11)
    
    # Plot 2: vz trajectory
    ax2 = axes[0, 1]
    ax2.plot(t_cl[:-1], x_cl[8, :-1], 'g-', linewidth=2, label='$v_z$(t)')
    ax2.axhline(0.0, color='r', linestyle='--', linewidth=1.5, label='Target $v_z$=0')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('$v_z$ (m/s)', fontsize=12)
    ax2.set_title('Vertical Velocity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Control input
    ax3 = axes[1, 0]
    ax3.plot(t_cl[:-1], u_cl[2, :], 'm-', linewidth=2, label='$P_{avg}$(t)')
    ax3.axhline(xs[2], color='r', linestyle='--', linewidth=1.5, label=f'Trim: {xs[2]:.1f}N')
    ax3.axhline(40, color='gray', linestyle=':', linewidth=1, label='Bounds')
    ax3.axhline(80, color='gray', linestyle=':', linewidth=1)
    ax3.fill_between(t_cl[:-1], 40, 80, alpha=0.1, color='gray')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('$P_{avg}$ (N)', fontsize=12)
    ax3.set_title('Control Input (Average Thrust)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(35, 85)
    
    # Plot 4: Phase portrait (vz vs z)
    ax4 = axes[1, 1]
    ax4.plot(x_cl[8, :-1], x_cl[11, :-1], 'b-', linewidth=2)
    ax4.plot(x_cl[8, 0], x_cl[11, 0], 'go', markersize=12, label='Initial')
    ax4.plot(x_cl[8, -2], x_cl[11, -2], 'ro', markersize=12, label='Final')
    ax4.plot(0, 3, 'k*', markersize=15, label='Target')
    ax4.set_xlabel('$v_z$ (m/s)', fontsize=12)
    ax4.set_ylabel('z (m)', fontsize=12)
    ax4.set_title('Phase Portrait', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print results
    print(f"\n{title}")
    print(f"  Final z = {x_cl[11, -1]:.3f} m (target: 3.0 m)")
    print(f"  Final vz = {x_cl[8, -1]:.3f} m/s")
    print(f"  Min z = {np.min(x_cl[11, :]):.3f} m")
    
    return t_cl, x_cl, u_cl


def plot_landing_maneuver(rocket, mpc, x0, x_ref, sim_time, H_sim, save_path=None):
    """
    Generate plots for full landing maneuver (4 subsystems).
    """
    t_cl, x_cl, u_cl, _, _, _ = rocket.simulate_land(mpc, sim_time, H_sim, x0, method='linear')
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    
    # Row 1: Positions
    ax = axes[0, 0]
    ax.plot(t_cl[:-1], x_cl[9, :-1], 'b-', linewidth=2)
    ax.axhline(x_ref[9], color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('x (m)')
    ax.set_title('X Position')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t_cl[:-1], x_cl[10, :-1], 'b-', linewidth=2)
    ax.axhline(x_ref[10], color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('y (m)')
    ax.set_title('Y Position')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(t_cl[:-1], x_cl[11, :-1], 'b-', linewidth=2)
    ax.axhline(x_ref[11], color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('z (m)')
    ax.set_title('Z Position (Altitude)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 3]
    ax.plot(t_cl[:-1], np.rad2deg(x_cl[5, :-1]), 'b-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('γ (deg)')
    ax.set_title('Roll Angle')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Velocities
    ax = axes[1, 0]
    ax.plot(t_cl[:-1], x_cl[6, :-1], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('$v_x$ (m/s)')
    ax.set_title('X Velocity')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t_cl[:-1], x_cl[7, :-1], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('$v_y$ (m/s)')
    ax.set_title('Y Velocity')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(t_cl[:-1], x_cl[8, :-1], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('$v_z$ (m/s)')
    ax.set_title('Z Velocity')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 3]
    ax.plot(t_cl[:-1], x_cl[2, :-1], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('$ω_z$ (rad/s)')
    ax.set_title('Angular Velocity (Roll)')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Control inputs
    ax = axes[2, 0]
    ax.plot(t_cl[:-1], np.rad2deg(u_cl[0, :]), 'm-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.fill_between(t_cl[:-1], -15, 15, alpha=0.1, color='gray')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$δ_1$ (deg)')
    ax.set_title('Gimbal X')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(t_cl[:-1], np.rad2deg(u_cl[1, :]), 'm-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.fill_between(t_cl[:-1], -15, 15, alpha=0.1, color='gray')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$δ_2$ (deg)')
    ax.set_title('Gimbal Y')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 2]
    ax.plot(t_cl[:-1], u_cl[2, :], 'm-', linewidth=2)
    ax.axhline(mpc.us[2], color='r', linestyle='--', linewidth=1.5)
    ax.fill_between(t_cl[:-1], 40, 80, alpha=0.1, color='gray')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$P_{avg}$ (N)')
    ax.set_title('Average Thrust')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(35, 85)
    
    ax = axes[2, 3]
    ax.plot(t_cl[:-1], u_cl[3, :], 'm-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    ax.fill_between(t_cl[:-1], -20, 20, alpha=0.1, color='gray')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$P_{diff}$ (%)')
    ax.set_title('Differential Thrust')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Deliverable 6.2: Full Landing Maneuver\n' + 
                 f'Initial: (3, 2, 10, 30°) → Target: (1, 0, 3, 0°)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print final results
    print("\n" + "="*60)
    print("DELIVERABLE 6.2 FINAL RESULTS")
    print("="*60)
    print(f"Position:")
    print(f"  x = {x_cl[9, -1]:.3f} m (target: {x_ref[9]:.1f} m)")
    print(f"  y = {x_cl[10, -1]:.3f} m (target: {x_ref[10]:.1f} m)")
    print(f"  z = {x_cl[11, -1]:.3f} m (target: {x_ref[11]:.1f} m)")
    print(f"\nOrientation:")
    print(f"  γ (roll) = {np.rad2deg(x_cl[5, -1]):.2f}° (target: 0°)")
    print("="*60)
    
    return t_cl, x_cl, u_cl


if __name__ == "__main__":
    print("="*60)
    print("GENERATING PLOTS FOR DELIVERABLE 6.1 & 6.2")
    print("="*60)
    
    # Setup
    rocket, A, B, xs, us, x_ref, Ts = setup_rocket()
    
    # Initial state
    x0 = np.array([0, 0, 0,  # angular velocities
                   0, 0, np.deg2rad(30),  # angles
                   0, 0, 0,  # linear velocities
                   3, 2, 10])  # positions
    
    sim_time = 15
    H_z = 5.0  # Horizon for z-subsystem
    H_full = 2.0  # Horizon for full system
    
    # Create controllers
    mpc_z = MPCControl_z(A, B, xs, us, Ts, H_z)
    mpc_full = MPCLandControl().new_controller(rocket, Ts, H_full, x_ref=x_ref)
    
    print("\n--- DELIVERABLE 6.1 ---")
    
    # Plot 1: Sets E, Xf, and constraints
    print("\n1. Generating E, Xf, and constraint plots...")
    plot_mrpi_terminal_constraints(mpc_z, us, save_path="plots_6_1_sets.png")
    
    # Plot 2: Closed-loop with random noise
    print("\n2. Generating closed-loop plot (random disturbance)...")
    plot_closed_loop_z(rocket, mpc_z, x0, sim_time, xs, 
                       "Deliverable 6.1: Z-Subsystem with Random Disturbance",
                       w_type='random', save_path="plots_6_1_random.png")
    
    # Plot 3: Closed-loop with extreme noise
    print("\n3. Generating closed-loop plot (extreme disturbance)...")
    plot_closed_loop_z(rocket, mpc_z, x0, sim_time, xs,
                       "Deliverable 6.1: Z-Subsystem with Extreme Disturbance (w=-15N)",
                       w_type='extreme', save_path="plots_6_1_extreme.png")
    
    print("\n--- DELIVERABLE 6.2 ---")
    
    # Plot 4: Full landing maneuver
    print("\n4. Generating full landing maneuver plot...")
    plot_landing_maneuver(rocket, mpc_full, x0, x_ref, sim_time, H_full,
                         save_path="plots_6_2_landing.png")
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60)
