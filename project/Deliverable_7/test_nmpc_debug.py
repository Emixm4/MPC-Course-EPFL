"""
Quick diagnostic script to test NMPC controller
"""
import sys
import os

# Get the project directory (parent of Deliverable_7)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.rocket import Rocket
from LandMPC.nmpc_land import NmpcCtrl
import numpy as np

# Setup
rocket_params_path = os.path.join(project_dir, "rocket.yaml")
Ts = 1/20
rocket = Rocket(Ts=Ts, model_params_filepath=rocket_params_path)
rocket.mass = 1.7
rocket.controller_type = 'NmpcCtrl'

# Initial and target states
x0 = np.array([0, 0, 0,  # angular velocities
               0, 0, np.deg2rad(30),  # angles (30° roll)
               0, 0, 0,  # linear velocities
               3, 2, 10])  # positions

x_ref = np.array([0.]*9 + [1., 0., 3.])  # Target

# Trim around target
xs, us = rocket.trim(x_ref)
print("="*60)
print("TRIM RESULTS")
print("="*60)
print(f"Target ref: {x_ref}")
print(f"Trim state xs: {xs}")
print(f"Trim input us: {us}")
print()

# Create NMPC
H = 4.0  # Longer horizon
nmpc = NmpcCtrl(rocket, Ts=Ts, H=H, x_ref=x_ref)
print("="*60)
print("NMPC INITIALIZATION")
print("="*60)
print(f"Horizon: {nmpc.N} steps ({H}s)")
print(f"State dim: {nmpc.nx}")
print(f"Input dim: {nmpc.nu}")
print(f"Reference: {nmpc.xs}")
print()

# Test single solve
print("="*60)
print("TESTING SINGLE get_u() CALL")
print("="*60)
print(f"Initial state x0: {x0}")
print()

try:
    u0, x_ol, u_ol, t_ol = nmpc.get_u(0.0, x0)
    
    print("✓ Solve succeeded!")
    print(f"\nFirst control u0: {u0}")
    print(f"  Expected (hover): {nmpc.us}")
    print(f"  Difference: {u0 - nmpc.us}")
    print()
    
    print(f"Trajectory shapes:")
    print(f"  States: {x_ol.shape} (expected: ({nmpc.nx}, {nmpc.N+1}))")
    print(f"  Inputs: {u_ol.shape} (expected: ({nmpc.nu}, {nmpc.N}))")
    print(f"  Times: {t_ol.shape} (expected: ({nmpc.N+1},))")
    print()
    
    print(f"Final state in trajectory: {x_ol[:, -1]}")
    print(f"Target state: {nmpc.xs}")
    print(f"Final error: {np.linalg.norm(x_ol[:, -1] - nmpc.xs):.4f}")
    print()
    
    # Check if control is reasonable
    if np.allclose(u0, [0, 0, 0, 0], atol=0.1):
        print("⚠️ WARNING: Control is nearly ZERO!")
        print("   This suggests the solver returned a bad solution.")
    elif np.max(np.abs(u0 - nmpc.us)) < 0.5:
        print("⚠️ WARNING: Control is too close to trim input.")
        print("   Controller might not be steering towards target.")
    else:
        print("✓ Control looks reasonable (significantly different from zero/trim)")
    
except Exception as e:
    print(f"✗ SOLVE FAILED!")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
