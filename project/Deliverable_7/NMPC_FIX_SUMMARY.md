# NMPC Controller Fix Summary

## Problem Identified
The rocket was crashing to z=0 instead of landing at z=3 because:
1. **Input constraints were wrong**: Pavg minimum was 10% instead of 40%
2. **Integration too crude**: Euler method caused large numerical errors
3. **Horizon too short**: 2.0s insufficient for 7m maneuver + 30° rotation
4. **Tuning too weak**: Low terminal cost meant no convergence guarantee

## Changes Made to `nmpc_land.py`

### 1. Fixed Input Constraints (Line ~130)
```python
# BEFORE (WRONG):
lbx.extend([np.deg2rad(-15), np.deg2rad(-15), 10.0, -20.0])  # Pavg min = 10%
ubx.extend([np.deg2rad(15), np.deg2rad(15), 90.0, 20.0])     # Pavg max = 90%

# AFTER (CORRECT):
lbx.extend([-0.26, -0.26, 40.0, -20.0])  # Pavg min = 40% (safety requirement)
ubx.extend([0.26, 0.26, 80.0, 20.0])     # Pavg max = 80% (structural limit)
```

### 2. Improved Integration: Euler → RK4 (Line ~95)
```python
# BEFORE: Euler (1st order, inaccurate)
x_next = X[:, k] + self.Ts * self.f(X[:, k], U[:, k])

# AFTER: RK4 (4th order, accurate)
k1 = self.f(X[:, k], U[:, k])
k2 = self.f(X[:, k] + (self.Ts/2)*k1, U[:, k])
k3 = self.f(X[:, k] + (self.Ts/2)*k2, U[:, k])
k4 = self.f(X[:, k] + self.Ts*k3, U[:, k])
x_next = X[:, k] + (self.Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
```

### 3. Improved Cost Function Tuning (Line ~60)
```python
# BEFORE (weak tuning):
Q = diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 10, 10, 10])
R = diag([0.1, 0.1, 0.1, 0.1])
P = Q  # Terminal cost same as stage cost

# AFTER (strong convergence):
Q = diag([1, 1, 1, 20, 20, 20, 10, 10, 10, 50, 50, 100])  # Higher weights on critical states
R = diag([0.01, 0.01, 0.01, 0.01])  # Lower input penalty (allow aggressive control)
P = Q * 10.0  # 10x terminal cost for guaranteed convergence
```

Key changes:
- **Angles (α, β, γ)**: 10 → 20 (avoid large tilts)
- **Velocities (vx, vy, vz)**: 1 → 10 (reach zero velocity)
- **Positions (x, y, z)**: 10 → [50, 50, **100**] (z most critical for landing)
- **Inputs**: 0.1 → 0.01 (allow aggressive maneuvers)
- **Terminal cost**: 1x → **10x** (ensure final convergence)

### 4. Improved Solver Settings (Line ~152)
```python
# BEFORE:
opts = {
    'ipopt.max_iter': 100,
    'ipopt.acceptable_tol': 1e-6,
}

# AFTER:
opts = {
    'ipopt.max_iter': 500,  # More iterations for convergence
    'ipopt.acceptable_tol': 1e-4,  # Relaxed tolerance (faster solves)
    'ipopt.tol': 1e-5,
    'expand': True  # Faster NLP evaluation
}
```

### 5. Increased Horizon (Notebook)
```python
# BEFORE:
H = 2.0  # 2 seconds (40 steps)

# AFTER:
H = 4.0  # 4 seconds (80 steps) - needed for 7m maneuver
```

## Expected Results After Fix

With these changes, the NMPC should:

✓ **Respect constraints**: Pavg ≥ 40%, z ≥ 0, |β| ≤ 80°
✓ **Land accurately**: Position error < 10 cm at target (1, 0, 3)
✓ **Converge within 4s**: Meet settling time requirement
✓ **Smooth trajectory**: RK4 integration eliminates numerical oscillations
✓ **No crashes**: Ground collision constraint properly enforced

## Next Steps

1. Run the notebook cells in order (the diagnostic cell will verify the fix)
2. If settling time > 4s, increase terminal cost further (P = 20*Q)
3. If solver too slow, reduce horizon to H = 3.0s
4. Compare performance with Part 6.2 linear MPC

## Files Modified

- `Deliverable_7/LandMPC/nmpc_land.py` - NMPC controller implementation
- `Deliverable_7/Deliverable_7_1.ipynb` - Added diagnostics + longer horizon
