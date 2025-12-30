# Linear MPC Velocity Control

## Vue d'ensemble

Ce package implémente 4 contrôleurs MPC découplés pour le contrôle de vitesse d'une fusée :

- **MPCControl_xvel** : Contrôle de la vitesse en X
- **MPCControl_yvel** : Contrôle de la vitesse en Y  
- **MPCControl_zvel** : Contrôle de la vitesse en Z
- **MPCControl_roll** : Contrôle de l'angle de roulis (gamma)

## Architecture

```
LinearMPC_template/
├── __init__.py
├── MPCControl_base.py      # Classe de base avec logique MPC commune
├── MPCControl_xvel.py       # Contrôleur pour sous-système X
├── MPCControl_yvel.py       # Contrôleur pour sous-système Y
├── MPCControl_zvel.py       # Contrôleur pour sous-système Z
├── MPCControl_roll.py       # Contrôleur pour angle de roulis
└── MPCVelControl.py         # Orchestrateur des 4 contrôleurs
```

## Sous-systèmes

### 1. Sous-système X (MPCControl_xvel)
- **États** : `[wy, beta, vx]` (vitesse angulaire pitch, angle pitch, vitesse x)
- **Entrée** : `[dP]` (angle de basculement pitch)
- **Objectif** : Stabiliser `vx` à 0 m/s
- **Contraintes** :
  - États : `-10° ≤ beta ≤ 10°`
  - Entrée : `-15° ≤ dP ≤ 15°`

### 2. Sous-système Y (MPCControl_yvel)
- **États** : `[wx, alpha, vy]` (vitesse angulaire roll, angle roll, vitesse y)
- **Entrée** : `[dR]` (angle de basculement roll)
- **Objectif** : Stabiliser `vy` à 0 m/s
- **Contraintes** :
  - États : `-10° ≤ alpha ≤ 10°`
  - Entrée : `-15° ≤ dR ≤ 15°`

### 3. Sous-système Z (MPCControl_zvel)
- **États** : `[vz]` (vitesse z uniquement, sans position)
- **Entrée** : `[Pavg]` (poussée moyenne)
- **Objectif** : Stabiliser `vz` à 0 m/s
- **Contraintes** :
  - Entrée : `40% ≤ Pavg ≤ 80%`

### 4. Sous-système Roll (MPCControl_roll)
- **États** : `[wz, gamma]` (vitesse angulaire yaw, angle roulis)
- **Entrée** : `[Pdiff]` (poussée différentielle)
- **Objectif** : Stabiliser `gamma` à 0°
- **Contraintes** :
  - Entrée : `-20% ≤ Pdiff ≤ 20%`

## Utilisation

### 1. Test d'un sous-système individuel

```python
from LinearMPC_template.MPCControl_xvel import MPCControl_xvel
from src.rocket import Rocket

# Créer le modèle
Ts = 0.05
H = 5.0
rocket = Rocket(Ts=Ts, model_params_filepath="rocket.yaml")

# Trouver le point d'équilibre et linéariser
xs, us = rocket.trim()
A, B = rocket.linearize(xs, us)

# Créer le contrôleur MPC
mpc_x = MPCControl_xvel(A, B, xs, us, Ts, H)

# Tester avec une condition initiale
x0 = np.array([0.0, 0.0, 5.0])  # [wy=0, beta=0, vx=5m/s]
u0, x_traj, u_traj = mpc_x.get_u(x0)

print(f"Contrôle optimal: {u0}")
print(f"État final: {x_traj[:, -1]}")
```

### 2. Contrôle complet avec les 4 sous-systèmes

```python
from LinearMPC_template.MPCVelControl import MPCVelControl

# Créer le contrôleur complet
Ts = 0.05
H = 5.0
rocket = Rocket(Ts=Ts, model_params_filepath="rocket.yaml")
mpc = MPCVelControl().new_controller(rocket, Ts, H)

# Condition initiale complète (12 états)
x0 = np.array([
    0, 0, 0,              # vitesses angulaires
    0, 0, np.deg2rad(40), # angles (gamma=40°)
    5.0, 5.0, 5.0,        # vitesses (5 m/s chacune)
    0, 0, 10              # positions
])

# Simulation en boucle fermée
sim_time = 10.0
t_cl, x_cl, u_cl, t_ol, x_ol, u_ol, _ = rocket.simulate_control(
    mpc, sim_time, H, x0, method="linear"
)
```

## Formulation MPC

Chaque contrôleur résout le problème d'optimisation :

```
min  Σ (xₖ-xᵣₑf)ᵀQ(xₖ-xᵣₑf) + (uₖ-uᵣₑf)ᵀR(uₖ-uᵣₑf) + (xₙ-xᵣₑf)ᵀP(xₙ-xᵣₑf)
u,x  k=0...N-1

s.t. x₀ = x(t)                          (condition initiale)
     xₖ₊₁ = Adxₖ + Bduₖ                (dynamique discrète)
     uₘᵢₙ ≤ uₖ ≤ uₘₐₓ                  (contraintes entrée)
     xₘᵢₙ ≤ xₖ ≤ xₘₐₓ                  (contraintes état)
```

Où :
- `Q` : matrice de coût sur les états
- `R` : matrice de coût sur les entrées
- `P` : matrice de coût terminal (calculée via LQR)
- `N` : nombre de pas de temps dans l'horizon

## Tuning des paramètres

### Horizon H
- **Valeur recommandée** : 5.0 - 10.0 secondes
- Augmenter pour de meilleures performances (mais augmente le temps de calcul)
- Réduire si le problème devient trop lent à résoudre

### Matrices de coût Q et R

**Pour une réponse plus rapide** :
- Augmenter les poids sur les états à contrôler dans Q
- Exemple : `Q = diag([1, 10, 200])` pour pénaliser fortement la vitesse

**Pour des entrées plus douces** :
- Augmenter R
- Exemple : `R = array([[10.0]])` au lieu de `R = array([[1.0]])`

### Valeurs actuelles

**MPCControl_xvel** :
- `Q = diag([1.0, 10.0, 100.0])`  # wy, beta, vx
- `R = array([[1.0]])`

**MPCControl_yvel** :
- `Q = diag([1.0, 10.0, 100.0])`  # wx, alpha, vy
- `R = array([[1.0]])`

**MPCControl_zvel** :
- `Q = array([[100.0]])`  # vz
- `R = array([[1.0]])`

**MPCControl_roll** :
- `Q = diag([1.0, 100.0])`  # wz, gamma
- `R = array([[1.0]])`

## Exigences de performance

✓ Stabilisation des vitesses et angle de roulis à zéro  
✓ Temps de stabilisation ≤ 7 secondes depuis :
  - 5 m/s pour vx, vy, vz
  - 40° pour gamma  
✓ Satisfaction récursive des contraintes d'état et d'entrée

## Dépendances

- `numpy` : calculs matriciels
- `cvxpy` : résolution du problème d'optimisation
- `control` : calcul LQR pour coût terminal
- `scipy` : discrétisation du système

## Notes d'implémentation

1. **Discrétisation** : Les matrices continues A et B sont discrétisées via `scipy.signal.cont2discrete`

2. **Solveur** : OSQP est utilisé pour résoudre le problème QP avec `warm_start=True` pour accélérer

3. **Robustesse** : En cas d'échec du solveur, le contrôleur retourne l'entrée d'équilibre comme fallback

4. **Coût terminal** : Calculé automatiquement via LQR pour garantir la stabilité

## Troubleshooting

**Le solveur ne converge pas** :
- Réduire l'horizon H
- Vérifier que les contraintes ne sont pas contradictoires
- Augmenter les tolérances du solveur

**Réponse trop lente** :
- Augmenter les poids dans Q pour les états importants
- Augmenter l'horizon H
- Réduire les poids dans R

**Entrées trop agressives** :
- Augmenter R
- Réduire Q
- Ajouter des contraintes de taux de variation (slew rate)
