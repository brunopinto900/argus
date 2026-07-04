# Argus — Quadrotor Circle-Tracking MPC

Model Predictive Control for a quadrotor tracking a circular trajectory. The controller is formulated as a Nonlinear MPC using [ACADOS](https://docs.acados.org/), which generates a C solver that is called from C++.

**Planned feature:** yaw/heading locked toward an object at the centre of the circle (active gimbal-like pointing). Not yet active — psi is currently excluded from the cost.

---

## Architecture

```
scripts/quadrotor_model.py   ← symbolic ODE (CasADi)
scripts/generate_mpc.py      ← OCP formulation + code generation
        │
        ▼  run once
c_generated_code/            ← C solver (auto-generated, do not edit)
        │
        ▼  link
src/mpc_controller.cpp       ← C++ application (to be added)
```

The Python scripts are a **build tool**, not runtime code. Run them once to produce the C solver; the solver is then compiled into your C++ binary.

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [ACADOS](https://github.com/acados/acados) | NLP solver and C code generation |
| [CasADi](https://web.casadi.org/) | Symbolic differentiation for dynamics and cost |
| Python 3 + `acados_template` | OCP formulation script |
| C++17 compiler | Application build |
| CMake ≥ 3.16 | Build system |

Assumed install path: `~/acados`. Adjust `_acados_lib` in [scripts/generate_mpc.py](scripts/generate_mpc.py) if yours differs.

---

## How to build

### Step 1 — Generate the C solver (run once, or after any model/cost change)

```bash
LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH python3 scripts/generate_mpc.py
```

This writes `c_generated_code/` with the compiled solver and a `Makefile`. The `LD_LIBRARY_PATH` prefix is required on WSL2 because the ACADOS Python wrapper loads shared libraries at generation time.

> **Tip:** Add `export LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH` to your `.bashrc` to avoid typing it every time.

### Step 2 — Build the C++ application (once `src/` exists)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Step 3 — Run

```bash
./build/argus_mpc
```

---

## Repository layout

```
argus/
├── scripts/
│   ├── quadrotor_model.py      # quadrotor ODE — edit to change the model
│   └── generate_mpc.py         # OCP setup (cost, constraints, horizon) — edit to tune
├── c_generated_code/           # auto-generated C solver — do not edit
├── src/                        # C++ application (to be added)
├── CMakeLists.txt              # top-level build (to be added)
├── quadrotor_ocp.json          # OCP spec snapshot — auto-generated
└── README.md
```

---

## How to change the model

Open [scripts/quadrotor_model.py](scripts/quadrotor_model.py).

**Physical parameters** (top of `export_quadrotor_ode_model`):

```python
m  = 1.0     # total mass      [kg]
Ix = 0.01    # roll  inertia   [kg·m²]
Iy = 0.01    # pitch inertia   [kg·m²]
Iz = 0.02    # yaw   inertia   [kg·m²]
```

**Dynamics** — the ODE is written explicitly as `f_expl`, a `ca.vertcat(...)` of 12 expressions in this order:

| Index | State | Governed by |
|---|---|---|
| 0–2 | x, y, z | position kinematics (`vx, vy, vz`) |
| 3–5 | vx, vy, vz | thrust in world frame divided by mass, minus gravity |
| 6–8 | φ, θ, ψ | Euler kinematics (ZYX convention) |
| 9–11 | p, q, r | Euler's equations for body rates |

To add aerodynamic drag, modify the `vx/vy/vz` rows. To switch to quaternions, replace rows 6–11 with the quaternion kinematics and update the state vector size.

**After any change here, re-run Step 1.**

---

## How to change the cost function and constraints

Open [scripts/generate_mpc.py](scripts/generate_mpc.py).

### Horizon and timing

```python
N  = 20      # number of prediction steps
Ts = 0.05    # sampling time [s] → total look-ahead = N × Ts = 1.0 s
```

### Cost weights

The cost is a weighted nonlinear least-squares:
`min Σ ‖h(x,u) − y_ref‖²_W`

The output vector `h` is defined near the top of `generate_mpc()`:

```python
cost_y_expr   = ca.vertcat(x_sym[:8], u_sym)   # [x,y,z, vx,vy,vz, φ,θ, T,τ_φ,τ_θ,τ_ψ]
cost_y_expr_e = x_sym[:8]                       # terminal: same without inputs
```

Weight matrices (diagonal, one entry per element of `cost_y_expr`):

```python
W_STAGE = np.diag([
    100.0, 100.0, 100.0,   # position  x, y, z
     10.0,  10.0,  10.0,   # velocity  vx, vy, vz
      5.0,   5.0,          # attitude  φ, θ  (keeps drone level)
      0.1,                 # thrust regularisation
      0.1,   0.1,   0.1,  # torque regularisation
])
```

To **add yaw pointing** toward a target at the origin, extend `cost_y_expr` with a yaw-error term:

```python
yaw_error = x_sym[8] - ca.atan2(-x_sym[1], -x_sym[0])   # psi - atan2(-y, -x)
cost_y_expr = ca.vertcat(x_sym[:8], yaw_error, u_sym)
```

Then add a corresponding weight row/column to `W_STAGE` and update `W_TERMINAL`.

### Input constraints

```python
T_MIN   = 0.0          # thrust lower bound [N]
T_MAX   = 2.0 * T_HOVER  # thrust upper bound (2× hover)
TAU_MAX = 0.5          # torque bound ±0.5 Nm
```

### State constraints (attitude)

```python
ANGLE_MAX = np.deg2rad(60.0)   # maximum roll / pitch angle
```

Applied to φ (state index 6) and θ (state index 7). Keeps the Euler-angle kinematics away from the gimbal-lock singularity at θ = ±90°.

**After any change here, re-run Step 1.**

---

## State and input reference

| Symbol | Index | Unit | Description |
|---|---|---|---|
| x, y, z | 0–2 | m | position in world frame |
| vx, vy, vz | 3–5 | m/s | velocity in world frame |
| φ (phi) | 6 | rad | roll (rotation about world X) |
| θ (theta) | 7 | rad | pitch (rotation about world Y) |
| ψ (psi) | 8 | rad | yaw (rotation about world Z) |
| p, q, r | 9–11 | rad/s | body-frame angular rates |
| T | 0 | N | collective thrust |
| τ_φ, τ_θ, τ_ψ | 1–3 | Nm | roll / pitch / yaw torques |

Attitude convention: ZYX Euler angles (yaw → pitch → roll). Singularity at θ = ±90°.
