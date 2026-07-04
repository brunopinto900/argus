# Argus — Quadrotor Circle-Tracking MPC

Model Predictive Control for a quadrotor tracking a circular trajectory. The controller is formulated as a Nonlinear MPC using [ACADOS](https://docs.acados.org/), which generates a C solver that is called from C++.

**Planned feature:** yaw/heading locked toward an object at the centre of the circle (active gimbal-like pointing). Not yet active — psi is currently excluded from the cost.

---

## Model

### State and inputs

The quadrotor is modelled with 12 states and 4 inputs, using ZYX Euler angles.

**State** `x ∈ ℝ¹²`:

| Symbol | Indices | Unit | Description |
|---|---|---|---|
| x, y, z | 0–2 | m | position in world frame |
| vx, vy, vz | 3–5 | m/s | velocity in world frame |
| φ, θ, ψ | 6–8 | rad | roll, pitch, yaw (ZYX Euler) |
| p, q, r | 9–11 | rad/s | body-frame angular rates |

**Input** `u ∈ ℝ⁴`:

| Symbol | Index | Unit | Description |
|---|---|---|---|
| T | 0 | N | collective thrust (sum of all rotor forces) |
| τ_φ | 1 | Nm | roll torque |
| τ_θ | 2 | Nm | pitch torque |
| τ_ψ | 3 | Nm | yaw torque |

The inputs are thrust and body torques — not individual motor speeds. This is a standard abstraction for MPC: the low-level motor mixing that maps (T, τ) to individual rotor RPMs is assumed to run on a separate, faster inner loop.

### Continuous-time ODE

**Position kinematics** — velocity integrates into position:
```
ẋ  = vx
ẏ  = vy
ż  = vz
```

**Translational dynamics** — thrust T acts along the body z-axis; rotating it to the world frame via R(φ,θ,ψ) gives the acceleration:
```
v̇x = T/m · (cos(ψ)·sin(θ)·cos(φ) + sin(ψ)·sin(φ))
v̇y = T/m · (sin(ψ)·sin(θ)·cos(φ) − cos(ψ)·sin(φ))
v̇z = T/m · cos(θ)·cos(φ) − g
```

The rotation is the ZYX convention: `R = Rz(ψ) · Ry(θ) · Rx(φ)`. When the drone is level (φ = θ = 0) this reduces to `v̇z = T/m − g`, so T = mg at hover.

**Euler angle kinematics** — relates body rates (p, q, r) to Euler rates:
```
φ̇ = p + sin(φ)·tan(θ)·q + cos(φ)·tan(θ)·r
θ̇ =     cos(φ)·q         − sin(φ)·r
ψ̇ =    (sin(φ)·q         + cos(φ)·r) / cos(θ)
```

This transformation has a kinematic singularity at θ = ±90°. The pitch constraint (±60°) in the OCP keeps the trajectory well away from it.

**Rotational dynamics** — Euler's equations with a diagonal inertia matrix (Ix, Iy, Iz):
```
ṗ = (τ_φ   − (Iz − Iy)·q·r) / Ix
q̇ = (τ_θ   − (Ix − Iz)·p·r) / Iy
ṙ = (τ_ψ   − (Iy − Ix)·p·q) / Iz
```

The cross-product (gyroscopic) terms `(Ij − Ik)·ω·ω` are included, so the model captures the coupling between axes that occurs when the drone spins.

### Physical parameters

All parameters are in [`config/quadrotor.yaml`](config/quadrotor.yaml):

| Parameter | Default | Description |
|---|---|---|
| mass | 1.0 kg | total vehicle mass |
| gravity | 9.81 m/s² | gravitational acceleration |
| Ix | 0.01 kg·m² | roll inertia |
| Iy | 0.01 kg·m² | pitch inertia |
| Iz | 0.02 kg·m² | yaw inertia (higher: yaw responds more slowly) |

---

## MPC formulation

### Cost function

ACADOS uses a **Nonlinear Least-Squares** cost:

```
min  Σ_{k=0}^{N-1} ‖h(x_k, u_k) − y_ref_k‖²_W  +  ‖h_e(x_N) − y_ref_N‖²_{W_e}
```

The output map `h` selects which states and inputs are penalised:

```
h(x, u)  = [x, y, z, vx, vy, vz, φ, θ, T, τ_φ, τ_θ, τ_ψ]   (ny = 12)
h_e(x)   = [x, y, z, vx, vy, vz, φ, θ]                       (ny_e = 8, terminal)
```

ψ (yaw) and body rates (p, q, r) are **excluded** from the cost. The drone tracks position and velocity, stays level, and minimises control effort — it does not try to point in any direction.

Stage weights `W` (diagonal):

| Output | Weight | Rationale |
|---|---|---|
| x, y, z | 100 | tight position tracking |
| vx, vy, vz | 10 | smooth velocity profile |
| φ, θ | 5 | keep drone roughly level |
| T, τ_φ, τ_θ, τ_ψ | 0.1 | regularise inputs |

Terminal weights `W_e` double the position and velocity weights (200 / 20) to encourage stability at the end of the horizon.

At each MPC step the reference `y_ref_k` is computed from the circle trajectory evaluated at `t + k·Ts`.

### Constraints

| Quantity | Bound | Reason |
|---|---|---|
| T | [0, 2·T_hover] | thrust is one-directional; 2× hover as saturation limit |
| τ_φ, τ_θ, τ_ψ | ±0.5 Nm | actuator limits |
| φ, θ | ±60° | keep away from kinematic singularity at ±90° |

### Horizon

| Setting | Value |
|---|---|
| N | 20 steps |
| Ts | 0.05 s |
| Look-ahead | 1.0 s |

### Solver

- **NLP solver:** SQP-RTI (one Real-Time Iteration per MPC cycle — suitable for embedded/fast execution)
- **QP solver:** PARTIAL_CONDENSING_HPIPM
- **Integrator:** ERK (explicit Runge-Kutta), RK4, 1 step per interval
- **Hessian:** Gauss-Newton approximation

---

## Architecture

```
config/quadrotor.yaml        ← single source of truth for all parameters

scripts/quadrotor_model.py   ← symbolic ODE (CasADi), reads yaml
scripts/generate_mpc.py      ← OCP + sim solver formulation, writes:
        │
        ├─▶  c_generated_code/acados_solver_quadrotor.c/.h    (OCP solver)
        ├─▶  c_generated_code/acados_sim_solver_quadrotor.c/.h (plant integrator)
        └─▶  c_generated_code/argus_params.h                  (C++ constants)
                 │
                 ▼  compile & link
        src/mpc_controller.cpp   ← C++ simulation loop
```

The Python scripts are a **build tool**, not runtime code. Run them once to produce the C solver; the solver is then compiled into your C++ binary. The plant integrator (`acados_sim_solver_quadrotor`) is generated from the same symbolic model as the OCP — if the dynamics change in Python, both the prediction model and the simulation automatically stay in sync.

---

## Design decisions

### Python for formulation, C++ for execution

The model, cost function, and OCP are defined in Python using CasADi's symbolic math and the `acados_template` API. Python is the right tool here: CasADi lets you write the ODE and cost map as readable symbolic expressions, and ACADOS can automatically differentiate them to generate exact Jacobians and Hessians. Writing this in C++ directly would mean hand-coding derivatives and losing the safety net of symbolic verification.

Python is a **build-time** tool only. The output is generated C code that is compiled into the binary — at runtime there is no Python dependency and no interpreter overhead.

### Separate model and MPC scripts

`quadrotor_model.py` contains only the physics: state definitions, ODE, and no control policy. `generate_mpc.py` is the control design layer: cost weights, constraints, horizon, and solver settings.

The split exists because the model is shared by two independent consumers — the OCP solver (`generate_mpc`) and the plant integrator (`generate_sim`) — and coupling the dynamics to a specific controller would make it harder to swap formulations later (e.g. adding yaw pointing changes the cost but not the ODE). The physics and the control design have different rates of change and different reasons to be modified.

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [ACADOS](https://github.com/acados/acados) | NLP solver and C code generation |
| [CasADi](https://web.casadi.org/) | Symbolic differentiation for dynamics and cost |
| Python 3 + `acados_template` | OCP formulation script |
| C++17 compiler | Application build |
| CMake ≥ 3.16 | Build system |

Assumed install path: `~/acados`. Adjust `ACADOS_ROOT` in `CMakeLists.txt` if yours differs.

---

## How to build

### Step 1 — Generate the C solver (run once, or after any change to the model, cost, or config)

```bash
LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH python3 scripts/generate_mpc.py
```

This writes `c_generated_code/` with the OCP solver, the plant integrator, and `argus_params.h`. The `LD_LIBRARY_PATH` prefix is required on WSL2 because the ACADOS Python wrapper loads shared libraries at generation time.

> **Tip:** Add `export LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH` to your `.bashrc`.

### Step 2 — Build the C++ application

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Step 3 — Run

```bash
./build/argus_mpc
# produces logs/trajectory.csv
python3 scripts/animate_xy.py
```

---

## How to tune

All tunable parameters live in **[`config/quadrotor.yaml`](config/quadrotor.yaml)**. Edit that file, then re-run Step 1 to regenerate the solver.

```yaml
model:          # physical properties (mass, inertia, gravity)
mpc:            # N (horizon steps) and Ts (sample time)
constraints:    # thrust limits, torque limits, angle limits
cost:           # W_stage and W_terminal diagonal weight vectors
circle:         # radius, altitude, lap period
```

After any change, re-run `python3 scripts/generate_mpc.py` and rebuild.

---

## Repository layout

```
argus/
├── config/
│   └── quadrotor.yaml              # all parameters — edit here
├── scripts/
│   ├── config.py                   # YAML loader (shared by Python scripts)
│   ├── quadrotor_model.py          # symbolic ODE (CasADi)
│   ├── generate_mpc.py             # OCP + sim solver code generation
│   └── animate_xy.py               # trajectory visualiser
├── src/
│   └── mpc_controller.cpp          # C++ simulation loop
├── c_generated_code/               # auto-generated C solver — do not edit
├── CMakeLists.txt
└── README.md
```
