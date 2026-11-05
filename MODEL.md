# Model & MPC Formulation

Full state/input definition, continuous-time dynamics, cost function,
constraints, and tuning reference for the quadrotor MPC. See
[README.md](README.md) for the project overview and quickstart, and
[paper/paper.pdf](paper/paper.pdf) for the occlusion-cost derivation and
main result.

## State and inputs

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
| p_cmd | 1 | rad/s | commanded roll rate |
| q_cmd | 2 | rad/s | commanded pitch rate |
| r_cmd | 3 | rad/s | commanded yaw rate |

The inputs are thrust and commanded body rates — matching the interface of a real flight controller such as PX4, which exposes a body-rate setpoint API. The dynamics of the inner rate controller are captured by a first-order lag on each body rate (time constants in `config/quadrotor.yaml` under `inner_loop`).

## Continuous-time ODE

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

This transformation has a kinematic singularity at θ = ±90°. The pitch constraint (±45°) in the OCP keeps the trajectory well away from it.

**Rotational dynamics** — first-order lag modelling the PX4 attitude-rate controller:
```
ṗ = (p_cmd − p) / τ_rp
q̇ = (q_cmd − q) / τ_rp
ṙ = (r_cmd − r) / τ_yaw
```

The actual PX4 rate controller is a second-order system (wn ≈ 25 rad/s for roll/pitch, wn ≈ 4 rad/s for yaw). A first-order lag with τ = 1/wn is a standard approximation that captures the dominant closed-loop pole. Since τ_rp = 0.04 s ≈ Ts, the inner-loop dynamics are not negligible and must be in the prediction model.

## Physical parameters

All parameters are in [`config/quadrotor.yaml`](config/quadrotor.yaml):

| Parameter | Default | Description |
|---|---|---|
| mass | 1.0 kg | total vehicle mass |
| gravity | 9.81 m/s² | gravitational acceleration |
| inner_loop.tau_rp | 0.04 s | roll/pitch rate time constant (1/wn, wn ≈ 25 rad/s) |
| inner_loop.tau_yaw | 0.25 s | yaw rate time constant (1/wn, wn ≈ 4 rad/s) |

Inertia (Ix, Iy, Iz) is no longer in the ODE — with body-rate inputs, inertia is implicitly captured by the inner-loop time constants.

## Cost function

ACADOS uses a **Nonlinear Least-Squares** cost:

```
min  Σ_{k=0}^{N-1} ‖h(x_k, u_k) − y_ref_k‖²_W  +  ‖h_e(x_N) − y_ref_N‖²_{W_e}
```

The output map `h` is built incrementally in `scripts/generate_mpc.py` from a
fixed core plus a handful of opt-in residual slots — `obstacle_avoidance.mode`,
`occlusion_avoidance.mode`, `trailer.distance_mode`, `camera.center_mode` in
`config/quadrotor.yaml` each add or remove one slot, so `ny`/`ny_e` are
derived, not hardcoded. In the **default config** — `trailer.distance_mode:
fixed`, `obstacle_avoidance.mode: slack`, `occlusion_avoidance.mode:
disabled`, `camera.center_mode: disabled` — the map is:

```
h(x, u)  = [x, y, z, vx, vy, vz, φ, θ, p, q, r, r_fov, r_dist, r_elev, r_azimuth, T, p_cmd, q_cmd, r_cmd]   (ny = 19)
h_e(x)   = [x, y, z, vx, vy, vz, φ, θ, p, q, r, r_fov, r_dist, r_elev, r_azimuth]                            (ny_e = 15)
```

Notes on what changed from a plain position/attitude tracker:

- **Yaw (ψ) has no dedicated cost term.** It settles wherever the FOV-cone
  residual `r_fov` allows — a soft "stay in frame" constraint, not a pull
  toward a specific heading.
- **Realized body rates `p, q, r`** (weight 5/5/5) are penalised separately
  from the **rate commands** `p_cmd, q_cmd, r_cmd` (weight 1/1/1, end of the
  vector) — the plant's second-order lag means a smooth commanded-rate
  sequence doesn't by itself guarantee a smooth realized rate.
- `r_fov = softplus(cos(α_FOV) − cos(γ))` — zero inside the FOV cone, grows
  smoothly outside it (γ = bearing angle to the leader).
- `r_dist = ‖p_drone,xy − p_leader,xy‖ − D_DESIRED` — ground-plane-only
  standoff distance (not 3-D range; see `r_elev`). `trailer.distance_mode:
  band` replaces it with two squared hinges (`r_dist_min`, `r_dist_max`), a
  dead zone over `[D_MIN, D_MAX]` (+1 to `ny`/`ny_e`).
- `r_elev = (z_drone − z_leader) − horiz_dist·tan(target_bearing_deg)` —
  look-down-angle residual, decoupled from `r_dist` so the solver can't
  trade standoff distance for elevation.
- `r_azimuth = 1 − cos(φ_rel − azimuth_target)` — "which side of the
  leader" residual (a position angle, not the drone's own heading).
- Collision avoidance is a slacked nonlinear **constraint** by default, not
  a cost term (see Constraints below). `obstacle_avoidance.mode: cost`
  swaps in a legacy squared-hinge `r_obs` residual instead (+1 to `ny`/`ny_e`),
  kept only for A/B comparison — see [TOOLING.md](TOOLING.md#config-variants-ab-comparisons).
- `occlusion_avoidance.mode: cost` adds an opt-in `r_occ` residual (+1) — see
  [TOOLING.md](TOOLING.md#occlusion-aware-cost-implementation) below.
- `camera.center_mode: cost` adds an opt-in `r_center = 1 − cos(γ)` residual
  (+1) — a continuous pull toward dead-center, distinct from `r_fov`'s
  keep-in-cone dead zone. Exploratory; see TOOLING.md.

Stage weights `W` (diagonal), shipped default (`config/quadrotor.yaml`):

| Output | Weight | Rationale |
|---|---|---|
| x, y, z | 100 | tight position tracking (trailer-point reference) |
| vx, vy, vz | 10 | smooth velocity profile |
| φ, θ | 5 | keep drone roughly level |
| p, q, r (realized rates) | 5 | damp oscillation the rate commands alone don't catch |
| r_fov | 300 | keep the leader inside the camera cone — high priority |
| r_dist, r_elev, r_azimuth | 0 | computed every solve, not penalised by default — x/y/z tracking is the sole position/azimuth authority; see the header comment in `config/quadrotor.yaml` |
| T, p_cmd, q_cmd, r_cmd | 0.1 / 1 / 1 / 1 | regularise inputs |

Terminal weights `W_e` roughly double the stage weights on position, velocity,
attitude, and rates to encourage stability at the end of the horizon
(`r_dist`/`r_elev`/`r_azimuth` stay at 0 for the same reason as above).

At each MPC step the reference `y_ref_k` (position, FOV target, azimuth
direction, ESDF linearization, …) is set at runtime from C++
(`acados_ocp_solver_set(..., "yref", ...)` / `"p"` for the runtime params) —
there is no fixed symbolic reference baked into the generated solver.

Ready-made `W_stage`/`W_terminal` pairs for every mode combination
(`obstacle_avoidance.mode`, `occlusion_avoidance.mode`,
`trailer.distance_mode`) live in [`config/variants.yaml`](config/variants.yaml).

## Constraints

| Quantity | Bound | Reason |
|---|---|---|
| T | [0, 2·T_hover] | thrust is one-directional; 2× hover as saturation limit |
| p_cmd, q_cmd | ±3.8397 rad/s (±220°/s) | PX4 stock `MC_ROLLRATE_MAX`/`MC_PITCHRATE_MAX` |
| r_cmd | ±3.4907 rad/s (±200°/s) | PX4 stock `MC_YAWRATE_MAX` |
| φ, θ | ±45° | PX4 stock `MPC_TILTMAX_AIR` |
| z | ≥ 0.3 m | hard ground-clearance floor — a true box constraint (`lbx`), not traded off against any cost term; clears rotor-downwash ground effect |
| `d_lin(p)` (ESDF, linearized) | ≥ `d_safe` (0.32 m default) | **collision avoidance** — a slacked nonlinear inequality constraint, not a cost residual: `obstacle_avoidance.mode: slack` (default). The slack carries an exact linear penalty (weight 1e4) so `d_safe` is enforced exactly wherever the problem is feasible, not just asymptotically approached the way a squared-hinge cost term would. `mode: cost` swaps in the legacy residual for A/B comparison only. |

Rate *commands* are bounded above; the *realized* rates `p, q, r` are only
shaped through the cost term above, not box-constrained.

## Horizon

| Setting | Value |
|---|---|
| N | 20 steps |
| Ts | 0.05 s |
| Look-ahead | 1.0 s |

## Solver

- **NLP solver:** SQP-RTI (one Real-Time Iteration per MPC cycle — suitable for embedded/fast execution)
- **QP solver:** PARTIAL_CONDENSING_HPIPM
- **Integrator:** ERK (explicit Runge-Kutta), RK4, 1 step per interval
- **Hessian:** Gauss-Newton approximation

## Design decisions

### Python for formulation, C++ for execution

The model, cost function, and OCP are defined in Python using CasADi's symbolic math and the `acados_template` API: CasADi lets the ODE and cost map stay readable symbolic expressions while ACADOS auto-differentiates exact Jacobians/Hessians, instead of hand-coding derivatives in C++.

Python is a **build-time** tool only — the output is generated C code compiled into the binary, so there's no Python dependency or interpreter overhead at runtime.

### Separate model and MPC scripts

`quadrotor_model.py` contains only the physics: state definitions, ODE, and no control policy. `generate_mpc.py` is the control design layer: cost weights, constraints, horizon, and solver settings.

The split exists because the model is shared by two independent consumers — the OCP solver (`generate_mpc`) and the plant integrator (`generate_sim`) — and coupling the dynamics to a specific controller would make it harder to swap formulations later (e.g. adding yaw pointing changes the cost but not the ODE).

## How to tune

Most tunable parameters live in **[`config/quadrotor.yaml`](config/quadrotor.yaml)**; the circle-trajectory shape is a separate file, **[`config/target.yaml`](config/target.yaml)**. Edit either, then re-run the generator to regenerate the solver.

```yaml
# config/quadrotor.yaml
model:               # physical properties (mass, inertia, gravity)
mpc:                 # N (horizon steps) and Ts (sample time)
constraints:         # thrust/rate/angle limits, ground-clearance floor
cost:                # W_stage and W_terminal diagonal weight vectors
camera:              # FOV half-angle, softplus sharpness, centering mode
trailer:             # standoff distance/band, look-down angle, azimuth smoothing
obstacle_avoidance:  # d_safe, slack weights, slack-vs-cost mode
occlusion_avoidance: # tau_occ, disabled-vs-cost mode

# config/target.yaml
circle:               # radius, altitude, lap period
```

After any change, re-run `./run.sh` (or `./run.sh --skip-gen` if only C++ source changed).
