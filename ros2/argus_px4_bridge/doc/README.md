# argus_px4_bridge

ROS2 node that drives PX4 (SITL or hardware) with argus's circle-tracking
MPC. It links the `quadrotor_mpc` target from the main `argus` repo directly
(via `add_subdirectory` in `CMakeLists.txt`) and calls
`argus_mpc::QuadrotorMpc::step()` every tick — argus itself has no ROS2 or
PX4 dependency; this package is the only place that bridges the two.

## How to launch

Prerequisites: `~/PX4-Autopilot` built (`make px4_sitl gz_x500_depth` once,
standalone — or `gz_x500` if you don't need the camera), `Micro-XRCE-DDS-Agent`
installed, `ros_gz_bridge` installed (`ros2 pkg prefix ros_gz_bridge` should
resolve), and a colcon workspace with `px4_msgs`, `px4_ros_com`, and this
package (`argus/ros2/argus_px4_bridge`, symlinked into the workspace's `src/`)
built:

```bash
source ~/ros2_jazzy/install/setup.bash   # or your ROS2 distro's setup.bash
source ~/ros2_ws/install/setup.bash
```

### Manual (recommended the first time — easier to see what's happening)

1. **Start the DDS agent:**
   ```bash
   MicroXRCEAgent udp4 -p 8888
   ```
2. **Start PX4 SITL** (separate terminal; `HEADLESS=1` to skip the Gazebo GUI).
   To get the depth camera + box world the launch file uses by default, first
   symlink the world in (one-time, or after editing `worlds/argus_box_world.sdf`):
   ```bash
   ln -sf ~/thesis_to_cpp_ws/argus/ros2/argus_px4_bridge/worlds/argus_box_world.sdf \
     ~/PX4-Autopilot/Tools/simulation/gz/worlds/argus_box_world.sdf
   cd ~/PX4-Autopilot && HEADLESS=1 PX4_GZ_WORLD=argus_box_world make px4_sitl gz_x500_depth
   ```
   (Plain `make px4_sitl gz_x500` with no `PX4_GZ_WORLD` set also still works —
   PX4's own `default.sdf`, no camera, no box.)
3. **Once it reaches the `pxh>` prompt**, set the two params that bypass
   SITL-only preflight checks (no simulated battery/GCS present — see
   `powerCheck.cpp`/`rcAndDataLinkCheck.cpp` in PX4 for what these guard):
   ```
   param set CBRK_SUPPLY_CHK 894281
   param set NAV_DLL_ACT 0
   ```
   Also worth checking once per vehicle model: `param show MPC_THR_HOVER`,
   and pass it via `--ros-args -p mpc_thr_hover:=<value>` below if it's not `0.6`.
4. **Bridge the cameras into ROS2** (separate terminal — only if you
   started `gz_x500_depth` above; skip for `gz_x500`). The RGB topic is
   namespaced under the world/vehicle-instance names — this assumes
   `argus_box_world` and the first (only) vehicle instance, matching the
   launch file's own default derivation:
   ```bash
   ros2 run ros_gz_bridge parameter_bridge \
     "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image" \
     "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" \
     "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
     "/world/argus_box_world/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image" \
     --ros-args -r /world/argus_box_world/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera/image_raw
   ```
5. **Run the bridge node** (separate terminal):
   ```bash
   ros2 run argus_px4_bridge argus_bridge_node --ros-args -p hover_only:=true
   ```
   It will stream a hold setpoint, arm, switch to offboard, climb to the
   circle's start point, then either hold there (`hover_only:=true`) or start
   MPC circle tracking (`hover_only:=false`, the parameter's own default) —
   watch its log for the state transitions (`Armed + offboard...` →
   `On station...`).

### Launch file (one command — does everything above automatically)

```bash
ros2 launch argus_px4_bridge argus_sitl.launch.py
```

Starts the Agent and PX4 SITL, then:

1. Symlinks `worlds/<px4_world>.sdf` from this package's share directory into
   `<px4_dir>/Tools/simulation/gz/worlds/`, re-linking on every launch so
   edits to the world file take effect without a PX4 rebuild. This has to be
   a physical file under `px4_dir` — PX4's own `gz_env.sh` unconditionally
   re-exports `PX4_GZ_WORLDS` back to `Tools/simulation/gz/worlds` at
   startup, so pointing that env var at a directory in this repo instead
   gets silently clobbered. (`PX4_GZ_WORLD`, the world *name* passed to
   `make`, isn't touched by `gz_env.sh` and works as a normal env var.)
2. Watches the SITL process's own stdout for the `pxh>` prompt and pushes
   `param set CBRK_SUPPLY_CHK 894281` / `param set NAV_DLL_ACT 0` into it the
   moment it appears (via a FIFO wired to the process's stdin — the
   interactive shell PX4 exposes on stdin never sees EOF, so it keeps
   accepting input for the rest of the run).
3. Polls `ros2 topic echo /fmu/out/vehicle_odometry --once` in a loop
   (bounded to 90 s) as a **real** readiness check — the earliest point
   PX4, Gazebo, and the DDS bridge are all confirmed up — rather than
   guessing a fixed boot delay. Fails loudly (`odometry_wait_failed`) instead
   of silently starting the node if that never happens.
4. Waits a further 5 s settle buffer after that (arming immediately on the
   first odometry sample was observed to trigger sensor-TIMEOUT failsafes in
   SITL), then starts `argus_bridge_node`.
5. Starts `gz_camera_bridge` (a `ros_gz_bridge parameter_bridge` instance)
   bridging `/depth_camera`, `/depth_camera/points`, and `/camera_info` from
   Gazebo into ROS2 — see [Depth camera](#depth-camera--world) below. Started
   unconditionally and immediately (it just waits for the gz topics to exist),
   regardless of `launch_gazebo`, so it also works against an
   already-running instance.
6. (GUI runs only, i.e. `headless=0`) Resends the model-follow camera command
   (`gz topic -t /gui/track ...`) every 2 s for five attempts. PX4's own init
   script only sends this once, right after spawning the model — if the GUI
   client is still starting up at that exact moment (it takes a few seconds
   to initialize rendering and subscribe), it misses the message entirely
   and the camera never finds the drone. There's no clean external signal
   for "the GUI has subscribed", so this just resends across a window
   comfortably longer than typical GUI startup instead of guessing one delay.

Launch arguments:

| Argument | Default | Description |
|---|---|---|
| `px4_dir` | `~/PX4-Autopilot` | Path to the PX4-Autopilot source tree. |
| `headless` | `0` | `0` shows the Gazebo GUI, `1` hides it. **Note:** PX4's own `px4-rc.gzsim` decides this by checking whether `HEADLESS` is *unset*, not whether it's `"0"` — the launch file handles that translation for you (leaves the env var unset entirely when `headless=0`). |
| `mpc_thr_hover` | `0.6` | Forwarded to the node's `mpc_thr_hover` param — see below. |
| `rviz` | `1` | `1` launches RViz with the checked-in visualization config (see below), `0` skips it. |
| `launch_gazebo` | `1` | `1` starts the Agent + PX4 SITL/Gazebo. `0` assumes they're already running externally (e.g. started by hand per the "Manual" section above, or left running from a previous launch) and only starts the bridge node (+ RViz) against them. |
| `px4_model` | `gz_x500_depth` | PX4 `make px4_sitl <target>` model target. `gz_x500` for the plain airframe with no camera. |
| `px4_world` | `argus_box_world` | Gazebo world name (no `.sdf`), looked up in this package's `worlds/` dir. `argus_box_world` is PX4's `default.sdf` plus one static box obstacle on the circle-start hover point's boresight — see [Depth camera & world](#depth-camera--world). Use `default` (or any other name under `PX4-Autopilot/Tools/simulation/gz/worlds/`) for PX4's stock worlds. |
| `hover_only` | `1` | `1` holds position at the circle-start point after takeoff instead of handing off to MPC circle tracking — set up for obstacle/depth-camera testing where the drone should stay put near `argus_box_world`'s box. `0` restores the original circle-tracking behavior. |

```bash
ros2 launch argus_px4_bridge argus_sitl.launch.py headless:=1 mpc_thr_hover:=0.5
```

`rviz:=0` skips launching RViz if you don't want it (see below).

Other common variations:

```bash
# Watch it in the Gazebo GUI instead of headless
ros2 launch argus_px4_bridge argus_sitl.launch.py headless:=0

# Go back to circle-tracking instead of hovering in place
ros2 launch argus_px4_bridge argus_sitl.launch.py hover_only:=0

# Original plain model/world — no camera, no box, for comparing against
# pre-depth-camera behavior
ros2 launch argus_px4_bridge argus_sitl.launch.py px4_model:=gz_x500 px4_world:=default
```

### Re-launching: check for stale processes first

If a Gazebo/PX4 process from a previous run doesn't fully exit (observed
happening even after what looked like a successful Ctrl-C or `pkill`), the
*next* launch's `px4-rc.gzsim` will silently detect that a world is "already
running" and just attach to it — you'll get the old world/model instead of
whatever `px4_world`/`px4_model` you just asked for, with no error. Before
re-launching, especially after killing a previous run, check for leftovers
and force-kill anything still there:

```bash
ps aux | grep -E "gz sim|px4_sitl_default"
kill -9 <pid> ...
```

(worth checking `MicroXRCEAgent`, `ros_gz_bridge`, and `rviz2` too if things
still look wrong after that — `ps aux | grep -E "MicroXRCEAgent|parameter_bridge|rviz2"`)

## Visualization (RViz)

The node also publishes RViz-ready topics, all in a fixed `map` frame
(ENU, no TF tree needed — `map` is set as RViz's own Fixed Frame, so no
lookup is required):

| Topic | Type | What it shows |
|---|---|---|
| `/argus/reference_path` | `nav_msgs/Path` | The fixed circle the drone is commanded to track (`ARGUS_CIRCLE_R`/`ARGUS_HOVER_ALTITUDE`), sampled once at 100 points. |
| `/argus/drone_path` | `nav_msgs/Path` | The drone's actual flown trajectory, accumulated tick by tick (capped at the last ~200 s / 4000 points so it doesn't grow unbounded on a long-running flight). |
| `/argus/fov_markers` | `visualization_msgs/MarkerArray` | A wireframe pyramid (`id=0`, 4 corners) showing the camera FOV — body `+x` boresight, half-angle `fov_half_angle_deg` (default `40°`, must match `config/quadrotor.yaml`'s `camera.fov_half_angle_deg`) — drawn out to the current range to the target, plus a sphere (`id=1`) marking the target itself at the circle's center. Mirrors the OCP's `r_fov` cost term directly: the cost is rotationally symmetric (a true circular cone, not an actual rectangular camera frustum), so the pyramid is a simplified stand-in at the same half-angle, not a literal FOV shape — empty of penalty exactly when the target sphere sits inside it. |
| `/argus/obstacle_markers` | `visualization_msgs/MarkerArray` | A `CUBE` marker (`id=0`) matching `argus_box_world`'s box obstacle — RViz has no visibility into Gazebo's own world geometry, so this is a second, hand-kept copy of the box's pose/size (`kObstacleBoxPos`/`kObstacleBoxSize` in `argus_bridge_node.cpp`) purely for visualization. Nothing keeps it in sync with `worlds/argus_box_world.sdf` automatically — update both if the box ever moves. |

Plus, when `gz_camera_bridge` is running (see below), an `Image` display on
`/camera/image_raw` (RGB), an `Image` display on `/depth_camera` (depth), and
a `PointCloud2` display on `/depth_camera/points`.

`ros2 launch argus_px4_bridge argus_sitl.launch.py` starts `rviz2` automatically
(`rviz:=0` to skip it) with a checked-in config
(`ros2/argus_px4_bridge/rviz/argus.rviz`) that already has all displays
set up. To view them against a node started manually instead, just run
`rviz2 -d $(ros2 pkg prefix argus_px4_bridge)/share/argus_px4_bridge/rviz/argus.rviz`
in a separate terminal.

## Depth camera & world

The default `px4_model` (`gz_x500_depth`) is PX4's stock `x500` airframe with
an OakD-Lite camera bolted on (`PX4-Autopilot/Tools/simulation/gz/models/x500_depth`).
It publishes on Gazebo-transport topics that `gz_camera_bridge` bridges into
ROS2 as-is (same names, `sensor_msgs` types):

| Gazebo topic | ROS2 topic | Type | Notes |
|---|---|---|---|
| `/depth_camera` | `/depth_camera` | `sensor_msgs/msg/Image` | `640x480`, `32FC1` (metres, float32). Named explicitly in the OakD-Lite model's SDF, so unlike PX4's other gz topics it isn't namespaced under `/world/<world>/model/<model>/...` — the name is stable regardless of world or vehicle instance. |
| `/depth_camera/points` | `/depth_camera/points` | `sensor_msgs/msg/PointCloud2` | Same sensor, as a point cloud. |
| `/camera_info` | `/camera_info` | `sensor_msgs/msg/CameraInfo` | Matches the depth image's intrinsics. |
| `/world/<px4_world>/model/<px4_model minus "gz_">_0/link/camera_link/sensor/IMX214/image` | `/camera/image_raw` | `sensor_msgs/msg/Image` | The OakD-Lite's plain RGB feed. Unlike the depth topics above, its gz topic *is* namespaced under `/world/<world>/model/<instance>/...`, so the launch file builds it from `px4_world`/`px4_model` and remaps it to the short name via the `Node`'s `remappings` (see [argus_sitl.launch.py:124-146](../launch/argus_sitl.launch.py)). Assumes a single vehicle instance (`_0`) — this launch file doesn't support spawning more than one. Only exists at all for a camera-equipped model (`gz_x500_depth`); harmless no-op if you're running plain `gz_x500`. |

`argus_box_world` (the default `px4_world`) is PX4's stock `default.sdf` with
one static `0.5 x 0.5 x 1.5 m` box added at `(0.3, 0.8, 0.75)` in ENU — near
the boresight from the drone's circle-start hover point
`(ARGUS_CIRCLE_R, 0, ARGUS_HOVER_ALTITUDE) = (2, 0, 1.5)`, nose toward the
origin, but offset laterally (`y=0.8`, not `0`). That offset isn't
cosmetic: the takeoff setpoint (`publishTakeoffSetpoint()` in
`argus_bridge_node.cpp`) is a single direct position command from spawn to
the hover point, and PX4 flies roughly a straight line to it holding
`y≈0` the whole climb — a `y=0` box with height matching the hover
altitude sits squarely on that line no matter its x position (confirmed by
an actual mid-takeoff collision during development; a first attempt at
`y=0.45` still clipped it, hence the larger offset here). With
`hover_only:=1` (the default) the drone parks at the hover point after
takeoff, so the box is visible to the depth camera immediately with no
circle tracking needed. The world's `<world name="...">`
is deliberately set to match the filename (`argus_box_world`, not PX4's
usual `default`) — PX4's world-readiness check polls a gz-transport topic
built from `PX4_GZ_WORLD`/the `px4_world` arg, so a mismatch there makes
SITL hang forever waiting for a world that never reports ready under that
name.

## Node: `argus_bridge_node`

A single node, `argus/ros2/argus_px4_bridge/src/argus_bridge_node.cpp`, built
around one `rclcpp::TimerBase` callback (`tick()`) and a 3-state state
machine:

```
INIT ──(armed + offboard)──▶ TAKEOFF ──(on station)──▶ MPC_TRACKING
                                  │
                                  └──(hover_only:=true)──▶ stays in TAKEOFF, holding position
```

- **INIT** — streams a hold setpoint, waits ~11 cycles, then requests
  offboard mode and arms. Transitions once PX4 reports both `ARMED` and
  `OFFBOARD`.
- **TAKEOFF** — holds a plain PX4 position/yaw setpoint at the circle's own
  `t=0` point (`(R, 0, hover_altitude)`, nose toward the origin) rather than
  straight up above the center. The MPC is only ever validated starting
  already on the circle (see `mpc_controller.cpp` in the main repo); handing
  off from a hover above the center would present it with a full
  radius-length position/yaw step it's never had to recover from. Once
  position error and velocity drop below tolerance: transitions to
  `MPC_TRACKING`, unless the `hover_only` parameter is set, in which case it
  just keeps re-publishing the same hold setpoint indefinitely instead.
- **MPC_TRACKING** — builds the `N`-step circle horizon for the current time,
  calls `mpc_.step()`, and publishes the resulting body-rate command.

## Timing

The node's tick rate is **20 Hz** (`Ts = 0.05 s`), set directly from
`ARGUS_MPC_TS` — the same `Ts` argus's own OCP was generated with
(`config/quadrotor.yaml` → `generate_mpc.py` → `argus_params.h`):

```cpp
const auto period = std::chrono::duration<double>(ARGUS_MPC_TS);
timer_ = this->create_wall_timer(..., std::bind(&ArgusBridgeNode::tick, this));
```

This isn't an arbitrary choice: the OCP's horizon (`N=20` steps of `Ts` each)
and its SQP-RTI solver (one real-time-iteration per call) both assume
`mpc.step()` is called exactly once per `Ts` — each horizon stage `j` is
implicitly `j * Ts` seconds ahead of "now". Ticking slower would desync the
horizon's time axis from wall-clock time; ticking faster wouldn't converge
the RTI solver any further per call. 20 Hz is also comfortably above the
≥2 Hz PX4 requires to stay in `OFFBOARD` without failing safe, but that's a
secondary constraint — argus's own control-design timing is the binding one.

Every tick, regardless of state, also publishes `OffboardControlMode` — PX4
needs that heartbeat continuously or it drops out of offboard.

## Pipeline (per tick)

1. **Read state** — latest `VehicleOdometry` (position, velocity, attitude
   quaternion, body rates), held from the last subscription callback.
2. **Frame conversion** — PX4's `/fmu` topics are NED world / FRD body.
   argus's model is ENU-ish (z-up) world / FLU body (thrust along body +z).
   Conversions go through `px4_ros_com::frame_transforms`:
   - position, velocity: `ned_to_enu_local_frame`
   - attitude: `px4_to_ros_orientation`, then converted to ZYX Euler angles
     with a **hand-rolled closed-form formula**, not
     `frame_transforms::quaternion_to_euler` — that helper wraps Eigen's
     generic `eulerAngles()`, which branch-flips to the equivalent
     `(roll+π, −pitch, yaw−π)` representation whenever yaw crosses ±90°
     (unrelated to the true ZYX gimbal lock at pitch=±90°). The default
     `gz_x500` spawn yaw (~96°) hits this every time.
   - body rates: `aircraft_to_baselink_body_frame` (incoming),
     `baselink_to_aircraft_body_frame` (outgoing) — a fixed π rotation about
     the forward axis, i.e. `(p,q,r) ↔ (p,−q,−r)`.
3. **State machine** — see above; builds the circle horizon
   (`argus_mpc::Reference` per stage) and the static `TargetState` (circle
   center) only in `MPC_TRACKING`.
4. **Solve** — `mpc_.step(x0, horizon, target)` → `argus_mpc::Command`
   (normalized thrust + body rates, `1.0` thrust == hover in argus's own
   convention).
5. **Publish** — converted back to PX4's FRD/normalized-thrust convention
   and sent as a `VehicleRatesSetpoint`.

## Subscribers

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | `rmw_qos_profile_sensor_data` (best-effort, depth 5) | Assumes NED pose/velocity frame (PX4 SITL default) — `pose_frame`/`velocity_frame` fields aren't checked per-message. |
| `/fmu/out/vehicle_status_v4` | `px4_msgs/VehicleStatus` | `rmw_qos_profile_sensor_data` | Version-suffixed topic name — `VehicleStatus` is schema version 4, PX4 does not publish it on the bare `/fmu/out/vehicle_status`. |

## Publishers

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | 20 Hz (every tick) | `position=true` in INIT/TAKEOFF, `body_rate=true` in MPC_TRACKING. |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | 20 Hz, INIT/TAKEOFF only | NED position + yaw hold at the circle's start point. |
| `/fmu/in/vehicle_rates_setpoint` | `px4_msgs/VehicleRatesSetpoint` | 20 Hz, MPC_TRACKING only | Body rates (FRD) + normalized thrust (`thrust_body.z`, negative = up). |
| `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | Once each, at the INIT→arm transition | `VEHICLE_CMD_DO_SET_MODE` (→ offboard) then `VEHICLE_CMD_COMPONENT_ARM_DISARM` (→ arm). |

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `mpc_thr_hover` | double | `0.6` | Normalized PX4 thrust at hover. Must match the vehicle's actual `MPC_THR_HOVER` param — argus's `Command.thrust` is normalized so `1.0 == hover` in its own convention, and `thrust_norm = mpc_thr_hover * cmd.thrust` assumes PX4's thrust curve is linear (`THR_MDL_FAC=0`, the SITL default). Check with `param show MPC_THR_HOVER` in the PX4 shell before trusting altitude hold on a new vehicle. |
| `fov_half_angle_deg` | double | `40.0` | Visualization only — half-angle of the FOV cone drawn on `/argus/fov_markers`. Must match `config/quadrotor.yaml`'s `camera.fov_half_angle_deg`; not auto-generated into `argus_params.h` since it doesn't feed the OCP itself. |
| `hover_only` | bool | `false` | When `true`, stays parked at the `TAKEOFF` hold setpoint instead of transitioning to `MPC_TRACKING` once on station — see the state machine above. The launch file's `hover_only` arg defaults to `1`/true, so this node-level default (`false`) only applies when running `argus_bridge_node` directly without it. |

## Known issues

- **Yaw-rate instability**: commanding `cmd.yaw_rate` as computed causes a
  runaway yaw spin-up (20–30 rad/s within ~2 s) in SITL. A diagnostic run
  with `yaw_rate` forced to 0 stayed bounded, confirming yaw is the dominant
  destabilizer — not yet root-caused (suspect PX4's actual rate-controller
  gains for `gz_x500` vs. argus's assumed `tau_yaw=0.25s`, or a remaining
  sign/scale issue).
- **Residual roll/pitch drift**: even with yaw disabled, roll/pitch alone
  still drift past the OCP's ±60° box constraint over tens of seconds,
  eventually diverging. Milder than the yaw issue but also unresolved.

Do not trust this node for anything beyond SITL experimentation until both
are fixed.
