# Controllers

Controllers are operator-input sources. They do not own the policy, environment lifecycle, or command targets.

| Deployment | Controller | Input source |
| --- | --- | --- |
| `sim` | `KeyboardCtrl` | local keyboard |
| `real` | `UnitreeCtrl` | wireless-remote bytes carried by G1 LowState |

[`Controller`](../g1_playground/controller/base_ctrl.py) consumes the queues shared by the input sources. The deployment
profile selects one `controller._target_`, and
[`scripts/pipeline.py`](../scripts/pipeline.py) constructs it with the shared `G1Env`.

## Read contract

`Controller.read()` drains queued updates and returns:

```python
control, shutdown_requested = controller.read()
# control == {"axes": {"LeftX": ..., "LeftY": ..., "RightX": ..., ...}}
# shutdown_requested is bool
```

The newest axis sample is retained. All pending button edges are consumed, and any pressed `A` edge makes the boolean
true for that read. The launcher checks it before policy inference and before the next target write. Other press edges
are available in `controller.pressed_buttons` for that frame.

The policy consumes three normalized axes:

- `LeftY`: forward/backward velocity;
- `LeftX`: lateral velocity;
- `RightX`: yaw rate.

The policy owns dead-zone, sign, and maximum-command scaling. Start each run with the sticks released.

## KeyboardCtrl

The two-process DDS simulation uses the keyboard by default.
[`KeyboardCtrl`](../g1_playground/controller/keyboard_ctrl.py)

```bash
# Terminal 1: mandatory-viewer simulator
python scripts/simulate.py
# Terminal 2: policy client and local keyboard
python scripts/pipeline.py deployment=sim
```

Up/Down controls longitudinal motion, Left/Right controls yaw, and Shift+Left/Right controls lateral motion.
Escape requests software shutdown.

## Policy switches

Enable Track with `track.enabled=true` and Metatron with `metatron.enabled=true`. The client starts in LeggedLab and
returns to it on request. Each handoff waits for the robot to settle. Metatron warms up while the current policy
keeps control. During the crossfade, both policies use live robot state to compute targets and the client sends one
blended command per tick. Press edges received during a gain transition are ignored.

| Action | Simulation keyboard | Unitree remote |
| --- | --- | --- |
| Enter Track | `t` | `L1` |
| Enter Metatron | `[` | `X` |
| Return to LeggedLab | `]` | `B` |
| Metatron `idle` | `i` | `Y` |
| Metatron `steer` | `k` | `R1` |

MuJoCo shows the active Metatron prompt in the world frame. Larger, brighter dots mark nearer position targets,
and lines connect the future positions of each body. The torso uses cyan, the left and right ankles use lime and
orange, and the left and right hands use violet and pink. Flat cyan markers indicate height-only targets. Arrows
show target directions, with RGB axes for the nearest full rotation and a yellow arrow for XY velocity. The legend
shows the active skill and time horizon. Press `p` in the MuJoCo window to toggle the overlay.

The real deployment requires an explicit `track.allow_real=true` or `metatron.allow_real=true` for each enabled handoff,
after checking the target robot's odometry and gains.

See the [Track and Metatron TODO](../TODO.md#track-and-metatron) for follow-up work.

## UnitreeCtrl

[`UnitreeCtrl`](../g1_playground/controller/unitree_ctrl.py) attaches a wireless-remote parser to
`env.remote_controller_handler`; it does not open a desktop device or retain a second environment reference. It is used
only by `deployment=real`:

```bash
python scripts/pipeline.py deployment=real env.net_if=ROBOT_NIC
```

> [!CAUTION]
> `A` is a software shutdown request, not an emergency stop. Verify it with the robot secured and zero commanded
> velocity, while a trained operator continuously guards the independent hardware emergency stop.

Do not remap or suppress the `A` edge on the real profile without a dedicated safety review and new simulation evidence.
