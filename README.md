# G1-Playground

G1-Playground runs the 50 Hz LeggedLab locomotion policy on the Unitree G1 29DoF. MuJoCo and the physical G1 share
one policy loop and the same Unitree DDS interface.

## Installation

Python 3.11 or newer is required.

```bash
cd G1-Playground
conda create -n g1-playground python=3.11 -y
conda activate g1-playground
python -m pip install -e .
python scripts/setup/install_third_party.py unitree_cpp
```

Install the architecture-specific TensorRT Python wheel on both the workstation and Jetson:

```bash
export TENSORRT_ROOT="$HOME/TensorRT"
python -m pip install "$TENSORRT_ROOT"/python/tensorrt-*-cp311-none-linux_"$(uname -m)".whl
```

The first run compiles ONNX models into `.cache/tensorrt/`; later runs reuse compatible engines. Do not copy engines
between x86_64 and aarch64 machines.

## Run Locomotion in MuJoCo

```bash
# Terminal 1: start the simulator and viewer
python scripts/simulate.py

# Terminal 2: start the locomotion policy
python scripts/pipeline.py deployment=sim
```

Use Up/Down for forward/backward motion, Left/Right for yaw, and Shift+Left/Right for lateral motion. Escape requests
software shutdown.

## Switch Policies in MuJoCo

Start the simulator as above. To run LeggedLab and Track in the same policy client:

```bash
python -m scripts.pipeline deployment=sim track.enabled=true
```

Metatron uses the same TensorRT installation as LeggedLab and Track. Git LFS stores its ONNX model. After installing
Git LFS, fetch the model and start all three policies. TensorRT builds and caches the engine on the first run.

```bash
git lfs pull --include="assets/models/metatron/unified_pipeline.onnx"
python -m scripts.pipeline deployment=sim track.enabled=true metatron.enabled=true
```

The client starts in LeggedLab. Press **t** for Track, **[** for Metatron, and **]** for LeggedLab. When both are enabled,
the same keys also switch directly between Track and Metatron. Track holds its standing reference, aligned to the heading
at entry. Metatron enters `stand`. Press **i** for `idle` or **k** for `steer`. Escape requests software shutdown.
[Controller mappings](docs/controller.md)

## Synchronize the G1

Run synchronization from the workstation repository root. Without `--apply`, the command is a dry run:

```bash
bash scripts/setup/sync_unitree_wifi.sh
bash scripts/setup/sync_unitree_wifi.sh --apply
```

The normal apply mode removes remote source/assets that no longer exist locally while preserving the remote Git metadata,
environment, native libraries, TensorRT cache, and logs. Use `--no-delete` only when remote-only source files must remain.

## Run Locomotion on the Real G1

```bash
ssh unitree-wifi
cd ~/G1-Playground
conda activate g1-playground
python scripts/pipeline.py deployment=real env.net_if=enP8p1s0
```

Set `env.net_if` to the robot-facing interface. The Unitree remote controls locomotion.
[Unitree setup guide](docs/unitree_setup.md)

## Inspect Configuration

Use `--cfg job --resolve` to print the configuration without opening DDS.

```bash
python scripts/pipeline.py deployment=sim --cfg job --resolve
python scripts/pipeline.py deployment=real env.net_if=enP8p1s0 --cfg job --resolve
```

## Replay a Recording

Enable recording with `recording.enabled=true` when launching the policy.

```bash
# Replay the newest logs/state_* recording
python scripts/replay.py

# Replay one recording explicitly
python scripts/replay.py logs/state_20260822-183322
```

## Development Checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check g1_playground scripts tests
ruff format --check g1_playground scripts tests
pre-commit run --all-files
```

The hardware-free tests cover locomotion joint ordering, MuJoCo support timing, and the DDS command watchdog.

## Configuration and Assets

- `configs/run_pipeline.yaml` defines the locomotion pipeline.
- `configs/deployment/` defines DDS and input settings for simulation and hardware.
- `configs/policy/leggedlab_g1.yaml` configures the bundled LeggedLab ONNX model.
- `configs/policy/track.yaml` configures the bundled Track model for standalone use and policy handoffs.
  [Track policy implementation](g1_playground/policy/track/track_policy.py)

## Safety

A learned policy can fall or command unexpected motion. Validate the policy and gains in MuJoCo before hardware testing.
Keep the robot supported, clear the workspace, and keep an independent hardware emergency stop guarded. Escape and the
Unitree remote's `A` button request software shutdown.

## Documentation

- [Deployment architecture](docs/architecture.md)
- [Policy contract](docs/policy.md)
- [Controller mappings](docs/controller.md)
- [G1Env contract](docs/environment.md)
- [Unitree setup](docs/unitree_setup.md)

## License and Citation

The repository is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

```bibtex
@misc{G1Playground,
  author = {Zihan Zhuang, Yi Dong, Peng Li},
  title = {G1-Playground: A deployment framework for the Unitree G1 29DoF},
  url = {https://github.com/xiaohu-art/G1-Playground},
  year = {2025}
}
```
