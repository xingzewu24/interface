from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

CONTROL_OUTPUTS = {"actions", "joint_pos_targets", "stiffness_targets", "damping_targets"}


@dataclass(frozen=True)
class HistorySpec:
    source: str
    steps: tuple[int, ...]
    fill: float | None


@dataclass(frozen=True)
class ModelInput:
    name: str
    key: str
    provider: str
    shape: tuple[int, ...]
    dtype: np.dtype
    history: HistorySpec | None

    @classmethod
    def load(cls, item):
        shape = tuple(item.shape)
        dtype = np.dtype(item.dtype)
        if not shape or shape[0] != 1 or min(shape) < 1 or dtype not in (np.dtype("float32"), np.dtype("bool")):
            raise ValueError(f"Unsupported Metatron input {item.name}: {shape}, {dtype}")
        history = None
        if "observation_history" in item:
            spec = item.observation_history
            if spec.reset_fill != "initial_measurement":
                raise ValueError("Metatron observation history must start with the first measurement")
            history = HistorySpec(spec.current_key, tuple(spec.steps_back), None)
        elif "output_key" in item:
            if item.fill_before_available != "zeros":
                raise ValueError("Metatron action history must start from a zero-action reference")
            history = HistorySpec(item.output_key, tuple(item.steps_back), float(item.get("reset_value", 0.0)))
        if history is not None:
            slots = shape[1] if len(shape) == 3 else 1
            if len(history.steps) != slots or min(history.steps) < 1:
                raise ValueError(f"Invalid Metatron history for {item.key}")
        return cls(item.name, item.key, item.provider, shape, dtype, history)


class MetatronContract:
    """Load the policy configuration exported alongside the ONNX model."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = OmegaConf.load(self.path)
        if raw.schema != "unified_pipeline/1":
            raise ValueError(f"Unsupported Metatron contract {raw.schema!r}")
        robot, spec = raw.robot, raw.control
        self.body_names = tuple(robot.body_names)
        self.joint_names = tuple(robot.joint_names)
        self.anchor_body = robot.anchor_body_name
        self.root_body = robot.root_body_name
        self.anchor_body_index = self.body_names.index(self.anchor_body)
        self.conditionable_bodies = tuple(robot.conditionable_body_names)
        if (
            tuple(raw.body_names) != self.body_names
            or tuple(raw.joint_names) != self.joint_names
            or self.anchor_body_index != robot.anchor_body_index
            or self.body_names.index(self.root_body) != robot.root_body_index
            or not set(self.conditionable_bodies).issubset(self.body_names)
            or self.anchor_body not in self.conditionable_bodies
        ):
            raise ValueError("Metatron robot and policy layouts disagree")
        self.control_dt = float(raw.timing.control_dt)
        if not (
            self.control_dt > 0
            and np.isclose(self.control_dt, float(raw.dt))
            and np.isclose(self.control_dt, raw.timing.physics_dt * raw.timing.decimation)
        ):
            raise ValueError("Metatron control timing disagrees with the export")

        self.inputs = tuple(ModelInput.load(item) for item in raw.policy_inputs)
        names = {item.name: item.key for item in self.inputs}
        if len(names) != len(self.inputs) or len(set(names.values())) != len(names):
            raise ValueError("Metatron input names and keys must be unique")
        if names != dict(raw._runtime.onnx_name_to_in_key):
            raise ValueError("Metatron name-to-key mapping disagrees with the export")
        self.outputs = {item.name: tuple(item.shape) for item in raw.policy_outputs}
        target_shape = (1, len(self.joint_names))
        if (
            self.outputs.keys() != CONTROL_OUTPUTS
            or len(self.outputs) != len(raw.policy_outputs)
            or set(self.outputs.values()) != {target_shape}
            or any(
                tuple(output.get("joint_names", self.joint_names)) != self.joint_names for output in raw.policy_outputs
            )
        ):
            raise ValueError("Invalid Metatron control output layout")
        if spec.action_fn != "bm_pd_action" or spec.action_ema_alpha != 1.0 or spec.pd_target_max_accel is not None:
            raise ValueError("Metatron requires direct exported PD targets")

        self.default_dof_pos = self._control_vector(spec.pd_action_offset)
        self.action_scale = self._control_vector(spec.action_scale)
        self.stiffness = self._control_vector(spec.stiffness)
        self.damping = self._control_vector(spec.damping)
        self.effort_limits = self._control_vector(spec.effort_limits)
        if (
            min(self.stiffness) <= 0
            or min(self.damping) < 0
            or min(self.action_scale) <= 0
            or min(self.effort_limits) <= 0
        ):
            raise ValueError("Invalid Metatron gains, action scales or effort limits")
        self._validate_sensors(raw.sensors)

    def _control_vector(self, values) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (len(self.joint_names),) or not np.isfinite(vector).all():
            raise ValueError("Metatron requires one finite control value per joint")
        return vector

    def _validate_sensors(self, sensors) -> None:
        imu, localization = sensors.imu, sensors.localization
        if (
            imu.body_name != self.root_body
            or localization.body_name != self.root_body
            or sensors.anchor_body_name != self.anchor_body
            or not sensors.w_last
            or imu.orientation_frame != "world"
            or imu.gyro_frame != "sensor"
            or tuple(imu.mount_sensor_to_body_xyzw) != (0.0, 0.0, 0.0, 1.0)
            or localization.position_frame != "world"
            or any(localization.offset_body_xyz)
        ):
            raise ValueError("Metatron export does not use the G1 pelvis localization and IMU frames")

    @property
    def onnx_path(self) -> Path:
        return self.path.with_suffix(".onnx")
