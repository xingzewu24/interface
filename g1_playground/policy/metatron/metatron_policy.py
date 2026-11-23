import logging

import numpy as np
from omegaconf import DictConfig

from g1_playground.interface.metatron.prompts import Prompt
from g1_playground.policy.base_policy import BasePolicy
from g1_playground.policy.metatron.metatron_contract import MetatronContract
from g1_playground.policy.metatron.metatron_observation import MetatronObservation, MetatronState
from g1_playground.policy.tensorrt_runner import TensorRTRunner
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import DoFAdapter

logger = logging.getLogger(__name__)


class MetatronPolicy(BasePolicy[dict[str, np.ndarray]]):
    """Run the exported Metatron policy on a robot state and a skill prompt."""

    def __init__(self, cfg_policy: DictConfig, dof_cfg: DictConfig, runner=None):
        self.contract = MetatronContract(resolve_repo_path(cfg_policy.contract))
        self.FREQ = round(1 / self.contract.control_dt)
        super().__init__()
        if not np.isclose(self.dt, self.contract.control_dt):
            raise ValueError("Metatron control period must correspond to an integral frequency")

        model_path = self.contract.onnx_path if cfg_policy.model is None else resolve_repo_path(cfg_policy.model)
        logger.debug("Loading TensorRT Metatron policy from %s", model_path)
        self.runner = runner or TensorRTRunner(
            model_path, input_shapes={item.name: item.shape for item in self.contract.inputs}
        )
        self._validate_signature()

        self.observation = MetatronObservation(self.contract, resolve_repo_path(cfg_policy.fk_xml), dof_cfg.joint_names)
        self.observation_adapter = DoFAdapter(dof_cfg.joint_names, self.contract.joint_names)
        self.action_adapter = DoFAdapter(self.contract.joint_names, dof_cfg.joint_names)
        self.default_pos = self.contract.default_dof_pos
        self.action_scale = self.contract.action_scale
        self.stiffness = self.action_adapter.fit(self.contract.stiffness)
        self.damping = self.action_adapter.fit(self.contract.damping)
        self.effort_limits = self.action_adapter.fit(self.contract.effort_limits)
        self.joint_limits = self.observation.joint_limits[self.action_adapter.indices]
        self._warmup_feed: dict[str, np.ndarray] | None = None
        self.reset()

    def _validate_signature(self) -> None:
        if (
            set(self.runner.input_names) != {item.name for item in self.contract.inputs}
            or set(self.runner.output_names) != self.contract.outputs.keys()
        ):
            raise ValueError("TensorRT model tensor names differ from the Metatron export")
        layouts = {item.name: (item.shape, item.dtype) for item in self.contract.inputs}
        layouts.update({name: (shape, np.dtype(np.float32)) for name, shape in self.contract.outputs.items()})
        for name, (shape, dtype) in layouts.items():
            actual = self.runner.shape(name)
            if actual != shape or self.runner.dtype(name) != dtype:
                raise ValueError(f"TensorRT tensor {name} does not match the Metatron export")

    def reset(self) -> None:
        self.observation.reset()
        self._pending = None
        self._raw_action = None

    @property
    def standing_target(self) -> np.ndarray:
        return self.action_adapter.fit(self.default_pos)

    def get_observation(self, env_data: MetatronState, control_data: Prompt) -> dict[str, np.ndarray]:
        return self.observation.build(env_data, control_data)

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        for item in self.contract.inputs:
            value = observation[item.name]
            if value.shape != item.shape or value.dtype != item.dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid Metatron input {item.key} ({value.shape}, {value.dtype})")
        outputs = self.runner.run(observation)
        for name, shape in self.contract.outputs.items():
            value = outputs[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid Metatron output {name}")
        for name, gains in (("stiffness_targets", self.contract.stiffness), ("damping_targets", self.contract.damping)):
            if not np.allclose(outputs[name][0], gains):
                raise RuntimeError(f"Metatron {name} differ from the export")
        self._raw_action = outputs["actions"]
        return self.action_adapter.fit(outputs["joint_pos_targets"][0])

    def act(self, env_data: MetatronState, control_data: Prompt) -> np.ndarray:
        target = self.get_action(self.get_observation(env_data, control_data))
        self._pending = (target.copy(), env_data.measurements(), self._raw_action)
        return target

    def warmup(self, state: MetatronState, prompt: Prompt) -> None:
        feed = self.get_observation(state, prompt)
        self.get_action(feed)
        self._warmup_feed = feed

    def prime(self) -> None:
        # The worker only touches the saved feed and runner, never the live policy history.
        if self._warmup_feed is None:
            raise RuntimeError("Prepare the Metatron warmup feed before handoff")
        self.runner.run(self._warmup_feed)

    def accept_applied_target(self, target: np.ndarray) -> None:
        if self._pending is None:
            raise RuntimeError("Metatron has no pending action to commit")
        requested, measurements, action = self._pending
        if not np.array_equal(target, requested):
            action = self._action_for_target(target)
        self.observation.accept_applied_action(measurements, action)
        self._pending = None

    def seed_applied_target(self, state: MetatronState, target: np.ndarray) -> None:
        self.observation.accept_applied_action(state.measurements(), self._action_for_target(target))

    def _action_for_target(self, target: np.ndarray) -> np.ndarray:
        position = self.observation_adapter.fit(target)
        action = ((position - self.default_pos) / self.action_scale)[None].astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("The applied Metatron action must be finite")
        return action
