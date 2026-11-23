from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from g1_playground.interface.metatron.prompts import Prompt, PromptPacker
from g1_playground.interface.metatron.skills import SkillContext
from g1_playground.policy.metatron.metatron_contract import MetatronContract
from g1_playground.utils.dof import DoFAdapter

if TYPE_CHECKING:
    from g1_playground.g1_env import G1Odometry, G1State


@dataclass(frozen=True)
class MetatronState:
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_local_ang_vel: np.ndarray
    body_pos: np.ndarray
    body_rot: np.ndarray
    anchor_index: int

    @property
    def anchor_pos(self) -> np.ndarray:
        return self.body_pos[self.anchor_index]

    def measurements(self) -> dict[str, np.ndarray]:
        return {
            "noisy.dof_pos": self.dof_pos[None],
            "noisy.dof_vel": self.dof_vel[None],
            "noisy.anchor_pos": self.anchor_pos[None],
            "noisy.anchor_rot": self.body_rot[self.anchor_index][None],
            "noisy.root_local_ang_vel": self.root_local_ang_vel[None],
        }


class MetatronObservation:
    """Build Metatron inputs from DDS joint state, base pose, and skill prompts."""

    def __init__(self, contract: MetatronContract, xml_path: str, runtime_joint_names):
        runtime_joint_names = tuple(runtime_joint_names)
        if len(runtime_joint_names) != len(contract.joint_names) or set(runtime_joint_names) != set(
            contract.joint_names
        ):
            raise ValueError("Metatron and DDS must use the same complete G1 joint set")
        self._joint_order = DoFAdapter(runtime_joint_names, contract.joint_names)
        self._anchor_index = contract.anchor_body_index
        self._inputs = contract.inputs
        self._history_inputs = tuple(item for item in self._inputs if item.history is not None)
        self._history: dict[str, deque[np.ndarray]] = {}
        self._prompts = PromptPacker(contract)
        self._scene = {
            item.key: np.zeros(item.shape, dtype=item.dtype) for item in self._inputs if item.provider == "scene"
        }
        self._scene["scene_boxes.object_rot"][..., 3] = 1.0
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        try:
            root = self.model.body(contract.root_body)
            joints = [self.model.joint(name).id for name in contract.joint_names]
            self._body_ids = [self.model.body(name).id for name in contract.body_names]
        except KeyError as error:
            raise ValueError(f"Metatron kinematics cannot resolve a required body or joint: {error}") from error

        if root.jntnum.item() != 1 or self.model.jnt_type[root.jntadr.item()] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Metatron kinematics requires a free pelvis joint")
        if any(self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE for joint in joints):
            raise ValueError("Metatron kinematics requires scalar G1 hinge joints")
        self._root_qpos = int(self.model.jnt_qposadr[root.jntadr.item()])
        self._joint_qpos = self.model.jnt_qposadr[joints]
        if not self.model.jnt_limited[joints].all():
            raise ValueError("Metatron kinematics requires G1 joint position limits")
        self.joint_limits = self.model.jnt_range[joints].copy()

        self.data.qpos[self._joint_qpos] = contract.default_dof_pos
        mujoco.mj_kinematics(self.model, self.data)
        self.body_names = contract.body_names
        self.anchor_body = contract.anchor_body
        self.default_pose = SkillContext(*self._pose(), self.body_names, self.anchor_body)

    @property
    def prompt_slots(self) -> int:
        return self._prompts.slots

    def reset(self) -> None:
        self._history.clear()

    def _initialize_history(self, measurements: dict[str, np.ndarray]) -> None:
        for item in self._history_inputs:
            history = item.history
            depth = max(history.steps)
            if history.source in self._history and self._history[history.source].maxlen >= depth:
                continue
            if history.fill is None:
                initial = measurements[history.source].copy()
            else:
                initial = np.full((1, item.shape[-1]), history.fill, dtype=item.dtype)
            self._history[history.source] = deque([initial] * depth, maxlen=depth)

    def build(self, state: MetatronState, prompt: Prompt) -> dict[str, np.ndarray]:
        measurements = state.measurements()
        if not self._history:
            self._initialize_history(measurements)
        values = {**self._scene, **self._prompts.pack(prompt), **measurements}
        for item in self._history_inputs:
            history = item.history
            buffer = self._history[history.source]
            values[item.key] = np.stack([buffer[step - 1] for step in history.steps]).reshape(item.shape)
        return {item.name: values[item.key] for item in self._inputs}

    def accept_applied_action(self, measurements: dict[str, np.ndarray], action: np.ndarray) -> None:
        if not self._history:
            self._initialize_history(measurements)
        values = {**measurements, "actions": action}
        updates = {source: values[source].copy() for source in self._history}
        for source, value in updates.items():
            self._history[source].appendleft(value)

    def _pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = self.data.xpos[self._body_ids].astype(np.float32)
        rotation = self.data.xquat[self._body_ids][:, [1, 2, 3, 0]].astype(np.float32)
        position.flags.writeable = rotation.flags.writeable = False
        return position, rotation

    def skill_context(self, state: MetatronState) -> SkillContext:
        return SkillContext(state.body_pos, state.body_rot, self.body_names, self.anchor_body)

    def read(self, state: G1State, odometry: G1Odometry) -> MetatronState:
        position = np.asarray(odometry.position, dtype=np.float32)
        rotation = np.asarray(state.base_quat, dtype=np.float32)
        dof_pos = self._joint_order.fit(state.dof_pos).astype(np.float32)
        dof_vel = self._joint_order.fit(state.dof_vel).astype(np.float32)
        gyro = np.asarray(state.base_ang_vel, dtype=np.float32)
        if (
            position.shape != (3,)
            or rotation.shape != (4,)
            or gyro.shape != (3,)
            or not all(np.isfinite(values).all() for values in (position, rotation, dof_pos, dof_vel, gyro))
            or position[2] <= 0.0
            or not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-3)
        ):
            raise ValueError("Metatron requires a finite DDS sample with positive pelvis height and a unit quaternion")

        root = self._root_qpos
        self.data.qpos[root : root + 3] = position
        self.data.qpos[root + 3 : root + 7] = rotation[[3, 0, 1, 2]]
        self.data.qpos[self._joint_qpos] = dof_pos
        mujoco.mj_kinematics(self.model, self.data)
        return MetatronState(
            dof_pos,
            dof_vel,
            gyro.copy(),
            *self._pose(),
            self._anchor_index,
        )
