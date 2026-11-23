from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyTarget:
    body: str
    slot: int
    time: float
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    planar: bool = True
    heading_only: bool = False


@dataclass(frozen=True)
class Prompt:
    """Sparse world-frame targets and an optional world-frame XY velocity."""

    targets: tuple[BodyTarget, ...] = ()
    velocity: np.ndarray | None = None
    ground_height: float = 0.0


class PromptPacker:
    """Encode named body targets using the canvas declared by the export."""

    def __init__(self, contract):
        self.inputs = {
            spec.key.removeprefix("masked_mimic."): spec for spec in contract.inputs if spec.provider == "masked_mimic"
        }
        self.bodies = {name: index for index, name in enumerate(contract.body_names)}
        self.conditionable = {name: index for index, name in enumerate(contract.conditionable_bodies)}
        self.anchor = contract.anchor_body
        self.slots = self.inputs["time_offsets"].shape[1]
        if self.inputs["ref_pos"].shape[1:3] != (self.slots, len(self.bodies)):
            raise ValueError("The prompt canvas does not match the export body layout")
        if np.prod(self.inputs["target_bodies_masks"].shape) != self.slots * len(self.conditionable) * 2:
            raise ValueError("The prompt masks do not match the export conditioning layout")

    def pack(self, prompt: Prompt) -> dict[str, np.ndarray]:
        values = {name: np.zeros(spec.shape, dtype=spec.dtype) for name, spec in self.inputs.items()}
        pos, rot = values["ref_pos"][0], values["ref_rot"][0]
        pos[..., 2] = prompt.ground_height
        rot[..., 3] = 1.0
        times = values["time_offsets"][0]
        active = values["target_poses_masks"][0]
        masks = values["target_bodies_masks"].reshape(self.slots, len(self.conditionable), 2)

        for goal in prompt.targets:
            slot = goal.slot
            if not 0 <= slot < self.slots or goal.body not in self.conditionable:
                raise ValueError(f"Invalid prompt target {goal.body!r} in slot {slot}")
            if not np.isfinite(goal.time) or (goal.position is None and goal.rotation is None):
                raise ValueError("A body target requires a finite arrival time and a pose component")
            body = self.bodies[goal.body]
            mask = masks[slot, self.conditionable[goal.body]]
            time = np.float32(max(0.0, goal.time))
            if mask.any() or (active[slot] and time != times[slot]):
                raise ValueError("Targets must use unique body/slot pairs and a shared time per slot")
            active[slot], times[slot] = True, time
            if goal.position is not None:
                pos[slot, body] = goal.position
                mask[0] = True
                if goal.body == self.anchor:
                    values["anchor_translation_mask"][0, slot] = (goal.planar, True)
            if goal.rotation is not None:
                if not np.isclose(np.linalg.norm(goal.rotation), 1.0, atol=1e-4):
                    raise ValueError("Prompt rotations must be unit xyzw quaternions")
                rot[slot, body] = goal.rotation
                mask[1] = True
                if goal.body == self.anchor:
                    values["anchor_rotation_mask"][0, slot] = (not goal.heading_only, goal.heading_only)

        if prompt.velocity is not None:
            values["average_velocity"][0] = prompt.velocity
            values["velocity_command_mask"].fill(True)
        # The graph handles frame conversion. Local/style masks stay clear.
        values["local_ref_pos"][:] = values["ref_pos"]
        values["local_ref_rot"][:] = values["ref_rot"]
        return {spec.key: values[name] for name, spec in self.inputs.items()}
