import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from g1_playground.controller.policy_ctrl import IDLE_BUTTON, STEER_BUTTONS
from g1_playground.interface.metatron.prompts import BodyTarget, Prompt

logger = logging.getLogger(__name__)
FEET = ("right_ankle_roll_link", "left_ankle_roll_link")


@dataclass(frozen=True)
class SkillContext:
    """Read-only world-frame body poses for one control frame."""

    positions: np.ndarray
    rotations: np.ndarray
    bodies: tuple[str, ...]
    anchor: str

    def position(self, body: str) -> np.ndarray:
        return self.positions[self.bodies.index(body)]

    @property
    def heading(self) -> float:
        x, y, z, w = self.rotations[self.bodies.index(self.anchor)]
        return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


class Skill(ABC):
    def __init__(self, dt: float, slots: int, default_pose: SkillContext):
        self.dt = dt
        self.times = 0.1 * np.arange(1, slots + 1)

    @abstractmethod
    def reset(self, context: SkillContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance(self, context: SkillContext) -> Prompt:
        raise NotImplementedError


class Idle(Skill):
    def reset(self, context: SkillContext) -> None:
        self.position = context.position(context.anchor).copy()
        self.heading = heading_quaternion(context.heading)

    def advance(self, context: SkillContext) -> Prompt:
        targets = tuple(
            BodyTarget(context.anchor, slot, time, self.position, self.heading, heading_only=True)
            for slot, time in enumerate(self.times)
        )
        return Prompt(targets, np.zeros(2))


class Stand(Skill):
    """Request the fixed world-frame stance from Metatron's smoke test."""

    def reset(self, context: SkillContext) -> None:
        position = np.array([0.0, 0.0, 0.8])
        rotation = heading_quaternion(0.0)
        targets = tuple(
            BodyTarget(context.anchor, slot, time, position, rotation, heading_only=True)
            for slot, time in enumerate(self.times)
        )
        self._prompt = Prompt(targets, np.zeros(2))

    def advance(self, context: SkillContext) -> Prompt:
        return self._prompt


class Steer(Skill):
    def __init__(self, dt: float, slots: int, default_pose: SkillContext):
        super().__init__(dt, slots, default_pose)
        torso = default_pose.position(default_pose.anchor)
        self.offsets = np.stack([default_pose.position(name) - torso for name in FEET])
        half_width = max(0.07, 0.5 * abs(self.offsets[0, 1] - self.offsets[1, 1]))
        self.offsets[:, 1] = (-half_width, half_width)
        self.height = torso[2] - default_pose.positions[:, 2].min()

    def reset(self, context: SkillContext) -> None:
        self.phase = 0.0
        self.velocity = np.zeros(2)
        self.facing = context.heading
        self.support = None

    def set_command(self, velocity: np.ndarray, facing: float) -> None:
        self.velocity = velocity
        self.facing = facing

    def advance(self, context: SkillContext) -> Prompt:
        speed = float(np.linalg.norm(self.velocity))
        if speed <= 0.05:
            if self.support is None:
                self.support = (context.position(context.anchor), *(context.position(name) for name in FEET))
            self.phase = 0.0
            velocity = np.zeros(2)
        else:
            self.support = None
            velocity = self.velocity

        targets = []
        rotation = heading_quaternion(self.facing)
        cadence = min(2.0, 0.83 + 0.93 * speed)
        for slot, time in enumerate(self.times):
            torso, feet = self._stance(context, time, cadence, speed)
            height = torso if slot in (1, 2) else None
            targets.append(BodyTarget(context.anchor, slot, time, height, rotation, planar=False, heading_only=True))
            targets.extend(BodyTarget(name, slot, time, position=foot) for name, foot in zip(FEET, feet, strict=True))
        if self.support is None:
            self.phase = (self.phase + math.tau * cadence * self.dt) % math.tau
        return Prompt(tuple(targets), velocity)

    def _stance(self, context: SkillContext, time: float, cadence: float, speed: float):
        if self.support is not None:
            return self.support[0], self.support[1:]
        torso = context.position(context.anchor).copy()
        torso[:2] += self.velocity * time
        torso[2] = context.positions[:, 2].min() + self.height
        angle = math.atan2(self.velocity[1], self.velocity[0]) - self.facing
        forward, lateral = math.cos(angle), math.sin(angle)
        phases = self.phase + math.tau * cadence * time + np.array([0.0, math.pi])
        swing = np.cos(phases)
        feet = self.offsets.copy()
        feet[:, 0] += 0.75 * (0.13 + 0.065 * speed) * forward * swing
        feet[:, 1] += 0.12 * (np.array([-1.0, 1.0]) * abs(lateral) + abs(forward) * lateral * swing)
        feet[:, 2] += 0.08 * np.maximum(0.0, -np.sin(phases))
        c, s = math.cos(self.facing), math.sin(self.facing)
        feet[:, :2] = feet[:, :2] @ np.array([[c, s], [-s, c]])
        return torso, torso + feet


class SkillLayer:
    def __init__(
        self,
        dt: float,
        slots: int,
        default_pose: SkillContext,
        registry=None,
        *,
        steer_speed: float = 0.6,
        steer_yaw_rate: float = 0.6,
    ):
        self.registry = {"stand": Stand, "idle": Idle, "steer": Steer} if registry is None else dict(registry)
        self._args = dt, slots, default_pose
        self.name = None
        self.active: Skill | None = None
        self.steer_speed = float(steer_speed)
        self.steer_yaw_rate = float(steer_yaw_rate)
        if not all(math.isfinite(value) and value > 0.0 for value in (self.steer_speed, self.steer_yaw_rate)):
            raise ValueError("Metatron steering settings must be positive and finite")

    def start(self, context: SkillContext) -> None:
        self.switch("stand", context)

    def switch(self, name: str, context: SkillContext) -> None:
        skill = self.registry[name](*self._args)
        skill.reset(context)
        self.name, self.active = name, skill

    def advance(self, context: SkillContext, control=None, *, settling: bool = False) -> Prompt:
        if control is not None:
            self._update_control(context, control, settling)
        if self.active is None:
            raise RuntimeError("Select a skill before building a prompt")
        return self.active.advance(context)

    def _update_control(self, context: SkillContext, control, settling: bool) -> None:
        buttons = control.get("buttons", ())
        if settling or IDLE_BUTTON in buttons:
            if self.name != "idle":
                self.switch("idle", context)
                logger.info("Metatron skill idle")
        elif STEER_BUTTONS.intersection(buttons) and self.name != "steer":
            self.switch("steer", context)
            logger.info("Metatron skill steer")

        if isinstance(self.active, Steer):
            axes = control["axes"]
            vx, vy, yaw = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]]).clip(-1.0, 1.0)
            heading = context.heading
            c, s = math.cos(heading), math.sin(heading)
            velocity = np.array([c * vx - s * vy, s * vx + c * vy]) * (self.steer_speed / max(1.0, math.hypot(vx, vy)))
            facing = math.remainder(self.active.facing + yaw * self.steer_yaw_rate * self.active.dt, math.tau)
            self.active.set_command(velocity, facing)


def heading_quaternion(heading: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * heading), math.cos(0.5 * heading)])
