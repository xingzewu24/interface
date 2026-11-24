from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import OmegaConf

from g1_playground.controller.policy_ctrl import POLICY_BUTTONS
from g1_playground.policy.leggedlab import LeggedLabPolicy
from g1_playground.utils.dof import compose_dof_config

if TYPE_CHECKING:
    from g1_playground.g1_env import G1Env, G1Odometry, G1State
    from g1_playground.interface.metatron.skills import SkillLayer
    from g1_playground.policy.metatron import MetatronPolicy
    from g1_playground.policy.metatron.metatron_observation import MetatronState
    from g1_playground.policy.track import TrackPolicy

logger = logging.getLogger(__name__)
LOCO = "loco"
NEUTRAL_CONTROL = {"axes": dict.fromkeys(("LeftX", "LeftY", "RightX"), 0.0)}


class LocomotionPolicy(LeggedLabPolicy):
    """Reuse the warmed LeggedLab state when taking control back at 50 Hz."""

    def reset(self) -> None:
        if not hasattr(self, "_standing_history"):
            super().reset()
            self._standing_history = self.history_buf.copy()
            self._standing_recurrent = self._hidden_state.copy(), self._cell_state.copy()
            return
        self.last_action.fill(0.0)
        self.history_buf = self._standing_history.copy()
        self._hidden_state, self._cell_state = (value.copy() for value in self._standing_recurrent)

    def accept_applied_target(self, target: np.ndarray) -> None:
        action = (self.observation_adapter.fit(target) - self.default_pos) / self.action_scale
        self.last_action = np.clip(action, -self.clip_action, self.clip_action).astype(np.float32)


@dataclass(frozen=True)
class Gains:
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray

    def mix(self, other: Gains, weight: float) -> Gains:
        return Gains(
            (1.0 - weight) * self.kp + weight * other.kp,
            (1.0 - weight) * self.kd + weight * other.kd,
            (1.0 - weight) * self.effort + weight * other.effort,
        )

    def limit_target(self, state: G1State, target: np.ndarray, limits: np.ndarray) -> np.ndarray:
        # The DDS actuator receives a position target. Bound the PD effort at the sampled state.
        center = state.dof_pos + self.kd * state.dof_vel / self.kp
        radius = self.effort / self.kp
        return np.clip(np.clip(target, center - radius, center + radius), limits[:, 0], limits[:, 1])


class PolicyWrapper:
    """Policy entry with its gains, switch timing, and optional skill layer."""

    def __init__(
        self,
        policy,
        cfg,
        gains: Gains,
        joint_limits: np.ndarray,
        *,
        skills: SkillLayer | None = None,
        publish_prompt=None,
    ):
        self.policy = policy
        self.gains = gains
        self.joint_limits = joint_limits
        self.skills = skills
        self._publish_prompt = publish_prompt
        self.transition_seconds = float(cfg.transition_seconds)
        self.settle_seconds = float(cfg.settle_seconds)
        self.settle_timeout = float(cfg.settle_timeout_seconds)
        self.max_switch_speed = float(cfg.max_switch_speed)
        settings = (self.transition_seconds, self.settle_seconds, self.settle_timeout, self.max_switch_speed)
        if not all(math.isfinite(value) and value > 0.0 for value in settings):
            raise ValueError("Handoff settings must be positive and finite")
        if self.settle_timeout <= self.settle_seconds:
            raise ValueError("The handoff timeout must exceed its settling duration")

    @property
    def skill(self) -> str | None:
        return None if self.skills is None else self.skills.name

    def enter(self, state: G1State, odometry: G1Odometry) -> MetatronState | None:
        self.policy.reset()
        if self.skills is None:
            return None
        observation = self.policy.observation
        frame = observation.read(state, odometry)
        self.skills.start(observation.skill_context(frame))
        return frame

    def act(self, state: G1State, odometry: G1Odometry, control, settling: bool) -> np.ndarray:
        if self.skills is None:
            return self.policy.act(state, control)
        observation = self.policy.observation
        frame = observation.read(state, odometry)
        prompt = self.skills.advance(observation.skill_context(frame), control, settling=settling)
        target = self.policy.act(frame, prompt)
        if self._publish_prompt is not None:
            self._publish_prompt(self.skills.name, prompt)
        return target

    def warmup(self, state: G1State, odometry: G1Odometry) -> None:
        frame = self.enter(state, odometry)
        if frame is None:
            self.policy.act(state, NEUTRAL_CONTROL)
        else:
            context = self.policy.observation.skill_context(frame)
            self.policy.warmup(frame, self.skills.advance(context))
        self.policy.reset()


@dataclass
class PolicyInterpolation:
    source_policy_id: str
    start_target: np.ndarray
    start_gains: Gains
    joint_limits: np.ndarray
    steps: int
    timestep: int = 0


@dataclass(frozen=True)
class AppliedFrame:
    target: np.ndarray
    mode: str
    skill: str | None


class PolicyInterpManager:
    """Select one policy and send one G1 command per control tick."""

    class InterpState(Enum):
        IDLE = auto()
        SETTLING = auto()
        PRIMING = auto()
        IN_PROGRESS = auto()

    def __init__(self, loco: LocomotionPolicy, loco_gains: Gains, policies: dict[str, PolicyWrapper]):
        self.loco = loco
        self.policies = policies
        self._gains = {LOCO: loco_gains, **{name: entry.gains for name, entry in policies.items()}}
        if any(not math.isclose(loco.dt, entry.policy.dt) for entry in policies.values()):
            raise ValueError("Policies must have the same control period")
        if any(
            not np.isfinite(gains.kp).all()
            or not np.isfinite(gains.kd).all()
            or not np.isfinite(gains.effort).all()
            or min(gains.kp) <= 0.0
            or min(gains.kd) < 0.0
            or min(gains.effort) <= 0.0
            for gains in self._gains.values()
        ):
            raise ValueError("Handoff needs finite positive stiffness and effort limits and nonnegative damping")

        self._current_policy_id = LOCO
        self._next_policy_id: str | None = None
        self._switch_requested_at = 0.0
        self._stable_since: float | None = None
        self._interpolation: PolicyInterpolation | None = None
        self._last_target = loco.standing_target
        self._applied_gains = loco_gains
        self._warmup_worker = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-warmup")
            if any(entry.skills is not None for entry in policies.values())
            else None
        )
        self._prime_future: Future | None = None

    @property
    def current_policy_id(self) -> str:
        return self._current_policy_id

    @property
    def interp_state(self) -> InterpState:
        """Report the phase from the current switch and interpolation fields."""
        if self._interpolation is not None:
            return self.InterpState.IN_PROGRESS
        if self._prime_future is not None:
            return self.InterpState.PRIMING
        if self._next_policy_id is not None:
            return self.InterpState.SETTLING
        return self.InterpState.IDLE

    def warmup(self, state: G1State, odometry: G1Odometry | None) -> None:
        if odometry is None:
            raise RuntimeError("Policy warmup requires fresh pelvis odometry")
        for entry in self.policies.values():
            if entry.skills is None:
                entry.warmup(state, odometry)
            else:
                self._warmup_worker.submit(entry.warmup, state, odometry).result()

    def close(self) -> None:
        if self._warmup_worker is not None:
            self._warmup_worker.shutdown(wait=False, cancel_futures=True)

    def _cancel_priming(self) -> None:
        if self._prime_future is not None:
            self._prime_future.cancel()
            self._prime_future = None

    def _request_policy_switch(self, buttons, odometry: G1Odometry | None, clock: float) -> None:
        policy_id = next(
            (name for button, name in POLICY_BUTTONS.items() if button in buttons and name in self._gains), None
        )
        if policy_id is None or policy_id == self._next_policy_id:
            return
        if policy_id == self.current_policy_id:
            self._next_policy_id = None
            self._cancel_priming()
            return
        if odometry is None:
            logger.warning("Handoff to %s declined because pelvis odometry is unavailable", policy_id)
            return
        self._next_policy_id = policy_id
        self._cancel_priming()
        self._switch_requested_at = clock
        self._stable_since = None
        logger.info("Handoff to %s requested, waiting for a stable stance", policy_id)

    def _is_settled(self, entry: PolicyWrapper, odometry: G1Odometry | None, clock: float) -> bool:
        if odometry is None or clock - self._switch_requested_at >= entry.settle_timeout:
            logger.warning(
                "Handoff to %s canceled because stable, fresh odometry was unavailable", self._next_policy_id
            )
            self._next_policy_id = None
            self._cancel_priming()
            return False
        if np.linalg.norm(odometry.velocity) > entry.max_switch_speed:
            self._stable_since = None
            self._cancel_priming()
            return False
        if self._stable_since is None:
            self._stable_since = clock
        return clock - self._stable_since >= entry.settle_seconds

    def _is_policy_primed(self, policy_id: str, entry: PolicyWrapper) -> bool:
        if policy_id == LOCO or entry.skills is None:
            return True
        if self._prime_future is None:
            self._prime_future = self._warmup_worker.submit(entry.policy.prime)
        if not self._prime_future.done():
            return False
        self._prime_future.result()
        self._prime_future = None
        return True

    def _update_policy_switch(
        self, state: G1State, odometry: G1Odometry | None, control, clock: float
    ) -> MetatronState | None:
        self._request_policy_switch(control.get("buttons", ()), odometry, clock)
        policy_id = self._next_policy_id
        if policy_id is None:
            return None
        # The alternate policy supplies the timing and limits for a switch back to locomotion.
        entry = self.policies[self.current_policy_id if policy_id == LOCO else policy_id]
        if self._is_settled(entry, odometry, clock) and self._is_policy_primed(policy_id, entry):
            return self._interpolate_start(policy_id, entry, state, odometry)
        return None

    def _interpolate_start(
        self, policy_id: str, entry: PolicyWrapper, state: G1State, odometry: G1Odometry
    ) -> MetatronState | None:
        self._interpolation = PolicyInterpolation(
            self.current_policy_id,
            self._last_target.copy(),
            self._applied_gains,
            entry.joint_limits,
            math.ceil(entry.transition_seconds / self.loco.dt),
        )
        self._current_policy_id = policy_id
        self._next_policy_id = None
        entry_frame = None
        if policy_id == LOCO:
            self.loco.reset()
        else:
            entry_frame = entry.enter(state, odometry)
            if entry.skill is not None:
                logger.info("%s skill %s", policy_id.capitalize(), entry.skill)
        logger.info("Transition to %s started", policy_id)
        return entry_frame

    def _interpolate_step(
        self,
        interp: PolicyInterpolation,
        state: G1State,
        odometry: G1Odometry,
        target: np.ndarray,
        gains: Gains,
    ) -> tuple[np.ndarray, Gains, LocomotionPolicy | TrackPolicy | MetatronPolicy | None]:
        progress = interp.timestep / interp.steps
        alpha = progress * progress * (3.0 - 2.0 * progress)
        start_gains = interp.start_gains
        blended_gains = start_gains if alpha == 0 else gains if alpha == 1 else start_gains.mix(gains, alpha)

        source_target = interp.start_target
        source_policy = None
        if 0 < alpha < 1:
            source = self.policies.get(interp.source_policy_id)
            if source is None:
                source_policy = self.loco
                source_target = source_policy.act(state, NEUTRAL_CONTROL)
            else:
                source_policy = source.policy
                source_target = source.act(state, odometry, NEUTRAL_CONTROL, settling=True)

        # Blend the two PD torques at the same state, including differing stiffness.
        target = ((1.0 - alpha) * start_gains.kp * source_target + alpha * gains.kp * target) / blended_gains.kp
        return blended_gains.limit_target(state, target, interp.joint_limits), blended_gains, source_policy

    def step(self, env: G1Env, state: G1State, odometry: G1Odometry | None, control, clock: float) -> AppliedFrame:
        interp = self._interpolation
        if odometry is None and (self.current_policy_id != LOCO or interp is not None):
            active = self.current_policy_id
            if interp is not None:
                active = f"{interp.source_policy_id} to {active}"
            raise RuntimeError(f"{active.capitalize()} lost fresh pelvis odometry")
        entry_frame = None
        if interp is None:
            entry_frame = self._update_policy_switch(state, odometry, control, clock)

        interp = self._interpolation
        policy_id = self.current_policy_id
        entry = self.policies.get(policy_id)
        gains = self._gains[policy_id]
        settling = self._next_policy_id is not None
        if entry_frame is not None:
            # The first blend tick has zero incoming weight. Seed its history from the command that is sent.
            target = interp.start_target
        elif entry is None:
            use_neutral = interp is not None or settling
            target = self.loco.act(state, NEUTRAL_CONTROL if use_neutral else control)
        else:
            active_control = NEUTRAL_CONTROL if interp is not None else control
            target = entry.act(state, odometry, active_control, settling=settling)

        phase = policy_id
        source_policy = None
        if interp is not None:
            target, gains, source_policy = self._interpolate_step(interp, state, odometry, target, gains)
            phase = f"to_{policy_id}"
        elif entry is not None:
            target = gains.limit_target(state, target, entry.joint_limits)

        if gains is not self._applied_gains:
            env.set_gains(gains.kp, gains.kd)
            self._applied_gains = gains
        env.step(target)

        if entry_frame is not None:
            entry.policy.seed_applied_target(entry_frame, target)
        elif entry is not None:
            entry.policy.accept_applied_target(target)
        elif interp is not None:
            self.loco.accept_applied_target(target)
        if source_policy is not None:
            source_policy.accept_applied_target(target)
        self._last_target = target.copy()

        if interp is not None:
            interp.timestep += 1
            if interp.timestep > interp.steps:
                self._interpolation = None
                logger.info("Transition to %s complete", policy_id)
        return AppliedFrame(target, phase, None if entry is None else entry.skill)


def setup_policy_fsm(cfg, dof, *, publish_prompt=None):
    enabled = {name: cfg.get(name, {}) for name in ("track", "metatron") if cfg.get(name, {}).get("enabled", False)}
    if not enabled:
        return None, cfg.controller
    if not cfg.env.enable_odometry:
        raise ValueError("Policy handoff requires pelvis odometry")
    for name, settings in enabled.items():
        if cfg.env.motion_switcher_required and not settings.allow_real:
            raise ValueError(f"Verify the real G1 odometry and gains before enabling {name} handoff on hardware")

    loco = LocomotionPolicy(cfg.policy, dof_cfg=dof)
    effort = np.asarray(cfg.robot.dof.torque_limits, dtype=float)
    policies = {}
    if "track" in enabled:
        from g1_playground.policy.track import TrackPolicy

        track_dof = compose_dof_config(cfg.robot.dof, cfg.track.dof)
        track = TrackPolicy(cfg.track, dof_cfg=track_dof)
        track_gains = Gains(np.asarray(track_dof.stiffness), np.asarray(track_dof.damping), effort)
        limits = np.column_stack((track.joint_pos_lower, track.joint_pos_upper))
        policies["track"] = PolicyWrapper(track, cfg.track, track_gains, limits)
    if "metatron" in enabled:
        from g1_playground.interface.metatron.skills import SkillLayer
        from g1_playground.policy.metatron import MetatronPolicy

        metatron = MetatronPolicy(cfg.metatron, dof_cfg=dof)
        observation = metatron.observation
        skills = SkillLayer(
            metatron.dt,
            observation.prompt_slots,
            observation.default_pose,
            steer_speed=cfg.metatron.steer_speed_limit,
            steer_yaw_rate=cfg.metatron.steer_yaw_rate_limit,
        )
        metatron_gains = Gains(metatron.stiffness, metatron.damping, np.minimum(effort, metatron.effort_limits))
        policies["metatron"] = PolicyWrapper(
            metatron,
            cfg.metatron,
            metatron_gains,
            metatron.joint_limits,
            skills=skills,
            publish_prompt=publish_prompt,
        )

    controller = cfg.controller
    if controller._target_ == "g1_playground.controller.keyboard_ctrl.KeyboardCtrl":
        controller = OmegaConf.merge(
            controller, {"_target_": "g1_playground.controller.policy_ctrl.PolicyKeyboardCtrl"}
        )
    return PolicyInterpManager(
        loco, Gains(np.asarray(dof.stiffness), np.asarray(dof.damping), effort), policies
    ), controller
