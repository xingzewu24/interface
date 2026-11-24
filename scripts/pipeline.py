# ruff: noqa: I001

import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time

# Apply the Jetson process settings before importing the numerical stack.
import g1_playground  # noqa: F401

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.policy import LeggedLabPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright
from g1_playground.utils.recorder import record, recorder, save_recording

logger = logging.getLogger("g1_playground")
RAMP_SECONDS = 3.0


def read_frame(env, controller):
    state = env.read()
    control, shutdown_requested = controller.read()
    if shutdown_requested:
        logger.warning("Emergency shutdown!")
        return None
    if not is_upright(state.base_quat):
        logger.error("Robot fallen! Shutting down for safety.")
        return None
    return state, control


def step(env, controller, policy, *, send_command: bool = True) -> bool:
    frame = read_frame(env, controller)
    if frame is None:
        return False
    state, control = frame
    target = policy.act(state, control)
    if send_command:
        env.step(target)
    return True


@hydra.main(version_base=None, config_path="../configs", config_name="run_pipeline")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    log = None
    policy_manager = None
    prompt_publisher = None
    try:
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        controller_cfg = cfg.controller
        if any(cfg.get(name, {}).get("enabled", False) for name in ("track", "metatron")):
            from g1_playground.policy_fsm import setup_policy_fsm

            if cfg.metatron.enabled and (port := cfg.get("sim_prompt_port")):
                if __package__:
                    from .telemetry import PromptPublisher
                else:
                    from telemetry import PromptPublisher

                try:
                    prompt_publisher = PromptPublisher(port)
                except OSError as error:
                    logger.warning("MuJoCo prompt publisher unavailable: %s", error)
            policy_manager, controller_cfg = setup_policy_fsm(cfg, dof, publish_prompt=prompt_publisher)
        policy = policy_manager.loco if policy_manager is not None else LeggedLabPolicy(cfg.policy, dof_cfg=dof)
        env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
        controller = instantiate(controller_cfg, env=env)
        if cfg.recording.enabled:
            log = recorder(int(cfg.recording.seconds * policy.freq))

        env.self_check()
        for _ in range(10):
            if not step(env, controller, policy, send_command=False):
                return

        frame = read_frame(env, controller)
        if frame is None:
            return
        state, _ = frame
        if policy_manager is not None:
            policy_manager.warmup(state, env.read_odometry())
        initial = state.dof_pos
        standing = policy.standing_target
        env.activate_commands()

        logger.warning("Ramping to the policy standing pose over %.1f seconds", RAMP_SECONDS)
        ramp_steps = int(RAMP_SECONDS * policy.freq)
        for index in range(ramp_steps):
            started = time.monotonic()
            if read_frame(env, controller) is None:
                return
            alpha = (index + 1) / ramp_steps
            env.step((1.0 - alpha) * initial + alpha * standing)
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -policy.dt:
                logger.warning("Control frame dropped during standing ramp")

        policy.reset()

        origin = time.monotonic()
        while True:
            started = time.monotonic()
            frame = read_frame(env, controller)
            if frame is None:
                break
            state, control = frame
            if policy_manager is None:
                target = policy.act(state, control)
                env.step(target)
                odometry = env.read_odometry()
            else:
                odometry = env.read_odometry()
                control.setdefault("buttons", getattr(controller, "pressed_buttons", ()))
                target = policy_manager.step(env, state, odometry, control, started - origin).target
            record(log, started - origin, state, target, odometry)
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            else:
                logger.error("Warning: frame drop -> %s", remaining)
                if remaining < -0.2:
                    logger.critical("Exiting due to excessive frame drop")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        try:
            if env is not None:
                env.shutdown()
        finally:
            if prompt_publisher is not None:
                prompt_publisher.close()
            if policy_manager is not None:
                policy_manager.close()
            if log is not None:
                save_recording(log, cfg.recording.directory, OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    run()
