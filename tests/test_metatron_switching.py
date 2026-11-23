import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from g1_playground.interface.metatron.prompts import Prompt
from g1_playground.interface.metatron.skills import SkillLayer
from g1_playground.policy.metatron import MetatronPolicy
from g1_playground.policy.metatron.metatron_contract import MetatronContract
from g1_playground.policy.track import TrackPolicy
from g1_playground.policy_handoff import Gains, HandoffLocomotion, HandoffPolicy, PolicyHandoff
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import compose_config
from tests.runner_helpers import FakeRunner, leggedlab_runner


class TestMetatronSwitching(unittest.TestCase):
    def setUp(self):
        self.cfg = compose_config(
            "sim",
            "metatron.settle_seconds=0.02",
            "metatron.transition_seconds=0.04",
            "track.settle_seconds=0.02",
            "track.transition_seconds=0.04",
        )
        self.dof = compose_dof_config(self.cfg.robot.dof, self.cfg.policy.dof)
        self.contract = MetatronContract(resolve_repo_path(self.cfg.metatron.contract))
        self.policy = MetatronPolicy(self.cfg.metatron, dof_cfg=self.dof, runner=self.runner())
        self.fk = self.policy.observation
        self.state = SimpleNamespace(
            dof_pos=np.asarray(self.dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(29, dtype=np.float32),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_ang_vel=np.zeros(3),
        )
        self.odometry = SimpleNamespace(position=np.array([0.0, 0.0, 0.78]), velocity=np.zeros(3))

    def runner(self):
        outputs = {
            "actions": np.zeros_like(self.contract.action_scale[None]),
            "joint_pos_targets": self.contract.default_dof_pos[None],
            "stiffness_targets": self.contract.stiffness[None],
            "damping_targets": self.contract.damping[None],
        }
        runner = FakeRunner(
            {item.name: item.shape for item in self.contract.inputs}, self.contract.outputs, lambda inputs: outputs
        )
        dtypes = {item.name: item.dtype for item in self.contract.inputs}
        dtypes.update({name: value.dtype for name, value in outputs.items()})
        runner.dtype = dtypes.__getitem__
        return runner

    def test_metatron_uses_exported_tensor_shapes_to_build_tensorrt_engine(self):
        runner = self.runner()
        with patch("g1_playground.policy.metatron.metatron_policy.TensorRTRunner", return_value=runner) as build:
            policy = MetatronPolicy(self.cfg.metatron, dof_cfg=self.dof)
        build.assert_called_once_with(
            policy.contract.onnx_path,
            input_shapes={item.name: item.shape for item in policy.contract.inputs},
        )

    def test_fk_matches_simulated_torso_and_feet_at_same_pose(self):
        model = mujoco.MjModel.from_xml_path(resolve_repo_path(self.cfg.robot.xml))
        data = mujoco.MjData(model)
        data.qpos[7:] = self.state.dof_pos
        bodies = ("torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
        for heading in (0.0, 0.6, -1.4):
            self.state.base_quat[:] = (0.0, 0.0, np.sin(heading / 2), np.cos(heading / 2))
            self.odometry.position[:2] = (heading, 2 * heading)
            data.qpos[:3] = self.odometry.position
            data.qpos[3:7] = self.state.base_quat[[3, 0, 1, 2]]
            mujoco.mj_kinematics(model, data)
            observed = self.fk.read(self.state, self.odometry)
            for name in bodies:
                body = self.contract.body_names.index(name)
                np.testing.assert_allclose(observed.body_pos[body], data.body(name).xpos, atol=2e-7)
                np.testing.assert_allclose(observed.body_rot[body], data.body(name).xquat[[1, 2, 3, 0]], atol=2e-7)

    def test_history_only_advances_when_applied_target_is_accepted(self):
        observed = self.fk.read(self.state, self.odometry)
        prompt = Prompt()
        target = self.policy.act(observed, prompt)
        observation = self.policy.get_observation(observed, prompt)
        self.assertEqual(observation.keys(), {item.name for item in self.contract.inputs})
        np.testing.assert_array_equal(observation["current_action"], 0)
        applied = target + 0.1 * self.contract.action_scale
        self.policy.accept_applied_target(applied)
        np.testing.assert_allclose(self.policy.get_observation(observed, prompt)["current_action"], 0.1, atol=3e-7)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.policy.accept_applied_target(applied)

    def handoff(self, *modes):
        loco = HandoffLocomotion(self.cfg.policy, dof_cfg=self.dof, runner=leggedlab_runner())
        effort = np.asarray(self.cfg.robot.dof.torque_limits)
        gains = Gains(np.asarray(self.dof.stiffness), np.asarray(self.dof.damping), effort)
        policies = {}
        if "metatron" in modes:
            skills = SkillLayer(self.policy.dt, self.fk.prompt_slots, self.fk.default_pose)
            metatron_gains = Gains(
                self.policy.stiffness, self.policy.damping, np.minimum(effort, self.policy.effort_limits)
            )
            policies["metatron"] = HandoffPolicy(
                self.policy,
                self.cfg.metatron,
                metatron_gains,
                self.policy.joint_limits,
                skills=skills,
            )
        if "track" in modes:
            dof = compose_dof_config(self.cfg.robot.dof, self.cfg.track.dof)
            runner = FakeRunner(
                {"obs": (1, 167), "obs_history": (1, 3, 167)},
                {"actions": (1, 29)},
                lambda inputs: {"actions": np.zeros((1, 29), dtype=np.float32)},
            )
            track = TrackPolicy(self.cfg.track, dof_cfg=dof, runner=runner)
            track_gains = Gains(np.asarray(dof.stiffness), np.asarray(dof.damping), effort)
            limits = np.column_stack((track.joint_pos_lower, track.joint_pos_upper))
            policies["track"] = HandoffPolicy(track, self.cfg.track, track_gains, limits)

        def run_inline(action, *args):
            result = Future()
            result.set_result(action(*args))
            return result

        with patch("g1_playground.policy_handoff.ThreadPoolExecutor") as executor:
            executor.return_value.submit.side_effect = run_inline
            handoff = PolicyHandoff(loco, gains, policies)
        handoff.warmup(self.state, self.odometry)
        return handoff

    def test_bidirectional_handoff_and_odometry_loss(self):
        handoff = self.handoff("metatron")
        sent, gains = [], []
        env = SimpleNamespace(step=sent.append, set_gains=lambda kp, kd: gains.append(kp))
        axes = {"LeftY": 0.0, "LeftX": 0.0, "RightX": 0.0}
        control = {"axes": axes, "buttons": {"X"}}
        self.assertEqual(handoff.step(env, self.state, None, control, 0.0).mode, "loco")
        frames = []
        loco = handoff.loco
        with (
            patch.object(loco, "act", wraps=loco.act) as loco_act,
            patch.object(self.policy, "act", wraps=self.policy.act) as metatron_act,
            patch.object(loco, "accept_applied_target", wraps=loco.accept_applied_target) as loco_accept,
            patch.object(self.policy, "accept_applied_target", wraps=self.policy.accept_applied_target) as meta_accept,
        ):
            for frame in range(1, 22):
                buttons = {1: {"X"}, 7: {"R1"}, 8: {"Y"}, 10: {"B"}, 17: {"X"}}.get(frame, set())
                for call in (loco_act, metatron_act, loco_accept, meta_accept):
                    call.reset_mock()
                frames.append(
                    handoff.step(env, self.state, self.odometry, {"axes": axes, "buttons": buttons}, frame * 0.03)
                )
                if frame in (3, 12, 19):
                    loco_act.assert_called_once()
                    metatron_act.assert_called_once()
                    loco_accept.assert_called_once()
                    meta_accept.assert_called_once()
                    np.testing.assert_array_equal(loco_accept.call_args.args[0], sent[-1])
                    np.testing.assert_array_equal(meta_accept.call_args.args[0], sent[-1])
                if frame == 11:
                    self.assertEqual(frames[-1].mode, "to_loco")
                    with self.assertRaisesRegex(RuntimeError, "Metatron to loco lost fresh pelvis odometry"):
                        handoff.step(env, self.state, None, {"axes": axes}, 0.34)
                    self.assertEqual(len(sent), frame + 1)
        modes = list(dict.fromkeys(frame.mode for frame in frames))
        self.assertEqual(modes, ["loco", "to_metatron", "metatron", "to_loco"])
        self.assertEqual([frame.skill for frame in frames if frame.mode == "to_metatron"], ["stand"] * 6)
        self.assertEqual([frames[index - 1].skill for index in (7, 8, 21)], ["steer", "idle", "stand"])
        self.assertEqual(len(sent), 22)
        self.assertGreater(len(gains), 4)
        with self.assertRaisesRegex(RuntimeError, "lost fresh pelvis odometry"):
            handoff.step(env, self.state, None, {"axes": axes}, 1.0)
        self.assertEqual(len(sent), 22)

    def test_metatron_keeps_loco_until_primed_and_seeds_history_without_inference(self):
        handoff = self.handoff("metatron")
        sent = []
        env = SimpleNamespace(step=sent.append, set_gains=lambda kp, kd: None)
        axes = {"LeftY": 0.0, "LeftX": 0.0, "RightX": 0.0}
        control = {"axes": axes}
        handoff.step(env, self.state, self.odometry, {**control, "buttons": {"X"}}, 0.0)
        priming = Future()
        handoff._warmup_worker.submit.side_effect = None
        handoff._warmup_worker.submit.return_value = priming
        with (
            patch.object(self.policy.runner, "run", wraps=self.policy.runner.run) as infer,
            patch.object(self.fk, "read", wraps=self.fk.read) as read_pose,
        ):
            waiting = handoff.step(env, self.state, self.odometry, control, 0.03)
            self.assertEqual(waiting.mode, "loco")
            self.assertEqual(len(sent), 2)
            read_pose.assert_not_called()
            infer.assert_not_called()

            priming.set_result(None)
            first = handoff.step(env, self.state, self.odometry, control, 0.06)
            self.assertEqual(first.mode, "to_metatron")
            read_pose.assert_called_once()
            infer.assert_not_called()
            np.testing.assert_allclose(sent[1], first.target)

            handoff.step(env, self.state, self.odometry, control, 0.09)
            infer.assert_called_once()
            model_target = self.policy.observation_adapter.fit(first.target)
            expected = ((model_target - self.policy.default_pos) / self.policy.action_scale)[None]
            np.testing.assert_allclose(infer.call_args.args[0]["current_action"], expected)
        self.assertEqual(len(sent), 4)

    def test_track_round_trip_and_mixed_handoffs(self):
        handoff = self.handoff("track", "metatron")
        sent, gains = [], []
        env = SimpleNamespace(step=sent.append, set_gains=lambda kp, kd: gains.append(kp))
        axes = {"LeftY": 0.0, "LeftX": 0.0, "RightX": 0.0}
        self.assertEqual(handoff.step(env, self.state, None, {"axes": axes, "buttons": {"L1"}}, 0.0).mode, "loco")
        commands = {1: "L1", 6: "B", 11: "L1", 16: "X", 21: "L1", 26: "B"}
        modes = []
        track = handoff.policies["track"].policy
        with (
            patch.object(handoff.loco, "act", wraps=handoff.loco.act) as loco_act,
            patch.object(track, "act", wraps=track.act) as track_act,
        ):
            for index in range(1, 32):
                button = commands.get(index)
                control = {"axes": axes, "buttons": () if button is None else (button,)}
                loco_act.reset_mock()
                track_act.reset_mock()
                modes.append(handoff.step(env, self.state, self.odometry, control, index * 0.03).mode)
                if index in (3, 8):
                    loco_act.assert_called_once()
                    track_act.assert_called_once()
        self.assertEqual(modes[3::5], ["to_track", "to_loco", "to_track", "to_metatron", "to_track", "to_loco"])
        self.assertEqual(modes[4:30:5], ["track", "loco", "track", "metatron", "track", "loco"])
        self.assertEqual(len(sent), 32)
        np.testing.assert_array_equal(gains[-1], self.dof.stiffness)
