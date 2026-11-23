import math
import unittest
from types import SimpleNamespace

import numpy as np

from g1_playground.controller.base_ctrl import Controller
from g1_playground.controller.policy_ctrl import PolicyKeyboardThread
from g1_playground.interface.metatron.prompts import PromptPacker
from g1_playground.interface.metatron.skills import SkillContext, SkillLayer, Steer, heading_quaternion
from g1_playground.policy.metatron.metatron_contract import MetatronContract
from g1_playground.utils import resolve_repo_path


class TestMetatronSkills(unittest.TestCase):
    def setUp(self):
        self.contract = MetatronContract(resolve_repo_path("assets/models/metatron/unified_pipeline.yaml"))
        self.packer = PromptPacker(self.contract)
        self.skills = SkillLayer(self.contract.control_dt, self.packer.slots, self.context(0.0))

    def context(self, x):
        body_count = len(self.contract.body_names)
        positions = np.zeros((body_count, 3))
        positions[:, 0] = x
        positions[self.contract.anchor_body_index, 2] = 0.82
        rotations = np.tile([0.0, 0.0, 0.0, 1.0], (body_count, 1))
        return SkillContext(positions, rotations, self.contract.body_names, self.contract.anchor_body)

    def test_idle_recaptures_pose_on_reentry(self):
        self.skills.switch("idle", self.context(1.0))
        prompt = self.skills.advance(self.context(2.0))
        np.testing.assert_allclose(prompt.targets[0].position, [1.0, 0.0, 0.82])
        self.skills.switch("idle", self.context(2.0))
        prompt = self.skills.advance(self.context(3.0))
        np.testing.assert_allclose(prompt.targets[0].position, [2.0, 0.0, 0.82])

    def test_stand_uses_smoke_world_pose_and_masks(self):
        self.skills.switch("stand", self.context(1.0))
        packed = self.packer.pack(self.skills.advance(self.context(2.0)))
        anchor = self.contract.anchor_body_index
        np.testing.assert_allclose(packed["masked_mimic.ref_pos"][0, :, anchor], [[0.0, 0.0, 0.8]] * 5)
        np.testing.assert_array_equal(packed["masked_mimic.ref_rot"][0, :, anchor], [[0.0, 0.0, 0.0, 1.0]] * 5)
        np.testing.assert_allclose(packed["masked_mimic.time_offsets"], [[0.1, 0.2, 0.3, 0.4, 0.5]])
        np.testing.assert_array_equal(packed["masked_mimic.anchor_translation_mask"][0], [[True, True]] * 5)
        np.testing.assert_array_equal(packed["masked_mimic.anchor_rotation_mask"][0], [[False, True]] * 5)
        np.testing.assert_array_equal(packed["masked_mimic.average_velocity"], [[0.0, 0.0]])
        self.assertTrue(packed["masked_mimic.velocity_command_mask"].all())

    def test_steer_updates_velocity_and_keeps_gait_phase(self):
        context = self.context(0.0)
        self.skills.switch("steer", context)
        steer = self.skills.active
        self.assertIsInstance(steer, Steer)
        steer.set_command(np.array([0.6, 0.0]), 0.0)
        self.skills.advance(context)
        phase = steer.phase
        steer.set_command(np.array([0.0, 0.6]), 0.0)
        packed = self.packer.pack(self.skills.advance(context))
        self.assertGreater(steer.phase, phase)
        np.testing.assert_allclose(packed["masked_mimic.average_velocity"], [[0.0, 0.6]])
        self.assertTrue(packed["masked_mimic.velocity_command_mask"].all())
        self.assertFalse(packed["masked_mimic.local_target_poses_masks"].any())
        self.assertFalse(packed["masked_mimic.style_frame_masks"].any())

    def test_operator_controls_produce_world_frame_prompts(self):
        context = self.context(0.0)
        heading = math.pi / 2
        context.rotations[:] = heading_quaternion(heading)
        self.skills.start(context)
        self.assertEqual(self.skills.name, "stand")

        control = {"axes": {"LeftX": 0.0, "LeftY": 1.0, "RightX": -0.5}, "buttons": {"R1"}}
        prompt = self.skills.advance(context, control)
        self.assertEqual(self.skills.name, "steer")
        np.testing.assert_allclose(prompt.velocity, [0.0, self.skills.steer_speed], atol=1e-7)
        facing = heading + 0.5 * self.skills.steer_yaw_rate * self.contract.control_dt
        np.testing.assert_allclose(prompt.targets[0].rotation, heading_quaternion(facing))

        stopped = self.skills.advance(context, control, settling=True)
        self.assertEqual(self.skills.name, "idle")
        np.testing.assert_array_equal(stopped.velocity, [0.0, 0.0])


class TestMetatronKeyboard(unittest.TestCase):
    def test_switch_edge_and_shutdown_use_the_existing_input_queue(self):
        controller = Controller(())
        keyboard = PolicyKeyboardThread(controller.state_queue, controller.event_queue, (), 0.5, 0.5, 0.4, 0.35)
        enter = SimpleNamespace(char="[")
        keyboard.on_press(enter)
        self.assertFalse(controller.read()[1])
        self.assertEqual(controller.pressed_buttons, {"X"})
        keyboard.on_press(enter)
        keyboard.on_press(SimpleNamespace(char="t"))
        keyboard.on_press(SimpleNamespace(name="esc"))
        self.assertTrue(controller.read()[1])
        self.assertEqual(controller.pressed_buttons, {"L1"})
