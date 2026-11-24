import socket
import time
import unittest
from contextlib import closing
from unittest.mock import patch

import mujoco
import numpy as np

from g1_playground.interface.metatron.prompts import BodyTarget, Prompt
from g1_playground.utils import resolve_repo_path
from scripts.prompt_overlay import draw_prompt
from scripts.telemetry import PromptFrame, PromptPublisher, PromptReceiver


class TestMetatronPromptOverlay(unittest.TestCase):
    def test_loopback_reads_latest_prompt_and_expires_it(self):
        torso = BodyTarget(
            "torso_link",
            2,
            0.3,
            np.array([1.0, 2.0, 0.8]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            planar=False,
            heading_only=True,
        )
        prompt = Prompt((torso,), np.array([-0.2, 0.1]), ground_height=0.1)
        with closing(PromptReceiver(0)) as receiver:
            port = receiver._socket.getsockname()[1]
            with closing(PromptPublisher(port)) as publish:
                publish("stand", Prompt())
                publish("steer", prompt)
            with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
                sock.sendto(b"invalid", ("127.0.0.1", port))
            frame = receiver.latest()
            self.assertEqual(frame.skill, "steer")
            self.assertEqual(frame.prompt.ground_height, 0.1)
            np.testing.assert_array_equal(frame.prompt.velocity, prompt.velocity)
            (received,) = frame.prompt.targets
            self.assertEqual((received.body, received.slot, received.time), ("torso_link", 2, 0.3))
            self.assertFalse(received.planar)
            self.assertTrue(received.heading_only)
            np.testing.assert_array_equal(received.position, torso.position)
            np.testing.assert_array_equal(received.rotation, torso.rotation)
            with patch("scripts.telemetry.time.monotonic", return_value=frame.sent_at + 1):
                self.assertIsNone(receiver.latest())

    def test_draws_future_positions_and_respects_height_mask(self):
        model = mujoco.MjModel.from_xml_path(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
        data = mujoco.MjData(model)
        data.qpos[:2] = [1.0, 2.0]
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=50)
        frame = PromptFrame(
            time.monotonic(),
            "steer",
            Prompt(
                (
                    BodyTarget("torso_link", 0, 0.1, np.array([9.0, 9.0, 0.75]), planar=False),
                    BodyTarget("left_ankle_roll_link", 0, 0.1, np.array([1.0, 2.0, 0.05])),
                    BodyTarget("left_ankle_roll_link", 1, 0.2, np.array([1.2, 2.0, 0.08])),
                    BodyTarget("left_rubber_hand", 0, 0.1, np.array([1.2, 2.2, 0.9]), np.array([0.0, 0.0, 0.0, 1.0])),
                ),
                np.array([0.2, 0.0]),
            ),
        )
        _, legend = draw_prompt(scene, data, frame)
        self.assertIn("0.1-0.2 s (2)", legend)
        self.assertIn("(+0.20, +0.00) m/s", legend)
        geoms = scene.geoms[: scene.ngeom]
        self.assertCountEqual(
            [geom.type for geom in geoms],
            [
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_ARROW,
                mujoco.mjtGeom.mjGEOM_ARROW,
                mujoco.mjtGeom.mjGEOM_ARROW,
                mujoco.mjtGeom.mjGEOM_ARROW,
            ],
        )
        np.testing.assert_allclose(geoms[0].pos, [*data.body("torso_link").xpos[:2], 0.75])
