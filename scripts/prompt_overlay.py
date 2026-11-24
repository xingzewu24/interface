"""Draw the active Metatron goals in the simulator's world frame."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from g1_playground.simulation.mujoco_viewer import add_geom
from g1_playground.utils.math import quat_rotate, yaw_quat

if TYPE_CHECKING:
    from scripts.telemetry import PromptFrame

BODY_COLORS = {
    "torso_link": (0.08, 0.86, 0.91),
    "left_ankle_roll_link": (0.55, 0.98, 0.17),
    "right_ankle_roll_link": (1.00, 0.56, 0.11),
    "left_rubber_hand": (0.58, 0.42, 1.00),
    "right_rubber_hand": (1.00, 0.34, 0.61),
}
HAND_WRISTS = {"left_rubber_hand": "left_wrist_yaw_link", "right_rubber_hand": "right_wrist_yaw_link"}
AXIS_COLORS = ((1.0, 0.20, 0.22, 1.0), (0.18, 0.96, 0.28, 1.0), (0.22, 0.50, 1.0, 1.0))
VELOCITY_COLOR = (1.0, 0.94, 0.26, 1.0)


def _connector(scene, kind, start, end, width, rgba):
    if np.linalg.norm(end - start) < 1e-5:
        return
    geom = add_geom(scene, kind, rgba, (0, 0, 0), start)
    if geom is not None:
        mujoco.mjv_connector(geom, kind, width, start, end)


def _measured_position(data, body):
    try:
        return data.body(body).xpos
    except KeyError:
        wrist = HAND_WRISTS.get(body)
        if wrist is None:
            return None
        # The simulator has wrist bodies. Metatron's FK puts each hand marker 12 cm ahead of its wrist.
        hand_frame = data.body(wrist)
        return hand_frame.xpos + 0.12 * hand_frame.xmat.reshape(3, 3)[:, 0]


def draw_prompt(scene, data, frame: PromptFrame) -> tuple[str, str]:
    goals_by_body = defaultdict(list)
    for goal in frame.prompt.targets:
        goals_by_body[goal.body].append(goal)

    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    times = sorted({goal.time for goal in frame.prompt.targets})
    for body, goals in goals_by_body.items():
        color = BODY_COLORS.get(body, (0.82, 0.82, 0.82))
        ordered = sorted(goals, key=lambda goal: (goal.time, goal.slot))
        nearest_rotation = next((goal for goal in ordered if goal.rotation is not None), None)
        previous_position = None
        for goal in reversed(ordered):
            rank = times.index(goal.time) / max(1, len(times) - 1)
            rgba = (*color, 1.0 - 0.65 * rank)
            radius = 0.023 - 0.009 * rank
            height_only = body == "torso_link" and not goal.planar
            position = goal.position
            if position is None or height_only:
                measured = _measured_position(data, body)
                if measured is not None:
                    position = measured if position is None else np.array([*measured[:2], position[2]])
            if position is None:
                continue
            if goal.position is not None:
                if height_only:
                    add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, rgba, (radius * 1.7, 0.004, 0.0), position)
                    previous_position = None
                else:
                    if previous_position is not None:
                        _connector(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, position, previous_position, 0.004, rgba)
                    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, rgba, (radius,) * 3, position)
                    previous_position = position

            if goal.rotation is not None:
                rotation = goal.rotation[[3, 0, 1, 2]]
                heading_only = body == "torso_link" and goal.heading_only
                if goal is nearest_rotation and not heading_only:
                    for axis, axis_color in zip(np.eye(3), AXIS_COLORS, strict=True):
                        tip = position + 0.095 * quat_rotate(rotation, axis)
                        _connector(scene, arrow, position, tip, 0.004, axis_color)
                else:
                    if heading_only:
                        rotation = yaw_quat(rotation)
                    forward = quat_rotate(rotation, np.array([1.0, 0.0, 0.0]))
                    tip = position + (0.15 - 0.05 * rank) * forward
                    _connector(scene, arrow, position, tip, 0.005, rgba)

    velocity = frame.prompt.velocity
    velocity_label = "unset"
    if velocity is not None:
        origin = np.array([*data.qpos[:2], frame.prompt.ground_height + 0.06])
        tip = origin + np.array([*velocity, 0.0])
        _connector(scene, arrow, origin, tip, 0.009, VELOCITY_COLOR)
        velocity_label = f"({velocity[0]:+.2f}, {velocity[1]:+.2f}) m/s"
    horizon = f"{times[0]:.3g}-{times[-1]:.3g} s ({len(times)})" if times else "none"
    return (
        "Metatron prompt\nFuture goals\nXY velocity\nFeet\nHands\nOverlay",
        f"{frame.skill}\n{horizon}, cyan torso\n{velocity_label}, yellow\n"
        "L lime / R orange\nL violet / R pink\nP to hide",
    )
