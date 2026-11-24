"""Live Metatron prompts for the local simulator viewer."""

import json
import math
import socket
import time
from dataclasses import dataclass

import numpy as np

from g1_playground.interface.metatron.prompts import BodyTarget, Prompt

PROMPT_MAX_AGE = 0.25


@dataclass(frozen=True)
class PromptFrame:
    sent_at: float
    skill: str
    prompt: Prompt


class PromptPublisher:
    def __init__(self, port: int):
        self._address = ("127.0.0.1", port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)

    def __call__(self, skill: str, prompt: Prompt) -> None:
        packet = {
            "sent_at": time.monotonic(),
            "skill": skill,
            "velocity": None if prompt.velocity is None else np.asarray(prompt.velocity).tolist(),
            "ground_height": float(prompt.ground_height),
            "targets": [
                {
                    "body": goal.body,
                    "slot": int(goal.slot),
                    "time": max(0.0, float(goal.time)),
                    "position": None if goal.position is None else np.asarray(goal.position).tolist(),
                    "rotation": None if goal.rotation is None else np.asarray(goal.rotation).tolist(),
                    "planar": bool(goal.planar),
                    "heading_only": bool(goal.heading_only),
                }
                for goal in prompt.targets
            ],
        }
        try:
            self._socket.sendto(json.dumps(packet, separators=(",", ":"), allow_nan=False).encode(), self._address)
        except (OSError, ValueError):
            # The viewer may be absent or slow. Never wait for it in the control loop.
            pass

    def close(self) -> None:
        self._socket.close()


class PromptReceiver:
    def __init__(self, port: int):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind(("127.0.0.1", port))
            self._socket.setblocking(False)
        except OSError:
            self._socket.close()
            raise
        self._frame: PromptFrame | None = None

    def latest(self) -> PromptFrame | None:
        # Prefer the newest snapshot and cap socket work per rendered frame.
        for _ in range(32):
            try:
                packet = self._socket.recv(65536)
            except BlockingIOError:
                break
            try:
                frame = self._decode(packet)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                continue
            age = time.monotonic() - frame.sent_at
            if 0 <= age <= PROMPT_MAX_AGE and (self._frame is None or frame.sent_at >= self._frame.sent_at):
                self._frame = frame
        if self._frame is not None and time.monotonic() - self._frame.sent_at > PROMPT_MAX_AGE:
            self._frame = None
        return self._frame

    @staticmethod
    def _decode(packet: bytes) -> PromptFrame:
        def vector(value, size):
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError("Invalid prompt vector")
            return array

        data = json.loads(packet)
        if not isinstance(data["skill"], str) or not math.isfinite(data["ground_height"]):
            raise ValueError("Invalid prompt metadata")
        targets = []
        for item in data["targets"]:
            if (
                not isinstance(item["body"], str)
                or not isinstance(item["slot"], int)
                or not math.isfinite(item["time"])
            ):
                raise ValueError("Invalid body target")
            rotation = vector(item["rotation"], 4)
            if rotation is not None and not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-4):
                raise ValueError("Invalid prompt rotation")
            targets.append(BodyTarget(**{**item, "position": vector(item["position"], 3), "rotation": rotation}))
        prompt = Prompt(tuple(targets), vector(data["velocity"], 2), data["ground_height"])
        return PromptFrame(float(data["sent_at"]), data["skill"], prompt)

    def close(self) -> None:
        self._socket.close()
