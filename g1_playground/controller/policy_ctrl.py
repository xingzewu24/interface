import time

from g1_playground.controller.keyboard_ctrl import KeyboardCtrl
from g1_playground.controller.utils.keyboard import KeyboardThread

POLICY_BUTTONS = {"B": "loco", "L1": "track", "LB": "track", "X": "metatron"}
IDLE_BUTTON = "Y"
STEER_BUTTON = "R1"
STEER_BUTTONS = frozenset((STEER_BUTTON, "RB"))


class PolicyKeyboardThread(KeyboardThread):
    BUTTONS = {"[": "X", "]": "B", "t": "L1", "i": IDLE_BUTTON, "k": STEER_BUTTON}

    def on_press(self, key) -> None:
        name = self.key_name(key)
        repeat = name in self.pressed_keys
        super().on_press(key)
        if not repeat and (button := self.BUTTONS.get(name)):
            self.event_queue.put({"type": "button", "name": button, "pressed": True, "timestamp": time.time()})


class PolicyKeyboardCtrl(KeyboardCtrl):
    THREAD_CLASS = PolicyKeyboardThread
