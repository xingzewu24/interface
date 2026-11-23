from queue import Empty, Queue

SHUTDOWN_BUTTON = "A"


class Controller:
    """Consume normalized axes and button edges from one input source."""

    def __init__(self, axes_names):
        self.state_queue = Queue(maxsize=2)
        self.event_queue = Queue(maxsize=100)
        self._axes_names = tuple(axes_names)
        self._last_axes = {name: 0.0 for name in self._axes_names}
        self.pressed_buttons: frozenset[str] = frozenset()

    def read(self) -> tuple[dict, bool]:
        """Return axes and shutdown status, and refresh the other press edges."""

        while True:
            try:
                state = self.state_queue.get_nowait()
            except Empty:
                break
            self._last_axes = state["axes"].copy()

        pressed_buttons = set()
        while True:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break
            if event["type"] == "button" and event["pressed"]:
                pressed_buttons.add(event["name"])

        self.pressed_buttons = frozenset(pressed_buttons - {SHUTDOWN_BUTTON})
        return {"axes": self._last_axes.copy()}, SHUTDOWN_BUTTON in pressed_buttons
