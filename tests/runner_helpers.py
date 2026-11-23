import numpy as np


class FakeRunner:
    def __init__(self, inputs, outputs, infer):
        self.input_names = tuple(inputs)
        self.output_names = tuple(outputs)
        self.shapes = {**inputs, **outputs}
        self.infer = infer

    def shape(self, name):
        return self.shapes[name]

    def run(self, inputs):
        return self.infer(inputs)


def leggedlab_runner():
    state_shape = (1, 1, 256)

    def infer(inputs):
        obs = np.asarray(inputs["obs"], dtype=np.float32)
        hidden = np.asarray(inputs["hidden_state"], dtype=np.float32)
        cell = np.asarray(inputs["cell_state"], dtype=np.float32)
        drive = obs.mean(axis=1, keepdims=True).reshape(1, 1, 1)
        next_hidden = 0.9 * hidden + drive + 0.01
        next_cell = 0.8 * cell + next_hidden
        actions = np.repeat(next_hidden.mean(axis=2), 29, axis=1).astype(np.float32)
        return {
            "actions": actions,
            "next_hidden_state": next_hidden.astype(np.float32),
            "next_cell_state": next_cell.astype(np.float32),
        }

    return FakeRunner(
        {"obs": (1, 96), "hidden_state": state_shape, "cell_state": state_shape},
        {"actions": (1, 29), "next_hidden_state": state_shape, "next_cell_state": state_shape},
        infer,
    )
