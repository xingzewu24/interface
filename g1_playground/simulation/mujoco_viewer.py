import glfw
import mujoco
import numpy as np


def add_geom(scene, kind, rgba, size, pos):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return geom


class MujocoViewer:
    """Interactive MuJoCo window for simulation and replay."""

    def __init__(
        self,
        model,
        data,
        *,
        draw_overlay=None,
        width: int = 1280,
        height: int = 720,
    ):
        self.model = model
        self.data = data
        self.window = None
        self.context = None
        self._left_pressed = False
        self._right_pressed = False
        self._middle_pressed = False
        self._last_cursor = (0.0, 0.0)
        self.show_frame_overlay = True

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        try:
            self.window = glfw.create_window(width, height, "G1 Playground", None, None)
            if self.window is None:
                raise RuntimeError("Failed to create MuJoCo viewer window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            self.option = mujoco.MjvOption()
            self.perturb = mujoco.MjvPerturb()
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self.cam)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            if draw_overlay is not None:
                draw_overlay(self.scene)
            self._overlay_geom_count = self.scene.ngeom
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

            glfw.set_key_callback(self.window, self._key_callback)
            glfw.set_cursor_pos_callback(self.window, self._cursor_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
        except BaseException:
            self.close()
            raise

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def render(self, *, draw_frame_overlay=None) -> None:
        if not self.is_running():
            return
        glfw.make_context_current(self.window)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        categories = mujoco.mjtCatBit.mjCAT_ALL.value
        if self._overlay_geom_count:
            # Keep static overlay geometry and rebuild only the model geometry each frame.
            self.scene.ngeom = self._overlay_geom_count
            mujoco.mjv_updateCamera(self.model, self.data, self.cam, self.scene)
            mujoco.mjv_addGeoms(self.model, self.data, self.option, self.perturb, categories, self.scene)
            mujoco.mjv_makeLights(self.model, self.data, self.scene)
        else:
            mujoco.mjv_updateScene(self.model, self.data, self.option, self.perturb, self.cam, categories, self.scene)
        overlay = draw_frame_overlay(self.scene) if draw_frame_overlay is not None and self.show_frame_overlay else None
        mujoco.mjr_render(viewport, self.scene, self.context)
        if overlay is not None:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, *overlay, self.context
            )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def _key_callback(self, window, key, _scancode, action, _mods) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_P and action == glfw.PRESS:
            self.show_frame_overlay = not self.show_frame_overlay

    def _mouse_button_callback(self, window, button, action, _mods) -> None:
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._middle_pressed = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _cursor_callback(self, window, xpos, ypos) -> None:
        if not (self._left_pressed or self._right_pressed or self._middle_pressed):
            return
        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        _, height = glfw.get_framebuffer_size(window)
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model,
            action,
            (xpos - last_x) / height,
            (ypos - last_y) / height,
            self.cam,
        )

    def _scroll_callback(self, _window, _x_offset, y_offset) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            self.cam,
        )
