import argparse
import importlib
import logging
import threading
import time

import mujoco
from omegaconf import OmegaConf

from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer
from g1_playground.simulation.mujoco_viewer import MujocoViewer
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.logger import setup_logger

if __package__:
    from .prompt_overlay import draw_prompt
    from .telemetry import PromptReceiver
else:
    from prompt_overlay import draw_prompt
    from telemetry import PromptReceiver

RENDER_HZ = 60.0
logger = logging.getLogger("g1_playground")


def build_server(sim=None) -> G1MujocoDdsServer:
    robot = OmegaConf.load(resolve_repo_path("configs/robot/g1.yaml"))
    if sim is None:
        sim = OmegaConf.load(resolve_repo_path("configs/deployment/sim.yaml"))
    try:
        unitree_cpp = importlib.import_module("unitree_cpp")
    except ImportError as error:
        raise RuntimeError(
            "The MuJoCo DDS server requires the vendored unitree_cpp binding; "
            "run 'python scripts/setup/install_third_party.py unitree_cpp' first"
        ) from error

    endpoint = {
        "domain_id": sim.env.domain_id,
        "net_if": sim.env.net_if,
        "lowcmd_topic": sim.env.lowcmd_topic,
        "lowstate_topic": sim.env.lowstate_topic,
        "sport_state_topic": sim.env.sport_state_topic,
        "mode_machine": 5,
    }

    backend = G1MujocoBackend(resolve_repo_path(robot.xml), elastic_support_scale=1.0)
    robot_endpoint = unitree_cpp.G1DdsRobotEndpoint(endpoint)
    try:
        return G1MujocoDdsServer(
            backend,
            robot_endpoint,
            unitree_cpp.DdsLowStateSnapshot,
            robot.dof.torque_limits,
            sport_state_factory=unitree_cpp.DdsSportStateSnapshot,
            sport_publish_hz=float(sim.sim_odometry.publish_hz),
        )
    except BaseException:
        robot_endpoint.close()
        raise


def run_with_viewer(server: G1MujocoDdsServer, *, prompt_port: int | None = None) -> None:
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []
    viewer = None
    prompt_receiver = None
    worker = None
    started = False

    def run_server() -> None:
        try:
            server.run(stop_event)
        except BaseException as error:
            worker_errors.append(error)
        finally:
            stop_event.set()

    try:
        if prompt_port:
            try:
                prompt_receiver = PromptReceiver(prompt_port)
            except OSError as error:
                logger.warning("MuJoCo prompt overlay unavailable: %s", error)
        render_data = mujoco.MjData(server.backend.model)  # pyright: ignore[reportAttributeAccessIssue]
        server.backend.copy_data_to(render_data)
        viewer = MujocoViewer(server.backend.model, render_data)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10.0
        viewer.cam.azimuth = 180.0
        worker = threading.Thread(target=run_server, name="G1MujocoDdsServer")
        worker.start()
        started = True
        frame_period = 1.0 / RENDER_HZ
        frame_deadline = time.monotonic()

        def draw_frame_overlay(scene):
            if prompt_receiver is not None and (frame := prompt_receiver.latest()) is not None:
                return draw_prompt(scene, render_data, frame)
            return None

        while viewer.is_running() and not stop_event.is_set():
            server.backend.copy_data_to(render_data)
            viewer.cam.lookat[:] = render_data.qpos[:3]
            viewer.render(draw_frame_overlay=draw_frame_overlay)
            frame_deadline += frame_period
            stop_event.wait(max(frame_deadline - time.monotonic(), 0.0))
    finally:
        stop_event.set()
        if started:
            if worker is not None:
                worker.join()
        else:
            server.shutdown()
        if viewer is not None:
            viewer.close()
        if prompt_receiver is not None:
            prompt_receiver.close()

    if worker_errors:
        raise worker_errors[0]


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(description="G1 MuJoCo DDS simulator with a mandatory viewer")
    parser.parse_args(argv)

    setup_logger()
    try:
        sim = OmegaConf.load(resolve_repo_path("configs/deployment/sim.yaml"))
        run_with_viewer(build_server(sim), prompt_port=sim.get("sim_prompt_port"))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    run()
