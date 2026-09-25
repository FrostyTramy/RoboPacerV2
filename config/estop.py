"""
Emergency-stop relay command - byte-identical across main/main.py,
manual_drive/manual_drive.py, cruise_control/cruise_control.py,
data_recorder/data_recorder.py, and model_runner/model_runner.py. Sends a
one-shot command to safety/estop_listener.py's Unix socket server, which
drives the physical relay (ESP32 GPIO13). Fire-and-forget: a socket error
here means the relay/estop_listener isn't reachable, in which case there's
nothing this process can do but keep running its own shutdown sequence.
"""
import socket

from config.ipc_config import RELAY_SOCKET


def relay_cmd(cmd: str) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(RELAY_SOCKET)
            s.sendall((cmd + "\n").encode())
    except (OSError, socket.timeout):
        pass
