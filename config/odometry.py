"""
ESP32 odometry (RPM) reader - byte-identical between main/main.py and
cruise_control/cruise_control.py (the version with staleness tracking).
manual_drive/manual_drive.py carried a simpler copy without staleness
tracking; this shared version is the superset and is safe for callers that
don't care about staleness to use as-is (they just never call is_odo_stale()).

Runs as a background thread for the lifetime of the program: connects to
safety/estop_listener.py's odometry socket, parses "RPM:<float>\n" lines,
and reconnects on any drop. If the connection is down, RPM falls back to 0
rather than staying frozen at the last known speed.
"""
import socket
import threading
import time

from config.hardware_config import WHEEL_CIRCUMFERENCE_M
from config.ipc_config import ODO_RECONNECT_SLEEP_SECONDS, ODO_SOCKET, ODO_STALE_TIMEOUT_SECONDS

_odo_state = {"rpm": 0.0, "last_update": 0.0}
_odo_lock = threading.Lock()


def set_rpm(value, fresh=True):
    with _odo_lock:
        _odo_state["rpm"] = value
        if fresh:
            _odo_state["last_update"] = time.time()


def get_rpm():
    with _odo_lock:
        return _odo_state["rpm"]


def is_odo_stale():
    with _odo_lock:
        return (time.time() - _odo_state["last_update"]) > ODO_STALE_TIMEOUT_SECONDS


def rpm_to_kmh(rpm):
    return rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000


def odo_reader_loop(stop_event):
    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(ODO_SOCKET)
        except OSError:
            stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)
            continue

        buf = ""
        try:
            while not stop_event.is_set():
                try:
                    data = sock.recv(64).decode("utf-8", errors="replace")
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("RPM:"):
                        continue
                    try:
                        set_rpm(float(line[4:]), fresh=True)
                    except ValueError:
                        continue
        finally:
            sock.close()
            set_rpm(0.0, fresh=False)
            if not stop_event.is_set():
                stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)
