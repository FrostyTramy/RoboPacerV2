"""
RoboPacerV2 Dashboard - system stats helpers
===============================================
Every function is defensive: if the hardware/service doesn't answer, this
returns None / a dict with a false "ok"/"connected" flag instead of raising -
the status page has to stay useful even if one data source is missing (e.g.
ESP32 disconnected, or the Hailo chip busy with main/main.py).
"""

import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
MAIN_DIR = os.path.join(REPO_ROOT, "main")

sys.path.append(REPO_ROOT)
from config.ipc_config import MAIN_CONTROL_SOCKET, MANUAL_DRIVE_CONTROL_SOCKET, RELAY_SOCKET  # noqa: E402

try:
    from config.battery import read_battery  # INA219 over I2C (smbus2)
except Exception:  # missing smbus2 etc. must never take the whole dashboard down
    def read_battery():
        return None

_JOYSTICK_NAME_HINTS = ("xbox", "shanwan", "gamepad", "joystick", "controller")


def get_cpu_temp():
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        m = re.search(r"([\d.]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


_HAILO_TEMP_CACHE_SECONDS = 30
_HAILO_READ_TIMEOUT_SECONDS = 3.0
_hailo_cache = {"value": None, "checked_at": 0.0}
_hailo_cache_lock = threading.Lock()
# Held for as long as a read has the Hailo device open - released by the
# reader thread itself, so a read that outlives its timeout still counts.
_hailo_device_lock = threading.Lock()


def get_hailo_temp(allowed=True):
    """Hailo-8 chip temperature, or None.

    allowed=False while any script runs: main/main.py needs the device
    exclusively, and an open VDevice here at the moment it starts makes its
    own VDevice() fail and the run abort. (While it runs, a read here would
    fail anyway.) Cached for _HAILO_TEMP_CACHE_SECONDS because HailoRT leaks
    native threads on every VDevice open/close - polling this every couple
    seconds from an open dashboard tab would otherwise slow the whole Flask
    process down within minutes."""
    if not allowed:
        return None
    now = time.time()
    with _hailo_cache_lock:
        if now - _hailo_cache["checked_at"] < _HAILO_TEMP_CACHE_SECONDS:
            return _hailo_cache["value"]
        _hailo_cache["checked_at"] = now
    if not _hailo_device_lock.acquire(blocking=False):
        return None  # a previous read is still stuck holding the device

    result = queue.Queue(maxsize=1)

    def _read():
        try:
            from hailo_platform import VDevice
            with VDevice() as vdevice:
                dev = vdevice.get_physical_devices()[0]
                t = dev.control.get_chip_temperature()
                result.put(round((t.ts0_temperature + t.ts1_temperature) / 2, 1))
        except Exception:
            result.put(None)
        finally:
            _hailo_device_lock.release()

    threading.Thread(target=_read, daemon=True).start()
    try:
        value = result.get(timeout=_HAILO_READ_TIMEOUT_SECONDS)
    except queue.Empty:
        value = None
    with _hailo_cache_lock:
        _hailo_cache["value"] = value
    return value


def wait_hailo_idle(timeout=_HAILO_READ_TIMEOUT_SECONDS + 2):
    """Blocks until no temperature read holds the Hailo device (call before
    starting a script, with new reads already disabled). True if idle."""
    if not _hailo_device_lock.acquire(timeout=timeout):
        return False
    _hailo_device_lock.release()
    return True


def get_controller_status():
    try:
        from evdev import InputDevice, list_devices
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            name = dev.name or ""
            if any(hint in name.lower() for hint in _JOYSTICK_NAME_HINTS):
                return {"connected": True, "name": name}
        return {"connected": False, "name": None}
    except Exception:
        return {"connected": False, "name": None}


def get_esp32_status():
    """Queries safety/estop_listener.py's relay socket - the only process
    holding the ESP32 serial port - for relay/ESP32/watchdog state."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.5)
            s.connect(RELAY_SOCKET)
            s.sendall(b"STATUS\n")
            raw = s.recv(1024).decode("utf-8", errors="replace").strip()
        state = json.loads(raw)
        state["service_running"] = True
        state["age_seconds"] = round(time.time() - state.get("last_update", time.time()), 1)
        return state
    except (OSError, socket.timeout, json.JSONDecodeError):
        return {
            "service_running": False,
            "esp32_connected": False,
            "relay_on": False,
            "watchdog_armed": False,
            "last_esp32_line": None,
            "age_seconds": None,
        }


def get_main_model_info():
    """Mirrors main/main.py's find_hef_path() "exactly one .hef" rule, so
    the web page can show/block Start before the script has even run."""
    try:
        hefs = [f for f in os.listdir(MAIN_DIR) if f.endswith(".hef")]
    except OSError as e:
        return {"ok": False, "error": f"Nu pot citi {MAIN_DIR}: {e}"}
    if len(hefs) == 0:
        return {"ok": False, "error": f"Niciun fisier .hef in {MAIN_DIR}."}
    if len(hefs) > 1:
        return {"ok": False, "error": f"Mai multe fisiere .hef in {MAIN_DIR}: {hefs}."}
    return {"ok": True, "model_name": hefs[0]}


def _query_control_socket(path, command):
    """One newline-terminated command to a script's control socket (only
    exists while that script is running), one JSON line back - or None."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(path)
            s.sendall(command.encode() + b"\n")
            raw = s.recv(1024).decode("utf-8", errors="replace").strip()
        return json.loads(raw)
    except (OSError, socket.timeout, json.JSONDecodeError):
        return None


def get_main_status():
    """main/main.py's live target/speed/distance telemetry."""
    return _query_control_socket(MAIN_CONTROL_SOCKET, "STATUS")


def get_manual_drive_status():
    """manual_drive/manual_drive.py's live distance/speed/pause state."""
    return _query_control_socket(MANUAL_DRIVE_CONTROL_SOCKET, "STATUS")


def reset_manual_drive_distance():
    """Zeroes manual_drive's displayed distance + avg/max speed (its final
    summary still reports the real total)."""
    return _query_control_socket(MANUAL_DRIVE_CONTROL_SOCKET, "RESET_DISTANCE") is not None


def get_all_stats(hailo_allowed=True):
    return {
        "cpu_temp_c": get_cpu_temp(),
        "hailo_temp_c": get_hailo_temp(allowed=hailo_allowed),
        "controller": get_controller_status(),
        "esp32": get_esp32_status(),
        "battery": read_battery(),
    }
