"""
RoboPacerV2 Dashboard - System stats helpers
==============================================
Fiecare functie e defensiva: daca hardware-ul/serviciul nu raspunde, se
intoarce None / un dict cu "available": False in loc sa arunce - pagina de
status trebuie sa ramana utila chiar daca o singura sursa de date lipseste
(ex: ESP32 nedeconectat, sau Hailo ocupat de main.py).
"""

import json
import queue
import re
import socket
import subprocess
import threading
import time

RELAY_SOCKET = "/tmp/esp32_relay.sock"

_JOYSTICK_NAME_HINTS = ("xbox", "shanwan", "gamepad", "joystick", "controller")


def _run_with_timeout(fn, timeout):
    """Ruleaza fn() intr-un thread separat si renunta daca depaseste timeout.
    Necesar pentru operatii care pot bloca (ex: deschiderea device-ului Hailo
    cat timp e deja folosit exclusiv de main.py)."""
    result = queue.Queue(maxsize=1)

    def _target():
        try:
            result.put(("ok", fn()))
        except Exception as e:
            result.put(("err", e))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        status, value = result.get(timeout=timeout)
    except queue.Empty:
        return None
    if status == "err":
        return None
    return value


def get_cpu_temp():
    """Temperatura CPU-ului Raspberry Pi, in grade Celsius."""
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        m = re.search(r"([\d.]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def get_hailo_temp():
    """Temperatura chip-ului Hailo-8. Poate esua/dura daca device-ul e
    deja deschis exclusiv de main.py - in acel caz se intoarce None dupa
    un timeout scurt, in loc sa blocheze pagina de status."""

    def _read():
        from hailo_platform import VDevice
        with VDevice() as vdevice:
            dev = vdevice.get_physical_devices()[0]
            t = dev.control.get_chip_temperature()
            return round((t.ts0_temperature + t.ts1_temperature) / 2, 1)

    return _run_with_timeout(_read, timeout=3.0)


def get_controller_status():
    """Primul device evdev care arata a gamepad/joystick, dupa nume."""
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
    """Interogheaza estop_listener.py (singurul care detine portul serial)
    pe socket-ul deja folosit pentru RELAY_ON/RELAY_OFF, cu comanda STATUS."""
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


def get_all_stats():
    return {
        "cpu_temp_c":   get_cpu_temp(),
        "hailo_temp_c": get_hailo_temp(),
        "controller":   get_controller_status(),
        "esp32":        get_esp32_status(),
    }
