"""
RoboPacerV2 Dashboard - System stats helpers
==============================================
Fiecare functie e defensiva: daca hardware-ul/serviciul nu raspunde, se
intoarce None / un dict cu "available": False in loc sa arunce - pagina de
status trebuie sa ramana utila chiar daca o singura sursa de date lipseste
(ex: ESP32 nedeconectat, sau Hailo ocupat de main.py).
"""

import json
import os
import queue
import re
import socket
import subprocess
import threading
import time

RELAY_SOCKET = "/tmp/esp32_relay.sock"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_MODEL_DIR = os.path.join(PROJECT_ROOT, "main")

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


_HAILO_TEMP_CACHE_SECONDS = 30  # vezi comentariul de mai jos
_hailo_cache = {"value": None, "checked_at": 0.0}
_hailo_cache_lock = threading.Lock()


def get_hailo_temp():
    """Temperatura chip-ului Hailo-8. Poate esua/dura daca device-ul e
    deja deschis exclusiv de main.py - in acel caz se intoarce None dupa
    un timeout scurt, in loc sa blocheze pagina de status.

    Fiecare citire reala deschide un hailo_platform.VDevice - iar biblioteca
    HailoRT lasa in urma cateva thread-uri native care nu se mai inchid
    niciodata dupa ce VDevice e eliberat (confirmat empiric: thread count-ul
    procesului creste cu fiecare apel si nu mai coboara - nu e ceva
    controlabil din partea noastra, tine de biblioteca vendor inchisa).
    Pagina de status cheruie stats-urile la fiecare 2s din fiecare tab
    deschis - fara cache, asta ar acumula thread-uri suficient de repede
    incat sa incetineasca vizibil tot procesul Flask dupa doar cateva
    minute de dashboard deschis (asta a fost cauza lag-ului raportat).
    De-aia se refolosc ultima valoare cunoscuta si se reinterogheaza
    hardware-ul doar o data la _HAILO_TEMP_CACHE_SECONDS."""
    now = time.time()
    with _hailo_cache_lock:
        if now - _hailo_cache["checked_at"] < _HAILO_TEMP_CACHE_SECONDS:
            return _hailo_cache["value"]
        _hailo_cache["checked_at"] = now  # marcat imediat - nu pornim mai multe citiri in paralel

    def _read():
        from hailo_platform import VDevice
        with VDevice() as vdevice:
            dev = vdevice.get_physical_devices()[0]
            t = dev.control.get_chip_temperature()
            return round((t.ts0_temperature + t.ts1_temperature) / 2, 1)

    value = _run_with_timeout(_read, timeout=3.0)
    with _hailo_cache_lock:
        _hailo_cache["value"] = value
    return value


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


MANUAL_DRIVE_SOCKET = "/tmp/manual_drive_control.sock"


def get_manual_drive_status():
    """Interogheaza socket-ul de control al manual_drive.py (exista doar
    cat timp scriptul chiar ruleaza) - distanta/viteza live pentru pagina
    lui din dashboard."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(MANUAL_DRIVE_SOCKET)
            s.sendall(b"STATUS\n")
            raw = s.recv(1024).decode("utf-8", errors="replace").strip()
        return json.loads(raw)
    except (OSError, socket.timeout, json.JSONDecodeError):
        return None


def reset_manual_drive_distance():
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(MANUAL_DRIVE_SOCKET)
            s.sendall(b"RESET_DISTANCE\n")
            s.recv(1024)
        return True
    except (OSError, socket.timeout):
        return False


CRUISE_CONTROL_SOCKET = "/tmp/cruise_control_control.sock"


def get_cruise_control_status():
    """Interogheaza socket-ul de control al cruise_control.py (exista doar
    cat timp scriptul chiar ruleaza) - tinta/viteza live pentru pagina lui
    din dashboard."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(CRUISE_CONTROL_SOCKET)
            s.sendall(b"STATUS\n")
            raw = s.recv(1024).decode("utf-8", errors="replace").strip()
        return json.loads(raw)
    except (OSError, socket.timeout, json.JSONDecodeError):
        return None


MAIN_CONTROL_SOCKET = "/tmp/main_autopilot_control.sock"


def get_main_model_info():
    """Lista .hef din main/ (mirror-uieste regula "exact unul" din
    main/main.py:find_hef_path()) - ca pagina web sa poata arata/bloca
    Start-ul FARA sa astepte ca scriptul sa fi pornit."""
    try:
        hefs = [f for f in os.listdir(MAIN_MODEL_DIR) if f.endswith(".hef")]
    except OSError as e:
        return {"ok": False, "error": f"Nu pot citi {MAIN_MODEL_DIR}: {e}"}
    if len(hefs) == 0:
        return {"ok": False, "error": f"Niciun fisier .hef in {MAIN_MODEL_DIR}."}
    if len(hefs) > 1:
        return {"ok": False, "error": f"Mai multe fisiere .hef in {MAIN_MODEL_DIR}: {hefs}."}
    return {"ok": True, "model_name": hefs[0]}


def get_main_status():
    """Interogheaza socket-ul de control al main/main.py (exista doar cat
    timp scriptul chiar ruleaza) - tinta/viteza/distanta live."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(MAIN_CONTROL_SOCKET)
            s.sendall(b"STATUS\n")
            raw = s.recv(1024).decode("utf-8", errors="replace").strip()
        return json.loads(raw)
    except (OSError, socket.timeout, json.JSONDecodeError):
        return None


def get_all_stats():
    return {
        "cpu_temp_c":   get_cpu_temp(),
        "hailo_temp_c": get_hailo_temp(),
        "controller":   get_controller_status(),
        "esp32":        get_esp32_status(),
    }
