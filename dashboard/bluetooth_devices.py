"""
RoboPacerV2 Dashboard - Bluetooth devices
============================================
Like a phone's Bluetooth screen: the devices the Pi has saved (paired) with
connect / disconnect / forget, and a scan for new ones. Backend is BlueZ's
`bluetoothctl` (the Pi OS default).

Security: same as wifi.py - every call is subprocess.run([...]) with a list,
never a shell string, and every MAC address that comes from the HTTP request
is checked against MAC_RE first.
"""

import re
import subprocess
import threading

MAC_RE = re.compile(r"[0-9A-F]{2}(:[0-9A-F]{2}){5}")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x01|\x02")
_DEVICE_LINE_RE = re.compile(r"^Device ((?:[0-9A-F]{2}:){5}[0-9A-F]{2}) (.*)$")

SCAN_SECONDS = 8
_CTL_TIMEOUT_S = 10
_PAIR_TIMEOUT_S = 40
_CONNECT_TIMEOUT_S = 25

# One slow operation (scan / pair / connect) at a time - BlueZ handles
# concurrent ones badly, and a second tap while one is running is a mistake.
_busy = threading.Lock()


class Busy(Exception):
    pass


def valid_mac(mac):
    return isinstance(mac, str) and MAC_RE.fullmatch(mac) is not None


def _ctl(args, timeout=_CTL_TIMEOUT_S):
    """Runs bluetoothctl, returns (exit_code, cleaned output)."""
    try:
        r = subprocess.run(["bluetoothctl", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "bluetoothctl nu e instalat."
    except subprocess.TimeoutExpired:
        return 124, "Timeout."
    return r.returncode, _ANSI_RE.sub("", (r.stdout + r.stderr)).strip()


def _last_line(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _parse_devices(text):
    """[(mac, name)] from `bluetoothctl devices` output."""
    out = []
    for line in text.splitlines():
        m = _DEVICE_LINE_RE.match(_ANSI_RE.sub("", line).strip())
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def _is_unnamed(mac, name):
    # BlueZ falls back to the address with dashes when a device has no name.
    return not name or name == mac.replace(":", "-")


def _info(mac):
    _, text = _ctl(["info", mac])
    fields = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith("\t\t"):
            key, _, value = line.strip().partition(":")
            fields.setdefault(key.strip(), value.strip())
    return fields


def adapter_status():
    code, text = _ctl(["show"])
    if code != 0 or "Powered:" not in text:
        return {"ok": False, "error": _last_line(text) or "Niciun adaptor Bluetooth gasit."}
    return {"ok": True, "powered": "Powered: yes" in text}


def ensure_powered():
    status = adapter_status()
    if status["ok"] and not status["powered"]:
        _ctl(["power", "on"])
        status = adapter_status()
    return status


def list_paired():
    _, text = _ctl(["devices", "Paired"])
    devices = []
    for mac, name in _parse_devices(text):
        info = _info(mac)
        devices.append({
            "mac": mac,
            "name": name if not _is_unnamed(mac, name) else mac,
            "connected": info.get("Connected") == "yes",
            "trusted": info.get("Trusted") == "yes",
        })
    devices.sort(key=lambda d: (not d["connected"], d["name"].lower()))
    return devices


def scan(seconds=SCAN_SECONDS):
    """Scans for `seconds`, returns named devices that are NOT already paired,
    strongest signal first. (Nameless ones are other people's phones/beacons
    advertising random addresses - nothing you could pick anyway.)"""
    if not _busy.acquire(blocking=False):
        raise Busy()
    try:
        status = ensure_powered()
        if not status["ok"]:
            return None, status["error"]
        _ctl(["--timeout", str(seconds), "scan", "on"], timeout=seconds + 5)
        _, paired_text = _ctl(["devices", "Paired"])
        paired = {mac for mac, _ in _parse_devices(paired_text)}
        _, text = _ctl(["devices"])
        found = []
        for mac, name in _parse_devices(text):
            if mac in paired or _is_unnamed(mac, name):
                continue
            rssi = _info(mac).get("RSSI")
            try:
                rssi_val = int(rssi.split()[0], 0) if rssi else None
            except ValueError:
                rssi_val = None
            found.append({"mac": mac, "name": name, "rssi": rssi_val})
        found.sort(key=lambda d: -(d["rssi"] if d["rssi"] is not None else -999))
        return found, None
    finally:
        _busy.release()


def pair_and_connect(mac):
    """Pair + trust + connect - what tapping a scanned device does."""
    if not _busy.acquire(blocking=False):
        raise Busy()
    try:
        status = ensure_powered()
        if not status["ok"]:
            return False, status["error"]
        code, text = _ctl(["pair", mac], timeout=_PAIR_TIMEOUT_S)
        if code != 0 and "AlreadyExists" not in text and "Already Paired" not in text:
            return False, _last_line(text) or "Imperecherea a esuat."
        _ctl(["trust", mac])  # lets it reconnect by itself when switched on
        code, text = _ctl(["connect", mac], timeout=_CONNECT_TIMEOUT_S)
        if code != 0:
            return False, _last_line(text) or "Conectarea a esuat."
        return True, None
    finally:
        _busy.release()


def connect(mac):
    if not _busy.acquire(blocking=False):
        raise Busy()
    try:
        code, text = _ctl(["connect", mac], timeout=_CONNECT_TIMEOUT_S)
        return (True, None) if code == 0 else (False, _last_line(text) or "Conectarea a esuat.")
    finally:
        _busy.release()


def disconnect(mac):
    code, text = _ctl(["disconnect", mac])
    return (True, None) if code == 0 else (False, _last_line(text) or "Deconectarea a esuat.")


def forget(mac):
    code, text = _ctl(["remove", mac])
    return (True, None) if code == 0 else (False, _last_line(text) or "Stergerea a esuat.")
