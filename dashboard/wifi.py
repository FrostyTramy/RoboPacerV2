"""
RoboPacerV2 Dashboard - WiFi scan/connect
============================================
Lets the robot join a home WiFi network from a phone, with no display/
keyboard attached to the Pi. Backend is NetworkManager (`nmcli`) - the
default network stack on current Raspberry Pi OS (Bookworm+). If the real
Pi turns out to run the older dhcpcd/wpa_supplicant stack instead, this
module's calls will simply fail with "nmcli not found" (see
backend_available()) and this feature needs a different backend - flagged,
not silently assumed correct.

The robot has two WiFi radios: a USB adapter dedicated to the RoboPacer
hotspot (AP mode, serves the phone this dashboard is loaded from) and the
Pi's onboard chip, which is the one this module should scan/join with.
Which physical adapter is which isn't guaranteed stable across boots (USB
enumeration order), so find_station_interface() detects it by mode
(exclude whichever device is actively running an AP-mode connection)
rather than by a hardcoded interface name.

Security: every nmcli invocation uses subprocess.run([...]) with the
command as a list, never a shell string - SSID/password come from the
HTTP request body, and list-form subprocess calls go straight to execve()
so there's no shell to inject into.
"""

import re
import subprocess

_NMCLI_TIMEOUT_S = 10
_SCAN_TIMEOUT_S = 15
_CONNECT_TIMEOUT_S = 30


def backend_available():
    try:
        result = subprocess.run(
            ["nmcli", "--version"], capture_output=True, text=True, timeout=3)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _nmcli(args, timeout=_NMCLI_TIMEOUT_S):
    return subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout)


def _split_terse_line(line):
    """nmcli -t output escapes literal ':' inside a field as '\\:' - split
    only on unescaped colons, then unescape."""
    parts = re.split(r"(?<!\\):", line)
    return [p.replace("\\:", ":").replace("\\\\", "\\") for p in parts]


def list_wifi_devices():
    result = _nmcli(["-t", "-f", "DEVICE,TYPE,STATE", "device", "status"])
    devices = []
    for line in result.stdout.strip().splitlines():
        parts = _split_terse_line(line)
        if len(parts) < 3:
            continue
        device, dtype, state = parts[0], parts[1], parts[2]
        if dtype == "wifi":
            devices.append({"device": device, "state": state})
    return devices


def _active_connection_for(device):
    result = _nmcli(["-t", "-f", "DEVICE,CONNECTION", "device", "status"])
    for line in result.stdout.strip().splitlines():
        parts = _split_terse_line(line)
        if len(parts) >= 2 and parts[0] == device:
            conn = parts[1]
            return conn if conn and conn != "--" else None
    return None


def _connection_mode(conn_name):
    result = _nmcli(["-t", "-f", "802-11-wireless.mode", "connection", "show", conn_name])
    line = result.stdout.strip()
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return None


def find_station_interface():
    """Returns (interface_name, error). Picks the WiFi device that is NOT
    currently running in AP mode (that one is the robot's own hotspot)."""
    if not backend_available():
        return None, "nmcli nu e disponibil - verifica daca NetworkManager ruleaza pe acest Pi."
    wifi_devices = list_wifi_devices()
    if not wifi_devices:
        return None, "Niciun adaptor WiFi gasit (nmcli device status)."
    candidates = []
    for dev in wifi_devices:
        conn = _active_connection_for(dev["device"])
        mode = _connection_mode(conn) if conn else None
        if mode == "ap":
            continue
        candidates.append(dev["device"])
    if not candidates:
        return None, "Toate adaptoarele WiFi gasite ruleaza in modul AP (hotspot)."
    return candidates[0], None


def scan_networks(interface):
    result = _nmcli(
        ["-t", "-f", "SSID,SIGNAL,SECURITY,IN-USE", "device", "wifi", "list",
         "ifname", interface, "--rescan", "yes"],
        timeout=_SCAN_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None, (result.stderr.strip() or "nmcli a esuat la scanare.")
    best = {}
    for line in result.stdout.strip().splitlines():
        parts = _split_terse_line(line)
        if len(parts) < 4:
            continue
        ssid, signal, security, in_use = parts[0], parts[1], parts[2], parts[3]
        if not ssid:
            continue  # hidden network - nothing to connect to by name
        try:
            signal_val = int(signal)
        except ValueError:
            signal_val = 0
        existing = best.get(ssid)
        if existing is None or signal_val > existing["signal"]:
            best[ssid] = {
                "ssid": ssid,
                "signal": signal_val,
                "security": security or "open",
                "in_use": in_use.strip() == "*",
            }
    networks = sorted(best.values(), key=lambda n: -n["signal"])
    return networks, None


def connect_network(interface, ssid, password):
    args = ["device", "wifi", "connect", ssid, "ifname", interface]
    if password:
        args += ["password", password]
    try:
        result = subprocess.run(["nmcli", *args], capture_output=True, text=True,
                                 timeout=_CONNECT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "Conectare expirata (timeout)."
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, (result.stderr.strip() or result.stdout.strip() or "Conectare esuata.")


def get_wifi_status(interface):
    result = _nmcli(["-t", "-f", "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS", "device", "show", interface])
    state, connection, ip = None, None, None
    for line in result.stdout.strip().splitlines():
        parts = _split_terse_line(line)
        if len(parts) < 2:
            continue
        key, value = parts[0], ":".join(parts[1:])
        if key == "GENERAL.STATE":
            state = value
        elif key == "GENERAL.CONNECTION":
            connection = value if value and value != "--" else None
        elif key.startswith("IP4.ADDRESS"):
            ip = value or ip
    return {"interface": interface, "state": state, "connection": connection, "ip": ip}
