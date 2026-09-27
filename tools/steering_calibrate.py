"""
RoboPacerV2 - Steering calibration (test tool)
================================================
Sets the three servo numbers in config/hardware_config.py - SERVO_MIN_ANGLE,
SERVO_STRAIGHT_ANGLE, SERVO_MAX_ANGLE - with the car in front of you and the
Xbox controller, then saves them back into that file. Meant to be run from
the dashboard's Tools page, which shows the live values and has the choice
of what to calibrate; only the physical controller drives the servo.

Pick what to calibrate on the web page - Mijloc (middle), Stanga (left) or
Dreapta (right) - then use the controller:

    Stick dreapta/stanga   steers normally, using the values as they stand
                            right now (in memory - nothing is written to
                            disk until Save), so you can immediately feel
                            the effect of an edit.
    D-pad stanga/dreapta   -1 / +1 degree on the value currently picked on
                            the page:
                              Mijloc  -> SERVO_STRAIGHT_ANGLE (center point).
                              Stanga  -> SERVO_MAX_ANGLE (servo angle at full
                                         left lock - stick fully left).
                              Dreapta -> SERVO_MIN_ANGLE (servo angle at full
                                         right lock - stick fully right).

Calibrating Stanga/Dreapta: leave the stick centered and the wheels sit at
the current middle value, untouched. Hold the stick at that lock (full left
for Stanga, full right for Dreapta) and tap the D-pad - the wheel moves
right away with each tap, so you can walk it out to the physical steering
limit and see the number update live.

Starts from whatever is in config/hardware_config.py right now (not a fixed
90 degrees). Nothing is written to that file until you press Save on the
page; Reset on the page reverts unsaved edits back to the last saved values
without restarting this script.

Only the steering servo moves - the ESC/motor is never armed or pulsed.
Still needs the relay ON (servo shares the 3S LiPo feed through it, see
HARDWARE.md), so this turns the relay on at start and off at exit exactly
like the driving scripts, and can't run at the same time as main /
data_recorder / manual_drive (one script at a time, dashboard-enforced).

Usage:
    python3 tools/steering_calibrate.py
    python3 tools/steering_calibrate.py --device /dev/input/event6
"""

import argparse
import json
import os
import re
import select
import signal
import socket
import sys
import threading
import time

from evdev import InputDevice, ecodes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from config.estop import relay_cmd  # noqa: E402
from config.hardware_config import (  # noqa: E402
    JOYSTICK_DEADZONE,
    SERVO_MAX_ANGLE,
    SERVO_MIN_ANGLE,
    SERVO_STRAIGHT_ANGLE,
)
from config.input_devices import find_xbox_controller  # noqa: E402
from config.ipc_config import STEERING_CALIBRATE_CONTROL_SOCKET  # noqa: E402
from config.joystick_steering import AXIS_CENTER, AXIS_MAX  # noqa: E402
from config.pca9685_init import init_pca9685  # noqa: E402
from config.servo_esc import SteeringServo  # noqa: E402

HARDWARE_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "hardware_config.py")

STEP_DEG = 1            # degrees per D-pad press
MARGIN_DEG = 2          # min gap kept between MIN/STRAIGHT/MAX while editing
ABS_ANGLE_MIN, ABS_ANGLE_MAX = 1, 179  # stay inside the servo's 0..180 range
RECONNECT_SECONDS = 0.5
TICK_SECONDS = 0.02

TARGETS = ("middle", "left", "right")

# Which servo lock each page target edits - see steering_axis_to_label()/
# steering_label_to_angle() in config/joystick_steering.py and
# config/servo_esc.py: stick fully left -> label -1 -> SERVO_MAX_ANGLE;
# stick fully right -> label +1 -> SERVO_MIN_ANGLE. Applied in _adjust().

_state_lock = threading.Lock()
_state = {
    "target": "middle",
    "values": {"min": SERVO_MIN_ANGLE, "max": SERVO_MAX_ANGLE, "straight": SERVO_STRAIGHT_ANGLE},
    "saved": {"min": SERVO_MIN_ANGLE, "max": SERVO_MAX_ANGLE, "straight": SERVO_STRAIGHT_ANGLE},
    "label": 0.0,
    "angle": SERVO_STRAIGHT_ANGLE,
    "controller_connected": True,
}


def _update_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


def _get_state():
    with _state_lock:
        return json.loads(json.dumps(_state))  # cheap deep copy - all JSON-safe values


def axis_to_label(x_value):
    """Mirrors config/joystick_steering.py's steering_axis_to_label() - kept
    separate because that one always reads the *saved* config constants,
    and here the values being calibrated are edited live, in memory."""
    raw = -1.0 + 2.0 * (x_value / AXIS_MAX)
    if abs(raw) <= JOYSTICK_DEADZONE:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    return sign * (abs(raw) - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)


def label_to_angle(label, min_angle, max_angle, straight_angle):
    """Mirrors config/servo_esc.py's steering_label_to_angle() - same reason."""
    label = max(-1.0, min(1.0, label))
    if label >= 0:
        return straight_angle - label * (straight_angle - min_angle)
    return straight_angle - label * (max_angle - straight_angle)


def _adjust(target, step):
    """+-STEP_DEG on the value `target` edits, clamped so MIN < STRAIGHT <
    MAX (with MARGIN_DEG of travel left on each side) never breaks - the
    same invariant config/hardware_config.py itself enforces on import."""
    with _state_lock:
        v = _state["values"]
        if target == "middle":
            v["straight"] = max(v["min"] + MARGIN_DEG, min(v["max"] - MARGIN_DEG, v["straight"] + step))
        elif target == "left":
            v["max"] = max(v["straight"] + MARGIN_DEG, min(ABS_ANGLE_MAX, v["max"] + step))
        elif target == "right":
            v["min"] = max(ABS_ANGLE_MIN, min(v["straight"] - MARGIN_DEG, v["min"] + step))


_ANGLE_LINE_RE = {
    "min": re.compile(r"^(SERVO_MIN_ANGLE\s*=\s*)-?\d+", re.MULTILINE),
    "max": re.compile(r"^(SERVO_MAX_ANGLE\s*=\s*)-?\d+", re.MULTILINE),
    "straight": re.compile(r"^(SERVO_STRAIGHT_ANGLE\s*=\s*)-?\d+", re.MULTILINE),
}


def _write_hardware_config(values):
    """Rewrites just the three SERVO_*_ANGLE lines in place - every comment,
    the other constants, and their order are left untouched. Atomic (write
    to a temp file, then os.replace) so a crash mid-write can't corrupt the
    file every other script on the car imports."""
    with open(HARDWARE_CONFIG_PATH, "r") as f:
        text = f.read()
    for key, pattern in _ANGLE_LINE_RE.items():
        text, n = pattern.subn(lambda m: m.group(1) + str(int(values[key])), text, count=1)
        if n != 1:
            raise RuntimeError(f"Linia pentru {key} nu a fost gasita in hardware_config.py")
    tmp_path = HARDWARE_CONFIG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, HARDWARE_CONFIG_PATH)


def _status_reply(ok=True, error=None):
    s = _get_state()
    reply = {
        "ok": ok, "target": s["target"], "values": s["values"], "saved": s["saved"],
        "dirty": s["values"] != s["saved"], "label": round(s["label"], 2),
        "angle": round(s["angle"], 1), "controller_connected": s["controller_connected"],
    }
    if error is not None:
        reply["error"] = error
    return reply


def _handle_command(data):
    if data == "STATUS":
        return _status_reply()
    if data.startswith("SET_TARGET "):
        target = data.split(" ", 1)[1]
        if target not in TARGETS:
            return _status_reply(ok=False, error="invalid_target")
        _update_state(target=target)
        return _status_reply()
    if data == "RESET":
        with _state_lock:
            _state["values"] = dict(_state["saved"])
        return _status_reply()
    if data == "SAVE":
        with _state_lock:
            values = dict(_state["values"])
        if not (ABS_ANGLE_MIN <= values["min"] < values["straight"] < values["max"] <= ABS_ANGLE_MAX):
            return _status_reply(ok=False, error="invalid_values")
        try:
            _write_hardware_config(values)
        except OSError as e:
            return _status_reply(ok=False, error=str(e))
        with _state_lock:
            _state["saved"] = dict(values)
        return _status_reply()
    return {"ok": False, "error": "unknown_command"}


def _control_server_loop(stop_event):
    """Local socket for the web dashboard - see main.py's _control_server_loop
    for the identical accept/recv/reply pattern."""
    try:
        os.unlink(STEERING_CALIBRATE_CONTROL_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(STEERING_CALIBRATE_CONTROL_SOCKET)
    srv.listen(5)
    srv.settimeout(1.0)
    try:
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    data = conn.recv(64).decode("utf-8", errors="replace").strip()
                except OSError:
                    continue
                try:
                    conn.sendall((json.dumps(_handle_command(data)) + "\n").encode())
                except OSError:
                    pass
    finally:
        srv.close()
        try:
            os.unlink(STEERING_CALIBRATE_CONTROL_SOCKET)
        except OSError:
            pass


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def _initial_axis_x(controller):
    absinfo = dict(controller.capabilities(absinfo=True).get(ecodes.EV_ABS, []))
    return absinfo[ecodes.ABS_X].value if ecodes.ABS_X in absinfo else AXIS_CENTER


def main():
    ap = argparse.ArgumentParser(description="Steering calibration (dashboard Tools page controls it).")
    ap.add_argument("--device", metavar="PATH", help="use this controller instead of searching for an Xbox pad")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    pca = None
    steering = None
    controller = None
    control_thread = None
    control_stop_event = None

    try:
        pca = init_pca9685()
        steering = SteeringServo(pca)

        if args.device:
            controller = InputDevice(args.device)
        else:
            try:
                controller = find_xbox_controller()
            except OSError:
                controller = None
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd
        joystick_x = _initial_axis_x(controller)

        relay_cmd("RELAY_ON")

        control_stop_event = threading.Event()
        control_thread = threading.Thread(target=_control_server_loop, args=(control_stop_event,), daemon=True)
        control_thread.start()

        print(f"Calibrare directie pornita. Valori curente: stanga(max)={SERVO_MAX_ANGLE} "
              f"mijloc={SERVO_STRAIGHT_ANGLE} dreapta(min)={SERVO_MIN_ANGLE}")
        print("Alege ce calibrezi pe pagina web. Stick = testeaza virajul, D-pad stanga/dreapta = -1/+1 grad. "
              "Salveaza / Reseteaza tot de pe pagina.")

        last_hat0x = 0
        controller_lost = False
        last_reconnect_try = 0.0

        while True:
            now = time.time()
            if controller_lost and now - last_reconnect_try >= RECONNECT_SECONDS:
                last_reconnect_try = now
                try:
                    new_controller = find_xbox_controller()
                except OSError:
                    new_controller = None
                if new_controller is not None:
                    try:
                        controller.close()
                    except OSError:
                        pass
                    controller = new_controller
                    controller_fd = controller.fd
                    joystick_x = _initial_axis_x(controller)
                    controller_lost = False
                    last_hat0x = 0
                    _update_state(controller_connected=True)
                    print("Controller reconectat.")

            ready, _, _ = select.select([] if controller_lost else [controller_fd], [], [], TICK_SECONDS)
            for _ in ready:
                try:
                    events = list(controller.read())
                except OSError as e:
                    events = []
                    controller_lost = True
                    last_reconnect_try = 0.0
                    _update_state(controller_connected=False)
                    print(f"Controller pierdut ({e}) - astept reconectarea...")
                for event in events:
                    if event.type != ecodes.EV_ABS:
                        continue
                    if event.code == ecodes.ABS_X:
                        joystick_x = event.value
                    elif event.code == ecodes.ABS_HAT0X:
                        step = 0
                        if event.value == -1 and last_hat0x == 0:
                            step = -STEP_DEG
                        elif event.value == 1 and last_hat0x == 0:
                            step = STEP_DEG
                        last_hat0x = event.value
                        if step:
                            _adjust(_get_state()["target"], step)

            # Re-applied every tick (not just on a new event) so a D-pad edit
            # while the stick is held at a lock moves the wheel immediately.
            label = axis_to_label(joystick_x)
            with _state_lock:
                v = _state["values"]
                angle = label_to_angle(label, v["min"], v["max"], v["straight"])
            steering.set_angle(angle)
            _update_state(label=label, angle=angle)
    except ConnectionError as e:
        print(f"Eroare: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        relay_cmd("RELAY_OFF")
        if steering is not None:
            steering.release()
        if pca is not None:
            try:
                pca.deinit()
            except OSError:
                pass
        if control_stop_event is not None:
            control_stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2)
        if controller is not None:
            try:
                controller.close()
            except OSError:
                pass
        print("Calibrare directie oprita.")


if __name__ == "__main__":
    main()
