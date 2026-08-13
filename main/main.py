"""
RoboPacerV2 - Autopilot (model steering + cruise-control speed)
===================================================================
Combines two already-proven scripts into one autonomous run:
  - Steering: 100% from the Hailo model (identical pipeline to
    model_runner/model_runner.py) - no manual stick input is ever read.
  - Speed: held constant by the exact same feedforward+PI regulator and
    10-second pace-recovery loop as cruise_control/cruise_control.py.

You give it a target pace (--target-kmh) and a target distance
(--distance-m). It arms the ESC, opens the camera/model, and *waits* -
the car does not move until you press [A] on the controller (same
deliberate "go" moment cruise_control.py uses). Once engaged, it drives
itself for --distance-m meters at --target-kmh, then stops itself and
prints a summary. [B] stops immediately at any point, same as every
other script here.

Usage (normally launched from the dashboard's "Autopilot" page, which
fills these in for you from the speed/distance form):
    python3 main.py --target-kmh 10 --distance-m 500
    python3 main.py --target-kmh 10 --distance-m 500 --display
    python3 main.py --target-kmh 10 --distance-m 500 --raw-steering

Put exactly one *.hef file in this folder next to this script (same rule
as model_runner.py's find_hef_path() - ambiguity about which model is
driving is not something to guess at). By default this folder has a
symlink to model_runner/'s model file, so retraining only has one place
to drop a new .hef.

--------------------------------------------------------------------------
Logging philosophy - quiet terminal, verbose file
--------------------------------------------------------------------------
Stdout (and therefore the dashboard's live console) only ever gets
one-time messages: model/camera/ESC init, the [A]/[B] banner, the "PORNIT"
line when you engage, and the final summary. No live status line, no
per-second FPS spam, no per-split prints - everything that happens every
moment of the drive (speed AND steering) is written instead to a dense,
fixed-cadence CSV in logs/, at TICK_INTERVAL_SECONDS. The cadence is a
gated check (like cruise_control.py's own tick), not a sleep() - sleeping
inside the steering loop would add real reaction latency, which is not
acceptable here.

--------------------------------------------------------------------------
Deliberate behavior difference from cruise_control.py
--------------------------------------------------------------------------
If the speed sensor goes stale mid-run, cruise_control.py just disengages
and idles - reasonable for a tool an operator is actively working the
D-pad on. This script is meant to run unattended over a fixed distance,
so idling forever with no summary would be a silent hang. Instead, a
stale sensor here stops the whole program (ESC already goes neutral
either way) and reports why in the summary.

--------------------------------------------------------------------------
How the ESC is actually controlled / IMPORTANT - ESC safety watchdog
--------------------------------------------------------------------------
See model_runner/model_runner.py's docstring for the full RC-PWM/arming
explanation - identical hardware, identical reasoning, not repeated here.

The esc-watchdog systemd service forces the ESC off unless a script in its
ALLOWED_SCRIPTS list (safety/esc_watchdog.sh) is running, matched by full
path. This script's full path must be in that list, or the watchdog will
fight it and the motor will never actually respond. Always launch with:

    python3 /home/pi/RoboPacerV2/main/main.py
--------------------------------------------------------------------------
"""

import argparse
import json
import logging
import os
import select
import signal
import socket
import threading
import time
from collections import deque
from datetime import datetime

import board
import busio
import cv2
import numpy as np
from adafruit_motor import servo as adafruit_servo
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, ff, list_devices
from picamera2 import Picamera2
from hailo_platform import (
    VDevice, HEF, FormatType, InferVStreams,
    InputVStreamParams, OutputVStreamParams,
    ConfigureParams, HailoStreamInterface,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "main.log")
TICK_LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(TICK_LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("picamera2", "libcamera", "PIL"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Camera - identical configuration to model_runner.py/data_recorder.py, so
# the model sees exactly what it was trained on.
# ---------------------------------------------------------------------------
TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0

# ---------------------------------------------------------------------------
# Steering (servo on PCA9685 channel 0) - identical to model_runner.py
# ---------------------------------------------------------------------------
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900
SERVO_MAX_PULSE = 2200
SERVO_MIN_ANGLE = 45
SERVO_MAX_ANGLE = 135
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 4

# ---------------------------------------------------------------------------
# Throttle (ESC on PCA9685 channel 1) - identical range to both source
# scripts. Unlike model_runner.py (D-pad throttle stepping), throttle here
# comes entirely from the cruise-control regulator below.
# ---------------------------------------------------------------------------
ESC_CHANNEL = 1
PCA_FREQUENCY_HZ = 50
ESC_ARM_PULSE_US = 1500
ESC_ARM_HOLD_SECONDS = 3.0
ESC_NEUTRAL_US = 1500
ESC_MIN_US = 1000
ESC_MAX_US = 1700

# Cat de aproape de ESC_MAX_US conteaza "puls la maxim" si peste ce procent
# din esantioane adaugam nota in sumarul final - vezi _format_final_summary.
# Identic cu cruise_control.py.
ESC_PULSE_SATURATION_EPSILON_US = 2.0
ESC_PULSE_SATURATION_NOTE_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Model - identical to model_runner.py
# ---------------------------------------------------------------------------
MODEL_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SMOOTH_ALPHA = 0.5
STEERING_DEADZONE = 0.06

FRAME_STACK_N = 3
FRAME_STACK_GAP_SECONDS = 0.1

DISPLAY_EVERY_N_FRAMES = 4

# ---------------------------------------------------------------------------
# Cruise-control regulator - copied verbatim (same tuning) from
# cruise_control/cruise_control.py. See that file for the full reasoning
# behind every constant below - not repeated here to avoid the two drifting
# out of sync in prose while staying in sync in code.
# ---------------------------------------------------------------------------
CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH = 23.0

CRUISE_KP_US_PER_KMH = 10.0
CRUISE_KI_US_PER_KMH_S = 3.0
CRUISE_INTEGRAL_MAX_US = 50.0

CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S = 120.0
CRUISE_MAX_PULSE_STEP_US_PER_S = 180.0
CRUISE_LAUNCH_SECONDS = 2.0

CRUISE_SPEED_FILTER_TAU_S = 0.7

TARGET_SPEED_STEP_KMH = 1.0
TARGET_SPEED_MAX_KMH = 100.0  # plafonul real vine din puls (ESC_MAX_US) - vezi cruise_control.py

CRUISE_CATCHUP_WINDOW_S = 10.0
CRUISE_CATCHUP_MAX_EXTRA_KMH = 3.0

# ---------------------------------------------------------------------------
# Odometrie (RPM de la ESP32 via estop_listener) - identic cu
# cruise_control.py/manual_drive.py/tools/odometry.py.
# ---------------------------------------------------------------------------
ODO_SOCKET = "/tmp/esp32_odometry.sock"
WHEEL_CIRCUMFERENCE_M = 0.1369
ODO_RECONNECT_SLEEP_SECONDS = 2.0
ODO_STALE_TIMEOUT_SECONDS = 1.0
ODO_STALE_GRACE_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Tick logging + splits
# ---------------------------------------------------------------------------
TICK_INTERVAL_SECONDS = 0.02  # 50Hz - vezi "Logging philosophy" in docstring
SPLIT_DISTANCE_M = 100.0

# ---------------------------------------------------------------------------
# Controller - doar [A] (angajeaza) si [B] (opreste). Fara [Y]: asta e o
# cursa cu o singura "tura" (pana la distanta tinta), nu un instrument
# multi-tura ca cruise_control.py, deci nu are sens un reset de "leg".
# D-pad SUS/JOS tot ajusteaza target_kmh live, exact ca in cruise_control.py.
# ---------------------------------------------------------------------------
BTN_ENGAGE = ecodes.BTN_A
BTN_STOP = ecodes.BTN_B

_TRANSIENT_I2C_ERRNOS = (121, 19)


class SteeringServo:
    def __init__(self, pca):
        self._servo = adafruit_servo.Servo(
            pca.channels[SERVO_CHANNEL],
            min_pulse=SERVO_MIN_PULSE,
            max_pulse=SERVO_MAX_PULSE,
        )
        self.angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET
        self.center()

    def set_angle(self, angle):
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        self.angle = angle
        try:
            self._servo.angle = angle
        except OSError as e:
            if e.errno in _TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting servo angle {angle}: {e}")
            else:
                raise

    def center(self):
        self.set_angle(SERVO_NEUTRAL_ANGLE + SERVO_OFFSET)

    def release(self):
        try:
            self._servo.angle = None
        except OSError as e:
            if e.errno not in _TRANSIENT_I2C_ERRNOS:
                raise


class ESC:
    def __init__(self, pca):
        self._channel = pca.channels[ESC_CHANNEL]
        self._period_us = 1_000_000 / pca.frequency

    def _pulse_to_duty_cycle(self, pulse_us):
        pulse_us = max(ESC_MIN_US, min(ESC_MAX_US, pulse_us))
        return int((pulse_us / self._period_us) * 0xFFFF)

    def set_pulse_us(self, pulse_us):
        try:
            self._channel.duty_cycle = self._pulse_to_duty_cycle(pulse_us)
        except OSError as e:
            if e.errno in _TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting ESC pulse {pulse_us}: {e}")
            else:
                raise

    def neutral(self):
        self.set_pulse_us(ESC_NEUTRAL_US)

    def stop(self):
        try:
            self._channel.duty_cycle = 0
        except OSError as e:
            if e.errno not in _TRANSIENT_I2C_ERRNOS:
                raise

    def arm(self):
        print(f"Armare ESC: puls {ESC_ARM_PULSE_US}us timp de {ESC_ARM_HOLD_SECONDS:.0f}s...")
        self.set_pulse_us(ESC_ARM_PULSE_US)
        time.sleep(ESC_ARM_HOLD_SECONDS)
        self.neutral()
        print(f"ESC armat, setat la neutru ({ESC_NEUTRAL_US}us).")


def find_hef_path():
    hefs = [f for f in os.listdir(BASE_DIR) if f.endswith(".hef")]
    if len(hefs) == 0:
        raise RuntimeError(f"Niciun fisier .hef gasit in {BASE_DIR}. Pune exact un model acolo.")
    if len(hefs) > 1:
        raise RuntimeError(f"Mai multe fisiere .hef gasite in {BASE_DIR}: {hefs}. Trebuie sa fie exact unul.")
    return os.path.join(BASE_DIR, hefs[0])


def find_xbox_controller():
    for path in list_devices():
        dev = InputDevice(path)
        if "xbox" in dev.name.lower():
            return dev
    return None


def make_camera():
    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(tuning=tuning)
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT},
        controls={"FrameRate": FRAME_RATE, "AnalogueGain": ANALOGUE_GAIN},
    )
    picam2.configure(config)
    return picam2


def preprocess(frame_bgr):
    """Must stay pixel-for-pixel identical to model_runner.py's preprocess()
    (and therefore to the trainer's load_and_preprocess()) - see that
    file's docstring for the full explanation."""
    frame_rgb = frame_bgr[:, :, ::-1]
    img = cv2.resize(frame_rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def build_frame_stack(history, now, frame_stack_n=FRAME_STACK_N):
    """Identical to model_runner.py's build_frame_stack() - see there."""
    frames = [history[-1][1]]
    for k in range(1, frame_stack_n):
        target_ts = now - k * FRAME_STACK_GAP_SECONDS
        best = history[0][1]
        for ts, img in reversed(history):
            if ts <= target_ts:
                best = img
                break
        frames.append(best)
    return np.concatenate(frames, axis=-1)


def quantize_input(img_float_nhwc, scale, zero_point):
    return np.clip(np.round(img_float_nhwc / scale + zero_point), 0, 255).astype(np.uint8)


def steering_label_to_angle(label):
    label = max(-1.0, min(1.0, label))
    angle = SERVO_NEUTRAL_ANGLE - label * (SERVO_MAX_ANGLE - SERVO_NEUTRAL_ANGLE) + SERVO_OFFSET
    return int(max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle)))


def cruise_feedforward_offset_us(target_kmh):
    """Identical to cruise_control.py - see there for the reasoning."""
    fraction = target_kmh / CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH
    return fraction * (ESC_MAX_US - ESC_NEUTRAL_US)


def cruise_pulse_us(target_kmh, current_kmh, integral, dt, prev_pulse_us, max_step_us_per_s):
    """Identical to cruise_control.py's regulator - feedforward + PI + slope
    limit. Pulse never drops below neutral - see cruise_control.py's module
    docstring for why (ESC brake/reverse ambiguity)."""
    error = target_kmh - current_kmh
    integral += error * dt
    integral = max(-CRUISE_INTEGRAL_MAX_US / CRUISE_KI_US_PER_KMH_S,
                   min(CRUISE_INTEGRAL_MAX_US / CRUISE_KI_US_PER_KMH_S, integral))
    ff_offset = cruise_feedforward_offset_us(target_kmh)
    trim = CRUISE_KP_US_PER_KMH * error + CRUISE_KI_US_PER_KMH_S * integral
    desired_pulse_us = max(ESC_NEUTRAL_US, min(ESC_MAX_US, ESC_NEUTRAL_US + ff_offset + trim))

    max_step = max_step_us_per_s * dt
    step = max(-max_step, min(max_step, desired_pulse_us - prev_pulse_us))
    pulse_us = prev_pulse_us + step

    return pulse_us, integral


def _rpm_to_kmh(rpm):
    return rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000


def _format_pace(kmh):
    if kmh <= 0.05:
        return "--:--"
    pace_sec = 3600 / kmh
    return f"{int(pace_sec // 60)}:{int(pace_sec % 60):02d}"


def _format_time_ms(total_sec):
    total_sec = max(0.0, total_sec)
    minutes = int(total_sec // 60)
    secs = total_sec % 60
    return f"{minutes}:{secs:06.3f}"


def _format_pace_ms(pace_sec_per_km):
    if pace_sec_per_km <= 0:
        return "--:--.---"
    return _format_time_ms(pace_sec_per_km)


_odo_state = {"rpm": 0.0, "last_update": 0.0}
_odo_lock = threading.Lock()


def _set_rpm(value, fresh=True):
    with _odo_lock:
        _odo_state["rpm"] = value
        if fresh:
            _odo_state["last_update"] = time.time()


def _get_rpm():
    with _odo_lock:
        return _odo_state["rpm"]


def _odo_is_stale():
    with _odo_lock:
        return (time.time() - _odo_state["last_update"]) > ODO_STALE_TIMEOUT_SECONDS


def _odo_reader_loop(stop_event):
    """Identic cu cruise_control.py - vezi acolo."""
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
                        _set_rpm(float(line[4:]), fresh=True)
                    except ValueError:
                        continue
        finally:
            sock.close()
            _set_rpm(0.0, fresh=False)
            if not stop_event.is_set():
                stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)


_CONTROL_SOCKET = "/tmp/main_autopilot_control.sock"

_live_state = {
    "target_kmh": 0.0, "kmh": 0.0, "engaged": False,
    "distance_m": 0.0, "distance_target_m": 0.0,
    "last_split_m": 0.0, "last_split_pace_sec": 0.0,
    "model_name": "",
}
_live_lock = threading.Lock()


def _update_live(**kwargs):
    with _live_lock:
        _live_state.update(kwargs)


def _get_live_state():
    with _live_lock:
        return dict(_live_state)


def _control_server_loop(stop_event):
    """Socket local pentru dashboard-ul web - STATUS intoarce tinta/viteza/
    distanta/model curente. Acelasi tipar ca cruise_control.py."""
    try:
        os.unlink(_CONTROL_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(_CONTROL_SOCKET)
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
                if data == "STATUS":
                    live = _get_live_state()
                    payload = {
                        "target_kmh": round(live["target_kmh"], 1),
                        "kmh": round(live["kmh"], 2),
                        "pace": _format_pace(live["kmh"]),
                        "engaged": live["engaged"],
                        "distance_m": round(live["distance_m"], 1),
                        "distance_target_m": round(live["distance_target_m"], 1),
                        "last_split_m": live["last_split_m"],
                        "last_split_pace": (
                            _format_pace_ms(live["last_split_pace_sec"])
                            if live["last_split_pace_sec"] else None
                        ),
                        "model_name": live["model_name"],
                    }
                    try:
                        conn.sendall((json.dumps(payload) + "\n").encode())
                    except OSError:
                        pass
    finally:
        srv.close()
        try:
            os.unlink(_CONTROL_SOCKET)
        except OSError:
            pass


def _format_splits(splits):
    """Identic cu cruise_control.py - vezi acolo."""
    if not splits:
        return ""
    lines = ["\n--- SUTE (100m) ---"]
    for mark_m, elapsed_s, split_duration, pace_sec in splits:
        lines.append(
            f"{mark_m:5.0f}m | total {_format_time_ms(elapsed_s)} | "
            f"suta {split_duration:6.3f}s | pace {_format_pace_ms(pace_sec)}/km"
        )
    return "\n".join(lines) + "\n"


def _format_final_summary(target_kmh, distance_target_m, elapsed_s, tick_samples,
                           distance_m, model_name, frame_stack_n, raw_steering, stop_reason):
    """Sumarul bogat cerut: viteza (ambele metrici), distanta, si un sumar
    al comportamentului modelului (steering) - nu doar al vitezei ca in
    cruise_control.py."""
    lines = ["\n--- SUMAR FINAL ---"]
    model_desc = "necunoscut" if not frame_stack_n else (
        "clasic, un cadru" if frame_stack_n == 1 else f"{frame_stack_n} cadre stacked"
    )
    lines.append(f"Model: {model_name or 'necunoscut'} ({model_desc}) | "
                 f"steering {'BRUT (fara EMA/deadzone)' if raw_steering else 'EMA+deadzone'}")
    lines.append(f"Motiv oprire: {stop_reason}")
    lines.append(
        f"Tinta: {target_kmh:.1f} km/h ({_format_pace(target_kmh)}/km) | "
        f"{distance_target_m:.0f}m tinta | Durata: {elapsed_s:.1f}s"
    )

    if not tick_samples:
        lines.append("Niciun esantion inregistrat (oprit inainte de angajare/[A]).")
        return "\n".join(lines) + "\n"

    kmh_vals = [s[1] for s in tick_samples]
    pulse_vals = [s[2] for s in tick_samples]
    steer_vals = [s[3] for s in tick_samples]
    fps_vals = [s[4] for s in tick_samples if s[4] > 0]

    moving_kmh_vals = [k for k in kmh_vals if k > 0]
    avg_kmh = sum(moving_kmh_vals) / len(moving_kmh_vals) if moving_kmh_vals else 0.0
    max_kmh = max(kmh_vals) if kmh_vals else 0.0
    deviation = avg_kmh - target_kmh

    saturated = sum(1 for p in pulse_vals if p >= ESC_MAX_US - ESC_PULSE_SATURATION_EPSILON_US)
    saturation_fraction = saturated / len(pulse_vals) if pulse_vals else 0.0

    abs_steer_vals = [abs(s) for s in steer_vals]
    avg_abs_steer = sum(abs_steer_vals) / len(abs_steer_vals) if abs_steer_vals else 0.0
    max_abs_steer = max(abs_steer_vals) if abs_steer_vals else 0.0
    straight_frames = sum(1 for s in steer_vals if s == 0.0)
    straight_fraction = straight_frames / len(steer_vals) if steer_vals else 0.0
    left_frames = sum(1 for s in steer_vals if s > 0)
    right_frames = sum(1 for s in steer_vals if s < 0)

    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0
    min_fps = min(fps_vals) if fps_vals else 0.0

    lines.append(
        f"Distanta -> {distance_m:.1f}m din {distance_target_m:.0f}m tinta "
        f"(diferenta {distance_m - distance_target_m:+.1f}m)"
    )
    lines.append(f"km/h  -> mediu {avg_kmh:.2f} (cat timp s-a miscat) | maxim {max_kmh:.2f}")
    lines.append(f"Pace  -> mediu {_format_pace(avg_kmh)}/km | cel mai bun {_format_pace(max_kmh)}/km")
    lines.append(f"Abatere fata de tinta: {deviation:+.2f} km/h")
    if saturation_fraction >= ESC_PULSE_SATURATION_NOTE_FRACTION:
        lines.append(
            f"NOTA: puls la maxim ({ESC_MAX_US:.0f}us) in {saturation_fraction:.0%} din cursa - "
            f"tinta a fost probabil peste plafonul fizic real al motorului/bateriei in acel moment, "
            f"nu o eroare a regulatorului."
        )
    lines.append(
        f"Model -> steer mediu |{avg_abs_steer:.3f}| | maxim |{max_abs_steer:.3f}| | "
        f"drept (deadzone) {straight_fraction:.0%} din timp | "
        f"stanga {left_frames} cadre / dreapta {right_frames} cadre"
    )
    lines.append(
        f"FPS inferenta -> mediu {avg_fps:.1f} | minim {min_fps:.1f} | "
        f"{len(tick_samples)} esantioane (la {TICK_INTERVAL_SECONDS * 1000:.0f}ms)"
    )
    return "\n".join(lines) + "\n"


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


_RELAY_SOCKET = "/tmp/esp32_relay.sock"


def _relay_cmd(cmd: str) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(_RELAY_SOCKET)
            s.sendall((cmd + "\n").encode())
    except (OSError, socket.timeout):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", action="store_true",
                     help="Open a live cv2 preview window (costs a few ms/frame)")
    ap.add_argument("--raw-steering", action="store_true",
                     help="Bypass EMA smoothing and the deadzone - send the model's "
                          "raw per-frame prediction straight to the servo")
    ap.add_argument("--target-kmh", type=float, required=True,
                     help=f"Viteza tinta in km/h (0.1-{TARGET_SPEED_MAX_KMH:.0f})")
    ap.add_argument("--distance-m", type=float, required=True,
                     help="Distanta tinta in metri (1-50000) - programul se opreste singur la final")
    args = ap.parse_args()

    if not (0.1 <= args.target_kmh <= TARGET_SPEED_MAX_KMH):
        raise SystemExit(
            f"--target-kmh trebuie sa fie intre 0.1 si {TARGET_SPEED_MAX_KMH:.0f} "
            f"(primit: {args.target_kmh})"
        )
    if not (1.0 <= args.distance_m <= 50000.0):
        raise SystemExit(f"--distance-m trebuie sa fie intre 1 si 50000 (primit: {args.distance_m})")

    show_display = args.display
    raw_steering = args.raw_steering
    target_kmh = args.target_kmh
    distance_target_m = args.distance_m

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Definite inaintea oricarui cod care poate arunca exceptie - `finally`
    # le foloseste chiar daca pornirea a picat devreme (acelasi motiv ca in
    # cruise_control.py/manual_drive.py).
    pca = None
    esc = None
    steering = None
    picam2 = None
    controller = None
    last_rumble_effect_id = None
    odo_thread = None
    odo_stop_event = None
    control_thread = None
    control_stop_event = None
    tick_csv_file = None
    model_name = None
    frame_stack_n = 0
    leg_start_time = None
    distance_m = 0.0
    tick_samples = []  # (elapsed_s, kmh, pulse_us, steer_cmd, fps) la fiecare TICK_INTERVAL_SECONDS
    splits = []  # (marcaj_m, elapsed_s, durata_suta_s, pace_sec_per_km)
    stop_reason = "oprit manual (Ctrl+C / semnal)"

    def rumble(duration_ms):
        nonlocal last_rumble_effect_id
        if controller is None:
            return
        if last_rumble_effect_id is not None:
            try:
                controller.erase_effect(last_rumble_effect_id)
            except OSError:
                pass
            last_rumble_effect_id = None
        try:
            effect = ff.Effect(
                ecodes.FF_RUMBLE, -1, 0,
                ff.Trigger(0, 0),
                ff.Replay(duration_ms, 0),
                ff.EffectType(ff_rumble_effect=ff.Rumble(strong_magnitude=0xFFFF, weak_magnitude=0xFFFF)),
            )
            last_rumble_effect_id = controller.upload_effect(effect)
            controller.write(ecodes.EV_FF, last_rumble_effect_id, 1)
        except OSError as e:
            logging.warning(f"Rumble esuat: {e}")

    try:
        hef_path = find_hef_path()
        model_name = os.path.basename(hef_path)
        print(f"[Hailo] Model: {model_name}")
        hef = HEF(hef_path)
        input_info = hef.get_input_vstream_infos()[0]
        input_name = input_info.name
        output_name = hef.get_output_vstream_infos()[0].name
        input_scale = input_info.quant_info.qp_scale
        input_zero_point = input_info.quant_info.qp_zp

        input_channels = input_info.shape[-1]
        if input_channels % 3 != 0:
            raise RuntimeError(f"Unexpected HEF input shape {input_info.shape} - "
                                f"channel count must be a multiple of 3 (RGB frames).")
        frame_stack_n = input_channels // 3
        print(f"[Hailo] Input: {input_info.shape}  ->  "
              f"{'classic, single-frame' if frame_stack_n == 1 else f'{frame_stack_n}-frame stacked'} model")

        # --- I2C / PCA9685 --------------------------------------------------
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = PCA_FREQUENCY_HZ

        esc = ESC(pca)
        esc.neutral()
        steering = SteeringServo(pca)

        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        _relay_cmd("RELAY_ON")
        esc.arm()

        # --- Camera -----------------------------------------------------------
        picam2 = make_camera()
        picam2.start()
        logging.info("Camera started - 640x480 @ 120fps")

        # --- Log detaliat (CSV, ritm fix) --------------------------------
        tick_csv_path = os.path.join(TICK_LOG_DIR, f"main_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        tick_csv_file = open(tick_csv_path, "w")
        tick_csv_file.write(
            "timestamp,elapsed_s,engaged,target_kmh,effective_target_kmh,rpm,kmh,pace_mmss,"
            "pulse_us,steer_raw,steer_smoothed,steer_cmd,servo_angle,distance_m,fps\n"
        )
        tick_csv_file.flush()
        print(f"Log detaliat: {tick_csv_path}")

        odo_stop_event = threading.Event()
        odo_thread = threading.Thread(target=_odo_reader_loop, args=(odo_stop_event,), daemon=True)
        odo_thread.start()

        _update_live(model_name=model_name, target_kmh=target_kmh, distance_target_m=distance_target_m)
        control_stop_event = threading.Event()
        control_thread = threading.Thread(
            target=_control_server_loop, args=(control_stop_event,), daemon=True)
        control_thread.start()

        print("\n-----------------------------------------------------")
        print(f"Autopilot gata. Tinta: {target_kmh:.1f} km/h ({_format_pace(target_kmh)}/km), "
              f"{distance_target_m:.0f}m.")
        print("Virajul vine 100% din model. Viteza e tinuta constant (cruise control).")
        print("[A] PORNESTE cursa. [B] OPRESTE oricand (la fel ca oricare alt script).")
        print("D-pad SUS/JOS ajusteaza tinta live, ca la cruise_control.")
        print("Se opreste singur automat cand atinge distanta tinta.")
        print("-----------------------------------------------------")

        engaged = False
        engage_time = 0.0
        integral = 0.0
        prev_pulse_us = ESC_NEUTRAL_US
        filtered_kmh = 0.0
        smooth_label = 0.0
        last_hat0y = 0
        last_control_time = time.time()
        last_tick_time = time.time()

        next_split_mark_m = SPLIT_DISTANCE_M
        last_split_elapsed_s = 0.0
        scheduled_distance_m = 0.0
        actual_distance_m = 0.0

        current_fps = 0.0
        t_prev = time.time()
        frame_counter = 0
        frame_history = deque()

        with VDevice() as device:
            cfg_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(hef, cfg_params)[0]
            ng_params = network_group.create_params()
            in_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

            with network_group.activate(ng_params):
                with InferVStreams(network_group, in_vstream_params, out_vstream_params) as pipeline:
                    while True:
                        ready, _, _ = select.select([controller_fd], [], [], 0.001)

                        for _ in ready:
                            for event in controller.read():
                                if event.type == ecodes.EV_ABS and event.code == ecodes.ABS_HAT0Y:
                                    if event.value == -1 and last_hat0y == 0:
                                        target_kmh = min(TARGET_SPEED_MAX_KMH, target_kmh + TARGET_SPEED_STEP_KMH)
                                    elif event.value == 1 and last_hat0y == 0:
                                        target_kmh = max(0.0, target_kmh - TARGET_SPEED_STEP_KMH)
                                    last_hat0y = event.value
                                elif event.type == ecodes.EV_KEY and event.value == 1:
                                    if event.code == BTN_ENGAGE:
                                        if not engaged:
                                            engaged = True
                                            engage_time = time.time()
                                            leg_start_time = engage_time
                                            integral = 0.0
                                            prev_pulse_us = ESC_NEUTRAL_US
                                            scheduled_distance_m = 0.0
                                            actual_distance_m = 0.0
                                            distance_m = 0.0
                                            next_split_mark_m = SPLIT_DISTANCE_M
                                            last_split_elapsed_s = 0.0
                                            last_tick_time = engage_time
                                            splits = []
                                            tick_samples = []
                                            print(f"\n>>> PORNIT (A) - tinta {target_kmh:.1f} km/h "
                                                  f"({_format_pace(target_kmh)}/km), {distance_target_m:.0f}m <<<")
                                            logging.info(
                                                f"Angajat: tinta {target_kmh:.1f} km/h, "
                                                f"distanta {distance_target_m:.0f}m")
                                            rumble(300)
                                    elif event.code == BTN_STOP:
                                        stop_reason = "buton B"
                                        raise KeyboardInterrupt

                        try:
                            frame = picam2.capture_array()
                        except RuntimeError:
                            logging.warning("Camera capture failed, skipping frame")
                            time.sleep(0.05)
                            continue

                        img_float = preprocess(frame)
                        now = time.time()
                        if frame_stack_n > 1:
                            frame_history.append((now, img_float))
                            cutoff = now - (frame_stack_n - 1) * FRAME_STACK_GAP_SECONDS - 0.5
                            while len(frame_history) > 1 and frame_history[0][0] < cutoff:
                                frame_history.popleft()
                            stack = build_frame_stack(frame_history, now, frame_stack_n)
                        else:
                            stack = img_float
                        inp = quantize_input(stack[np.newaxis], input_scale, input_zero_point)
                        result = pipeline.infer({input_name: inp})
                        raw_label = float(np.array(result[output_name]).reshape(-1)[0])
                        smooth_label = SMOOTH_ALPHA * raw_label + (1 - SMOOTH_ALPHA) * smooth_label
                        if raw_steering:
                            steer_cmd = raw_label
                        else:
                            steer_cmd = 0.0 if abs(smooth_label) < STEERING_DEADZONE else smooth_label

                        # Steering se aplica mereu, indiferent de `engaged` -
                        # asa operatorul poate vedea vizual ca modelul
                        # urmareste drumul corect INAINTE sa apese [A] si sa
                        # dea gaz (rotile nu sunt actionate pana la angajare).
                        servo_angle = steering_label_to_angle(steer_cmd)
                        steering.set_angle(servo_angle)

                        t_now = time.time()
                        current_fps = 0.9 * current_fps + 0.1 / max(t_now - t_prev, 1e-9)
                        t_prev = t_now

                        # --- viteza (regulator cruise-control) -------------
                        rpm = _get_rpm()
                        raw_kmh = _rpm_to_kmh(rpm)
                        dt = now - last_control_time
                        filter_weight = min(1.0, dt / CRUISE_SPEED_FILTER_TAU_S) if dt > 0 else 0.0
                        filtered_kmh += (raw_kmh - filtered_kmh) * filter_weight
                        kmh = filtered_kmh
                        effective_target_kmh = target_kmh

                        if engaged:
                            past_grace = (now - engage_time) > ODO_STALE_GRACE_SECONDS
                            if past_grace and _odo_is_stale():
                                engaged = False
                                esc.neutral()
                                prev_pulse_us = ESC_NEUTRAL_US
                                stop_reason = "senzor de viteza indisponibil"
                                logging.warning("Oprire automata: date RPM invechite")
                                raise KeyboardInterrupt
                            else:
                                in_launch = (now - engage_time) <= CRUISE_LAUNCH_SECONDS
                                max_step = (CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S if in_launch
                                            else CRUISE_MAX_PULSE_STEP_US_PER_S)

                                prev_split_distance_m = distance_m
                                distance_m += (kmh / 3.6) * dt
                                while distance_m >= next_split_mark_m:
                                    if distance_m > prev_split_distance_m:
                                        frac = ((next_split_mark_m - prev_split_distance_m) /
                                                (distance_m - prev_split_distance_m))
                                    else:
                                        frac = 1.0
                                    crossing_time = last_control_time + frac * dt
                                    crossing_elapsed_s = crossing_time - leg_start_time
                                    split_duration = crossing_elapsed_s - last_split_elapsed_s
                                    pace_sec = split_duration * (1000.0 / SPLIT_DISTANCE_M)
                                    splits.append((next_split_mark_m, crossing_elapsed_s, split_duration, pace_sec))
                                    _update_live(last_split_m=next_split_mark_m, last_split_pace_sec=pace_sec)
                                    logging.info(
                                        f"Suta {next_split_mark_m:.0f}m: {_format_time_ms(crossing_elapsed_s)} "
                                        f"total, {split_duration:.3f}s, pace {_format_pace_ms(pace_sec)}/km")
                                    last_split_elapsed_s = crossing_elapsed_s
                                    next_split_mark_m += SPLIT_DISTANCE_M

                                if in_launch:
                                    effective_target_kmh = target_kmh
                                else:
                                    scheduled_distance_m += (target_kmh / 3.6) * dt
                                    actual_distance_m += (kmh / 3.6) * dt
                                    deficit_m = scheduled_distance_m - actual_distance_m
                                    extra_kmh = (deficit_m / CRUISE_CATCHUP_WINDOW_S) * 3.6
                                    extra_kmh = max(-CRUISE_CATCHUP_MAX_EXTRA_KMH,
                                                     min(CRUISE_CATCHUP_MAX_EXTRA_KMH, extra_kmh))
                                    effective_target_kmh = max(0.0, min(TARGET_SPEED_MAX_KMH,
                                                                         target_kmh + extra_kmh))

                                pulse_us, integral = cruise_pulse_us(
                                    effective_target_kmh, kmh, integral, dt, prev_pulse_us, max_step)
                                esc.set_pulse_us(pulse_us)
                                prev_pulse_us = pulse_us

                                # Logul CSV detaliat scrie doar cat timp cursa
                                # chiar ruleaza (engaged si nu stale) - altfel,
                                # in asteptarea lui [A], elapsed_s ar ramane
                                # blocat la 0 pe zeci/sute de randuri fara sens.
                                if now - last_tick_time >= TICK_INTERVAL_SECONDS:
                                    last_tick_time = now
                                    elapsed_s = now - leg_start_time
                                    pace = _format_pace(kmh)
                                    tick_samples.append((elapsed_s, kmh, prev_pulse_us, steer_cmd, current_fps))
                                    tick_csv_file.write(
                                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{elapsed_s:.3f},"
                                        f"{engaged},{target_kmh:.1f},{effective_target_kmh:.2f},{rpm:.2f},{kmh:.2f},"
                                        f"{pace},{prev_pulse_us:.1f},{raw_label:+.4f},{smooth_label:+.4f},"
                                        f"{steer_cmd:+.4f},{servo_angle},{distance_m:.2f},{current_fps:.1f}\n"
                                    )
                                    tick_csv_file.flush()

                                if distance_m >= distance_target_m:
                                    stop_reason = f"distanta tinta atinsa ({distance_target_m:.0f}m)"
                                    raise KeyboardInterrupt
                        last_control_time = now

                        _update_live(target_kmh=target_kmh, kmh=kmh, engaged=engaged,
                                     distance_m=distance_m, distance_target_m=distance_target_m)

                        frame_counter += 1

                        if show_display:
                            if frame_counter % DISPLAY_EVERY_N_FRAMES == 0:
                                display = frame
                                cv2.putText(
                                    display,
                                    f"Steer: {steer_cmd:+.2f} (model {raw_label:+.2f})  FPS: {current_fps:.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                                cv2.putText(
                                    display,
                                    f"{'CRUISE' if engaged else 'astept [A]'} | {kmh:.2f} km/h "
                                    f"(tinta {effective_target_kmh:.1f}) | {_format_pace(kmh)}/km | "
                                    f"{distance_m:.0f}/{distance_target_m:.0f}m",
                                    (10, display.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                                cv2.imshow("RoboPacerV2 - Autopilot", display)

                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    stop_reason = "tasta 'q' in fereastra preview"
                                    raise KeyboardInterrupt

    except (KeyboardInterrupt, ConnectionError) as e:
        if isinstance(e, ConnectionError):
            print(f"\n{e}")
            stop_reason = "eroare de conexiune"
        print(f"\nOprire ({stop_reason})...")
    except Exception as e:
        logging.exception("Eroare majora neasteptata")
        print(f"\nEroare majora neasteptata: {e}")
        stop_reason = f"eroare: {e}"
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        _relay_cmd("RELAY_OFF")
        if esc is not None:
            esc.neutral()
            time.sleep(0.1)
            esc.stop()
            rumble(150)
            time.sleep(0.25)
            rumble(150)
            time.sleep(0.2)
        if steering is not None:
            steering.release()
        if pca is not None:
            try:
                pca.deinit()
            except OSError as e:
                logging.warning(f"I2C error during pca.deinit(): {e}")
        if picam2 is not None and getattr(picam2, "started", False):
            picam2.stop()
        cv2.destroyAllWindows()

        if odo_stop_event is not None:
            odo_stop_event.set()
        if odo_thread is not None:
            odo_thread.join(timeout=2)

        if control_stop_event is not None:
            control_stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2)

        elapsed_s = (time.time() - leg_start_time) if leg_start_time is not None else 0.0
        summary = _format_final_summary(
            target_kmh=target_kmh, distance_target_m=distance_target_m, elapsed_s=elapsed_s,
            tick_samples=tick_samples, distance_m=distance_m, model_name=model_name,
            frame_stack_n=frame_stack_n, raw_steering=raw_steering, stop_reason=stop_reason,
        ) + _format_splits(splits)

        if tick_csv_file is not None:
            tick_csv_file.write(summary)
            tick_csv_file.close()

        print(summary)
        logging.info("Autopilot stopped\n" + summary)
        print("Hardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
