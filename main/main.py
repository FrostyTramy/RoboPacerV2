"""
RoboPacerV2 - Autopilot (model steering + selectable speed mode)
===================================================================
Combines three previously separate scripts (on `main`) into one:
  - model_runner.py's model-only steering pipeline.
  - cruise_control.py's D-pad-adjustable speed + PI regulator.
  - main.py's fixed-target speed, distance-target stop, and run summary.

Steering is 100% from the Hailo model, every frame, regardless of speed
mode or engage state - identical pipeline to model_runner.py. Speed is
selected with --speed-mode:

    cruise      Fixed target speed for the whole run (--target-kmh
                required). D-pad is a no-op - the target never changes
                mid-run. Uses the same feedforward+PI regulator and
                10-second pace-recovery loop as cruise_control.py, so the
                *average* speed converges on the target even after
                transients (curves, launch ramp).
    controller  Target speed starts at 0.0 km/h. D-pad up/down adjusts it
                live in 1 km/h steps (same regulator underneath as
                cruise). --target-kmh is not accepted in this mode.
    none        ESC stays at neutral for the entire run - only the model
                steers. No D-pad throttle stepping either. Useful for
                bench-testing the model or pushing the car by hand.
                --target-kmh is not accepted in this mode.

--distance-m is optional in every mode: if given, the run stops itself at
that distance; if omitted, it runs until you press [B] or Ctrl+C. [A]
engages (starts the run/log/distance-accumulation) - the car does not move
until then. [B] stops immediately at any point.

Steering defaults to RAW (the model's per-frame prediction goes straight
to the servo, no EMA smoothing, no deadzone) - pass --smooth-steering for
EMA+deadzone. Display defaults to off (headless) - pass --display for a
live cv2 preview window.

Put exactly one *.hef file in this folder next to this script (same rule
as model_runner.py's find_hef_path()).

Usage:
    python3 main.py --speed-mode cruise --target-kmh 10 --distance-m 500
    python3 main.py --speed-mode controller
    python3 main.py --speed-mode none --distance-m 200 --display

--------------------------------------------------------------------------
Logging philosophy - quiet terminal, verbose file
--------------------------------------------------------------------------
Stdout (and therefore the dashboard's live console) only ever gets
one-time messages: model/camera/ESC init, the [A]/[B] banner, the "PORNIT"
line when you engage, and the final summary. Everything that happens every
moment of the drive (speed AND steering) is written instead to a dense,
fixed-cadence CSV in logs/, at TICK_INTERVAL_SECONDS.

--------------------------------------------------------------------------
Deliberate behavior difference from cruise_control.py
--------------------------------------------------------------------------
If the speed sensor goes stale mid-run, this script stops the whole
program (ESC already goes neutral either way) and reports why in the
summary, rather than idling forever waiting for a D-pad/[Y] press - this
script is meant to run unattended.

--------------------------------------------------------------------------
How the ESC is actually controlled / IMPORTANT - ESC safety watchdog
--------------------------------------------------------------------------
See model_runner/model_runner.py's docstring for the full RC-PWM/arming
explanation - identical hardware, identical reasoning, not repeated here.

The esc-watchdog systemd service forces the ESC off unless a script in its
ALLOWED_SCRIPTS list is running, matched by full path. This script's full
path must be in that list, or the watchdog will fight it and the motor
will never actually respond. Always launch with:

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
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from evdev import ecodes, ff
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
# Shared config/ package (repo root) - see config/ for the single source of
# truth on every constant/class below, consolidated from main.py/
# cruise_control.py/model_runner.py where they used to be duplicated.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from config.camera_config import make_camera
from config.hardware_config import (
    ESC_MAX_US,
    ESC_NEUTRAL_US,
    ESC_PULSE_SATURATION_EPSILON_US,
    ESC_PULSE_SATURATION_NOTE_FRACTION,
)
from config.cruise_config import (
    CRUISE_CATCHUP_MAX_EXTRA_KMH,
    CRUISE_CATCHUP_WINDOW_S,
    CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S,
    CRUISE_LAUNCH_SECONDS,
    CRUISE_MAX_PULSE_STEP_US_PER_S,
    CRUISE_SPEED_FILTER_TAU_S,
    TARGET_SPEED_MAX_KMH,
    TARGET_SPEED_STEP_KMH,
)
from config.ipc_config import MAIN_CONTROL_SOCKET, ODO_STALE_GRACE_SECONDS
from config.pca9685_init import init_pca9685
from config.servo_esc import ESC, SteeringServo, steering_label_to_angle
from config.cruise_pi import cruise_pulse_us
from config.estop import relay_cmd
from config.odometry import get_rpm, is_odo_stale, odo_reader_loop, rpm_to_kmh
from config.vision import build_frame_stack, preprocess, quantize_input
from config.input_devices import find_xbox_controller

# ---------------------------------------------------------------------------
# Model - identical to model_runner.py
# ---------------------------------------------------------------------------
SMOOTH_ALPHA = 0.5
STEERING_DEADZONE = 0.06
FRAME_STACK_GAP_SECONDS = 0.1

DISPLAY_EVERY_N_FRAMES = 4

# ---------------------------------------------------------------------------
# Tick logging + splits
# ---------------------------------------------------------------------------
TICK_INTERVAL_SECONDS = 0.02  # 50Hz - see "Logging philosophy" in docstring
SPLIT_DISTANCE_M = 100.0

# ---------------------------------------------------------------------------
# Controller - [A] engages, [B] stops. D-pad UP/DOWN adjusts target_kmh
# live, but only in --speed-mode controller (no-op in cruise/none).
# ---------------------------------------------------------------------------
BTN_ENGAGE = ecodes.BTN_A
BTN_STOP = ecodes.BTN_B

SPEED_MODES = ("cruise", "controller", "none")


def find_hef_path(explicit=None):
    """--hef PATH (picked in the dashboard) wins; without it, the one .hef
    next to this script, as before."""
    if explicit:
        if not (explicit.endswith(".hef") and os.path.isfile(explicit)):
            raise RuntimeError(f"Modelul ales nu exista sau nu e un fisier .hef: {explicit}")
        return explicit
    hefs = [f for f in os.listdir(BASE_DIR) if f.endswith(".hef")]
    if len(hefs) == 0:
        raise RuntimeError(f"Niciun fisier .hef gasit in {BASE_DIR}. Pune exact un model acolo.")
    if len(hefs) > 1:
        raise RuntimeError(f"Mai multe fisiere .hef gasite in {BASE_DIR}: {hefs}. Trebuie sa fie exact unul.")
    return os.path.join(BASE_DIR, hefs[0])


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

_live_state = {
    "engaged": False, "speed_mode": "", "target_kmh": 0.0, "effective_target_kmh": 0.0,
    "kmh": 0.0, "distance_m": 0.0, "distance_target_m": None, "stop_reason": None,
}
_live_lock = threading.Lock()


def _update_live(**kwargs):
    with _live_lock:
        _live_state.update(kwargs)


def _get_live_state():
    with _live_lock:
        return dict(_live_state)


def _control_server_loop(stop_event):
    """Local socket for the web dashboard - STATUS returns the current
    target/speed/distance/mode, same pattern as cruise_control.py."""
    try:
        os.unlink(MAIN_CONTROL_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(MAIN_CONTROL_SOCKET)
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
                        "engaged": live["engaged"],
                        "speed_mode": live["speed_mode"],
                        "target_kmh": round(live["target_kmh"], 1),
                        "effective_target_kmh": round(live["effective_target_kmh"], 2),
                        "kmh": round(live["kmh"], 2),
                        "pace_sec_per_km": (3600.0 / live["kmh"]) if live["kmh"] > 0.05 else None,
                        "distance_m": round(live["distance_m"], 1),
                        "distance_target_m": live["distance_target_m"],
                        "stop_reason": live["stop_reason"],
                    }
                    try:
                        conn.sendall((json.dumps(payload) + "\n").encode())
                    except OSError:
                        pass
    finally:
        srv.close()
        try:
            os.unlink(MAIN_CONTROL_SOCKET)
        except OSError:
            pass


def _format_splits(splits):
    if not splits:
        return ""
    lines = ["\n--- SUTE (100m) ---"]
    for mark_m, elapsed_s, split_duration, pace_sec in splits:
        lines.append(
            f"{mark_m:5.0f}m | total {_format_time_ms(elapsed_s)} | "
            f"suta {split_duration:6.3f}s | pace {_format_pace_ms(pace_sec)}/km"
        )
    return "\n".join(lines) + "\n"


def _format_final_summary(speed_mode, target_kmh, distance_target_m, elapsed_s, tick_samples,
                           distance_m, model_name, frame_stack_n, raw_steering, stop_reason):
    lines = ["\n--- SUMAR FINAL ---"]
    model_desc = "necunoscut" if not frame_stack_n else (
        "clasic, un cadru" if frame_stack_n == 1 else f"{frame_stack_n} cadre stacked"
    )
    lines.append(f"Model: {model_name or 'necunoscut'} ({model_desc}) | "
                 f"speed-mode {speed_mode} | "
                 f"steering {'BRUT (fara EMA/deadzone)' if raw_steering else 'EMA+deadzone'}")
    lines.append(f"Motiv oprire: {stop_reason}")
    distance_target_desc = f"{distance_target_m:.0f}m tinta" if distance_target_m is not None else "fara tinta (pana la oprire)"
    if speed_mode == "none":
        lines.append(f"Fara tinta de viteza (doar steering) | {distance_target_desc} | Durata: {elapsed_s:.1f}s")
    else:
        lines.append(
            f"Tinta: {target_kmh:.1f} km/h ({_format_pace(target_kmh)}/km) | "
            f"{distance_target_desc} | Durata: {elapsed_s:.1f}s"
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
        f"Distanta -> {distance_m:.1f}m" +
        (f" din {distance_target_m:.0f}m tinta (diferenta {distance_m - distance_target_m:+.1f}m)"
         if distance_target_m is not None else "")
    )
    if speed_mode != "none":
        deviation = avg_kmh - target_kmh
        lines.append(f"km/h  -> mediu {avg_kmh:.2f} (cat timp s-a miscat) | maxim {max_kmh:.2f}")
        lines.append(f"Pace  -> mediu {_format_pace(avg_kmh)}/km | cel mai bun {_format_pace(max_kmh)}/km")
        lines.append(f"Abatere fata de tinta: {deviation:+.2f} km/h")
        saturated = sum(1 for p in pulse_vals if p >= ESC_MAX_US - ESC_PULSE_SATURATION_EPSILON_US)
        saturation_fraction = saturated / len(pulse_vals) if pulse_vals else 0.0
        if saturation_fraction >= ESC_PULSE_SATURATION_NOTE_FRACTION:
            lines.append(
                f"NOTA: puls la maxim ({ESC_MAX_US:.0f}us) in {saturation_fraction:.0%} din cursa - "
                f"tinta a fost probabil peste plafonul fizic real al motorului/bateriei in acel moment, "
                f"nu o eroare a regulatorului."
            )
    else:
        lines.append(f"km/h  -> mediu {avg_kmh:.2f} (miscare externa, ESC la neutru) | maxim {max_kmh:.2f}")
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-mode", choices=SPEED_MODES, required=True,
                     help="cruise = fixed target speed (needs --target-kmh); "
                          "controller = D-pad up/down adjusts speed live, starts at 0; "
                          "none = ESC stays neutral, only the model steers")
    ap.add_argument("--target-kmh", type=float, default=None,
                     help=f"Viteza tinta in km/h (0.1-{TARGET_SPEED_MAX_KMH:.0f}) - "
                          "obligatoriu pentru --speed-mode cruise, interzis in celelalte moduri")
    ap.add_argument("--distance-m", type=float, default=None,
                     help="Distanta tinta in metri (1-50000) - daca lipseste, ruleaza pana la oprire "
                          "([B]/Ctrl+C)")
    ap.add_argument("--smooth-steering", dest="raw_steering", action="store_false", default=True,
                     help="Foloseste EMA smoothing + deadzone in loc de predictia bruta a modelului "
                          "(implicit: raw)")
    ap.add_argument("--display", action="store_true",
                     help="Deschide o fereastra cv2 de preview live")
    ap.add_argument("--hef", metavar="PATH", default=None,
                     help="Model .hef de folosit (cale absoluta, oriunde pe Pi). Implicit: "
                          "singurul .hef din folderul main/")
    args = ap.parse_args()

    if args.speed_mode == "cruise":
        if args.target_kmh is None:
            raise SystemExit("--target-kmh este obligatoriu pentru --speed-mode cruise")
    elif args.target_kmh is not None:
        raise SystemExit(f"--target-kmh nu poate fi folosit cu --speed-mode {args.speed_mode}")

    if args.target_kmh is not None and not (0.1 <= args.target_kmh <= TARGET_SPEED_MAX_KMH):
        raise SystemExit(
            f"--target-kmh trebuie sa fie intre 0.1 si {TARGET_SPEED_MAX_KMH:.0f} "
            f"(primit: {args.target_kmh})"
        )
    if args.distance_m is not None and not (1.0 <= args.distance_m <= 50000.0):
        raise SystemExit(f"--distance-m trebuie sa fie intre 1 si 50000 (primit: {args.distance_m})")

    return args


def main():
    args = parse_args()
    speed_mode = args.speed_mode
    show_display = args.display
    raw_steering = args.raw_steering
    target_kmh = args.target_kmh if speed_mode == "cruise" else 0.0
    distance_target_m = args.distance_m

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Defined before any code that can raise - `finally` uses these even if
    # startup fails early (same reasoning as cruise_control.py/main.py).
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
    tick_samples = []  # (elapsed_s, kmh, pulse_us, steer_cmd, fps) at each TICK_INTERVAL_SECONDS
    splits = []  # (mark_m, elapsed_s, split_duration_s, pace_sec_per_km)
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
        hef_path = find_hef_path(args.hef)
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
        pca = init_pca9685()
        esc = ESC(pca)
        esc.neutral()
        steering = SteeringServo(pca)

        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        relay_cmd("RELAY_ON")
        esc.arm()

        # --- Camera -----------------------------------------------------------
        picam2 = make_camera()
        picam2.start()
        logging.info("Camera started")

        # --- Log detaliat (CSV, ritm fix) --------------------------------
        tick_csv_path = os.path.join(TICK_LOG_DIR, f"main_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        tick_csv_file = open(tick_csv_path, "w")
        tick_csv_file.write(
            "timestamp,elapsed_s,engaged,speed_mode,target_kmh,effective_target_kmh,rpm,kmh,pace_mmss,"
            "pulse_us,steer_raw,steer_smoothed,steer_cmd,servo_angle,distance_m,fps\n"
        )
        tick_csv_file.flush()
        print(f"Log detaliat: {tick_csv_path}")

        odo_stop_event = threading.Event()
        odo_thread = threading.Thread(target=odo_reader_loop, args=(odo_stop_event,), daemon=True)
        odo_thread.start()

        _update_live(speed_mode=speed_mode, target_kmh=target_kmh, distance_target_m=distance_target_m)
        control_stop_event = threading.Event()
        control_thread = threading.Thread(
            target=_control_server_loop, args=(control_stop_event,), daemon=True)
        control_thread.start()

        print("\n-----------------------------------------------------")
        print(f"Autopilot gata. Mod viteza: {speed_mode}.")
        if speed_mode == "cruise":
            print(f"Tinta: {target_kmh:.1f} km/h ({_format_pace(target_kmh)}/km). D-pad nu are efect.")
        elif speed_mode == "controller":
            print("Tinta porneste de la 0 km/h. D-pad SUS/JOS ajusteaza tinta live.")
        else:
            print("Fara tinta de viteza - ESC ramane la neutru, doar modelul da directia.")
        print(f"Distanta: {f'{distance_target_m:.0f}m' if distance_target_m is not None else 'fara tinta (ruleaza pana oprire)'}")
        print("Virajul vine 100% din model, mereu (indiferent de angajare).")
        print("[A] PORNESTE cursa. [B] OPRESTE oricand.")
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
        effective_target_kmh = target_kmh

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
                                    if speed_mode == "controller":
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
                                            if speed_mode == "none":
                                                esc.neutral()
                                            print(f"\n>>> PORNIT (A) - mod {speed_mode}"
                                                  + (f", tinta {target_kmh:.1f} km/h ({_format_pace(target_kmh)}/km)"
                                                     if speed_mode != "none" else "")
                                                  + (f", {distance_target_m:.0f}m" if distance_target_m is not None else "")
                                                  + " <<<")
                                            logging.info(f"Angajat: mod {speed_mode}, tinta {target_kmh:.1f} km/h, "
                                                          f"distanta {distance_target_m}")
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

                        # Steering applies always, regardless of `engaged` -
                        # so the operator can see the model tracking the
                        # road correctly BEFORE pressing [A] (wheels aren't
                        # driven until engaged).
                        servo_angle = steering_label_to_angle(steer_cmd)
                        steering.set_angle(servo_angle)

                        t_now = time.time()
                        current_fps = 0.9 * current_fps + 0.1 / max(t_now - t_prev, 1e-9)
                        t_prev = t_now

                        # --- speed (cruise-control regulator, cruise/controller only) ---
                        rpm = get_rpm()
                        raw_kmh = rpm_to_kmh(rpm)
                        dt = now - last_control_time
                        filter_weight = min(1.0, dt / CRUISE_SPEED_FILTER_TAU_S) if dt > 0 else 0.0
                        filtered_kmh += (raw_kmh - filtered_kmh) * filter_weight
                        kmh = filtered_kmh
                        # implicit - suprascris mai jos cat timp e angajat
                        effective_target_kmh = target_kmh if speed_mode != "none" else 0.0

                        if engaged:
                            # ESP32 tace cand roata sta (un singur RPM:0.00, apoi
                            # nimic), deci "stale" = masina oprita SAU senzor picat.
                            # In "none" ESC-ul nu e condus - nimic de protejat.
                            past_grace = (now - engage_time) > ODO_STALE_GRACE_SECONDS
                            odo_stale = speed_mode != "none" and past_grace and is_odo_stale()
                            if odo_stale and speed_mode == "cruise":
                                engaged = False
                                esc.neutral()
                                prev_pulse_us = ESC_NEUTRAL_US
                                stop_reason = "senzor de viteza indisponibil"
                                logging.warning("Oprire automata: date RPM invechite")
                                raise KeyboardInterrupt
                            elif odo_stale:
                                # controller: ca cruise_control.py pe main - doar
                                # dezangajeaza, scriptul ramane pornit ([A] reia).
                                engaged = False
                                esc.neutral()
                                prev_pulse_us = ESC_NEUTRAL_US
                                print("\n!!! Senzor de viteza indisponibil / masina oprita - "
                                      "dezangajat (apasa [A] ca sa reiei) !!!")
                                logging.warning("Dezangajat automat: date RPM invechite")
                            else:
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
                                    logging.info(
                                        f"Suta {next_split_mark_m:.0f}m: {_format_time_ms(crossing_elapsed_s)} "
                                        f"total, {split_duration:.3f}s, pace {_format_pace_ms(pace_sec)}/km")
                                    last_split_elapsed_s = crossing_elapsed_s
                                    next_split_mark_m += SPLIT_DISTANCE_M

                                if speed_mode == "none":
                                    effective_target_kmh = 0.0
                                else:
                                    in_launch = (now - engage_time) <= CRUISE_LAUNCH_SECONDS
                                    max_step = (CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S if in_launch
                                                else CRUISE_MAX_PULSE_STEP_US_PER_S)

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

                                # CSV detaliat scrie doar cat timp cursa chiar
                                # ruleaza (engaged si nu stale).
                                if now - last_tick_time >= TICK_INTERVAL_SECONDS:
                                    last_tick_time = now
                                    elapsed_s = now - leg_start_time
                                    pace = _format_pace(kmh)
                                    tick_samples.append((elapsed_s, kmh, prev_pulse_us, steer_cmd, current_fps))
                                    tick_csv_file.write(
                                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{elapsed_s:.3f},"
                                        f"{engaged},{speed_mode},{target_kmh:.1f},{effective_target_kmh:.2f},{rpm:.2f},"
                                        f"{kmh:.2f},{pace},{prev_pulse_us:.1f},{raw_label:+.4f},{smooth_label:+.4f},"
                                        f"{steer_cmd:+.4f},{servo_angle},{distance_m:.2f},{current_fps:.1f}\n"
                                    )
                                    tick_csv_file.flush()

                                if distance_target_m is not None and distance_m >= distance_target_m:
                                    stop_reason = f"distanta tinta atinsa ({distance_target_m:.0f}m)"
                                    raise KeyboardInterrupt
                        last_control_time = now

                        _update_live(target_kmh=target_kmh, effective_target_kmh=effective_target_kmh,
                                     kmh=kmh, engaged=engaged, distance_m=distance_m)

                        frame_counter += 1

                        if show_display:
                            if frame_counter % DISPLAY_EVERY_N_FRAMES == 0:
                                display = frame
                                cv2.putText(
                                    display,
                                    f"Steer: {steer_cmd:+.2f} (model {raw_label:+.2f})  FPS: {current_fps:.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                                dist_label = (f"{distance_m:.0f}/{distance_target_m:.0f}m"
                                              if distance_target_m is not None else f"{distance_m:.0f}m")
                                cv2.putText(
                                    display,
                                    f"{'PORNIT' if engaged else 'astept [A]'} | {speed_mode} | {kmh:.2f} km/h "
                                    f"(tinta {effective_target_kmh:.1f}) | {_format_pace(kmh)}/km | {dist_label}",
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
        _update_live(stop_reason=stop_reason)

        relay_cmd("RELAY_OFF")
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
            speed_mode=speed_mode, target_kmh=target_kmh, distance_target_m=distance_target_m,
            elapsed_s=elapsed_s, tick_samples=tick_samples, distance_m=distance_m, model_name=model_name,
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
