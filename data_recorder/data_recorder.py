"""
RoboPacerV2 - Data Recorder
============================
Captures camera frames paired with steering-angle labels (behavioral-cloning
style dataset) while an Xbox controller drives the car through a PCA9685
(servo = steering on channel 0, ESC = throttle on channel 1 — verify this
against your actual wiring; see SERVO_CHANNEL / ESC_CHANNEL below).

Camera settings match camera.py (640x480 RGB888 @ 120fps, IMX219 NoIR
tuning; AnalogueGain 12 here vs camera.py's 16, reduced for outdoor
daylight) so the live-view app and this recorder agree on what the model
will actually see.

--------------------------------------------------------------------------
How the ESC is actually controlled
--------------------------------------------------------------------------
An ESC (Electronic Speed Controller) is driven by a standard 50Hz RC PWM
signal: every 20ms it expects one pulse whose *width* (not amplitude)
encodes the throttle command:

    ~1000us  -> full reverse / brake   (ESC_MIN_US)
    ~1500us  -> neutral / stop         (ESC_NEUTRAL_US)
    ~1700us  -> full forward           (ESC_MAX_US, our configured ceiling)

The PCA9685 doesn't output microseconds directly - it's a 12-bit PWM chip,
so every pulse width has to be converted into a 16-bit duty cycle relative
to the 20ms (50Hz) period:

    duty_cycle = (pulse_us / period_us) * 65535   where period_us = 1e6 / freq

On power-up almost every ESC refuses to spin the motor until it has seen a
*stable* signal for a second or two - this is the "arming" sequence, and it
exists so the motor can't jump to whatever value the PWM line happened to
be at when power was applied. We replicate that below by holding a steady
pulse for ESC_ARM_HOLD_SECONDS before ever handing control to the joystick.

That arm pulse must be NEUTRAL (ESC_ARM_PULSE_US = ESC_NEUTRAL_US), not an
endpoint. Confirmed against the QuicRun 10BL120 manual: there's no special
"arm at an extreme" concept - 1000us (ESC_MIN_US) is a real brake/reverse
command the ESC will act on, not a neutral arming cue. In "Forward/Reverse
with Brake" mode (this ESC's setting), holding 1000us steady is read as
brake, and depending on the ESC's internal state from the previous run it
can register as the second push of the brake-then-reverse sequence - i.e.
holding ESC_MIN_US at startup could launch the car into reverse at full
throttle. Don't change this away from ESC_NEUTRAL_US.
--------------------------------------------------------------------------
Why frame writes happen on a background thread
--------------------------------------------------------------------------
Writing every recorded frame to disk (cv2.imwrite) used to happen inline in
the same loop that reads the controller and writes servo/ESC pulses. An SD
card can stall for several seconds under sustained continuous write load
(internal garbage collection/wear-leveling) - when that write call itself
was blocking the loop, the *entire* control loop froze along with it: no
controller events got read, no new PWM value got written, so the car kept
driving at whatever throttle/steering it had going into the stall (the
PCA9685 holds its last programmed duty cycle in hardware until told
otherwise - this looked like a stuck ESC/servo but wasn't one). The
esc_watchdog safety net doesn't catch this either - it only reacts to the
process dying, and a frozen-but-alive loop still shows up as running.

Recording now hands frames off to a queue that a separate writer thread
drains - the control loop's per-iteration cost is just a fast in-memory
enqueue, so a slow/stalled disk can delay when a frame gets *saved* but can
never block reading the controller or updating the ESC/servo again.
--------------------------------------------------------------------------
"""

import argparse
import json
import logging
import os
import queue
import select
import signal
import socket
import threading
import time

import board
import busio
import cv2
import numpy as np
from adafruit_motor import servo as adafruit_servo
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, ff, list_devices
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Paths - everything this script reads/writes lives next to it, in its own
# folder, separate from the rest of RoboPacerV2.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "set1")
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
LOG_JSON_PATH = os.path.join(DATASET_DIR, "driving_log.json")
LOG_FILE_PATH = os.path.join(BASE_DIR, "data_recorder.log")

os.makedirs(FRAMES_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("picamera2", "libcamera", "PIL"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Camera - identical configuration to camera.py
# ---------------------------------------------------------------------------
TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0  # 16.0 * 0.75 - reduced 25% for outdoor daylight use

SAVED_FRAME_SIZE = (640, 480)  # frame size written to disk for training (matches FRAME_SIZE)

# ---------------------------------------------------------------------------
# Steering (servo on PCA9685 channel 0)
# ---------------------------------------------------------------------------
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900
SERVO_MAX_PULSE = 2200
SERVO_MIN_ANGLE = 45
SERVO_MAX_ANGLE = 135
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 4

# ---------------------------------------------------------------------------
# Throttle (ESC on PCA9685 channel 1) - pulse widths in microseconds
# ---------------------------------------------------------------------------
ESC_CHANNEL = 1
PCA_FREQUENCY_HZ = 50
ESC_ARM_PULSE_US = 1500  # = ESC_NEUTRAL_US - the QuicRun 10BL120 manual has no
                          # concept of a special "arm at an extreme" pulse; it
                          # just needs a stable valid signal, and 1000 (our
                          # ESC_MIN_US, i.e. brake/reverse) is a real command
                          # the ESC will actually act on, not a neutral arm cue
ESC_ARM_HOLD_SECONDS = 3.0
ESC_NEUTRAL_US = 1500
ESC_MIN_US = 1000
ESC_MAX_US = 1700

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
BTN_START_RECORDING = ecodes.BTN_A  # start / resume
BTN_PAUSE = ecodes.BTN_Y  # pause
BTN_SAVE_AND_STOP = ecodes.BTN_B  # save + quit

AXIS_CENTER = 32767
AXIS_MAX = 65535


# Transient I2C bus faults worth logging-and-continuing instead of crashing a
# multi-hour recording session over: 121 (Remote I/O error - device didn't
# ACK) and 19 (No such device - the device briefly vanished from the bus
# entirely). Both are typically a momentary bus/wiring glitch on a vibrating
# RC chassis - loose SDA/SCL/power wiring to the PCA9685, or motor/ESC
# electrical noise - not something the program should die over.
_TRANSIENT_I2C_ERRNOS = (121, 19)


class SteeringServo:
    """Servo on the PCA9685, with I2C-error handling centralized here."""

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
    """
    Car ESC on the PCA9685, controlled by microsecond pulse width. See the
    module docstring above for how RC PWM and arming actually work.
    """

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
        """Cut PWM entirely (used on shutdown)."""
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

    @staticmethod
    def pulse_from_gas_brake(gas_value, brake_value):
        gas_offset = (gas_value / 1023.0) * (ESC_MAX_US - ESC_NEUTRAL_US)
        brake_offset = -(brake_value / 1023.0) * (ESC_NEUTRAL_US - ESC_MIN_US)
        pulse = ESC_NEUTRAL_US + gas_offset + brake_offset
        return max(ESC_MIN_US, min(ESC_MAX_US, int(pulse)))


LABEL_DEADZONE = 0.2


def steering_axis_to_label(x_value):
    raw = np.interp(x_value, [0, AXIS_MAX], [-1.0, 1.0])
    if abs(raw) <= LABEL_DEADZONE:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    label = sign * (abs(raw) - LABEL_DEADZONE) / (1.0 - LABEL_DEADZONE)
    return round(float(label), 2)


def steering_axis_to_angle(x_value):
    raw = np.interp(x_value, [0, AXIS_MAX], [1.0, -1.0])
    if abs(raw) <= LABEL_DEADZONE:
        norm = 0.0
    else:
        sign = 1.0 if raw > 0 else -1.0
        norm = sign * (abs(raw) - LABEL_DEADZONE) / (1.0 - LABEL_DEADZONE)
    angle = SERVO_NEUTRAL_ANGLE + norm * (SERVO_MAX_ANGLE - SERVO_NEUTRAL_ANGLE) + SERVO_OFFSET
    return int(max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle)))


def find_xbox_controller():
    for path in list_devices():
        dev = InputDevice(path)
        if "xbox" in dev.name.lower():
            return dev
    return None


def load_driving_log():
    if os.path.exists(LOG_JSON_PATH) and os.path.getsize(LOG_JSON_PATH) > 0:
        try:
            with open(LOG_JSON_PATH) as f:
                log = json.load(f)
            print(f"Log existent incarcat: {len(log)} intrari.")
            return log
        except (json.JSONDecodeError, OSError) as e:
            print(f"Log JSON corupt/ilizibil ({e}); pornesc unul nou.")
    return []


def save_driving_log(log):
    with open(LOG_JSON_PATH, "w") as f:
        json.dump(log, f, indent=4)
    print(f"Log salvat in {LOG_JSON_PATH} ({len(log)} intrari).")


WRITE_QUEUE_MAXSIZE = 400  # ~5s of buffering at 80fps before frames start
                            # getting dropped - generous enough to absorb an
                            # SD card stall without ever blocking the caller


def _writer_loop(write_queue, driving_log, frames_dir, legacy_format):
    """Runs on a background thread for the whole program lifetime. The only
    thing that ever touches disk for a recorded frame - see the module
    docstring for why this isn't inline in the control loop.

    legacy_format=True omits "timestamp" from each record (classic format:
    just image_path + steering_angle). The trainer (trainer/engine/train.py)
    detects which format a dataset is by whether "timestamp" is present, and
    picks single-frame vs frame-stacked training accordingly - main.py does
    the equivalent detection at inference time from the compiled model's own
    input shape. See --legacy's help text below for why you'd choose either."""
    while True:
        item = write_queue.get()
        if item is None:  # sentinel - drain requested, stop
            write_queue.task_done()
            return
        image_filename, saved_frame, steering_label, ts = item
        cv2.imwrite(os.path.join(frames_dir, image_filename), saved_frame)
        record = {
            "image_path": f"{os.path.basename(frames_dir)}/{image_filename}",
            "steering_angle": steering_label,
        }
        if not legacy_format:
            record["timestamp"] = ts
        driving_log.append(record)
        write_queue.task_done()


def next_frame_index():
    existing = [f for f in os.listdir(FRAMES_DIR) if f.startswith("frame_") and f.endswith(".jpg")]
    indices = []
    for name in existing:
        try:
            indices.append(int(name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 0


def make_camera():
    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(tuning=tuning)
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT},
        controls={"FrameRate": FRAME_RATE, "AnalogueGain": ANALOGUE_GAIN},
    )
    picam2.configure(config)
    return picam2


def _handle_sigterm(signum, frame):
    """
    SIGTERM (plain `kill <pid>`, `systemctl stop`, ...) has no default
    Python handler, so without this it would kill the process without ever
    reaching the `finally` block below - leaving the ESC at whatever pulse
    it last had. Converting it into a KeyboardInterrupt reuses the same
    graceful shutdown path as Ctrl+C.

    This still cannot save us from SIGKILL, a crash, or a power loss - that
    class of failure is what the separate esc_watchdog service is for.
    """
    raise KeyboardInterrupt


_RELAY_SOCKET = "/tmp/esp32_relay.sock"

def _relay_cmd(cmd: str) -> None:
    """Trimite RELAY_ON / RELAY_OFF la estop_listener care detine portul serial.
    Best-effort: daca ESP32-ul nu e conectat sau serviciul nu ruleaza, ignora silentios."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(_RELAY_SOCKET)
            s.sendall((cmd + "\n").encode())
    except (OSError, socket.timeout):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                     help="Record in the classic format (image_path + steering_angle "
                          "only, no timestamp). The default records a 'timestamp' per "
                          "frame too, which the trainer needs for frame-stacked "
                          "temporal input (see trainer/engine/train.py) - use --legacy "
                          "only if you specifically want single-frame training/inference.")
    ap.add_argument("--display", action="store_true",
                     help="Open a live cv2 preview window (costs a few ms/frame). "
                          "Default is headless - status is printed to the console instead.")
    args = ap.parse_args()
    legacy_format = args.legacy
    show_display = args.display

    signal.signal(signal.SIGTERM, _handle_sigterm)

    pca = None
    esc = None
    steering = None
    picam2 = None
    controller = None
    driving_log = load_driving_log()
    frame_index = next_frame_index()

    last_rumble_effect_id = None

    def rumble(duration_ms):
        """One-shot rumble, non-blocking - the controller plays it on its own
        timer while the caller keeps running (no time.sleep() needed here,
        same reasoning as why frame writes moved off the control loop - see
        module docstring). Xbox pads only have a handful of FF effect slots,
        so the previous effect is erased before uploading the next one -
        otherwise a long session full of start/pause/resume presses would
        eventually exhaust them and go silent. Defined here (before
        `controller` is ever assigned) rather than after find_xbox_controller()
        succeeds, so it's always safe to call from the shutdown/finally path
        too, even if startup failed before a controller was found."""
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

    if driving_log:
        # Keep one driving_log.json internally consistent - a dataset that
        # mixes records with and without "timestamp" can't be reliably
        # trained (train.py's _detect_format() would just reject it), so an
        # existing log's format wins over a mismatched --legacy flag rather
        # than silently corrupting it.
        existing_legacy = not any("timestamp" in r for r in driving_log)
        if existing_legacy != legacy_format:
            print(f"Atentie: {os.path.basename(LOG_JSON_PATH)} exista deja in format "
                  f"{'CLASIC (fara timestamp)' if existing_legacy else 'CU TIMESTAMP'} - "
                  f"continui in acelasi format ca sa nu amestec formate in acelasi fisier.")
            legacy_format = existing_legacy

    print(f"Format de inregistrare: {'CLASIC (fara timestamp)' if legacy_format else 'CU TIMESTAMP (frame-stacking posibil la antrenare)'}")

    # Started unconditionally (even before recording ever starts) and kept
    # running for the whole program - idle-waiting on an empty queue costs
    # nothing. See _writer_loop()/module docstring for why frame writes
    # don't happen inline in the control loop below.
    write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
    writer_thread = threading.Thread(
        target=_writer_loop, args=(write_queue, driving_log, FRAMES_DIR, legacy_format), daemon=True)
    writer_thread.start()

    try:
        _relay_cmd("RELAY_ON")
        # --- I2C / PCA9685 --------------------------------------------------
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = PCA_FREQUENCY_HZ

        esc = ESC(pca)
        steering = SteeringServo(pca)

        # --- Controller: fail fast before we spend 3s arming the ESC --------
        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        # --- Arm the ESC (see module docstring for why this matters) --------
        esc.arm()

        # --- Camera -----------------------------------------------------------
        picam2 = make_camera()
        picam2.start()
        logging.info("Camera started - 640x480 @ 120fps gain=16")

        print("\n-----------------------------------------------------")
        print("Apasa [A] pentru a INCEPE INREGISTRAREA/RESUME.")
        print("Apasa [Y] pentru a PAUZA inregistrarea.")
        print("Apasa [B] pentru a OPRI si SALVA log-ul.")
        if show_display:
            print("Apasa 'q' in fereastra pentru oprire (sau [B] pe controller).")
        else:
            print("Mod headless (fara fereastra). Ctrl+C sau [B] pe controller pentru oprire.")
        print("-----------------------------------------------------")

        gas_value = 0
        brake_value = 0
        is_recording = False
        is_paused = False
        current_fps = 0
        frame_count_fps = 0
        start_time_fps = time.time()
        steering_axis_raw = AXIS_CENTER
        steering_label = steering_axis_to_label(steering_axis_raw)

        while True:
            ready, _, _ = select.select([controller_fd], [], [], 0.001)

            for _ in ready:
                for event in controller.read():
                    if event.type == ecodes.EV_ABS:
                        abs_name = ecodes.ABS.get(event.code)

                        if abs_name == "ABS_GAS":
                            gas_value = event.value
                        elif abs_name == "ABS_BRAKE":
                            brake_value = event.value
                        elif abs_name == "ABS_X":
                            steering_axis_raw = event.value
                            angle = steering_axis_to_angle(event.value)
                            # Servo follows the stick even while paused - pause only
                            # stops writing to disk, not driving. Lets you steer the
                            # car (e.g. off-track for deliberate recovery-data setup)
                            # without it being recorded, then hit Resume to start
                            # capturing the correction back.
                            steering.set_angle(angle)

                        if abs_name in ("ABS_GAS", "ABS_BRAKE"):
                            pulse = ESC.pulse_from_gas_brake(gas_value, brake_value)
                            esc.set_pulse_us(pulse)

                    elif event.type == ecodes.EV_KEY and event.value == 1:
                        if event.code == BTN_START_RECORDING:
                            if is_paused:
                                is_paused = False
                                print("\n>>> RESUME (A) <<<")
                                rumble(500)
                            elif not is_recording:
                                is_recording = True
                                start_time_fps = time.time()
                                frame_count_fps = 0
                                print("\n>>> START INREGISTRARE <<<")
                        elif event.code == BTN_PAUSE:
                            if is_recording and not is_paused:
                                is_paused = True
                                # One-off safety snap to neutral/center the instant
                                # pause is pressed (in case your hand was mid-motion
                                # on the stick) - after this, the stick drives
                                # normally again, same as unpaused (see ABS_X/ABS_GAS
                                # handling above).
                                esc.neutral()
                                steering.center()
                                print("\n>>> PAUZA (Y) <<<")
                                rumble(1000)
                        elif event.code == BTN_SAVE_AND_STOP:
                            raise KeyboardInterrupt

            try:
                frame = picam2.capture_array()
            except RuntimeError:
                logging.warning("Camera capture failed, skipping frame")
                time.sleep(0.05)
                continue

            steering_label = steering_axis_to_label(steering_axis_raw)

            if is_recording and not is_paused:
                frame_count_fps += 1
                elapsed = time.time() - start_time_fps
                if elapsed > 0:
                    current_fps = frame_count_fps / elapsed

                image_filename = f"frame_{frame_index:05d}.jpg"
                saved_frame = frame if SAVED_FRAME_SIZE == FRAME_SIZE else cv2.resize(frame, SAVED_FRAME_SIZE, interpolation=cv2.INTER_AREA)
                try:
                    # .copy() - the writer thread reads this after capture_array()
                    # may already be filling the next frame's buffer.
                    write_queue.put_nowait((image_filename, saved_frame.copy(), steering_label, time.time()))
                except queue.Full:
                    logging.warning(f"Write queue full - dropped frame {frame_index:05d} (disk falling behind)")
                print(
                    f"REC {len(driving_log)} | frame {frame_index:05d} | "
                    f"steer {steering_label:+.2f} | {current_fps:.1f} fps | coada={write_queue.qsize()}",
                    end="\r",
                )
                frame_index += 1

            if show_display:
                display = frame
                status, color = "OFF", (0, 0, 255)
                if is_recording and is_paused:
                    status, color = "PAUZA (Y)", (255, 255, 0)
                elif is_recording:
                    status, color = "REC (A)", (0, 255, 0)

                cv2.putText(display, f"Stare: {status} | Viraj: {steering_label:+.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                if current_fps > 0:
                    cv2.putText(display, f"FPS: {current_fps:.1f} | Cadre: {len(driving_log)}",
                                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("RoboPacerV2 - Data Recorder", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    except (KeyboardInterrupt, ConnectionError) as e:
        if isinstance(e, ConnectionError):
            print(f"\n{e}")
        print("\nOprire...")
    except Exception as e:
        logging.exception("Eroare majora neasteptata")
        print(f"\nEroare majora neasteptata: {e}")
    finally:
        _relay_cmd("RELAY_OFF")
        # Motor to neutral first, before anything that can take a while
        # (like draining a backlog of queued disk writes) - stopping the
        # car is more urgent than finishing the save.
        if esc is not None:
            esc.neutral()

        print("\nSe salveaza cadrele ramase in coada de scriere pe disc...")
        write_queue.put(None)
        writer_thread.join(timeout=30)
        if writer_thread.is_alive():
            logging.warning("Writer thread did not stop within 30s - some recent frames may be missing from the log.")
            print("Atentie: thread-ul de scriere nu s-a oprit la timp - unele cadre recente pot lipsi din log.")

        if driving_log:
            save_driving_log(driving_log)
        if esc is not None:
            time.sleep(0.1)
            esc.stop()
            # Two short pulses = "program stopped" - the controller's own
            # timer plays each one, so these sleeps are just spacing/letting
            # them finish before the fd potentially closes on process exit
            # (this is shutdown cleanup, not the control loop, so blocking
            # briefly here doesn't cost anything).
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
                # Unlike SteeringServo/ESC above, adafruit_pca9685's own
                # deinit()/reset() does a raw I2C write with no error
                # handling of its own - if the bus is already unstable
                # (see _TRANSIENT_I2C_ERRNOS), this would otherwise crash
                # cleanup itself instead of just skipping a courtesy reset
                # on a program that's already shutting down.
                logging.warning(f"I2C error during pca.deinit(): {e}")
        if picam2 is not None and getattr(picam2, "started", False):
            picam2.stop()
        cv2.destroyAllWindows()
        logging.info("Data recorder stopped")
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
