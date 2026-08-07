"""
RoboPacerV2 - Autonomous Drive
================================
Runs the .hef model found in this same folder on the Hailo AI Kit, steers
the car from its live prediction, and lets the Xbox controller's D-pad
adjust throttle (Up = a bit faster, Down = a bit slower, down to 0/stop).
No manual steering input - curves are entirely up to the model.

Usage:
    python3 main.py                  headless (no window) - fastest, prints
                                      average FPS once per second to the console
    python3 main.py --display        opens a live cv2 preview window (costs a
                                      few ms/frame - see DISPLAY_EVERY_N_FRAMES)
    python3 main.py --raw-steering   sends the model's raw per-frame prediction
                                      straight to the servo - no EMA smoothing,
                                      no deadzone. For testing/debugging what the
                                      model itself outputs, not for normal driving.

Put exactly one *.hef file in this folder next to this script. If there
are zero or more than one, this refuses to start (see find_hef_path()) -
ambiguity about which model is driving the car is not something to guess
at.

Camera and hardware configuration (servo/ESC channels, pulse ranges,
camera resolution/format/framerate/gain/tuning file) are copied 1:1 from
data_recorder/data_recorder.py, since the model was trained on exactly
what that script captures - any mismatch here would mean the model sees
something different at inference time than it did during training.

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
pulse for ESC_ARM_HOLD_SECONDS before ever handing control to the model/D-pad.

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
IMPORTANT - ESC safety watchdog
--------------------------------------------------------------------------
The esc-watchdog systemd service forces the ESC off unless a script in its
ALLOWED_SCRIPTS list (safety/esc_watchdog.sh) is running, matched by full
path. This script's full path must be in that list, or the watchdog will
fight it and the motor will never actually respond. Always launch with:

    python3 /home/pi/RoboPacerV2/main/main.py
--------------------------------------------------------------------------
"""

import argparse
import logging
import os
import select
import signal
import socket
import time
from collections import deque

import board
import busio
import cv2
import numpy as np
from adafruit_motor import servo as adafruit_servo
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, list_devices
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

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("picamera2", "libcamera", "PIL"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Camera - identical configuration to data_recorder/data_recorder.py, so
# the model sees exactly what it was trained on.
# ---------------------------------------------------------------------------
TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0

# ---------------------------------------------------------------------------
# Steering (servo on PCA9685 channel 0) - identical to data_recorder.py
# ---------------------------------------------------------------------------
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900
SERVO_MAX_PULSE = 2200
SERVO_MIN_ANGLE = 45
SERVO_MAX_ANGLE = 135
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 4

# ---------------------------------------------------------------------------
# Throttle (ESC on PCA9685 channel 1) - pulse widths in microseconds,
# identical range to data_recorder.py. Only the D-pad step logic below is
# new (data_recorder.py used analog gas/brake triggers instead).
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

THROTTLE_STEP_US = 20  # per D-pad press
THROTTLE_MAX_LEVEL = int((ESC_MAX_US - ESC_NEUTRAL_US) / THROTTLE_STEP_US)  # 10 steps to full

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_SIZE = 224  # ResNet-18 trained at 224x224 (see RoboPacerTrainer)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SMOOTH_ALPHA = 0.5  # EMA smoothing on the predicted steering label. Higher =
                     # less lag (faster reaction) but more raw model jitter
                     # passed through; lower = smoother but more delay. Delay
                     # matters more at higher throttle (car covers more
                     # ground per frame), so this is tuned toward "less lag"
                     # and jitter is instead handled by STEERING_DEADZONE
                     # below, not by over-smoothing.
STEERING_DEADZONE = 0.06  # |smoothed label| below this is treated as 0 (go
                           # straight). Model output on straight sections is
                           # never exactly 0, it drifts a little frame to
                           # frame - without this, that drift alone causes
                           # visible weaving even on a straight track.

# ---------------------------------------------------------------------------
# Frame stacking (temporal memory) - must match trainer/engine/train.py's
# copies of these two constants exactly. The model's input is the current
# frame plus FRAME_STACK_N - 1 frames pulled from a rolling buffer, each
# roughly FRAME_STACK_GAP_SECONDS further in the past, concatenated as extra
# channels. Consecutive camera captures (~10ms apart at 100fps) look nearly
# identical, so we don't stack raw consecutive frames - the buffer is
# searched by wall-clock time, same as the training-side lookback.
# ---------------------------------------------------------------------------
FRAME_STACK_N = 3
FRAME_STACK_GAP_SECONDS = 0.1

DISPLAY_EVERY_N_FRAMES = 4  # cv2.imshow costs ~4-5ms/call - update it less often
                             # than the control loop (measured: capture+
                             # preprocess+infer alone hit ~120fps, imshow was
                             # the single biggest avoidable per-frame cost)


# Transient I2C bus faults worth logging-and-continuing instead of crashing
# an autonomous drive over: 121 (Remote I/O error - device didn't ACK) and 19
# (No such device - the device briefly vanished from the bus entirely). Both
# are typically a momentary bus/wiring glitch on a vibrating RC chassis -
# loose SDA/SCL/power wiring to the PCA9685, or motor/ESC electrical noise -
# not something the program should die over.
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
    """
    frame_bgr comes from picamera2 in "RGB888" format, which is actually
    BGR-ordered in memory (a picamera2 naming quirk - confirmed against
    its source). data_recorder.py saves training frames straight from
    that same array with no channel conversion, and the trainer
    (trainer/engine/train.py, on the training machine) reads them back
    with cv2.imread + the same BGR->RGB flip, so the model was trained on
    true-RGB tensors. Must stay pixel-for-pixel identical to that file's
    load_and_preprocess()/SteeringDataset (same RGB order, same
    cv2.resize/INTER_LINEAR, same normalize math) - if you change one,
    change the other.
    """
    frame_rgb = frame_bgr[:, :, ::-1]
    img = cv2.resize(frame_rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img  # HWC, RGB, normalized (batch dim added after frame-stacking)


def build_frame_stack(history, now, frame_stack_n=FRAME_STACK_N):
    """
    history is a deque of (timestamp, preprocessed_HWC_frame) tuples, newest
    last - history[-1] is always the current frame. Picks frame_stack_n
    frames spaced ~FRAME_STACK_GAP_SECONDS apart in wall-clock time (not by
    a fixed frame-count stride, since consecutive captures are only ~10ms
    apart at 100fps and would look nearly identical). See
    trainer/engine/train.py's _history_indices() for the matching
    training-side logic.

    Pads with the oldest frame currently in the buffer when there isn't
    enough history yet (e.g. right after startup) - the same fallback
    train.py uses when it hits a session boundary.
    """
    frames = [history[-1][1]]
    for k in range(1, frame_stack_n):
        target_ts = now - k * FRAME_STACK_GAP_SECONDS
        best = history[0][1]  # fallback: oldest available
        for ts, img in reversed(history):
            if ts <= target_ts:
                best = img
                break
        frames.append(best)
    return np.concatenate(frames, axis=-1)  # H, W, 3 * frame_stack_n


def quantize_input(img_float_nhwc, scale, zero_point):
    """Hailo's native input dtype is UINT8 - feeding FLOAT32 works but makes
    HailoRT do this same conversion per inference call (confirmed in
    hailort.log: a Quantization filter element on the input vstream).
    Doing it here once with the HEF's own scale/zero_point (from
    hef.get_input_vstream_infos()[0].quant_info) skips that."""
    return np.clip(np.round(img_float_nhwc / scale + zero_point), 0, 255).astype(np.uint8)


def steering_label_to_angle(label):
    """Inverse of data_recorder.py's steering_axis_to_label/to_angle pair -
    same relationship (a +1.0 label matches what a full-right stick would
    have produced there), so the model's output drives the servo exactly
    like the joystick did while recording."""
    label = max(-1.0, min(1.0, label))
    angle = SERVO_NEUTRAL_ANGLE - label * (SERVO_MAX_ANGLE - SERVO_NEUTRAL_ANGLE) + SERVO_OFFSET
    return int(max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle)))


def throttle_level_to_pulse(level):
    return ESC_NEUTRAL_US + level * THROTTLE_STEP_US


def _handle_sigterm(signum, frame):
    """See data_recorder.py's identical handler - without this, SIGTERM
    skips the `finally` block below and leaves the ESC at its last pulse."""
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
    ap.add_argument("--display", action="store_true",
                     help="Open a live cv2 preview window (costs a few ms/frame)")
    ap.add_argument("--raw-steering", action="store_true",
                     help="Bypass EMA smoothing and the deadzone - send the model's "
                          "raw per-frame prediction straight to the servo (for "
                          "testing/debugging what the model itself actually outputs)")
    args = ap.parse_args()
    show_display = args.display
    raw_steering = args.raw_steering

    signal.signal(signal.SIGTERM, _handle_sigterm)

    pca = None
    esc = None
    steering = None
    picam2 = None
    controller = None

    try:
        _relay_cmd("RELAY_ON")
        hef_path = find_hef_path()
        print(f"[Hailo] Model: {os.path.basename(hef_path)}")
        hef = HEF(hef_path)
        input_info = hef.get_input_vstream_infos()[0]
        input_name = input_info.name
        output_name = hef.get_output_vstream_infos()[0].name
        input_scale = input_info.quant_info.qp_scale
        input_zero_point = input_info.quant_info.qp_zp

        # The HEF's own input shape tells us whether this model was trained
        # classic (single frame, data_recorder.py --legacy - no temporal
        # input) or on a timestamped dataset (frame-stacked, the trainer's
        # default) - see trainer/engine/train.py's build_model(). Detecting
        # it here means this script doesn't need a separate flag to keep in
        # sync with whatever model happens to be dropped in this folder.
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
        logging.info("Camera started - 640x480 @ 120fps")

        with VDevice() as device:
            cfg_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(hef, cfg_params)[0]
            ng_params = network_group.create_params()
            in_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

            print("\n-----------------------------------------------------")
            print("Conducere autonoma pornita. Virajul vine din model.")
            print("D-pad SUS = mareste viteza, D-pad JOS = scade viteza (pana la 0).")
            if show_display:
                print("Apasa 'q' in fereastra pentru oprire.")
            else:
                print("Mod headless (fara fereastra). Ctrl+C pentru oprire.")
            print("-----------------------------------------------------")

            throttle_level = 0
            smooth_label = 0.0
            current_fps = 0.0
            t_prev = time.time()
            frame_counter = 0
            frames_since_report = 0
            last_report_time = time.time()
            frame_history = deque()  # (timestamp, preprocessed HWC frame)

            with network_group.activate(ng_params):
                with InferVStreams(network_group, in_vstream_params, out_vstream_params) as pipeline:
                    while True:
                        ready, _, _ = select.select([controller_fd], [], [], 0.001)

                        for _ in ready:
                            for event in controller.read():
                                if event.type == ecodes.EV_ABS and event.code == ecodes.ABS_HAT0Y:
                                    if event.value == -1:
                                        throttle_level = min(THROTTLE_MAX_LEVEL, throttle_level + 1)
                                    elif event.value == 1:
                                        throttle_level = max(0, throttle_level - 1)
                                elif event.type == ecodes.EV_KEY and event.value == 1:
                                    if event.code == ecodes.BTN_DPAD_UP:
                                        throttle_level = min(THROTTLE_MAX_LEVEL, throttle_level + 1)
                                    elif event.code == ecodes.BTN_DPAD_DOWN:
                                        throttle_level = max(0, throttle_level - 1)

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
                            # Keep only what build_frame_stack() could possibly need, plus
                            # a margin - bounds memory without needing a fixed-count maxlen
                            # (capture fps can vary a bit frame to frame).
                            cutoff = now - (frame_stack_n - 1) * FRAME_STACK_GAP_SECONDS - 0.5
                            while len(frame_history) > 1 and frame_history[0][0] < cutoff:
                                frame_history.popleft()
                            stack = build_frame_stack(frame_history, now, frame_stack_n)
                        else:
                            stack = img_float  # classic model - no temporal history needed
                        inp = quantize_input(stack[np.newaxis], input_scale, input_zero_point)
                        result = pipeline.infer({input_name: inp})
                        raw_label = float(np.array(result[output_name]).reshape(-1)[0])
                        smooth_label = SMOOTH_ALPHA * raw_label + (1 - SMOOTH_ALPHA) * smooth_label
                        if raw_steering:
                            steer_cmd = raw_label
                        else:
                            steer_cmd = 0.0 if abs(smooth_label) < STEERING_DEADZONE else smooth_label

                        steering.set_angle(steering_label_to_angle(steer_cmd))
                        esc.set_pulse_us(throttle_level_to_pulse(throttle_level))

                        t_now = time.time()
                        current_fps = 0.9 * current_fps + 0.1 / max(t_now - t_prev, 1e-9)
                        t_prev = t_now
                        frame_counter += 1
                        frames_since_report += 1

                        if show_display:
                            # cv2.imshow costs several ms/call (measured) - update the
                            # window less often than the control loop runs, so display
                            # doesn't cap steering/throttle responsiveness.
                            if frame_counter % DISPLAY_EVERY_N_FRAMES == 0:
                                pulse = throttle_level_to_pulse(throttle_level)
                                display = frame
                                cv2.putText(display, f"Steer: {steer_cmd:+.2f} (model {raw_label:+.2f} smoothed {smooth_label:+.2f})  Throttle: {throttle_level}/{THROTTLE_MAX_LEVEL} ({pulse}us)",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                                cv2.putText(display, f"FPS: {current_fps:.1f}", (10, display.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                                cv2.imshow("RoboPacerV2 - Autonomous Drive", display)

                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    raise KeyboardInterrupt
                        else:
                            if t_now - last_report_time >= 1.0:
                                avg_fps = frames_since_report / (t_now - last_report_time)
                                print(f"fps={avg_fps:.1f}  steer={steer_cmd:+.2f} (model {raw_label:+.2f} smoothed {smooth_label:+.2f})  "
                                      f"throttle={throttle_level}/{THROTTLE_MAX_LEVEL}")
                                frames_since_report = 0
                                last_report_time = t_now

    except (KeyboardInterrupt, ConnectionError) as e:
        if isinstance(e, ConnectionError):
            print(f"\n{e}")
        print("\nOprire...")
    except Exception as e:
        logging.exception("Eroare majora neasteptata")
        print(f"\nEroare majora neasteptata: {e}")
    finally:
        _relay_cmd("RELAY_OFF")
        if esc is not None:
            esc.neutral()
            time.sleep(0.1)
            esc.stop()
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
        logging.info("Autonomous drive stopped")
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
