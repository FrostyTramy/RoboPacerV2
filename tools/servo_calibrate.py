"""
RoboPacerV2 - Servo Calibration Tool (+ drive test)
======================================================
Interactively jog the steering servo's PWM pulse width to find its real
MIN / CENTER / MAX pulses, in microseconds, AND arms the ESC so you can
actually drive forward with the Xbox controller's gas/brake triggers
while adjusting the center trim - static "wheels look straight" isn't
the same as "goes straight while moving" (torque steer, alignment,
suspension slop only show up in motion). Use this whenever the physical
servo changes (e.g. swapping to an MG958) instead of reusing the old
servo's calibration values blindly - different servo models/units don't
necessarily share the same pulse-to-angle mapping or mechanical center.

Controls:
    Up / w      increase servo pulse (steering trim) by <step>
    Down / s    decrease servo pulse (steering trim) by <step>
    Right / d   increase step size
    Left / a    decrease step size
    c           mark current pulse as CENTER (wheels visually straight)
    n           mark current pulse as MIN endpoint
    x           mark current pulse as MAX endpoint
    r           reset steering to the safe default (1500us)
    q           quit and print a summary + ready-to-paste constants

    Xbox right trigger (ABS_GAS)    throttle forward
    Xbox left trigger (ABS_BRAKE)   brake/reverse

Safety: steering pulse is hard-clamped to 500-2500us the whole time, a
range that essentially every hobby RC servo can handle without straining
against its mechanical stop. Widen the endpoints slowly and stop marking
MIN/MAX the moment you hear/feel the servo buzzing against its end of
travel. The ESC only arms if an Xbox controller is found - without one
this still works as a servo-only calibration tool, no throttle.

IMPORTANT: this script's full path must be in ALLOWED_SCRIPTS in
safety/esc_watchdog.sh (it already is) - launch it with:
    python3 /home/pi/RoboPacerV2/tools/servo_calibrate.py
A relative path won't match, and the watchdog will fight it.
"""

import os
import select
import signal
import sys
import termios
import time
import tty

import board
import busio
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, list_devices

SERVO_CHANNEL = 0  # must match SERVO_CHANNEL in data_recorder/data_recorder.py
ESC_CHANNEL = 1  # must match ESC_CHANNEL in data_recorder/data_recorder.py
PCA_FREQUENCY_HZ = 50

PULSE_HARD_MIN_US = 500
PULSE_HARD_MAX_US = 2500
DEFAULT_PULSE_US = 1500

STEP_SIZES_US = [1, 5, 10, 25, 50, 100]

ESC_ARM_PULSE_US = 1000
ESC_ARM_HOLD_SECONDS = 3.0
ESC_NEUTRAL_US = 1500
ESC_MIN_US = 1000
ESC_MAX_US = 1700


def find_xbox_controller():
    for path in list_devices():
        dev = InputDevice(path)
        if "xbox" in dev.name.lower():
            return dev
    return None


def pulse_to_duty_cycle(pulse_us, period_us, lo, hi):
    pulse_us = max(lo, min(hi, pulse_us))
    return int((pulse_us / period_us) * 0xFFFF)


def esc_pulse_from_gas_brake(gas_value, brake_value):
    """Same mapping as data_recorder.py's ESC.pulse_from_gas_brake."""
    gas_offset = (gas_value / 1023.0) * (ESC_MAX_US - ESC_NEUTRAL_US)
    brake_offset = -(brake_value / 1023.0) * (ESC_NEUTRAL_US - ESC_MIN_US)
    pulse = ESC_NEUTRAL_US + gas_offset + brake_offset
    return max(ESC_MIN_US, min(ESC_MAX_US, int(pulse)))


def _handle_sigterm(signum, frame):
    """See data_recorder.py's identical handler - without this, SIGTERM
    skips the `finally` block below and leaves the ESC at its last pulse."""
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    stdin_fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = PCA_FREQUENCY_HZ
    servo_channel = pca.channels[SERVO_CHANNEL]
    esc_channel = pca.channels[ESC_CHANNEL]
    period_us = 1_000_000 / pca.frequency

    def apply_servo(p):
        servo_channel.duty_cycle = pulse_to_duty_cycle(p, period_us, PULSE_HARD_MIN_US, PULSE_HARD_MAX_US)

    def apply_esc(p):
        esc_channel.duty_cycle = pulse_to_duty_cycle(p, period_us, ESC_MIN_US, ESC_MAX_US)

    pulse = DEFAULT_PULSE_US
    step_index = 2  # STEP_SIZES_US[2] == 10us
    marks = {"min": None, "center": None, "max": None}
    apply_servo(pulse)

    controller = find_xbox_controller()
    controller_fd = None
    gas_value = 0
    brake_value = 0

    try:
        print(__doc__)

        if controller is None:
            print("\nControllerul Xbox nu a fost gasit - doar calibrare servo, fara accelerare.\n")
        else:
            controller_fd = controller.fd
            print(f"\nArmare ESC: puls {ESC_ARM_PULSE_US}us timp de {ESC_ARM_HOLD_SECONDS:.0f}s...")
            apply_esc(ESC_ARM_PULSE_US)
            time.sleep(ESC_ARM_HOLD_SECONDS)
            apply_esc(ESC_NEUTRAL_US)
            print("ESC armat - poti da gaz/frana ca sa conduci in fata cat calibrezi.\n")

        print("Jogging from the safe default center (1500us). Use arrows/wsad, "
              "c/n/x to mark, q to quit.\n")

        while True:
            step = STEP_SIZES_US[step_index]
            marked = ", ".join(f"{k}={v}us" for k, v in marks.items() if v is not None) or "none yet"
            print(f"\rservo={pulse:4d}us  step={step:3d}us  marked: {marked}  gas={gas_value:4d}          ",
                  end="", flush=True)

            wait_fds = [stdin_fd] + ([controller_fd] if controller_fd is not None else [])
            ready, _, _ = select.select(wait_fds, [], [], 0.05)

            if controller_fd is not None and controller_fd in ready:
                for event in controller.read():
                    if event.type == ecodes.EV_ABS:
                        abs_name = ecodes.ABS.get(event.code)
                        if abs_name == "ABS_GAS":
                            gas_value = event.value
                        elif abs_name == "ABS_BRAKE":
                            brake_value = event.value
                        if abs_name in ("ABS_GAS", "ABS_BRAKE"):
                            apply_esc(esc_pulse_from_gas_brake(gas_value, brake_value))

            if stdin_fd in ready:
                ch = os.read(stdin_fd, 1).decode(errors="ignore")
                key = ch
                if ch == "\x1b":
                    ready2, _, _ = select.select([stdin_fd], [], [], 0.01)
                    if ready2:
                        seq = os.read(stdin_fd, 2).decode(errors="ignore")
                        key = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "")
                    else:
                        key = ""

                if key in ("up", "w"):
                    pulse += step
                elif key in ("down", "s"):
                    pulse -= step
                elif key in ("right", "d"):
                    step_index = min(step_index + 1, len(STEP_SIZES_US) - 1)
                elif key in ("left", "a"):
                    step_index = max(step_index - 1, 0)
                elif key == "c":
                    marks["center"] = pulse
                elif key == "n":
                    marks["min"] = pulse
                elif key == "x":
                    marks["max"] = pulse
                elif key == "r":
                    pulse = DEFAULT_PULSE_US
                elif key == "q":
                    break
                else:
                    continue

                pulse = max(PULSE_HARD_MIN_US, min(PULSE_HARD_MAX_US, pulse))
                apply_servo(pulse)

    except KeyboardInterrupt:
        pass
    finally:
        apply_esc(ESC_NEUTRAL_US)
        time.sleep(0.1)
        esc_channel.duty_cycle = 0
        servo_channel.duty_cycle = 0
        pca.deinit()
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        print("\n\nServo si ESC oprite.\n")

        min_p, center_p, max_p = marks["min"], marks["center"], marks["max"]
        if min_p is not None and center_p is not None and max_p is not None and max_p > min_p:
            angle_at_center = (center_p - min_p) / (max_p - min_p) * 180
            offset = round(angle_at_center - 90)
            print("Paste into data_recorder/data_recorder.py AND main/main.py (both must match):")
            print(f"    SERVO_MIN_PULSE = {min_p}")
            print(f"    SERVO_MAX_PULSE = {max_p}")
            print(f"    SERVO_OFFSET = {offset}   # SERVO_NEUTRAL_ANGLE stays 90")
        else:
            print(f"Not all of min/center/max were marked (min={min_p}, center={center_p}, max={max_p}).")
            print("Run again and press n / c / x at each position before quitting.")


if __name__ == "__main__":
    main()
