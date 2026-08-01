"""
RoboPacerV2 - Servo Calibration Tool
======================================
Interactively jog the steering servo's PWM pulse width to find its real
MIN / CENTER / MAX pulses, in microseconds. Use this whenever the physical
servo changes (e.g. swapping to an MG958) instead of reusing the old
servo's calibration values blindly - different servo models/units don't
necessarily share the same pulse-to-angle mapping or mechanical center.

Controls:
    Up / w      increase pulse by <step>
    Down / s    decrease pulse by <step>
    Right / d   increase step size
    Left / a    decrease step size
    c           mark current pulse as CENTER (wheels visually straight)
    n           mark current pulse as MIN endpoint
    x           mark current pulse as MAX endpoint
    r           reset to the safe default (1500us)
    q           quit and print a summary + ready-to-paste constants

Safety: pulse is hard-clamped to 500-2500us the whole time, a range that
essentially every hobby RC servo can handle without straining against its
mechanical stop. Widen the endpoints slowly and stop marking MIN/MAX the
moment you hear/feel the servo buzzing against its end of travel.
"""

import sys
import termios
import tty

import board
import busio
from adafruit_pca9685 import PCA9685

SERVO_CHANNEL = 0  # must match SERVO_CHANNEL in data_recorder/data_recorder.py
PCA_FREQUENCY_HZ = 50

PULSE_HARD_MIN_US = 500
PULSE_HARD_MAX_US = 2500
DEFAULT_PULSE_US = 1500

STEP_SIZES_US = [1, 5, 10, 25, 50, 100]


def read_key():
    """Read one keypress, resolving arrow-key escape sequences."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3, "")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def pulse_to_duty_cycle(pulse_us, period_us):
    pulse_us = max(PULSE_HARD_MIN_US, min(PULSE_HARD_MAX_US, pulse_us))
    return int((pulse_us / period_us) * 0xFFFF)


def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = PCA_FREQUENCY_HZ
    channel = pca.channels[SERVO_CHANNEL]
    period_us = 1_000_000 / pca.frequency

    pulse = DEFAULT_PULSE_US
    step_index = 2  # STEP_SIZES_US[2] == 10us
    marks = {"min": None, "center": None, "max": None}

    def apply(p):
        channel.duty_cycle = pulse_to_duty_cycle(p, period_us)

    apply(pulse)

    print(__doc__)
    print("\nJogging from the safe default center (1500us). Use arrows/wsad, "
          "c/n/x to mark, q to quit.\n")

    try:
        while True:
            step = STEP_SIZES_US[step_index]
            marked = ", ".join(f"{k}={v}us" for k, v in marks.items() if v is not None) or "none yet"
            print(f"\rpulse={pulse:4d}us  step={step:3d}us  marked: {marked}          ", end="", flush=True)

            key = read_key()
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
            apply(pulse)

    finally:
        channel.duty_cycle = 0
        pca.deinit()
        print("\n\nServo released.\n")

        min_p, center_p, max_p = marks["min"], marks["center"], marks["max"]
        if min_p is not None and center_p is not None and max_p is not None and max_p > min_p:
            angle_at_center = (center_p - min_p) / (max_p - min_p) * 180
            offset = round(angle_at_center - 90)
            print("Paste into data_recorder/data_recorder.py:")
            print(f"    SERVO_MIN_PULSE = {min_p}")
            print(f"    SERVO_MAX_PULSE = {max_p}")
            print(f"    SERVO_OFFSET = {offset}   # SERVO_NEUTRAL_ANGLE stays 90")
        else:
            print(f"Not all of min/center/max were marked (min={min_p}, center={center_p}, max={max_p}).")
            print("Run again and press n / c / x at each position before quitting.")


if __name__ == "__main__":
    main()
