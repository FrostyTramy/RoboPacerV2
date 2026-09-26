"""
SteeringServo + ESC control classes - byte-identical across main/main.py,
manual_drive/manual_drive.py, cruise_control/cruise_control.py,
data_recorder/data_recorder.py, and model_runner/model_runner.py.
pulse_from_gas_brake() was only present on manual_drive.py/data_recorder.py's
copy of ESC (main.py drives the ESC via cruise_pi.py instead) but is kept
here as part of the one shared ESC class.
"""
import logging
import time

from adafruit_motor import servo as adafruit_servo

from config.hardware_config import (
    ESC_ARM_HOLD_SECONDS,
    ESC_ARM_PULSE_US,
    ESC_CHANNEL,
    ESC_MAX_US,
    ESC_MIN_US,
    ESC_NEUTRAL_US,
    SERVO_CHANNEL,
    SERVO_MAX_ANGLE,
    SERVO_MIN_ANGLE,
    SERVO_MIN_PULSE,
    SERVO_MAX_PULSE,
    SERVO_STRAIGHT_ANGLE,
    TRANSIENT_I2C_ERRNOS,
)


class SteeringServo:
    def __init__(self, pca):
        self._servo = adafruit_servo.Servo(
            pca.channels[SERVO_CHANNEL],
            min_pulse=SERVO_MIN_PULSE,
            max_pulse=SERVO_MAX_PULSE,
        )
        self.angle = SERVO_STRAIGHT_ANGLE
        self.center()

    def set_angle(self, angle):
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        self.angle = angle
        try:
            self._servo.angle = angle
        except OSError as e:
            if e.errno in TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting servo angle {angle}: {e}")
            else:
                raise

    def center(self):
        self.set_angle(SERVO_STRAIGHT_ANGLE)

    def release(self):
        try:
            self._servo.angle = None
        except OSError as e:
            if e.errno not in TRANSIENT_I2C_ERRNOS:
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
            if e.errno in TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting ESC pulse {pulse_us}: {e}")
            else:
                raise

    def neutral(self):
        self.set_pulse_us(ESC_NEUTRAL_US)

    def stop(self):
        try:
            self._channel.duty_cycle = 0
        except OSError as e:
            if e.errno not in TRANSIENT_I2C_ERRNOS:
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


def steering_label_to_angle(label):
    """-1..1 label -> servo angle, scaled per side so +1 lands exactly on
    SERVO_MIN_ANGLE and -1 on SERVO_MAX_ANGLE (no clamped dead travel on
    the shorter side). Shared by the model (main.py) and the joystick."""
    label = max(-1.0, min(1.0, label))
    if label >= 0:
        angle = SERVO_STRAIGHT_ANGLE - label * (SERVO_STRAIGHT_ANGLE - SERVO_MIN_ANGLE)
    else:
        angle = SERVO_STRAIGHT_ANGLE - label * (SERVO_MAX_ANGLE - SERVO_STRAIGHT_ANGLE)
    return int(round(angle))
