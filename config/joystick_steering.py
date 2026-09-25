"""
Xbox left-stick (ABS_X) steering conversion - byte-identical across
manual_drive/manual_drive.py, data_recorder/data_recorder.py, and
cruise_control/cruise_control.py on `main`. ABS_X reports 0..AXIS_MAX; a
small deadzone around center absorbs stick noise before the rest of the
travel maps linearly to a steering label (-1..1, used for recorded
training labels) or directly to a servo angle (manual driving).
"""
from config.hardware_config import SERVO_MAX_ANGLE, SERVO_MIN_ANGLE, SERVO_NEUTRAL_ANGLE, SERVO_OFFSET

AXIS_CENTER = 32767
AXIS_MAX = 65535
LABEL_DEADZONE = 0.2


def steering_axis_to_label(x_value):
    raw = -1.0 + 2.0 * (x_value / AXIS_MAX)
    if abs(raw) <= LABEL_DEADZONE:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    label = sign * (abs(raw) - LABEL_DEADZONE) / (1.0 - LABEL_DEADZONE)
    return round(float(label), 2)


def steering_axis_to_angle(x_value):
    raw = 1.0 - 2.0 * (x_value / AXIS_MAX)
    if abs(raw) <= LABEL_DEADZONE:
        norm = 0.0
    else:
        sign = 1.0 if raw > 0 else -1.0
        norm = sign * (abs(raw) - LABEL_DEADZONE) / (1.0 - LABEL_DEADZONE)
    angle = SERVO_NEUTRAL_ANGLE + norm * (SERVO_MAX_ANGLE - SERVO_NEUTRAL_ANGLE) + SERVO_OFFSET
    return int(max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle)))
