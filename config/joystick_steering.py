"""
Xbox left-stick (ABS_X) steering conversion - shared by
manual_drive/manual_drive.py and data_recorder/data_recorder.py. ABS_X
reports 0..AXIS_MAX; a small deadzone around center (JOYSTICK_DEADZONE in
config/hardware_config.py) absorbs stick noise before the rest of the travel
maps linearly to a steering label (-1..1, used for recorded training labels)
and from there to a servo angle through the same steering_label_to_angle()
the model uses - so a label means the same wheel angle whether the stick or
the model produced it.
"""
from config.hardware_config import JOYSTICK_DEADZONE
from config.servo_esc import steering_label_to_angle

AXIS_CENTER = 32767
AXIS_MAX = 65535


def steering_axis_to_label(x_value):
    raw = -1.0 + 2.0 * (x_value / AXIS_MAX)
    if abs(raw) <= JOYSTICK_DEADZONE:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    label = sign * (abs(raw) - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
    return round(float(label), 2)


def steering_axis_to_angle(x_value):
    return steering_label_to_angle(steering_axis_to_label(x_value))
