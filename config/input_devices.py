"""
Xbox controller lookup - byte-identical across data_recorder/data_recorder.py,
manual_drive/manual_drive.py, cruise_control/cruise_control.py,
main/main.py, and tools/servo_calibrate.py.
"""
from evdev import InputDevice, list_devices


def find_xbox_controller():
    for path in list_devices():
        dev = InputDevice(path)
        if "xbox" in dev.name.lower():
            return dev
    return None
