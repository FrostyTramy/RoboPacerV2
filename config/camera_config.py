"""
Single source of truth for the camera setup. Every script that opens the
camera (main, manual_drive, cruise_control, data_recorder, model_runner)
must call make_camera() from here instead of re-declaring these settings -
see main's CAMERA_SPEC.md for the full rationale behind each value.
"""
from picamera2 import Picamera2

TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"
SENSOR_MODE = {"output_size": (1640, 1232), "bit_depth": 8}
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0


def make_camera(extra_controls=None):
    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(tuning=tuning)
    controls = {"FrameRate": FRAME_RATE, "AnalogueGain": ANALOGUE_GAIN}
    if extra_controls:
        controls.update(extra_controls)
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT},
        sensor=SENSOR_MODE,
        controls=controls,
    )
    picam2.configure(config)
    return picam2
