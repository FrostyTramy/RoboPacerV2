"""
Single source of truth for the camera setup.

camera/camera.py, data_recorder/data_recorder.py, main/main.py,
model_runner/model_runner.py and tools/camera_calibrate.py all build their
camera through make_camera() below, so recording, calibration and driving
always see exactly the same image. Change a value here and it changes
everywhere - do not re-declare any of these in the individual scripts.
"""
from picamera2 import Picamera2

TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"

# Output frame handed to the scripts (and saved / fed to the model as-is).
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"  # BGR-ordered in memory - picamera2 naming quirk

# Full-FOV sensor mode (2x2 binned, no crop); the ISP downscales it to
# FRAME_SIZE. Without this, picamera2 picks the 640x480 sensor mode, which is
# a zoomed centre crop. 8-bit = 83.7 fps max (10-bit is 41.85 fps).
SENSOR_MODE = {"output_size": (1640, 1232), "bit_depth": 8}

# Requested rate; the sensor mode above caps it at ~83.7 fps.
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0  # 16.0 * 0.75 - reduced 25% for outdoor daylight use


def make_camera(extra_controls=None):
    """Return a configured (not yet started) Picamera2.

    extra_controls: extra libcamera controls merged on top of the shared ones
    (e.g. tools/camera_calibrate.py adds AwbEnable).
    """
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
