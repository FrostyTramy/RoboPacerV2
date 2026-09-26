"""
THE camera settings file - the only place any camera setting lives.

main/main.py (driving), data_recorder/data_recorder.py (recording training
data) and tools/camera_view.py (live preview) all open the camera with
make_camera() below and never change a setting themselves - so the robot
always sees the road exactly the way its training data was recorded.
Change a value here and every one of them changes together.

Values are the ones every dataset so far was recorded with - changing one
changes what the model sees compared to its training data, so re-record
(and retrain) after changing anything below.
"""
from picamera2 import Picamera2

# Camera tuning (colour/lens profile) for the IMX219 NoIR (no IR filter) module.
TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"

# Sensor mode: the full 3280x2464 sensor area, 2x2 binned to 1640x1232, 8-bit.
# Full field of view - nothing is cropped. This mode's maximum is 83.7 fps.
SENSOR_MODE = {"output_size": (1640, 1232), "bit_depth": 8}

# Frame handed to the scripts: the whole sensor view scaled to 640x480.
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"  # picamera2 naming quirk: the array is BGR in memory (what cv2 expects)

# Requested 120 fps; the sensor mode above caps it at 83.7 fps (11.95 ms per
# frame) - that's the real rate, in recordings and while driving alike.
FRAME_RATE = 120.0

# Requested gain 12; the IMX219 caps analogue gain at 10.67. Exposure time is
# automatic (auto-exposure), limited to one frame time (~11.9 ms).
ANALOGUE_GAIN = 12.0

# data_recorder saves each frame as a FRAME_SIZE JPEG at this quality (95 is
# also OpenCV's default, which older datasets were saved with).
SAVED_JPEG_QUALITY = 95


def make_camera():
    """Configured (not started) Picamera2 - call .start() on it."""
    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(tuning=tuning)
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT},
        sensor=SENSOR_MODE,
        controls={"FrameRate": FRAME_RATE, "AnalogueGain": ANALOGUE_GAIN},
    )
    picam2.configure(config)
    return picam2


def describe_settings():
    """One-line summary of the settings above, for startup logs."""
    return (f"camera {FRAME_SIZE[0]}x{FRAME_SIZE[1]} {FRAME_FORMAT} from sensor mode "
            f"{SENSOR_MODE['output_size'][0]}x{SENSOR_MODE['output_size'][1]} "
            f"{SENSOR_MODE['bit_depth']}-bit (full view), FrameRate {FRAME_RATE:g}, "
            f"AnalogueGain {ANALOGUE_GAIN:g}, tuning {TUNING_FILE.rsplit('/', 1)[-1]}")
