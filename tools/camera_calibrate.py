"""
RoboPacerV2 - Camera White Balance Calibration Tool
======================================================
Live preview with the camera's current auto white-balance gains overlaid,
plus a manual purple/magenta reduction filter for the leftover cast (the
standard IMX219 tuning still doesn't fully neutralize it in some lighting).
Point at a neutral white/grey card (plain white paper works) in the actual
lighting you'll be recording in.

Controls:
    x           toggle LOCK - freezes AwbEnable/ColourGains at the
                currently displayed values (auto-AWB stops adjusting), so
                you can pan the camera around and check the locked
                setting looks right elsewhere in the scene. Press x again
                to unlock and resume auto.
    Up / w      reduce purple (subtracts more from red/blue vs. green)
    Down / s    add purple back (subtracts less, or goes negative to add)
    q           quit - prints last-seen gains + purple offset as
                ready-to-paste constants.

Why the purple filter: the purple/pink cast on this sensor comes from
near-infrared light adding onto the red (and to a lesser extent blue)
channel - an additive contamination, not a white-balance gain mismatch,
so AWB/ColourGains alone can't fully remove it (confirmed - even black
areas showed the tint). This filter subtracts a flat, adjustable amount
from red and blue instead, which matches how the error actually behaves.

Why lock white balance at all: camera.py currently runs with
AwbEnable=True (continuous auto). Auto-AWB re-adjusts based on whatever's
in frame - a scene dominated by one color (e.g. grass) drags the whole
image's color balance around scene-to-scene, which is bad for consistent
training data. Locking ColourGains to a fixed value measured against a
neutral reference removes that inconsistency.

Run this in the same lighting (e.g. outdoors in sun) you'll actually be
recording in - indoor-calibrated settings will look wrong outdoors and
vice versa.
"""

import cv2
import numpy as np

from picamera2 import Picamera2

TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219.json"
FRAME_SIZE = (640, 480)
FRAME_FORMAT = "RGB888"
FRAME_RATE = 120.0
ANALOGUE_GAIN = 12.0  # match data_recorder.py's current daylight value

PURPLE_STEP = 5
PURPLE_OFFSET_LIMIT = 80

# cv2.waitKeyEx() arrow-key codes vary by backend (GTK/Qt/Windows) - cover
# the common ones instead of guessing one.
UP_KEYS = {82, 65362, 2490368, 63232}
DOWN_KEYS = {84, 65364, 2621440, 63233}


def apply_purple_reduction(frame, offset):
    """frame is BGR-ordered (picamera2's 'RGB888' format is BGR in memory).
    Moves energy out of red/blue and into green instead of just subtracting
    it (which would only darken the image) - this keeps overall brightness
    roughly constant while shifting the cast away from magenta/purple."""
    if offset == 0:
        return frame
    adjusted = frame.astype(np.int16)
    adjusted[:, :, 2] -= offset       # red
    adjusted[:, :, 0] -= offset // 2  # blue
    adjusted[:, :, 1] += offset       # green - compensates the brightness loss
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def main():
    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(tuning=tuning)
    config = picam2.create_video_configuration(
        main={"size": FRAME_SIZE, "format": FRAME_FORMAT},
        controls={"FrameRate": FRAME_RATE, "AnalogueGain": ANALOGUE_GAIN, "AwbEnable": True},
    )
    picam2.configure(config)
    picam2.start()

    print(__doc__)

    locked = False
    last_gains = None
    purple_offset = 0

    try:
        while True:
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()

            gains = metadata.get("ColourGains")
            lux = metadata.get("Lux")
            if gains:
                last_gains = gains

            display = apply_purple_reduction(frame, purple_offset)

            status = "LOCKED" if locked else "LIVE"
            color = (0, 255, 255) if locked else (0, 255, 0)
            gain_text = f"red={last_gains[0]:.3f} blue={last_gains[1]:.3f}" if last_gains else "no reading yet"
            cv2.putText(display, f"{status}  {gain_text}  lux={lux}  purple_offset={purple_offset}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(display, "Up/Down = purple filter   X = lock/unlock AWB   Q = quit",
                        (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("RoboPacerV2 - Camera Calibration", display)

            key = cv2.waitKeyEx(1)
            if key == ord("x"):
                locked = not locked
                if locked and last_gains:
                    picam2.set_controls({
                        "AwbEnable": False,
                        "ColourGains": (float(last_gains[0]), float(last_gains[1])),
                    })
                    print(f"\nLocked: red={last_gains[0]:.3f}  blue={last_gains[1]:.3f}")
                else:
                    picam2.set_controls({"AwbEnable": True})
                    print("\nUnlocked - AWB auto again.")
            elif key in UP_KEYS or key == ord("w"):
                purple_offset = min(PURPLE_OFFSET_LIMIT, purple_offset + PURPLE_STEP)
                print(f"\rpurple_offset={purple_offset}          ", end="", flush=True)
            elif key in DOWN_KEYS or key == ord("s"):
                purple_offset = max(-PURPLE_OFFSET_LIMIT, purple_offset - PURPLE_STEP)
                print(f"\rpurple_offset={purple_offset}          ", end="", flush=True)
            elif key == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("\n\nCamera stopped.\n")

        if last_gains:
            r, b = last_gains
            print("Paste into camera.py controls dict:")
            print('    "AwbEnable": False,')
            print(f'    "ColourGains": ({r:.3f}, {b:.3f}),')
        else:
            print("No gains captured - run again and wait for a reading before quitting.")
        print(f"purple_offset = {purple_offset}   # subtract from red, offset//2 from blue")


if __name__ == "__main__":
    main()
