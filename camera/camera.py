import os
import time
import logging
import cv2
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TUNING_FILE = "/usr/share/libcamera/ipa/rpi/pisp/imx219.json"
RECORDS_DIR = os.path.join(BASE_DIR, "records")
LOG_FILE = os.path.join(BASE_DIR, "camera.log")

os.makedirs(RECORDS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("picamera2", "libcamera", "PIL"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

tuning = Picamera2.load_tuning_file(TUNING_FILE)
picam2 = Picamera2(tuning=tuning)

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 120.0},
)
picam2.configure(config)
picam2.start()
logging.info("Camera started — 640x480 @ 120fps, auto exposure/gain/AWB")

encoder = H264Encoder(bitrate=8_000_000, repeat=True, iperiod=60)

fps_history = []
capture_times = []
rec_fps_samples = []
prev_time = time.time()
recording = False
current_file = None
last_saved = None
rec_start_time = None

while True:
    t0 = time.perf_counter()
    frame = picam2.capture_array()
    capture_ms = (time.perf_counter() - t0) * 1000

    now = time.time()
    fps_history.append(1.0 / max(now - prev_time, 1e-6))
    if len(fps_history) > 60:
        fps_history.pop(0)
    prev_time = now
    fps = sum(fps_history) / len(fps_history)

    capture_times.append(capture_ms)
    if len(capture_times) > 60:
        capture_times.pop(0)
    avg_capture_ms = sum(capture_times) / len(capture_times)

    if recording:
        rec_fps_samples.append(fps)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"capture: {avg_capture_ms:.1f}ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    if recording:
        elapsed = time.time() - rec_start_time
        cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (570, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"{elapsed:.1f}s", (570, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, os.path.basename(current_file), (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    else:
        if last_saved:
            cv2.putText(frame, f"Last: {os.path.basename(last_saved)}", (10, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, "R=record  S=save  Q=quit", (10, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("RoboPacerV2", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        if not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_file = os.path.join(RECORDS_DIR, f"rec_{timestamp}.mp4")
            picam2.start_encoder(encoder, FfmpegOutput(current_file, audio=False))
            recording = True
            rec_start_time = time.time()
            rec_fps_samples = []
            logging.info(f"REC START  {os.path.basename(current_file)}")
            print(f"Recording started → {current_file}")
    elif key == ord("s"):
        if recording:
            picam2.stop_encoder()
            recording = False
            duration = time.time() - rec_start_time
            size_mb = os.path.getsize(current_file) / 1_000_000 if os.path.exists(current_file) else 0
            avg_fps = sum(rec_fps_samples) / len(rec_fps_samples) if rec_fps_samples else 0
            logging.info(
                f"REC SAVE   {os.path.basename(current_file)} | "
                f"duration={duration:.1f}s | size={size_mb:.1f}MB | avg_fps={avg_fps:.1f}"
            )
            last_saved = current_file
            print(f"Saved → {current_file}")
            current_file = None
            rec_start_time = None

if recording:
    picam2.stop_encoder()

logging.info("Camera stopped")
picam2.stop()
cv2.destroyAllWindows()
