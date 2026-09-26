"""
RoboPacerV2 - Camera view (test tool)
======================================
Live view of what the robot sees, opened with the exact camera settings the
robot uses (config/camera_config.py - make_camera(), the same call main.py
and data_recorder.py make). Shows side by side:

  DATA RECORDER  the 640x480 frame exactly as recording saves it to disk
                 (JPEG at SAVED_JPEG_QUALITY, decoded back) - what training
                 data looks like.
  MAIN           the 224x224 image main.py feeds the model, for each frame
                 of its stack: now, 0.1s ago, 0.2s ago (same resize and same
                 frame selection as main.py).

plus live camera FPS, exposure, gain and light level. Doesn't touch the motor,
servo or relay.

The camera can only be open in one program at a time - stop main /
data_recorder (dashboard Stop) before running this.

Usage:
    python3 tools/camera_view.py            window on the Pi's desktop (HDMI/VNC)
    python3 tools/camera_view.py --web      in a browser: http://<pi-ip>:8081
    python3 tools/camera_view.py --frames 1 single-frame model view

Keys in the window: q / Esc = quit, s = save a snapshot to tools/snapshots/.
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from config.camera_config import FRAME_SIZE, SAVED_JPEG_QUALITY, describe_settings, make_camera  # noqa: E402
from config.vision import select_stack_frames  # noqa: E402
from config.vision_config import FRAME_STACK_GAP_SECONDS, FRAME_STACK_N, MODEL_SIZE  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
WEB_PORT = 8081
FPS_WINDOW_SECONDS = 1.0

BG = (24, 24, 24)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
GREEN = (80, 220, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org, color=TEXT, scale=0.5):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def compose(saved_view, tiles, tile_names, info_lines):
    """One image: recorder view on top, main's model-input tiles below, text bar."""
    width = max(FRAME_SIZE[0], MODEL_SIZE * len(tiles))
    bar_h = 22 * len(info_lines) + 12
    canvas = np.full((bar_h + FRAME_SIZE[1] + 28 + MODEL_SIZE + 28, width, 3), BG, np.uint8)

    y = 22
    for text, color in info_lines:
        label(canvas, text, (10, y), color)
        y += 22

    top = bar_h
    x0 = (width - FRAME_SIZE[0]) // 2
    canvas[top:top + FRAME_SIZE[1], x0:x0 + FRAME_SIZE[0]] = saved_view
    label(canvas, f"DATA RECORDER - saved frame {FRAME_SIZE[0]}x{FRAME_SIZE[1]} "
                  f"(JPEG q{SAVED_JPEG_QUALITY}, full view)", (x0 + 8, top + 20), GREEN)

    top += FRAME_SIZE[1] + 28
    label(canvas, f"MAIN - model input {MODEL_SIZE}x{MODEL_SIZE}, {len(tiles)}-frame stack",
          (10, top - 8), GREEN)
    x0 = (width - MODEL_SIZE * len(tiles)) // 2
    for i, (tile, name) in enumerate(zip(tiles, tile_names)):
        x = x0 + i * MODEL_SIZE
        canvas[top:top + MODEL_SIZE, x:x + MODEL_SIZE] = tile
        label(canvas, name, (x + 6, top + MODEL_SIZE - 8))
    return canvas


class MjpegServer:
    """Latest composed view as an MJPEG stream on http://<pi>:WEB_PORT/."""

    def __init__(self, port):
        self._jpeg = None
        self._cond = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    page = (b"<html><body style='margin:0;background:#181818'>"
                            b"<img src='/stream' style='width:100%;max-width:720px;display:block;margin:auto'>"
                            b"</body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with server._cond:
                            server._cond.wait(timeout=2)
                            jpeg = server._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def publish(self, image):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def close(self):
        self._httpd.shutdown()


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description="Live view of the robot's camera, as recording and driving see it.")
    ap.add_argument("--web", action="store_true", help=f"serve the view on http://<pi-ip>:{WEB_PORT} instead of a window")
    ap.add_argument("--frames", type=int, default=FRAME_STACK_N, choices=(1, 2, 3, 4, 5),
                    help=f"frames in main's model stack (default {FRAME_STACK_N}, like the current .hef models)")
    args = ap.parse_args()

    if not args.web and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        sys.exit("No desktop display here (e.g. over SSH). Run it on the Pi's desktop, "
                 "use `DISPLAY=:0 python3 tools/camera_view.py`, or add --web and open it in a browser.")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        picam2 = make_camera()
    except (RuntimeError, IndexError) as e:
        sys.exit(f"Can't open the camera ({e}). Is main / data_recorder still running? Stop it first.")

    web = None
    try:
        picam2.start()
        cfg = picam2.camera_configuration()
        print("Settings (config/camera_config.py): " + describe_settings())
        print(f"Active configuration: sensor {cfg['sensor']}, main {cfg['main']['size']} {cfg['main']['format']}")
        if args.web:
            web = MjpegServer(WEB_PORT)
            print(f"Open http://<pi-ip>:{WEB_PORT} in a browser (robot hotspot or home WiFi). Ctrl+C to stop.")
        else:
            print("Window open - q / Esc to quit, s to save a snapshot.")

        history = deque()  # (time, 224x224 frame) - main.py's frame history, for the stack tiles
        tile_names = ["now"] + [f"-{k * FRAME_STACK_GAP_SECONDS:.1f}s" for k in range(1, args.frames)]
        window_start, window_frames, window_sensor_first = time.time(), 0, None
        loop_fps = camera_fps = 0.0
        last_seq = None
        skipped = 0

        while True:
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                md = request.get_metadata()
                seq = request.request.sequence
            finally:
                request.release()
            now = time.time()

            # Camera FPS from the sensor's own timestamps; skipped = frames the
            # camera made that this viewer didn't get to (display is slower
            # than the robot's loops - it doesn't change the camera itself).
            if last_seq is not None:
                skipped += max(0, seq - last_seq - 1)
            last_seq = seq
            if window_sensor_first is None:
                window_sensor_first = (md["SensorTimestamp"], seq)
            window_frames += 1
            if now - window_start >= FPS_WINDOW_SECONDS:
                loop_fps = window_frames / (now - window_start)
                ts0, seq0 = window_sensor_first
                if md["SensorTimestamp"] > ts0:
                    camera_fps = (seq - seq0) / ((md["SensorTimestamp"] - ts0) / 1e9)
                window_start, window_frames, window_sensor_first = now, 0, None

            # DATA RECORDER: exactly what goes to disk - JPEG round trip.
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SAVED_JPEG_QUALITY])
            saved_view = cv2.imdecode(jpeg, cv2.IMREAD_COLOR) if ok else frame

            # MAIN: the same resize main.py does (config/vision.py), and the same
            # stack frame selection. The model gets these pixels normalized.
            small = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
            history.append((now, small))
            cutoff = now - (args.frames - 1) * FRAME_STACK_GAP_SECONDS - 0.5
            while len(history) > 1 and history[0][0] < cutoff:
                history.popleft()
            tiles = select_stack_frames(history, now, args.frames)

            exposure_ms = md.get("ExposureTime", 0) / 1000
            frame_ms = md.get("FrameDuration", 0) / 1000
            info = [
                (f"camera {camera_fps:5.1f} FPS (max {1000 / frame_ms if frame_ms else 0:.1f})  |  "
                 f"view {loop_fps:5.1f} FPS  |  skipped by view: {skipped}", TEXT),
                (f"exposure {exposure_ms:.1f} ms  |  gain {md.get('AnalogueGain', 0):.2f} "
                 f"x {md.get('DigitalGain', 1):.2f} digital  |  light {md.get('Lux', 0):.1f} lux", TEXT),
                ("settings: config/camera_config.py - same as main.py and data_recorder.py", DIM),
            ]
            view = compose(saved_view, tiles, tile_names, info)

            if web is not None:
                web.publish(view)
                continue
            cv2.imshow("RoboPacerV2 - camera view", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(SNAPSHOT_DIR, time.strftime("view_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, view)
                print(f"Snapshot: {path}")
    except KeyboardInterrupt:
        pass
    finally:
        if web is not None:
            web.close()
        if getattr(picam2, "started", False):
            picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()
