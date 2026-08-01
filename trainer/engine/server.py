"""
RoboPacerV2 Trainer - web server.
Run: python engine/server.py
Then open: http://localhost:5000
"""
import json
import sys
import threading
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
ENGINE_DIR = Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))

app = Flask(__name__)

_lock = threading.Lock()
_running = False
_events = []
_stop_requested = False


def _push(event):
    _events.append(event)


def _should_stop():
    return _stop_requested


@app.route("/")
def index():
    return send_file(ROOT / "index.html")


@app.route("/api/status")
def status():
    MODELS_DIR.mkdir(exist_ok=True)
    hefs = sorted(f.name for f in MODELS_DIR.glob("*.hef"))
    return jsonify({"ok": True, "running": _running, "models": hefs})


@app.route("/api/train", methods=["POST"])
def train():
    global _running, _events, _stop_requested
    import train as train_module

    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        config = request.json or {}
        if not config.get("json_path"):
            return jsonify({"error": "json_path is required."}), 400
        _running = True
        _stop_requested = False
        _events = []

    def worker():
        global _running
        try:
            train_module.run(config, _push, _should_stop)
        except Exception:
            _push({"type": "log", "level": "error", "text": traceback.format_exc()})
            _push({"type": "done"})
        finally:
            _running = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop():
    global _stop_requested
    _stop_requested = True
    return jsonify({"status": "stopping"})


@app.route("/api/stream")
def stream():
    def generate():
        last = 0
        waited = 0
        while not _events and waited < 60:
            threading.Event().wait(0.2)
            waited += 0.2
        while True:
            chunk = _events[last:]
            done = False
            for ev in chunk:
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") == "done":
                    done = True
            last += len(chunk)
            if done:
                break
            threading.Event().wait(0.1)

    return Response(generate(), mimetype="text/event-stream",
                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    print("=" * 50)
    print("  RoboPacerV2 Trainer  ->  http://localhost:5000")
    print("  Ctrl+C to stop.")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
