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
import tkinter
import tkinter.filedialog

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


@app.route("/api/compile", methods=["POST"])
def compile_only():
    """Recompile an already-trained model to HEF without retraining - for
    recovering from a Docker/DFC failure after training already finished."""
    global _running, _events, _stop_requested
    import train as train_module

    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        config = request.json or {}
        if not config.get("model_name"):
            return jsonify({"error": "model_name is required."}), 400
        _running = True
        _stop_requested = False
        _events = []

    def worker():
        global _running
        try:
            train_module.retry_compile(config, _push)
        except Exception:
            _push({"type": "log", "level": "error", "text": traceback.format_exc()})
            _push({"type": "done"})
        finally:
            _running = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/smoketest", methods=["POST"])
def smoketest():
    """0-epoch run through the full pipeline (ONNX export + Docker + Hailo
    compile) with an untrained model - takes seconds, not hours, to catch
    export/Docker/DFC problems. Never touches your real named models."""
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
            train_module.run_smoke_test(config, _push)
        except Exception:
            _push({"type": "log", "level": "error", "text": traceback.format_exc()})
            _push({"type": "done"})
        finally:
            _running = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/browse/folder", methods=["GET"])
def browse_folder():
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = tkinter.filedialog.askdirectory(title="Selecteaza folderul cu dataset")
    root.destroy()
    return jsonify({"path": path or ""})


@app.route("/api/browse/file", methods=["GET"])
def browse_file():
    ft = request.args.get("type", "pth")
    filetypes = {"pth": [("PyTorch checkpoint", "*.pth"), ("All", "*.*")],
                 "npy": [("NumPy array", "*.npy"), ("All", "*.*")]}
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = tkinter.filedialog.askopenfilename(title="Selecteaza fisierul", filetypes=filetypes.get(ft, [("All", "*.*")]))
    root.destroy()
    return jsonify({"path": path or ""})


@app.route("/api/compile_pth", methods=["POST"])
def compile_pth():
    global _running, _events, _stop_requested
    import train as train_module

    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        config = request.json or {}
        if not config.get("pth_path"):
            return jsonify({"error": "pth_path is required."}), 400
        _running = True
        _stop_requested = False
        _events = []

    def worker():
        global _running
        try:
            train_module.compile_from_pth(config, _push)
        except Exception:
            _push({"type": "log", "level": "error", "text": traceback.format_exc()})
            _push({"type": "done"})
        finally:
            _running = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/smoketest_pth", methods=["POST"])
def smoketest_pth():
    """Test compile cu model neantrenat - fara dataset necesar."""
    global _running, _events, _stop_requested
    import train as train_module

    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        _running = True
        _stop_requested = False
        _events = []

    def worker():
        global _running
        try:
            import torch, numpy as np
            from pathlib import Path
            push = _push
            MODELS_DIR.mkdir(exist_ok=True)
            name = "__smoketest_pth__"
            push({"type": "log", "level": "info", "text": "Generez model de test (fara antrenament)..."})
            device = torch.device("cpu")
            model = train_module.build_model(train_module.FRAME_STACK_N).to(device)
            onnx_path = MODELS_DIR / f"{name}.onnx"
            calib_path = MODELS_DIR / f"{name}_calib_data_nhwc.npy"
            train_module.export_onnx(model, device, onnx_path)
            calib = np.random.randn(train_module.CALIB_N, train_module.IMG_SIZE, train_module.IMG_SIZE, 9).astype("float32")
            np.save(str(calib_path), calib)
            push({"type": "log", "level": "info", "text": "ONNX exportat. Pornesc compilare Docker..."})
            if not train_module.check_compile_prereqs(push):
                push({"type": "done"}); return
            ok = train_module.compile_hef(name, push)
            for f in [onnx_path, calib_path,
                      MODELS_DIR / f"{name}.har",
                      MODELS_DIR / f"{name}_optimized.har",
                      MODELS_DIR / f"{name}.hef"]:
                try: Path(f).unlink()
                except: pass
            if ok:
                push({"type": "log", "level": "success", "text": "Test compile reusit!"})
            push({"type": "done"})
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
