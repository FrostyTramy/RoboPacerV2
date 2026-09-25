"""
RoboPacerV2 Trainer - web server (Windows only).
Run: engine/start.bat (or `python engine/server.py` directly)
Then open: http://localhost:5000

Thin orchestration layer over train_core.py (pure training engine) and
compile_pipeline.py (Docker/Hailo compile) - this file owns Flask routing,
job/thread bookkeeping, and the "Full" action's train-then-compile chaining;
it contains no training or compile logic of its own.
"""
import json
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
import tkinter
import tkinter.filedialog

ENGINE_DIR = Path(__file__).parent
ROOT_DIR = ENGINE_DIR.parent
sys.path.insert(0, str(ENGINE_DIR))

import defaults

app = Flask(__name__, static_folder=str(ENGINE_DIR / "static"), static_url_path="")

_lock = threading.Lock()
_running = False
_events = []
_stop_requested = False
_current_job = None  # {"action": str, "model_name": str, "started_at": float} while a job runs


def _push(event):
    _events.append(event)


def _should_stop():
    return _stop_requested


def _start_job(action, job_fn, config, required_fields=()):
    """Handles the lock check, required-field validation, event reset,
    thread spawn, and the top-level exception->_push fallback shared by
    every job-launching route - see job_fn for what actually runs."""
    global _running, _events, _stop_requested, _current_job

    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        missing = [f for f in required_fields if not str(config.get(f) or "").strip()]
        if missing:
            return jsonify({"error": f"Required field(s) missing: {', '.join(missing)}"}), 400
        _running = True
        _stop_requested = False
        _events = []
        _current_job = {"action": action, "model_name": config.get("model_name", ""), "started_at": time.time()}

    def worker():
        global _running, _current_job
        try:
            job_fn()
        except Exception:
            _push({"type": "log", "level": "error", "text": traceback.format_exc()})
            _push({"type": "done"})
        finally:
            _running = False
            _current_job = None

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"}), 200


# ── Support routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    import train_core
    train_core.MODELS_DIR.mkdir(exist_ok=True)
    hefs = sorted(f.name for f in train_core.MODELS_DIR.glob("*.hef"))
    return jsonify({"ok": True, "running": _running, "models": hefs, "current_job": _current_job})


@app.route("/api/defaults")
def get_defaults():
    return jsonify(defaults.CORE_RECIPE_DEFAULTS)


@app.route("/api/validate")
def validate():
    kind = request.args.get("kind", "dataset")
    path = request.args.get("path", "")
    if not path:
        return jsonify({"ok": False, "error": "No path given."})
    import train_core

    try:
        if kind == "dataset":
            records, _data_root, is_stacked = train_core.resolve_dataset(path)
            frame_stack_n = train_core.FRAME_STACK_N_DEFAULT if is_stacked else 1
            return jsonify({"ok": True, "record_count": len(records),
                             "format": "timestamped" if is_stacked else "classic",
                             "frame_stack_n": frame_stack_n})
        elif kind == "pth":
            p = Path(path)
            if not p.exists():
                return jsonify({"ok": False, "error": f"Not found: {p}"})
            state = train_core.torch.load(str(p), map_location="cpu")
            frame_stack_n = state["conv1.weight"].shape[1] // 3
            return jsonify({"ok": True, "frame_stack_n": frame_stack_n})
        elif kind == "npy":
            p = Path(path)
            if not p.exists():
                return jsonify({"ok": False, "error": f"Not found: {p}"})
            arr = train_core.np.load(str(p), mmap_mode="r")
            return jsonify({"ok": True, "record_count": int(arr.shape[0])})
        else:
            return jsonify({"ok": False, "error": f"Unknown kind: {kind}"})
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/browse/folder", methods=["GET"])
def browse_folder():
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = tkinter.filedialog.askdirectory(title="Select dataset folder")
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
    path = tkinter.filedialog.askopenfilename(title="Select file", filetypes=filetypes.get(ft, [("All", "*.*")]))
    root.destroy()
    return jsonify({"path": path or ""})


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
        while not _events and _running and waited < 60:
            time.sleep(0.2)
            waited += 0.2
        last_activity = time.time()
        while True:
            chunk = _events[last:]
            done = False
            for ev in chunk:
                yield f"data: {json.dumps(ev)}\n\n"
                last_activity = time.time()
                if ev.get("type") == "done":
                    done = True
            last += len(chunk)
            if done:
                break
            if not _running:
                # job ended without a "done" event (shouldn't happen, but
                # don't hang the client forever if it ever does)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            if time.time() - last_activity > 5:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                last_activity = time.time()
            time.sleep(0.1)

    return Response(generate(), mimetype="text/event-stream",
                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Job-launching routes ────────────────────────────────────────────────

@app.route("/api/run/full", methods=["POST"])
def run_full():
    config = request.json or {}

    def job():
        import train_core
        import compile_pipeline
        # Before training, not after - a stopped Docker / missing wheel should
        # fail in a second, not after hours of training (same as main's train.py).
        if not compile_pipeline.check_compile_prereqs(_push):
            _push({"type": "done"})
            return
        ok = train_core.run(config, _push, _should_stop, emit_done=False)
        if not ok:
            _push({"type": "done"})
            return
        # Stop only ends training early - the best checkpoint so far is still
        # exported + compiled, same as main's train.py.
        model_name = (config.get("model_name") or "model").strip() or "model"
        compile_pipeline.compile_from_pth({
            "pth_path": str(train_core.MODELS_DIR / f"{model_name}.pth"),
            "model_name": model_name,
            "json_path": config.get("json_path"),
        }, _push)

    return _start_job("full", job, config, required_fields=("json_path", "model_name"))


@app.route("/api/run/train-only", methods=["POST"])
def run_train_only():
    config = dict(request.json or {})
    config["pth_only"] = True

    def job():
        import train_core
        train_core.run(config, _push, _should_stop, emit_done=True)

    return _start_job("train-only", job, config, required_fields=("json_path", "model_name"))


@app.route("/api/run/compile-only", methods=["POST"])
def run_compile_only():
    config = request.json or {}

    def job():
        import compile_pipeline
        compile_pipeline.compile_from_pth(config, _push)

    return _start_job("compile-only", job, config, required_fields=("pth_path", "model_name"))


@app.route("/api/run/retry-compile", methods=["POST"])
def run_retry_compile():
    config = request.json or {}

    def job():
        import compile_pipeline
        compile_pipeline.retry_compile(config, _push)

    return _start_job("retry-compile", job, config, required_fields=("model_name",))


@app.route("/api/run/smoketest", methods=["POST"])
def run_smoketest():
    config = request.json or {}

    def job():
        import compile_pipeline
        compile_pipeline.run_smoke_test(config, _push)

    return _start_job("smoketest", job, config, required_fields=("json_path",))


@app.route("/api/run/smoketest-compile", methods=["POST"])
def run_smoketest_compile():
    config = request.json or {}

    def job():
        import compile_pipeline
        compile_pipeline.run_smoke_test_no_dataset(config, _push)

    return _start_job("smoketest-compile", job, config)


if __name__ == "__main__":
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    print("=" * 50)
    print("  RoboPacerV2 Trainer  ->  http://localhost:5000")
    print("  Ctrl+C to stop.")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
