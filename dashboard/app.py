"""
RoboPacerV2 Dashboard
=======================
Web UI to start/stop the robot's scripts (main, data_recorder,
manual_drive) and monitor the robot from a phone
connected to the robot's hotspot (or a home WiFi joined via the WiFi panel
on the home page).

Only one script can run at a time, regardless of how many browser tabs are
connected: start/stop/relay actions are serialized by `_action_lock`, and
Start is refused (409) while any registered script is alive - including one
launched outside the dashboard (e.g. over SSH), which is found by scanning
processes with psutil and "adopted" so it can be seen and stopped here.

Each run's stdout+stderr is written to a dedicated file in run_logs/,
truncated at the start of every new run - that file is the single source
of truth for the live console: a newly-connected browser tab gets the last
CONSOLE_LINES lines of the current run, then new lines as they're written
(tail -f style). Nothing is kept in the Flask process's memory, so the
console survives a restart of this process (systemd Restart=always).

Relay safety: every script turns the relay ON at start and OFF in its own
`finally`. The relay card can also switch it ON by hand (only with nothing
running); a Start that finds it already ON power-cycles it first - see
api_relay/api_start. Stopping from here (SIGTERM, grace period, then SIGKILL) also
sends RELAY_OFF afterwards, so the motor loses power even if the script had
to be killed. Turning the relay OFF from here cuts power first, then stops
the running script - and safety/estop_listener.py independently kills every
driving script when the ESP32 reports the relay going off (!!ESTOP!!).
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time

import psutil
from flask import Flask, Response, jsonify, render_template, request

import bluetooth_devices
import filebrowser
import system_stats
import wifi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(REPO_ROOT)
from config.cruise_config import TARGET_SPEED_MAX_KMH  # noqa: E402
from config.estop import relay_cmd  # noqa: E402
from config.ipc_config import DATA_RECORDER_STOP_GRACE_SECONDS, SCRIPT_STOP_GRACE_SECONDS  # noqa: E402

RUN_LOGS_DIR = os.path.join(BASE_DIR, "run_logs")
PORT = 8080
STATUS_STREAM_INTERVAL_SECONDS = 1
CONSOLE_LINES = 50  # console shows only the last N lines (the page caps too)
LOG_KEEPALIVE_SECONDS = 5

# main-only bounds, mirroring main/main.py's own parse_args() validation -
# kept in sync manually since main.py can't be imported here (it pulls in
# evdev/hailo_platform/picamera2 at module level, none of which need to be
# on the dashboard's box).
DISTANCE_M_MIN = 1.0
DISTANCE_M_MAX = 50000.0
SPEED_MODES = ("cruise", "controller", "none")

# Same rule as data_recorder.py's DATASET_NAME_RE - a folder with any other
# name can't be picked here (and the recorder would refuse it anyway).
DATASET_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
FRAME_NAME_RE = re.compile(r"frame_(\d+)\.jpg")

SCRIPTS = {
    "main": {
        "id": "main",
        "name": "Autopilot (main)",
        "description": "Modelul la volan + viteza: cruise, controller sau doar steering",
        "path": os.path.join(REPO_ROOT, "main", "main.py"),
        "template": "run_main.html",
    },
    "data_recorder": {
        "id": "data_recorder",
        "name": "Data Recorder",
        "description": "Condus manual, inregistreaza cadre + steering pentru antrenare",
        "path": os.path.join(REPO_ROOT, "data_recorder", "data_recorder.py"),
        "template": "run_data_recorder.html",
    },
    "manual_drive": {
        "id": "manual_drive",
        "name": "Manual Drive",
        "description": "Condus manual cu controller-ul, distanta/viteza live",
        "path": os.path.join(REPO_ROOT, "manual_drive", "manual_drive.py"),
        "template": "run_manual_drive.html",
    },
}

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask(__name__)

_lock = threading.Lock()           # guards _running
_action_lock = threading.Lock()    # serializes start/stop/relay actions
_starting = threading.Event()      # set while a Start is launching a script
_run_seq = {}                      # script_id -> starts so far; log streams reset on change
_running = {"id": None, "pid": None, "popen": None, "started_at": None}


def log_path_for(script_id):
    return os.path.join(RUN_LOGS_DIR, f"{script_id}.log")


def list_datasets():
    """Folders in data_recorder/ holding frames/frame_NNNNN.jpg - the
    candidates for "continue recording". Each: {name, frames, last_index}."""
    base = os.path.dirname(SCRIPTS["data_recorder"]["path"])
    datasets = []
    try:
        entries = list(os.scandir(base))
    except OSError:
        return datasets
    for entry in entries:
        if not entry.is_dir() or not DATASET_NAME_RE.fullmatch(entry.name):
            continue
        count, last_index = 0, -1
        try:
            with os.scandir(os.path.join(entry.path, "frames")) as frames:
                for f in frames:
                    m = FRAME_NAME_RE.fullmatch(f.name)
                    if m:
                        count += 1
                        last_index = max(last_index, int(m.group(1)))
        except OSError:  # no frames/ - not a dataset folder
            continue
        if count:
            datasets.append({"name": entry.name, "frames": count, "last_index": last_index})
    # Natural sort (set2 before set10).
    datasets.sort(key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d["name"])])
    return datasets


def _is_alive(running):
    if running["id"] is None:
        return False
    if running["popen"] is not None:
        return running["popen"].poll() is None
    return psutil.pid_exists(running["pid"])


def _adopt_running_script():
    """Find a registered script that's running without us tracking it -
    Flask restarted by systemd mid-run, or a script launched over SSH - so
    it counts for the one-at-a-time rule and can be stopped from here."""
    self_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        if proc.pid == self_pid:
            continue
        try:
            script_id = _script_id_for_cmdline(proc.info.get("cmdline") or [], proc.cwd)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if script_id is not None:
            return {"id": script_id, "pid": proc.pid, "popen": None,
                    "started_at": proc.info.get("create_time")}
    return None


# Interpreter options that take a separate value (`-X dev`, `-W ignore`).
_PYTHON_OPTS_WITH_VALUE = {"-X", "-W", "--check-hash-based-pycs"}
# `python -m <one of these> script.py ...` really runs script.py.
_SCRIPT_RUNNER_MODULES = {"pdb", "cProfile", "profile", "trace"}


def _script_id_for_cmdline(cmdline, get_cwd):
    """Registered script this command line actually RUNS, or None. Must be a
    python interpreter whose script argument is the script's file (also under
    `-m pdb`/`-m cProfile`) - not just any process mentioning the path, e.g.
    esc_watchdog.sh's `pgrep -f .../main/main.py`, an editor, `tail`,
    `python -c ... <path>` or `python -m py_compile <path>`."""
    if not cmdline or not os.path.basename(cmdline[0]).lower().startswith("python"):
        return None
    args = cmdline[1:]
    i = 0
    while i < len(args):  # skip interpreter options
        a = args[i]
        if a == "-c":
            return None  # inline code - whatever follows is just its argv
        if a == "-m":
            if i + 1 >= len(args) or args[i + 1] not in _SCRIPT_RUNNER_MODULES:
                return None
            i += 2
            while i < len(args) and args[i].startswith("-"):  # runner's own options
                i += 1
            break
        if a in _PYTHON_OPTS_WITH_VALUE:
            i += 2
            continue
        if not a.startswith("-"):
            break
        i += 1
    if i >= len(args):
        return None
    script_arg = args[i]
    if not os.path.isabs(script_arg):  # e.g. `cd main && python3 main.py` over SSH
        script_arg = os.path.join(get_cwd(), script_arg)
    script_arg = os.path.realpath(script_arg)
    for script in SCRIPTS.values():
        if script_arg == os.path.realpath(script["path"]):
            return script["id"]
    return None


def _status_payload():
    with _lock:
        running = dict(_running)
    if not _is_alive(running):
        adopted = _adopt_running_script()
        with _lock:
            if _running["pid"] == running["pid"]:  # not replaced by a new Start meanwhile
                _running.update(adopted or dict(id=None, pid=None, popen=None, started_at=None))
            running = dict(_running)
        if running["id"] is None:
            return None
    return {
        "id": running["id"],
        "name": SCRIPTS[running["id"]]["name"],
        "external": running["popen"] is None,
        "pid": running["pid"],
        "started_at": running["started_at"],
        "uptime_seconds": round(time.time() - running["started_at"], 1) if running["started_at"] else None,
    }


def _stop_running(running):
    if running["popen"] is not None:
        terminate, kill, wait = (running["popen"].terminate, running["popen"].kill,
                                  running["popen"].wait)
    else:
        try:
            p = psutil.Process(running["pid"])
        except psutil.NoSuchProcess:
            return
        terminate, kill, wait = p.terminate, p.kill, p.wait

    # Power off FIRST - a script hung in native code may never run its own
    # finally, and the watchdog doesn't step in while its process is alive.
    # (The ESP32 then reports !!ESTOP!!, so estop_listener SIGTERMs it too.)
    relay_cmd("RELAY_OFF")
    try:
        terminate()
    except Exception:
        pass
    try:
        # Safe to wait: no power. data_recorder needs this to save its log.
        wait(timeout=DATA_RECORDER_STOP_GRACE_SECONDS if running["id"] == "data_recorder"
             else SCRIPT_STOP_GRACE_SECONDS)
    except Exception:
        try:
            kill()
            wait(timeout=2)  # reap it - no zombie left behind
        except Exception:
            pass

    with _lock:
        _running.update(id=None, pid=None, popen=None, started_at=None)


def _stats_payload(running):
    # Hailo temp only while nothing runs or is launching - see get_hailo_temp.
    return system_stats.get_all_stats(hailo_allowed=running is None and not _starting.is_set())


def _stop_current(reason):
    """Stops whatever registered script is running. Caller holds _action_lock."""
    current = _status_payload()
    if current is None:
        return None
    with _lock:
        running_copy = dict(_running)
    logging.info(f"Stopping {current['id']} (PID {current['pid']}): {reason}")
    _stop_running(running_copy)
    return current["id"]


@app.route("/")
def home():
    return render_template("home.html", scripts=list(SCRIPTS.values()))


@app.route("/run/<script_id>")
def run_page(script_id):
    script = SCRIPTS.get(script_id)
    if script is None:
        return f"Script necunoscut: {script_id}", 404
    return render_template(script["template"], script=script, target_speed_max_kmh=TARGET_SPEED_MAX_KMH,
                            distance_min=DISTANCE_M_MIN, distance_max=DISTANCE_M_MAX,
                            main_dir=os.path.join(REPO_ROOT, "main"))


@app.route("/api/status")
def api_status():
    return jsonify({"running": _status_payload()})


@app.route("/api/stats")
def api_stats():
    return jsonify(_stats_payload(_status_payload()))


@app.route("/api/data_recorder/datasets")
def api_data_recorder_datasets():
    return jsonify({"datasets": list_datasets()})


@app.route("/api/manual_drive/status")
def api_manual_drive_status():
    status = system_stats.get_manual_drive_status()
    return jsonify(status or {
        "distance_m": None, "kmh": None, "pace": None, "paused": None,
        "avg_kmh": None, "avg_pace": None, "max_kmh": None, "max_pace": None,
    })


@app.route("/api/manual_drive/reset_distance", methods=["POST"])
def api_manual_drive_reset_distance():
    return jsonify({"ok": system_stats.reset_manual_drive_distance()})


@app.route("/api/main/model_info")
def api_main_model_info():
    return jsonify(system_stats.get_main_model_info())


@app.route("/api/main/status")
def api_main_status():
    status = system_stats.get_main_status()
    return jsonify(status or {
        "engaged": None, "speed_mode": None, "target_kmh": None, "effective_target_kmh": None,
        "kmh": None, "pace_sec_per_km": None, "distance_m": None, "distance_target_m": None,
        "stop_reason": None, "pwm_us": None, "fps": None,
    })


# Per-script argv builders. Each validates the client's JSON and rebuilds the
# argv server-side from known flags only - a client string is never passed
# through to the subprocess. Return (args, None) or (None, (error, status)).

def _bool_param(data, key, default=False):
    value = data.get(key, default)
    return value if isinstance(value, bool) else None


def _main_args(data):
    speed_mode = data.get("speed_mode")
    if speed_mode not in SPEED_MODES:
        return None, ("invalid_speed_mode", 400)

    target_kmh = data.get("target_kmh")
    if speed_mode == "cruise":
        if isinstance(target_kmh, bool) or not isinstance(target_kmh, (int, float)) or not math.isfinite(target_kmh):
            return None, ("missing_target_kmh", 400)
        target_kmh = float(target_kmh)
        if not (0.1 <= target_kmh <= TARGET_SPEED_MAX_KMH):
            return None, ("target_kmh_out_of_range", 400)
    elif target_kmh is not None:
        return None, ("target_kmh_not_allowed", 400)

    distance_m = data.get("distance_m")
    if distance_m is not None:
        if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)) or not math.isfinite(distance_m):
            return None, ("invalid_distance_m", 400)
        distance_m = float(distance_m)
        if not (DISTANCE_M_MIN <= distance_m <= DISTANCE_M_MAX):
            return None, ("distance_m_out_of_range", 400)

    raw_steering = _bool_param(data, "raw_steering", True)
    if raw_steering is None:
        return None, ("invalid_raw_steering", 400)
    display = _bool_param(data, "display")
    if display is None:
        return None, ("invalid_display", 400)

    # Optional model picked in the file browser; absent = main/'s own .hef.
    hef_path = data.get("hef_path")
    if hef_path is not None:
        hef_path = filebrowser.resolve_hef(hef_path)
        if hef_path is None:
            return None, ("invalid_hef_path", 400)

    args = ["--speed-mode", speed_mode]
    if hef_path is not None:
        args += ["--hef", hef_path]
    if speed_mode == "cruise":
        args += ["--target-kmh", str(target_kmh)]
    if distance_m is not None:
        args += ["--distance-m", str(distance_m)]
    if not raw_steering:
        args.append("--smooth-steering")
    if display:
        args.append("--display")
    return args, None


def _data_recorder_args(data):
    # Explicit choice required: {"mode": "new"} or {"mode": "existing",
    # "name": ...}, the name checked against folders that actually hold frames.
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        return None, ("missing_dataset", 400)
    if dataset.get("mode") == "new":
        args = ["--new-dataset"]
    elif dataset.get("mode") == "existing":
        name = dataset.get("name")
        if not isinstance(name, str) or name not in {d["name"] for d in list_datasets()}:
            return None, ("unknown_dataset", 400)
        args = ["--dataset", name]
    else:
        return None, ("invalid_dataset", 400)

    for key, flag in (("legacy", "--legacy"), ("display", "--display")):
        value = _bool_param(data, key)
        if value is None:
            return None, (f"invalid_{key}", 400)
        if value:
            args.append(flag)
    return args, None


def _manual_drive_args(data):
    return [], None


ARG_BUILDERS = {
    "main": _main_args,
    "data_recorder": _data_recorder_args,
    "manual_drive": _manual_drive_args,
}


RELAY_CYCLE_WAIT_SECONDS = 3.0
RELAY_CYCLE_SETTLE_SECONDS = 1.0


def _cycle_relay_if_on():
    """A script only powers the ESC up correctly from a cold start (neutral
    signal running first, THEN relay ON) - and the ESP32 ignores RELAY_ON
    while already ON. So if the relay was left ON (switched on by hand from
    the card, or by a script that died), cut it now, before the script exists:
    the ESP32's !!ESTOP!! from this OFF then has nothing to kill, and the
    script's own RELAY_ON is a real power-up. Caller holds _action_lock."""
    if not system_stats.get_esp32_status().get("relay_on"):
        return
    logging.info("Relay was ON at Start - power-cycling it first")
    relay_cmd("RELAY_OFF")
    deadline = time.time() + RELAY_CYCLE_WAIT_SECONDS
    while time.time() < deadline and system_stats.get_esp32_status().get("relay_on"):
        time.sleep(0.1)
    time.sleep(RELAY_CYCLE_SETTLE_SECONDS)  # let the listener finish handling that ESTOP


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    script_id = data.get("id")

    script = SCRIPTS.get(script_id)
    if script is None:
        return jsonify({"error": "unknown_script"}), 404

    args, err = ARG_BUILDERS[script_id](data)
    if err is not None:
        error, code = err
        return jsonify({"error": error}), code

    with _action_lock:
        # One script at a time, no exceptions - stop the running one first.
        current = _status_payload()
        if current is not None:
            return jsonify({"error": "already_running", "running": current}), 409

        _starting.set()  # no new Hailo temperature reads from here on
        try:
            # ...and let one already in flight release the device, or
            # main.py's own VDevice() would fail and abort the run.
            if not system_stats.wait_hailo_idle():
                return jsonify({"error": "hailo_busy"}), 503

            _cycle_relay_if_on()

            os.makedirs(RUN_LOGS_DIR, exist_ok=True)
            log_file = open(log_path_for(script_id), "wb", buffering=0)
            try:
                popen = subprocess.Popen(
                    ["python3", "-u", script["path"], *args],
                    stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    cwd=os.path.dirname(script["path"]),
                    start_new_session=True,
                )
            finally:
                log_file.close()

            with _lock:
                _running.update(id=script_id, pid=popen.pid, popen=popen, started_at=time.time())
                _run_seq[script_id] = _run_seq.get(script_id, 0) + 1
        finally:
            _starting.clear()
        logging.info(f"Started {script_id} (PID {popen.pid}) with args {args}")
        return jsonify({"ok": True, "running": _status_payload()})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _action_lock:
        stopped = _stop_current("stop button")
    return jsonify({"ok": True, "stopped": stopped, "running": None})


@app.route("/api/relay", methods=["POST"])
def api_relay():
    data = request.get_json(force=True, silent=True) or {}
    on = data.get("on")
    if not isinstance(on, bool):
        return jsonify({"error": "invalid_on"}), 400
    with _action_lock:
        if not on:
            # Power off first (instant), then stop the script cleanly.
            relay_cmd("RELAY_OFF")
            logging.info("Relay OFF from dashboard")
            stopped = _stop_current("relay turned off")
            return jsonify({"ok": True, "stopped": stopped})

        # Manual ON - only with nothing running: a running script owns the
        # relay (and a Start powers it up itself, in the right order).
        # NOTE: with no script running the watchdog holds the ESC channel at
        # "no signal", so a manually powered ESC (QuicRun) sits in failsafe -
        # api_start therefore power-cycles the relay before launching anything.
        if _status_payload() is not None:
            return jsonify({"error": "script_running"}), 409
        esp32 = system_stats.get_esp32_status()
        if not esp32.get("service_running") or not esp32.get("esp32_connected"):
            return jsonify({"error": "esp32_unavailable"}), 503
        if not esp32.get("relay_on"):
            relay_cmd("RELAY_ON")
            logging.info("Relay ON from dashboard")
        return jsonify({"ok": True})


@app.route("/api/stream/log/<script_id>")
def stream_log(script_id):
    if script_id not in SCRIPTS:
        return "unknown_script", 404
    path = log_path_for(script_id)

    def generate():
        while not os.path.exists(path):
            yield ": waiting\n\n"  # SSE comment - keeps the connection alive
            time.sleep(1.0)
        with open(path, "rb") as f:
            # Only the current run: the file is truncated on every new start.
            seq = _run_seq.get(script_id)
            lines, status = _console_tail(f, CONSOLE_LINES)
            for line in lines:
                yield _sse_line(line)
            if status is not None:
                yield _sse_line(status, "status")
            pending = b""
            last_yield = time.time()
            while True:
                if _run_seq.get(script_id) != seq or os.path.getsize(path) < f.tell():
                    # A new run started (file truncated) - clear the page and
                    # read the new run from its first line.
                    seq = _run_seq.get(script_id)
                    f.seek(0)
                    pending = b""
                    yield "event: reset\ndata: \n\n"
                    last_yield = time.time()
                    continue
                chunk = f.read(65536)
                if chunk:
                    pending += chunk
                    lines, status, pending = _split_console(pending)
                    for line in lines:
                        yield _sse_line(line)
                    if status is not None:
                        # Only the newest of this read's \r redraws - the
                        # scripts redraw ~100x/s, the page doesn't need that.
                        yield _sse_line(status, "status")
                    last_yield = time.time()
                    continue
                if time.time() - last_yield > LOG_KEEPALIVE_SECONDS:
                    # SSE comment. Also how a closed tab is noticed: the write
                    # fails and the server ends this generator and its thread.
                    yield ": keepalive\n\n"
                    last_yield = time.time()
                time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse_line(text, event=None):
    # Any stray \r/\n inside would end the SSE field early.
    text = text.replace("\r", "").replace("\n", "")
    return (f"event: {event}\n" if event else "") + f"data: {text}\n\n"


def _split_console(data):
    """Splits raw log bytes the way a terminal shows them: "\\n" ends a line,
    a lone "\\r" means the script redraws the same line (manual_drive and
    data_recorder print their live status with end="\\r"). Returns
    (finished lines, latest in-place status line or None, leftover bytes
    that aren't terminated yet)."""
    lines, status = [], None
    start = 0
    i = 0
    n = len(data)
    while i < n:
        c = data[i]
        if c == 0x0A:  # \n
            # (a \r right before it is part of the \r\n ending)
            lines.append(data[start:i].rstrip(b"\r").decode("utf-8", errors="replace"))
            status = None
            start = i + 1
        elif c == 0x0D:  # \r
            if i + 1 == n:
                break  # could be half of \r\n - wait for the next read
            if data[i + 1] == 0x0A:
                i += 1
                continue  # \r\n: handled as \n on the next step
            status = data[start:i].decode("utf-8", errors="replace")
            start = i + 1
        i += 1
    return lines, status, data[start:]


CONSOLE_REPLAY_MAX_BYTES = 256 * 1024  # 50 real lines never need more


def _console_tail(f, n):
    """(last n finished lines, current status line or None) of a file opened
    in binary mode, leaving f at EOF. Reads at most CONSOLE_REPLAY_MAX_BYTES
    from the end, backwards in blocks, each byte read once."""
    f.seek(0, os.SEEK_END)
    end = f.tell()
    pos, chunks, newlines = end, [], 0
    while pos > 0 and newlines <= n and end - pos < CONSOLE_REPLAY_MAX_BYTES:
        step = min(8192, pos)
        pos -= step
        f.seek(pos)
        block = f.read(step)
        chunks.append(block)
        newlines += block.count(b"\n")
    f.seek(end)
    data = b"".join(reversed(chunks))
    if pos > 0:
        # Drop the first segment - it's probably cut mid-way. A status-only
        # log (\r redraws, no \n for a long time) is cut at a \r instead.
        cuts = [c for c in (data.find(b"\n"), data.find(b"\r")) if c != -1]
        data = data[min(cuts) + 1:] if cuts else b""
    lines, status, rest = _split_console(data)
    if rest:
        # Unfinished last line: show it for now, and let the live tail
        # re-read it so the finished version isn't missing its start.
        status = rest.decode("utf-8", errors="replace").rstrip("\r")
        f.seek(end - len(rest))
    return lines[-n:], status


@app.route("/api/stream/status")
def stream_status():
    def generate():
        while True:
            running = _status_payload()
            if running is not None and running["id"] == "main":
                # Inference FPS for the status bar on every page - main.py
                # already averages it over 1s, this is one socket read/s.
                running["fps"] = (system_stats.get_main_status() or {}).get("fps")
            payload = {"running": running, "stats": _stats_payload(running)}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(STATUS_STREAM_INTERVAL_SECONDS)

    return Response(generate(), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# File browser for the model picker (main page) - read-only, see filebrowser.py
# ---------------------------------------------------------------------------

@app.route("/api/fs/list")
def api_fs_list():
    listing, err = filebrowser.list_dir(request.args.get("path") or os.path.expanduser("~"))
    if err is not None:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, **listing})


# ---------------------------------------------------------------------------
# Bluetooth - saved devices, scan, connect/forget (see bluetooth_devices.py)
# ---------------------------------------------------------------------------

def _bt_mac(data):
    mac = data.get("mac")
    mac = mac.upper() if isinstance(mac, str) else mac
    return mac if bluetooth_devices.valid_mac(mac) else None


@app.route("/api/bluetooth/devices")
def api_bluetooth_devices():
    status = bluetooth_devices.adapter_status()
    if not status["ok"]:
        return jsonify({"ok": False, "error": status["error"]})
    return jsonify({"ok": True, "powered": status["powered"], "devices": bluetooth_devices.list_paired()})


@app.route("/api/bluetooth/scan", methods=["POST"])
def api_bluetooth_scan():
    try:
        found, err = bluetooth_devices.scan()
    except bluetooth_devices.Busy:
        return jsonify({"ok": False, "error": "Alta operatie Bluetooth e in curs."}), 409
    if err is not None:
        return jsonify({"ok": False, "error": err}), 503
    return jsonify({"ok": True, "devices": found})


@app.route("/api/bluetooth/<action>", methods=["POST"])
def api_bluetooth_action(action):
    actions = {
        "pair": bluetooth_devices.pair_and_connect,
        "connect": bluetooth_devices.connect,
        "disconnect": bluetooth_devices.disconnect,
        "forget": bluetooth_devices.forget,
    }
    fn = actions.get(action)
    if fn is None:
        return jsonify({"ok": False, "error": "unknown_action"}), 404
    mac = _bt_mac(request.get_json(force=True, silent=True) or {})
    if mac is None:
        return jsonify({"ok": False, "error": "invalid_mac"}), 400
    try:
        ok, err = fn(mac)
    except bluetooth_devices.Busy:
        return jsonify({"ok": False, "error": "Alta operatie Bluetooth e in curs."}), 409
    return jsonify({"ok": True}) if ok else (jsonify({"ok": False, "error": err}), 502)


# ---------------------------------------------------------------------------
# WiFi scan/connect - see wifi.py for the nmcli backend + interface
# auto-detection (excludes whichever radio is serving the robot's hotspot).
# ---------------------------------------------------------------------------

@app.route("/api/wifi/status")
def api_wifi_status():
    interface, err = wifi.find_station_interface()
    if err is not None:
        return jsonify({"ok": False, "error": err})
    return jsonify({"ok": True, **wifi.get_wifi_status(interface)})


@app.route("/api/wifi/scan")
def api_wifi_scan():
    interface, err = wifi.find_station_interface()
    if err is not None:
        return jsonify({"ok": False, "error": err}), 503
    networks, err = wifi.scan_networks(interface)
    if err is not None:
        return jsonify({"ok": False, "error": err}), 502
    return jsonify({"ok": True, "interface": interface, "networks": networks})


@app.route("/api/wifi/connect", methods=["POST"])
def api_wifi_connect():
    data = request.get_json(force=True, silent=True) or {}
    ssid = data.get("ssid")
    password = data.get("password") or ""
    if not isinstance(ssid, str) or not ssid:
        return jsonify({"ok": False, "error": "missing_ssid"}), 400
    if not isinstance(password, str):
        return jsonify({"ok": False, "error": "invalid_password"}), 400

    interface, err = wifi.find_station_interface()
    if err is not None:
        return jsonify({"ok": False, "error": err}), 503
    success, message = wifi.connect_network(interface, ssid, password)
    if not success:
        return jsonify({"ok": False, "error": message}), 502
    return jsonify({"ok": True, "message": message})


if __name__ == "__main__":
    adopted = _adopt_running_script()
    if adopted:
        with _lock:
            _running.update(adopted)
    logging.info(f"Dashboard started on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
