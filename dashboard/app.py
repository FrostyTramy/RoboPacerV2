"""
RoboPacerV2 Dashboard
=======================
Interfata web pentru pornit/oprit scripturile din scripts.json de pe
telefon/laptop, conectat la hotspot-ul "RoboPacer" al Pi-ului (sau la
wifi-ul de acasa - vezi safety/wifi_fallback.py).

Un singur script poate rula la un moment dat (mutex global, tinut server-
side - vezi `_running`), indiferent cati clienti web sunt conectati.

Output-ul (stdout+stderr) fiecarei rulari e scris intr-un fisier dedicat
in run_logs/, trunchiat la fiecare pornire noua - asta e sursa unica de
adevar pentru consola live: un client nou sau reconectat primeste tot
fisierul de la inceput, apoi continua sa primeasca linii noi (tail -f),
fara sa tina nimic in memoria procesului Flask. Asta face consola sa
supravietuiasca si unui restart al acestui proces Flask (systemd
Restart=always) - la pornire, re-adoptam orice script deja pornit anterior
scanand procesele cu psutil, dupa acelasi model ca estop_listener.py.

Oprirea (SIGTERM, cu grace period, apoi SIGKILL) declanseaza automat
oprirea releului - main.py/data_recorder.py au deja in `finally` un
_relay_cmd("RELAY_OFF") catre estop_listener - deci nu trebuie duplicata
logica aici.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time

import psutil
from flask import Flask, Response, jsonify, render_template, request

import system_stats

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_JSON   = os.path.join(BASE_DIR, "scripts.json")
RUN_LOGS_DIR   = os.path.join(BASE_DIR, "run_logs")
PORT           = 8080
SIGTERM_WAIT_SECONDS = 5
STATUS_STREAM_INTERVAL_SECONDS = 2

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask(__name__)

_lock = threading.Lock()
_running = {"id": None, "pid": None, "popen": None, "started_at": None}


def load_scripts():
    with open(SCRIPTS_JSON) as f:
        return json.load(f)


def find_script(script_id, scripts):
    return next((s for s in scripts if s["id"] == script_id), None)


def log_path_for(script_id):
    return os.path.join(RUN_LOGS_DIR, f"{script_id}.log")


def _is_alive(running):
    if running["id"] is None:
        return False
    if running["popen"] is not None:
        return running["popen"].poll() is None
    return psutil.pid_exists(running["pid"])


def _adopt_running_script():
    """La pornirea Flask-ului, cauta daca vreun script din scripts.json e
    deja pornit (ex: Flask-ul a fost repornit de systemd in timp ce main.py
    rula) si il "adopta", ca sa nu pierdem controlul/mutex-ul asupra lui."""
    scripts = load_scripts()
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for s in scripts:
            if any(s["path"] in arg for arg in cmdline):
                logging.info(f"Adoptat script deja pornit: {s['id']} (PID {proc.pid})")
                return {"id": s["id"], "pid": proc.pid, "popen": None,
                        "started_at": proc.info.get("create_time")}
    return None


def _status_payload():
    with _lock:
        running = dict(_running)
    if not _is_alive(running):
        with _lock:
            if _running["id"] == running["id"]:
                _running.update(id=None, pid=None, popen=None, started_at=None)
        return None
    return {
        "id": running["id"],
        "pid": running["pid"],
        "started_at": running["started_at"],
        "uptime_seconds": round(time.time() - running["started_at"], 1) if running["started_at"] else None,
    }


def _stop_running(running):
    if running["popen"] is not None:
        proc_terminate = running["popen"].terminate
        proc_kill = running["popen"].kill
        def proc_wait(timeout): running["popen"].wait(timeout=timeout)
    else:
        try:
            p = psutil.Process(running["pid"])
        except psutil.NoSuchProcess:
            return
        proc_terminate = p.terminate
        proc_kill = p.kill
        proc_wait = p.wait

    try:
        proc_terminate()
    except Exception:
        pass
    try:
        proc_wait(SIGTERM_WAIT_SECONDS)
    except Exception:
        try:
            proc_kill()
        except Exception:
            pass

    with _lock:
        _running.update(id=None, pid=None, popen=None, started_at=None)


@app.route("/")
def home():
    return render_template("home.html", scripts=load_scripts())


@app.route("/run/<script_id>")
def run_page(script_id):
    scripts = load_scripts()
    script = find_script(script_id, scripts)
    if script is None:
        return f"Script necunoscut: {script_id}", 404
    return render_template("run.html", script=script)


@app.route("/api/scripts")
def api_scripts():
    return jsonify(load_scripts())


@app.route("/api/status")
def api_status():
    return jsonify({"running": _status_payload()})


@app.route("/api/stats")
def api_stats():
    return jsonify(system_stats.get_all_stats())


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    script_id = data.get("id")
    force = bool(data.get("force"))
    requested_args = data.get("args") or []

    scripts = load_scripts()
    script = find_script(script_id, scripts)
    if script is None:
        return jsonify({"error": "unknown_script"}), 404

    # Doar flag-urile listate in scripts.json pentru scriptul asta sunt
    # acceptate - evita sa injectam argumente arbitrare in subprocess.
    allowed_flags = {f["flag"] for f in script.get("flags", [])}
    if not isinstance(requested_args, list) or not all(isinstance(a, str) for a in requested_args):
        return jsonify({"error": "invalid_args"}), 400
    unknown = [a for a in requested_args if a not in allowed_flags]
    if unknown:
        return jsonify({"error": "unknown_flag", "flags": unknown}), 400

    current = _status_payload()
    if current is not None:
        if current["id"] == script_id:
            return jsonify({"ok": True, "running": current})
        if not force:
            return jsonify({"error": "conflict", "running": current}), 409
        with _lock:
            running_copy = dict(_running)
        logging.info(f"Opresc {current['id']} ca sa pornesc {script_id} (force)")
        _stop_running(running_copy)

    os.makedirs(RUN_LOGS_DIR, exist_ok=True)
    log_file = open(log_path_for(script_id), "wb", buffering=0)
    try:
        popen = subprocess.Popen(
            ["python3", "-u", script["path"], *requested_args],
            stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=os.path.dirname(script["path"]),
            start_new_session=True,
        )
    finally:
        log_file.close()

    with _lock:
        _running.update(id=script_id, pid=popen.pid, popen=popen, started_at=time.time())
    logging.info(f"Pornit {script_id} (PID {popen.pid})")
    return jsonify({"ok": True, "running": _status_payload()})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    current = _status_payload()
    if current is None:
        return jsonify({"ok": True, "running": None})
    with _lock:
        running_copy = dict(_running)
    logging.info(f"Opresc {current['id']} (PID {current['pid']}) la cerere")
    _stop_running(running_copy)
    return jsonify({"ok": True, "running": None})


@app.route("/api/stream/log/<script_id>")
def stream_log(script_id):
    path = log_path_for(script_id)

    def generate():
        for _ in range(50):
            if os.path.exists(path):
                break
            time.sleep(0.1)
        else:
            yield "data: [inca nu exista log - scriptul nu a fost pornit]\n\n"
            return
        with open(path, "r", errors="replace") as f:
            for line in f.read().splitlines():
                yield f"data: {line}\n\n"
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip(chr(10))}\n\n"
                else:
                    time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stream/status")
def stream_status():
    def generate():
        while True:
            payload = {"running": _status_payload(), "stats": system_stats.get_all_stats()}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(STATUS_STREAM_INTERVAL_SECONDS)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    adopted = _adopt_running_script()
    if adopted:
        with _lock:
            _running.update(adopted)
    logging.info(f"Dashboard pornit pe portul {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
