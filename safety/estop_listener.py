"""
RoboPacerV2 - Emergency Stop Listener
======================================
Watches a USB-serial link to an ESP32 for an emergency-stop signal and, on
receipt, kills every RoboPacerV2 Python process (main.py, data_recorder.py,
...) so the car stops immediately regardless of what it's doing.

Why this exists as its own always-on service, separate from the driving
scripts themselves: main.py/data_recorder.py can freeze, hang on an I2C
error, or otherwise stop reading their own controller input without the
process itself dying - in any of those cases the script can't react to
anything, including its own SIGTERM handler's trigger condition. This
listener is a second, independent process with exactly one job, so it keeps
working even if the driving script is stuck.

Hardware path (RX): user presses the DOWN button on a Garmin watch -> ESP32
(connected to the watch, however that link works) -> ESP32 writes the ASCII
line "!!ESTOP!!\n" over USB serial to this Pi -> this script reads it and
kills the driving process(es).

Kill sequence: SIGTERM first (every RoboPacerV2 script already has a SIGTERM
handler that centers the servo, cuts ESC PWM, and releases the PCA9685
cleanly - see main.py/data_recorder.py's _handle_sigterm()), then SIGKILL
after SIGTERM_WAIT_SECONDS for anything that didn't exit in time. SIGKILL
skips that cleanup entirely (the process dies mid-instruction, PCA9685 keeps
outputting whatever it was last told) - which is exactly why the separate
esc_watchdog.sh service exists as a further fallback: it forces the ESC off
at the hardware register level if no known driving script is running at all,
regardless of why.

Hardware path (TX): this script also writes "!!HB!!\n" every
HEARTBEAT_INTERVAL_SECONDS. The ESP32 has its own hardware watchdog on the
other end - if it stops receiving that heartbeat for 2000ms (Pi crashed,
froze, USB cable fell out, this script got killed, ...), it cuts a relay
itself, independent of anything the Pi-side software is (or isn't) doing at
that moment. This is deliberately the same script/process as the ESTOP
reader above rather than a second one, because only one process can hold
/dev/ttyUSB0 open at a time - one process owns the port and handles both
directions.

Runs as a systemd service (see estop-listener.service in this same folder)
so it starts at boot and restarts automatically if it ever crashes.
"""

import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import psutil
import serial

PROJECT_DIR = "/home/pi/RoboPacerV2"
ESTOP_MARKER = "!!ESTOP!!"
HEARTBEAT_MESSAGE = b"!!HB!!\n"
RELAY_SOCKET = "/tmp/esp32_relay.sock"  # Unix socket pentru comenzi relay de la main.py / data_recorder.py
SCRIPTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts.json")
HEARTBEAT_INTERVAL_SECONDS = 0.5  # ESP32-side watchdog trips at 2000ms with
                                  # no heartbeat - well above this interval,
                                  # so one or two missed beats from normal
                                  # scheduling jitter won't false-trigger it.
BAUD_RATE = 115200
SERIAL_CANDIDATES = ["/dev/ttyUSB0", "/dev/ttyACM0"]
SIGTERM_WAIT_SECONDS = 5
RECONNECT_SLEEP_SECONDS = 2
# How many failed port-search attempts between "still waiting" log lines -
# at RECONNECT_SLEEP_SECONDS=2 this is once a minute. Avoids flooding
# journald for however long the ESP32 happens to be unplugged/rebooting.
QUIET_RETRY_LOG_EVERY = 30

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _handle_sigterm(signum, frame):
    """Let `systemctl stop`/plain `kill` unwind through the normal
    try/finally below (closes the serial port) instead of dying mid-read."""
    raise KeyboardInterrupt


def find_serial_port():
    for candidate in SERIAL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def connect_serial():
    """Blocks until the ESP32's serial port exists and opens successfully.
    Called both on startup and after any mid-run disconnect, so this is the
    single place the "wait for the device to (re)appear" logic lives."""
    attempt = 0
    while True:
        port = find_serial_port()
        if port is None:
            if attempt % QUIET_RETRY_LOG_EVERY == 0:
                logging.info(f"Niciun ESP32 gasit ({' / '.join(SERIAL_CANDIDATES)}) - astept...")
            attempt += 1
            time.sleep(RECONNECT_SLEEP_SECONDS)
            continue
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            logging.info(f"Conectat la {port} @ {BAUD_RATE} baud.")
            return ser
        except serial.SerialException as e:
            logging.warning(f"Nu pot deschide {port}: {e} - reincerc in {RECONNECT_SLEEP_SECONDS}s.")
            time.sleep(RECONNECT_SLEEP_SECONDS)


def load_scripts():
    """Incarca scripts.json — lista de scripturi Python care pot fi pornite de pe Garmin.
    Returneaza lista de dictionare {"name": ..., "path": ...} sau [] la eroare."""
    try:
        with open(SCRIPTS_JSON) as f:
            data = json.load(f)
        logging.info(f"Scripturi incarcate: {[s['name'] for s in data]}")
        return data
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Nu pot incarca {SCRIPTS_JSON}: {e} - lista vida.")
        return []


def build_scripts_msg(scripts):
    """Construieste mesajul !!SCRIPTS!!name1:path1|name2:path2 pentru ESP32."""
    parts = [f"{s['name']}:{s['path']}" for s in scripts]
    return ("!!SCRIPTS!!" + "|".join(parts) + "\n").encode()


def _socket_relay_loop(relay_queue, stop_event):
    """Listens on RELAY_SOCKET for RELAY_ON / RELAY_OFF commands from
    main.py / data_recorder.py. Puts ("ON",) or ("OFF",) into relay_queue;
    the main serial-read loop drains it and writes !!ON!! / !!OFF!! to the
    ESP32. Runs as a global daemon thread for the entire process lifetime —
    not per serial connection, since the socket binding is independent of
    whether the ESP32 is currently plugged in."""
    try:
        os.unlink(RELAY_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(RELAY_SOCKET)
    srv.listen(5)
    srv.settimeout(1.0)
    try:
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    data = conn.recv(64).decode("utf-8", errors="replace").strip()
                except OSError:
                    continue
                if data == "RELAY_ON":
                    relay_queue.put("ON")
                elif data == "RELAY_OFF":
                    relay_queue.put("OFF")
    finally:
        srv.close()
        try:
            os.unlink(RELAY_SOCKET)
        except OSError:
            pass


def _heartbeat_loop(ser, stop_event):
    """Runs on its own daemon thread for the lifetime of one serial
    connection - writes HEARTBEAT_MESSAGE every HEARTBEAT_INTERVAL_SECONDS
    until told to stop. Reading (main thread) and writing (this thread) the
    same pyserial Serial object concurrently is fine - they're independent
    fd operations - so no lock is needed around the write itself.

    stop_event.wait() rather than time.sleep(): identical idle behavior (one
    thread parked, no busy-waiting) but it also wakes immediately if the
    main thread sets stop_event on disconnect, instead of finishing out
    whatever's left of the current 0.5s sleep first.
    """
    while not stop_event.is_set():
        try:
            ser.write(HEARTBEAT_MESSAGE)
        except (serial.SerialException, OSError) as e:
            logging.warning(f"Heartbeat: eroare la scriere pe serial ({e}) - opresc heartbeat-ul.")
            stop_event.set()
            return
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


def _find_target_processes():
    """Every Python process (other than this one) whose command line
    references a path under PROJECT_DIR - main.py, data_recorder.py, etc.

    Filtering on "is this a python process" as well as "does PROJECT_DIR
    appear in argv" matters: esc_watchdog.sh's own systemd ExecStart is
    literally `bash /home/pi/RoboPacerV2/safety/esc_watchdog.sh`, so
    PROJECT_DIR alone would also match it - and killing that watchdog on
    ESTOP would remove exactly the independent hardware-level fallback the
    module docstring above describes as working "regardless of why" the
    driving script died. Confirmed by testing against a virtual serial pair
    before this filter was added - ESTOP killed the real esc-watchdog
    service running on this Pi.
    """
    self_pid = os.getpid()
    targets = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == self_pid:
            continue
        try:
            cmdline = proc.info.get("cmdline") or []
            name = (proc.info.get("name") or "").lower()
            is_python = name.startswith("python") or (
                cmdline and os.path.basename(cmdline[0]).lower().startswith("python"))
            if is_python and any(PROJECT_DIR in arg for arg in cmdline):
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return targets


def kill_target_processes():
    procs = _find_target_processes()
    if not procs:
        logging.info("ESTOP primit, dar niciun proces RoboPacerV2 activ de oprit.")
        return

    for proc in procs:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = "?"
        logging.warning(f"ESTOP: trimit SIGTERM catre PID {proc.pid} ({cmdline})")
        try:
            proc.terminate()  # SIGTERM
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(procs, timeout=SIGTERM_WAIT_SECONDS)
    for proc in gone:
        logging.info(f"PID {proc.pid} oprit curat (SIGTERM).")

    for proc in alive:
        logging.warning(f"PID {proc.pid} nu a raspuns la SIGTERM in {SIGTERM_WAIT_SECONDS}s - trimit SIGKILL.")
        try:
            proc.kill()  # SIGKILL
        except psutil.NoSuchProcess:
            pass

    if alive:
        gone2, alive2 = psutil.wait_procs(alive, timeout=2)
        for proc in gone2:
            logging.info(f"PID {proc.pid} oprit fortat (SIGKILL).")
        for proc in alive2:
            logging.error(f"PID {proc.pid} tot nu a murit dupa SIGKILL - posibil in stare D (uninterruptible).")

    logging.info(f"ESTOP gestionat: {len(procs)} proces(e) vizate.")


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    logging.info(f"estop_listener pornit (PID {os.getpid()}) - astept {ESTOP_MARKER!r} pe serial pentru {PROJECT_DIR}")

    scripts     = load_scripts()
    scripts_msg = build_scripts_msg(scripts)
    logging.info(f"[DBG] scripts_msg ce va fi trimis: {scripts_msg!r}")

    relay_queue = queue.Queue()
    socket_stop = threading.Event()
    socket_thread = threading.Thread(
        target=_socket_relay_loop, args=(relay_queue, socket_stop),
        name="relay-socket", daemon=True)
    socket_thread.start()
    logging.info(f"[DBG] relay-socket thread pornit, asculta pe {RELAY_SOCKET}")

    managed_proc = None  # procesul Python pornit de pe Garmin via !!RUN!!

    try:
        while True:
            ser = connect_serial()
            logging.info(f"[DBG] Serial deschis: {ser.port} @ {ser.baudrate}")

            # Trimite lista de scripturi la ESP32 dupa fiecare conectare
            try:
                ser.write(scripts_msg)
                logging.info(f"[DBG] !!SCRIPTS!! trimis catre ESP32 ({len(scripts_msg)} bytes): {scripts_msg!r}")
            except (serial.SerialException, OSError) as e:
                logging.error(f"[DBG] Eroare la trimiterea !!SCRIPTS!!: {e}")

            stop_event = threading.Event()
            hb_thread = threading.Thread(
                target=_heartbeat_loop, args=(ser, stop_event), name="heartbeat", daemon=True)
            hb_thread.start()
            logging.info("[DBG] Heartbeat thread pornit")

            try:
                while True:
                    # Trimite comenzi relay primite de la main.py / data_recorder.py
                    while not relay_queue.empty():
                        cmd = relay_queue.get_nowait()
                        msg = b"!!ON!!\n" if cmd == "ON" else b"!!OFF!!\n"
                        try:
                            ser.write(msg)
                            logging.info(f"[DBG] Relay cmd trimis: {msg!r}")
                        except (serial.SerialException, OSError) as e:
                            logging.error(f"[DBG] Eroare relay cmd: {e}")

                    raw = ser.readline()
                    if not raw:
                        if stop_event.is_set():
                            raise serial.SerialException("Heartbeat thread reported a write failure")
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    # Log orice linie primita de la ESP32 (exceptie: HB echo care ar fi spam)
                    if not line.startswith("[WD]") and "HB" not in line:
                        logging.info(f"[DBG] ESP32→RPi: {line!r}")

                    if ESTOP_MARKER in line:
                        logging.warning(f"ESTOP PRIMIT: {line!r}")
                        managed_proc = None
                        kill_target_processes()

                    elif line.startswith("!!RUN!!"):
                        logging.info(f"[DBG] Primit !!RUN!! linie: {line!r}")
                        try:
                            idx = int(line[len("!!RUN!!"):])
                        except ValueError:
                            logging.error(f"[DBG] !!RUN!! index invalid (nu e int): {line!r}")
                            continue
                        logging.info(f"[DBG] !!RUN!! index={idx}, scripturi disponibile={len(scripts)}")
                        if 0 <= idx < len(scripts):
                            path = scripts[idx]["path"]
                            logging.info(f"[DBG] Pornesc: python3 {path}")
                            # Opreste procesul curent daca ruleaza
                            if managed_proc is not None and managed_proc.poll() is None:
                                logging.info(f"[DBG] Opresc procesul curent PID {managed_proc.pid}")
                                managed_proc.terminate()
                            managed_proc = subprocess.Popen(["python3", path])
                            logging.info(f"[DBG] Pornit {path} (PID {managed_proc.pid})")
                        else:
                            logging.warning(f"[DBG] !!RUN!! index {idx} out of range (0..{len(scripts)-1})")

                    elif line == "!!STOP!!":
                        logging.info(f"[DBG] Primit !!STOP!!")
                        if managed_proc is not None and managed_proc.poll() is None:
                            logging.info(f"[DBG] Oprire script PID {managed_proc.pid}")
                            managed_proc.terminate()
                        else:
                            logging.info("[DBG] !!STOP!! primit dar niciun script nu rula")
                        managed_proc = None

            except (serial.SerialException, OSError) as e:
                logging.warning(f"Port serial deconectat ({e}) - reconectare in {RECONNECT_SLEEP_SECONDS}s.")
                logging.info(f"[DBG] Serial exception detalii: {type(e).__name__}: {e}")
            finally:
                stop_event.set()
                hb_thread.join(timeout=2)
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(RECONNECT_SLEEP_SECONDS)
    except KeyboardInterrupt:
        socket_stop.set()
        if managed_proc is not None and managed_proc.poll() is None:
            managed_proc.terminate()
        logging.info("estop_listener oprit.")


if __name__ == "__main__":
    main()
