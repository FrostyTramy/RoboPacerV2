"""
RoboPacerV2 - Emergency Stop Listener (simplified)
====================================================
Doua responsabilitati:
  1. Trimite !!HB!! la ESP32 la fiecare 0.5s (watchdog-ul hardware al ESP32
     taie releul daca nu mai primeste heartbeat timp de 2s).
  2. Citeste !!ESTOP!! de la ESP32 si omoara toate procesele Python din
     PROJECT_DIR (main.py, data_recorder.py etc.) prin SIGTERM then SIGKILL.

Pe langa astea, tine o stare partajata (releu/ESP32/watchdog) interogabila
pe acelasi Unix socket ca RELAY_ON/RELAY_OFF, trimitand "STATUS" - folosita
de dashboard-ul web ca sa afiseze starea fara sa mai vorbeasca ea insasi cu
portul serial (portul e detinut exclusiv de procesul asta).

Ruleaza ca serviciu systemd (estop-listener.service) — porneste la boot,
se restarteaza automat daca se blocheaza.
"""

import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time

import psutil
import serial

PROJECT_DIR                = "/home/pi/RoboPacerV2"
ESTOP_MARKER               = "!!ESTOP!!"
HEARTBEAT_MESSAGE          = b"!!HB!!\n"
RELAY_SOCKET               = "/tmp/esp32_relay.sock"
HEARTBEAT_INTERVAL_SECONDS = 0.5
BAUD_RATE                  = 115200
SERIAL_CANDIDATES          = ["/dev/ttyUSB0", "/dev/ttyACM0"]
SIGTERM_WAIT_SECONDS       = 5
RECONNECT_SLEEP_SECONDS    = 2
QUIET_RETRY_LOG_EVERY      = 30

# Odometrie
ODO_SOCKET                 = "/tmp/esp32_odometry.sock"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Stare partajata intre thread-ul serial (scrie) si thread-ul de socket
# (citeste, ca sa raspunda la "STATUS"). Nu e nevoie de lock separat pentru
# citire/scriere de valori simple in Python (GIL), dar il folosim oricum ca
# sa nu trimitem un snapshot pe jumatate actualizat.
_state = {
    "esp32_connected":  False,
    "relay_on":         False,
    "watchdog_armed":   False,
    "last_esp32_line":  None,
    "last_update":      time.time(),
}
_state_lock = threading.Lock()


def _update_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)
        _state["last_update"] = time.time()


def _state_snapshot():
    with _state_lock:
        return dict(_state)


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def find_serial_port():
    for candidate in SERIAL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def connect_serial():
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
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = BAUD_RATE
            ser.timeout = 1
            # Fara asta placa ramane blocata in bootloader (GPIO0 tinut jos
            # de circuitul de auto-reset al CH340) si nu ajunge sa ruleze
            # firmware-ul deloc - simptom: nicio linie pe serial, ESP32
            # invizibil pe BLE, desi placa e alimentata si USB-ul e ok.
            ser.dtr = False
            ser.rts = False
            ser.open()
            logging.info(f"Conectat la {port} @ {BAUD_RATE} baud.")
            return ser
        except serial.SerialException as e:
            logging.warning(f"Nu pot deschide {port}: {e} - reincerc in {RECONNECT_SLEEP_SECONDS}s.")
            time.sleep(RECONNECT_SLEEP_SECONDS)


_odo_clients: list = []
_odo_lock = threading.Lock()


def _odo_server_loop(stop_event):
    """Accepta clienti pe ODO_SOCKET si le trimite fiecare linie RPM:xxx."""
    try:
        os.unlink(ODO_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(ODO_SOCKET)
    srv.listen(5)
    srv.settimeout(1.0)
    try:
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
                with _odo_lock:
                    _odo_clients.append(conn)
            except socket.timeout:
                continue
    finally:
        srv.close()
        try:
            os.unlink(ODO_SOCKET)
        except OSError:
            pass


def _odo_broadcast(line: str):
    """Trimite linia RPM:xxx tuturor clientilor conectati."""
    data = (line + "\n").encode()
    with _odo_lock:
        dead = []
        for conn in _odo_clients:
            try:
                conn.sendall(data)
            except OSError:
                dead.append(conn)
        for conn in dead:
            _odo_clients.remove(conn)


def _socket_relay_loop(relay_queue, stop_event):
    """Asculta pe RELAY_SOCKET comenzi RELAY_ON / RELAY_OFF de la scripturi Python.
    Pune 'ON' sau 'OFF' in relay_queue; loop-ul principal scrie !!ON!!/!!OFF!! la ESP32."""
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
                elif data == "STATUS":
                    try:
                        conn.sendall(json.dumps(_state_snapshot()).encode() + b"\n")
                    except OSError:
                        pass
    finally:
        srv.close()
        try:
            os.unlink(RELAY_SOCKET)
        except OSError:
            pass


def _heartbeat_loop(ser, stop_event):
    while not stop_event.is_set():
        try:
            ser.write(HEARTBEAT_MESSAGE)
        except (serial.SerialException, OSError) as e:
            logging.warning(f"Heartbeat: eroare la scriere ({e})")
            stop_event.set()
            return
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


# Scripturi excluse de la kill-ul de ESTOP desi ruleaza sub PROJECT_DIR:
# dashboard/app.py nu atinge niciodata motorul/servo-ul direct (doar
# porneste/opreste alte scripturi si citeste stare) - nu exista niciun
# motiv de siguranta sa fie omorat. Inainte era omorat la FIECARE ESTOP
# (inclusiv unul declansat de la ceas, complet independent de vreun script
# de condus), systemd il repornea imediat (Restart=always), iar in acel
# interval de haos dashboard-ul vechi/nou + un script de condus orfan
# ajungeau sa scrie concurrent pe acelasi canal PCA9685 - confirmat empiric
# cu un monitor live pe registrele PCA9685: un puls neutru corect scris de
# scriptul nou era suprascris cu "full-off" la cateva sute de ms, exact in
# fereastra de armare a ESC-ului.
_KILL_EXCLUDE_BASENAMES = ("dashboard/app.py",)


def _find_target_processes():
    self_pid = os.getpid()
    targets = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid == self_pid:
            continue
        try:
            cmdline = proc.info.get("cmdline") or []
            name    = (proc.info.get("name") or "").lower()
            is_python = name.startswith("python") or (
                cmdline and os.path.basename(cmdline[0]).lower().startswith("python"))
            if not is_python or not any(PROJECT_DIR in arg for arg in cmdline):
                continue
            if any(excl in arg for arg in cmdline for excl in _KILL_EXCLUDE_BASENAMES):
                continue
            targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return targets


def kill_target_processes():
    procs = _find_target_processes()
    if not procs:
        logging.info("ESTOP primit, dar niciun proces RoboPacerV2 activ.")
        return

    for proc in procs:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = "?"
        logging.warning(f"ESTOP: SIGTERM -> PID {proc.pid} ({cmdline})")
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(procs, timeout=SIGTERM_WAIT_SECONDS)
    for proc in gone:
        logging.info(f"PID {proc.pid} oprit (SIGTERM).")
    for proc in alive:
        logging.warning(f"PID {proc.pid} nu a raspuns — SIGKILL.")
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    logging.info(f"ESTOP gestionat: {len(procs)} proces(e) vizate.")


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    logging.info(f"estop_listener pornit (PID {os.getpid()})")

    relay_queue = queue.Queue()
    socket_stop = threading.Event()
    socket_thread = threading.Thread(
        target=_socket_relay_loop, args=(relay_queue, socket_stop),
        name="relay-socket", daemon=True)
    socket_thread.start()
    logging.info(f"Relay socket pornit pe {RELAY_SOCKET}")

    odo_thread = threading.Thread(
        target=_odo_server_loop, args=(socket_stop,),
        name="odo-server", daemon=True)
    odo_thread.start()
    logging.info(f"Odometry socket pornit pe {ODO_SOCKET}")

    try:
        while True:
            ser = connect_serial()
            _update_state(esp32_connected=True)

            stop_event = threading.Event()
            hb_thread  = threading.Thread(
                target=_heartbeat_loop, args=(ser, stop_event),
                name="heartbeat", daemon=True)
            hb_thread.start()
            logging.info("Heartbeat thread pornit.")

            try:
                while True:
                    # Trimite comenzi relay primite de la scripturi Python
                    while not relay_queue.empty():
                        cmd = relay_queue.get_nowait()
                        msg = b"!!ON!!\n" if cmd == "ON" else b"!!OFF!!\n"
                        try:
                            ser.write(msg)
                            logging.info(f"Relay cmd trimis: {msg!r}")
                        except (serial.SerialException, OSError) as e:
                            logging.error(f"Eroare relay cmd: {e}")

                    raw = ser.readline()
                    if not raw:
                        if stop_event.is_set():
                            raise serial.SerialException("Heartbeat thread raportat eroare")
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    if line.startswith("RPM:"):
                        _odo_broadcast(line)
                        continue

                    if "[WD]" not in line and "HB" not in line:
                        logging.info(f"ESP32: {line!r}")

                    _update_state(last_esp32_line=line)
                    if "[RELAY]" in line and "Ignorat" not in line:
                        _update_state(relay_on="ON" in line)
                    elif line == "[WD] Armat":
                        _update_state(watchdog_armed=True)
                    elif "[WD]" in line and "pierdut" in line:
                        _update_state(watchdog_armed=False)

                    if ESTOP_MARKER in line:
                        logging.warning(f"ESTOP PRIMIT: {line!r}")
                        kill_target_processes()

            except (serial.SerialException, OSError) as e:
                logging.warning(f"Port serial deconectat ({e}) - reconectare in {RECONNECT_SLEEP_SECONDS}s.")
            finally:
                stop_event.set()
                hb_thread.join(timeout=2)
                _update_state(esp32_connected=False, watchdog_armed=False)
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(RECONNECT_SLEEP_SECONDS)

    except KeyboardInterrupt:
        socket_stop.set()
        logging.info("estop_listener oprit.")


if __name__ == "__main__":
    main()
