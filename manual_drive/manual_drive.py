"""
RoboPacerV2 - Manual Drive
===========================
Doar control direct al masinii cu un controller Xbox, prin PCA9685 (servo =
viraj pe canalul 0, ESC = acceleratie pe canalul 1). Fara camera, fara
inregistrare de cadre - vezi data_recorder/data_recorder.py pentru varianta
care salveaza dataset.

Butoane (acelasi layout ca la data_recorder):
    [A] - RESUME (reia controlul cu stick-ul)
    [Y] - PAUZA  (servo -> centru, ESC -> neutru, stick-ul e ignorat complet
                  pana la RESUME - spre deosebire de data_recorder, unde
                  pauza opreste doar inregistrarea si masina tot raspunde la
                  stick; aici chiar nu se misca nimic in pauza)
    [B] - STOP   (opreste programul)

Porneste direct in stare ACTIVA (raspunde la stick din prima clipa, dupa
armarea ESC-ului) - [A]/[Y] doar comuta intre activ si pauza.

Foloseste config/ (ESC/SteeringServo, init PCA9685, odometrie, releu, cautare
controller, conversia axei de steering) in loc sa duplice boilerplate-ul -
vezi main/main.py pentru acelasi tipar. Formatul logului de viteza si
protocolul socket-ului de control raman neschimbate.
"""

import json
import logging
import os
import select
import signal
import socket
import sys
import threading
import time

from evdev import ecodes, ff

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "manual_drive.log")
SPEED_LOG_DIR = os.path.join(BASE_DIR, "speed_logs")
os.makedirs(SPEED_LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Shared config/ package (repo root) - see config/ for the single source of
# truth on every constant/class below, consolidated from manual_drive.py/
# data_recorder.py/cruise_control.py where they used to be duplicated.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from config.pca9685_init import init_pca9685
from config.servo_esc import ESC, SteeringServo
from config.estop import relay_cmd
from config.odometry import get_rpm, odo_reader_loop, rpm_to_kmh
from config.input_devices import find_xbox_controller
from config.joystick_steering import steering_axis_to_angle
from config.ipc_config import MANUAL_DRIVE_CONTROL_SOCKET

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
BTN_RESUME = ecodes.BTN_A
BTN_PAUSE = ecodes.BTN_Y
BTN_STOP = ecodes.BTN_B

SPEED_LOG_INTERVAL_SECONDS = 0.5


# Distanta traita separat de total_distance_m din main(): dashboard-ul web
# afiseaza _distance_state["total_m"] - _distance_state["reset_offset_m"]
# (ca un cronometru de trip), dar sumarul final de la sfarsitul rularii tot
# raporteaza distanta totala neatinsa de reset-uri - vezi total_distance_m.
_distance_state = {"total_m": 0.0, "reset_offset_m": 0.0}
_distance_lock = threading.Lock()

# Viteza/pauza curente, pentru afisajul live din pagina web (nu au nevoie
# de reset - se actualizeaza pur si simplu la fiecare tick al buclei).
_live_state = {"kmh": 0.0, "paused": False}
_live_lock = threading.Lock()

# Medie/maxim viteza de la ultimul reset - la fel ca in sumarul final,
# media ia in calcul doar cat timp masina chiar s-a miscat (kmh > 0).
_speed_stats_state = {"kmh_sum": 0.0, "kmh_count": 0, "max_kmh": 0.0}
_speed_stats_lock = threading.Lock()


def _update_distance(total_m):
    with _distance_lock:
        _distance_state["total_m"] = total_m


def _get_display_distance_m():
    with _distance_lock:
        return _distance_state["total_m"] - _distance_state["reset_offset_m"]


def _reset_display_distance():
    with _distance_lock:
        _distance_state["reset_offset_m"] = _distance_state["total_m"]


def _update_live(kmh, paused):
    with _live_lock:
        _live_state["kmh"] = kmh
        _live_state["paused"] = paused


def _get_live_state():
    with _live_lock:
        return dict(_live_state)


def _record_speed_sample(kmh):
    with _speed_stats_lock:
        if kmh > 0:
            _speed_stats_state["kmh_sum"] += kmh
            _speed_stats_state["kmh_count"] += 1
        if kmh > _speed_stats_state["max_kmh"]:
            _speed_stats_state["max_kmh"] = kmh


def _get_speed_stats():
    with _speed_stats_lock:
        count = _speed_stats_state["kmh_count"]
        avg_kmh = _speed_stats_state["kmh_sum"] / count if count else 0.0
        return avg_kmh, _speed_stats_state["max_kmh"]


def _reset_speed_stats():
    with _speed_stats_lock:
        _speed_stats_state["kmh_sum"] = 0.0
        _speed_stats_state["kmh_count"] = 0
        _speed_stats_state["max_kmh"] = 0.0


def _control_server_loop(stop_event):
    """Socket local pentru dashboard-ul web: STATUS intoarce distanta live
    (afisata pe pagina scriptului), RESET_DISTANCE o pune la zero (doar
    afisajul - sumarul final tot raporteaza distanta totala reala)."""
    try:
        os.unlink(MANUAL_DRIVE_CONTROL_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(MANUAL_DRIVE_CONTROL_SOCKET)
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
                if data == "STATUS":
                    live = _get_live_state()
                    avg_kmh, max_kmh = _get_speed_stats()
                    payload = {
                        "distance_m": round(_get_display_distance_m(), 2),
                        "kmh": round(live["kmh"], 2),
                        "pace": _format_pace(live["kmh"]),
                        "paused": live["paused"],
                        "avg_kmh": round(avg_kmh, 2),
                        "avg_pace": _format_pace(avg_kmh),
                        "max_kmh": round(max_kmh, 2),
                        "max_pace": _format_pace(max_kmh),
                    }
                    try:
                        conn.sendall((json.dumps(payload) + "\n").encode())
                    except OSError:
                        pass
                elif data == "RESET_DISTANCE":
                    _reset_display_distance()
                    _reset_speed_stats()
                    try:
                        conn.sendall(b'{"ok": true}\n')
                    except OSError:
                        pass
    finally:
        srv.close()
        try:
            os.unlink(MANUAL_DRIVE_CONTROL_SOCKET)
        except OSError:
            pass


def _format_pace(kmh):
    """mm:ss per km, ca la un ceas de alergare - "--:--" cand nu se misca."""
    if kmh <= 0:
        return "--:--"
    pace_sec = 3600 / kmh
    return f"{int(pace_sec // 60)}:{int(pace_sec % 60):02d}"


def _handle_sigterm(signum, frame):
    """Vezi data_recorder.py - fara asta SIGTERM sare peste `finally` si lasa
    ESC-ul la ultimul puls trimis."""
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    pca = None
    esc = None
    steering = None
    controller = None
    last_rumble_effect_id = None
    odo_thread = None
    odo_stop_event = None
    control_thread = None
    control_stop_event = None
    speed_log_file = None
    speed_samples = []  # (rpm, kmh) la fiecare SPEED_LOG_INTERVAL_SECONDS

    def rumble(duration_ms):
        """Vezi data_recorder.py - rumble non-blocant, un singur efect activ
        odata ca sa nu epuizam sloturile FF ale controller-ului."""
        nonlocal last_rumble_effect_id
        if controller is None:
            return
        if last_rumble_effect_id is not None:
            try:
                controller.erase_effect(last_rumble_effect_id)
            except OSError:
                pass
            last_rumble_effect_id = None
        try:
            effect = ff.Effect(
                ecodes.FF_RUMBLE, -1, 0,
                ff.Trigger(0, 0),
                ff.Replay(duration_ms, 0),
                ff.EffectType(ff_rumble_effect=ff.Rumble(strong_magnitude=0xFFFF, weak_magnitude=0xFFFF)),
            )
            last_rumble_effect_id = controller.upload_effect(effect)
            controller.write(ecodes.EV_FF, last_rumble_effect_id, 1)
        except OSError as e:
            logging.warning(f"Rumble esuat: {e}")

    # Definit inaintea oricarui cod care poate arunca exceptie - blocul
    # `finally` foloseste asta la calculul duratei chiar si daca pornirea
    # a picat devreme (ex: controller negasit, log de viteza nescriibil).
    run_start_time = time.time()
    total_distance_m = 0.0

    try:
        # PCA/ESC configurat SI semnal de neutru trimis inainte de a alimenta
        # ESC-ul prin releu - la fel ca in data_recorder.py/main.py. ESC-urile
        # asteapta semnal PWM valid din clipa in care primesc curent; daca stau
        # alimentate cateva secunde fara semnal, multe intra in failsafe.
        pca = init_pca9685()

        esc = ESC(pca)
        esc.neutral()
        steering = SteeringServo(pca)

        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        relay_cmd("RELAY_ON")

        esc.arm()

        speed_log_path = os.path.join(
            SPEED_LOG_DIR, f"speed_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        speed_log_file = open(speed_log_path, "w")
        speed_log_file.write("elapsed_s,rpm,kmh,pace_mmss\n")
        speed_log_file.flush()
        print(f"Log de viteza: {speed_log_path}")

        odo_stop_event = threading.Event()
        odo_thread = threading.Thread(
            target=odo_reader_loop, args=(odo_stop_event,), daemon=True)
        odo_thread.start()

        control_stop_event = threading.Event()
        control_thread = threading.Thread(
            target=_control_server_loop, args=(control_stop_event,), daemon=True)
        control_thread.start()

        print("\n-----------------------------------------------------")
        print("Masina e ACTIVA - raspunde la stick din prima clipa.")
        print("Apasa [Y] pentru PAUZA (servo+motor opresc, stick-ul e ignorat).")
        print("Apasa [A] pentru RESUME.")
        print("Apasa [B] pentru STOP.")
        print("Ctrl+C opreste programul la fel ca [B].")
        print("-----------------------------------------------------")

        gas_value = 0
        brake_value = 0
        is_paused = False
        last_speed_log_time = run_start_time
        last_distance_time = run_start_time

        while True:
            ready, _, _ = select.select([controller_fd], [], [], 0.001)

            for _ in ready:
                for event in controller.read():
                    if event.type == ecodes.EV_ABS and not is_paused:
                        abs_name = ecodes.ABS.get(event.code)

                        if abs_name == "ABS_GAS":
                            gas_value = event.value
                        elif abs_name == "ABS_BRAKE":
                            brake_value = event.value
                        elif abs_name == "ABS_X":
                            angle = steering_axis_to_angle(event.value)
                            steering.set_angle(angle)

                        if abs_name in ("ABS_GAS", "ABS_BRAKE"):
                            pulse = ESC.pulse_from_gas_brake(gas_value, brake_value)
                            esc.set_pulse_us(pulse)

                    elif event.type == ecodes.EV_KEY and event.value == 1:
                        if event.code == BTN_RESUME:
                            if is_paused:
                                is_paused = False
                                print("\n>>> RESUME (A) <<<")
                                rumble(500)
                        elif event.code == BTN_PAUSE:
                            if not is_paused:
                                is_paused = True
                                gas_value = 0
                                brake_value = 0
                                esc.neutral()
                                steering.center()
                                print("\n>>> PAUZA (Y) <<<")
                                rumble(1000)
                        elif event.code == BTN_STOP:
                            raise KeyboardInterrupt

            rpm = get_rpm()
            kmh = rpm_to_kmh(rpm)

            now = time.time()
            total_distance_m += (kmh / 3.6) * (now - last_distance_time)
            last_distance_time = now
            _update_distance(total_distance_m)
            _update_live(kmh, is_paused)
            _record_speed_sample(kmh)

            if now - last_speed_log_time >= SPEED_LOG_INTERVAL_SECONDS:
                last_speed_log_time = now
                pace = _format_pace(kmh)
                speed_samples.append((rpm, kmh))
                speed_log_file.write(f"{now - run_start_time:.1f},{rpm:.2f},{kmh:.2f},{pace}\n")
                speed_log_file.flush()

            print(
                f"{'PAUZA' if is_paused else 'ACTIV '} | viraj {steering.angle:3d} | "
                f"gas {gas_value:4d} | brake {brake_value:4d} | "
                f"{kmh:5.2f} km/h | pace {_format_pace(kmh)}/km",
                end="\r",
            )
            time.sleep(0.01)

    except (KeyboardInterrupt, ConnectionError) as e:
        if isinstance(e, ConnectionError):
            print(f"\n{e}")
        print("\nOprire...")
    except Exception as e:
        logging.exception("Eroare majora neasteptata")
        print(f"\nEroare majora neasteptata: {e}")
    finally:
        # Ignoram SIGTERM-uri ulterioare din acest punct incolo. RELAY_OFF
        # de mai jos face ESP32-ul sa emita !!ESTOP!! (trateaza orice
        # oprire de releu la fel, indiferent de cauza), care ne mai trimite
        # un SIGTERM noua - fara asta, al doilea semnal ridica un nou
        # KeyboardInterrupt chiar in mijlocul lui `finally`, fara alt
        # try/except in jur, si Python abandoneaza curatarea pe loc
        # (ESC/servo pot ramane neoprite, sumarul nu se mai scrie).
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        relay_cmd("RELAY_OFF")
        if esc is not None:
            esc.neutral()
            time.sleep(0.1)
            esc.stop()
            rumble(150)
            time.sleep(0.25)
            rumble(150)
            time.sleep(0.2)
        if steering is not None:
            steering.release()
        if pca is not None:
            try:
                pca.deinit()
            except OSError as e:
                logging.warning(f"I2C error during pca.deinit(): {e}")

        if odo_stop_event is not None:
            odo_stop_event.set()
        if odo_thread is not None:
            odo_thread.join(timeout=2)

        if control_stop_event is not None:
            control_stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2)

        # Distanta = integrala vitezei (km/h -> m/s) pe durata fiecarui tick al
        # buclei principale (vezi total_distance_m += ... mai sus) - nu vine
        # direct de la ESP32, care trimite doar RPM instantaneu.
        total_cm = round(total_distance_m * 100)
        distance_m, distance_cm = divmod(total_cm, 100)
        distance_line = (
            f"Distanta -> {distance_m} m {distance_cm} cm "
            f"({total_distance_m / 1000:.3f} km)\n"
        )

        if speed_samples:
            rpm_vals = [s[0] for s in speed_samples]
            kmh_vals = [s[1] for s in speed_samples]
            moving_kmh_vals = [k for k in kmh_vals if k > 0]
            avg_rpm = sum(rpm_vals) / len(rpm_vals)
            max_rpm = max(rpm_vals)
            avg_kmh_moving = sum(moving_kmh_vals) / len(moving_kmh_vals) if moving_kmh_vals else 0.0
            max_kmh = max(kmh_vals)
            summary = (
                "\n--- SUMAR ---\n"
                f"Durata: {time.time() - run_start_time:.1f}s | {len(speed_samples)} esantioane (la {SPEED_LOG_INTERVAL_SECONDS}s)\n"
                f"RPM   -> mediu {avg_rpm:.2f} | maxim {max_rpm:.2f}\n"
                f"km/h  -> mediu {avg_kmh_moving:.2f} (cat timp s-a miscat) | maxim {max_kmh:.2f}\n"
                f"Pace  -> mediu {_format_pace(avg_kmh_moving)}/km | cel mai bun {_format_pace(max_kmh)}/km\n"
                f"{distance_line}"
            )
        else:
            summary = f"\n--- SUMAR ---\nNiciun esantion inregistrat.\n{distance_line}"

        if speed_log_file is not None:
            speed_log_file.write(summary)
            speed_log_file.close()
            print(summary)

        logging.info("Manual drive stopped\n" + summary)
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
