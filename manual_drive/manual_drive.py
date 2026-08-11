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
"""

import logging
import os
import select
import signal
import socket
import threading
import time

import board
import busio
from adafruit_motor import servo as adafruit_servo
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, ff, list_devices

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
# Steering (servo on PCA9685 channel 0)
# ---------------------------------------------------------------------------
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900
SERVO_MAX_PULSE = 2200
SERVO_MIN_ANGLE = 45
SERVO_MAX_ANGLE = 135
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 4

# ---------------------------------------------------------------------------
# Throttle (ESC on PCA9685 channel 1) - pulse widths in microseconds. Vezi
# docstring-ul din data_recorder.py pentru explicatia completa a armarii.
# ---------------------------------------------------------------------------
ESC_CHANNEL = 1
PCA_FREQUENCY_HZ = 50
ESC_ARM_PULSE_US = 1500  # = ESC_NEUTRAL_US, nu ESC_MIN_US - vezi data_recorder.py
ESC_ARM_HOLD_SECONDS = 3.0
ESC_NEUTRAL_US = 1500
ESC_MIN_US = 1000
ESC_MAX_US = 1700

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
BTN_RESUME = ecodes.BTN_A
BTN_PAUSE = ecodes.BTN_Y
BTN_STOP = ecodes.BTN_B

AXIS_CENTER = 32767
AXIS_MAX = 65535
LABEL_DEADZONE = 0.2

# ---------------------------------------------------------------------------
# Odometrie (RPM de la ESP32 via estop_listener) - vezi tools/odometry.py,
# acelasi socket si aceeasi formula de conversie RPM -> km/h.
# ---------------------------------------------------------------------------
ODO_SOCKET = "/tmp/esp32_odometry.sock"
WHEEL_CIRCUMFERENCE_M = 0.1257  # diametru 40mm -> pi x 0.04
ODO_RECONNECT_SLEEP_SECONDS = 2.0
SPEED_LOG_INTERVAL_SECONDS = 0.5

# Vezi data_recorder.py - erori I2C tranzitorii de logat-si-continuat, nu de
# crash-uit programul pentru ele (bus/wiring glitch pe un sasiu care vibreaza).
_TRANSIENT_I2C_ERRNOS = (121, 19)


class SteeringServo:
    def __init__(self, pca):
        self._servo = adafruit_servo.Servo(
            pca.channels[SERVO_CHANNEL],
            min_pulse=SERVO_MIN_PULSE,
            max_pulse=SERVO_MAX_PULSE,
        )
        self.angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET
        self.center()

    def set_angle(self, angle):
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        self.angle = angle
        try:
            self._servo.angle = angle
        except OSError as e:
            if e.errno in _TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting servo angle {angle}: {e}")
            else:
                raise

    def center(self):
        self.set_angle(SERVO_NEUTRAL_ANGLE + SERVO_OFFSET)

    def release(self):
        try:
            self._servo.angle = None
        except OSError as e:
            if e.errno not in _TRANSIENT_I2C_ERRNOS:
                raise


class ESC:
    def __init__(self, pca):
        self._channel = pca.channels[ESC_CHANNEL]
        self._period_us = 1_000_000 / pca.frequency

    def _pulse_to_duty_cycle(self, pulse_us):
        pulse_us = max(ESC_MIN_US, min(ESC_MAX_US, pulse_us))
        return int((pulse_us / self._period_us) * 0xFFFF)

    def set_pulse_us(self, pulse_us):
        try:
            self._channel.duty_cycle = self._pulse_to_duty_cycle(pulse_us)
        except OSError as e:
            if e.errno in _TRANSIENT_I2C_ERRNOS:
                logging.warning(f"I2C error setting ESC pulse {pulse_us}: {e}")
            else:
                raise

    def neutral(self):
        self.set_pulse_us(ESC_NEUTRAL_US)

    def stop(self):
        try:
            self._channel.duty_cycle = 0
        except OSError as e:
            if e.errno not in _TRANSIENT_I2C_ERRNOS:
                raise

    def arm(self):
        print(f"Armare ESC: puls {ESC_ARM_PULSE_US}us timp de {ESC_ARM_HOLD_SECONDS:.0f}s...")
        self.set_pulse_us(ESC_ARM_PULSE_US)
        time.sleep(ESC_ARM_HOLD_SECONDS)
        self.neutral()
        print(f"ESC armat, setat la neutru ({ESC_NEUTRAL_US}us).")

    @staticmethod
    def pulse_from_gas_brake(gas_value, brake_value):
        gas_offset = (gas_value / 1023.0) * (ESC_MAX_US - ESC_NEUTRAL_US)
        brake_offset = -(brake_value / 1023.0) * (ESC_NEUTRAL_US - ESC_MIN_US)
        pulse = ESC_NEUTRAL_US + gas_offset + brake_offset
        return max(ESC_MIN_US, min(ESC_MAX_US, int(pulse)))


def steering_axis_to_angle(x_value):
    raw = 1.0 - 2.0 * (x_value / AXIS_MAX)  # 0..65535 -> +1..-1
    if abs(raw) <= LABEL_DEADZONE:
        norm = 0.0
    else:
        sign = 1.0 if raw > 0 else -1.0
        norm = sign * (abs(raw) - LABEL_DEADZONE) / (1.0 - LABEL_DEADZONE)
    angle = SERVO_NEUTRAL_ANGLE + norm * (SERVO_MAX_ANGLE - SERVO_NEUTRAL_ANGLE) + SERVO_OFFSET
    return int(max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle)))


def find_xbox_controller():
    for path in list_devices():
        dev = InputDevice(path)
        if "xbox" in dev.name.lower():
            return dev
    return None


_odo_state = {"rpm": 0.0}
_odo_lock = threading.Lock()


def _set_rpm(value):
    with _odo_lock:
        _odo_state["rpm"] = value


def _get_rpm():
    with _odo_lock:
        return _odo_state["rpm"]


def _odo_reader_loop(stop_event):
    """Thread separat, cat tine tot programul: tine la zi ultimul RPM primit
    de la estop_listener (acelasi socket/format ca tools/odometry.py). Daca
    socket-ul pica sau estop_listener nu ruleaza inca, RPM cade la 0 (nu
    ramane "inghetat" la ultima viteza cunoscuta) si se reincearca periodic -
    logul de viteza tot iese, doar cu valori 0 cat nu exista conexiune."""
    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(ODO_SOCKET)
        except OSError:
            stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)
            continue

        buf = ""
        try:
            while not stop_event.is_set():
                try:
                    data = sock.recv(64).decode("utf-8", errors="replace")
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("RPM:"):
                        continue
                    try:
                        _set_rpm(float(line[4:]))
                    except ValueError:
                        continue
        finally:
            sock.close()
            _set_rpm(0.0)
            if not stop_event.is_set():
                stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)


def _rpm_to_kmh(rpm):
    return rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000


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


_RELAY_SOCKET = "/tmp/esp32_relay.sock"


def _relay_cmd(cmd: str) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(_RELAY_SOCKET)
            s.sendall((cmd + "\n").encode())
    except (OSError, socket.timeout):
        pass


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    pca = None
    esc = None
    steering = None
    controller = None
    last_rumble_effect_id = None
    odo_thread = None
    odo_stop_event = None
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

    try:
        _relay_cmd("RELAY_ON")

        speed_log_path = os.path.join(
            SPEED_LOG_DIR, f"speed_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        speed_log_file = open(speed_log_path, "w")
        speed_log_file.write("elapsed_s,rpm,kmh,pace_mmss\n")
        speed_log_file.flush()
        print(f"Log de viteza: {speed_log_path}")

        odo_stop_event = threading.Event()
        odo_thread = threading.Thread(
            target=_odo_reader_loop, args=(odo_stop_event,), daemon=True)
        odo_thread.start()

        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = PCA_FREQUENCY_HZ

        esc = ESC(pca)
        steering = SteeringServo(pca)

        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        esc.arm()

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

            rpm = _get_rpm()
            kmh = _rpm_to_kmh(rpm)

            now = time.time()
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
        _relay_cmd("RELAY_OFF")
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

        if speed_log_file is not None:
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
                )
            else:
                summary = "\n--- SUMAR ---\nNiciun esantion inregistrat.\n"
            speed_log_file.write(summary)
            speed_log_file.close()
            print(summary)

        logging.info("Manual drive stopped")
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
