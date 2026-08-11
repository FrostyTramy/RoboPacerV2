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
import time

import board
import busio
from adafruit_motor import servo as adafruit_servo
from adafruit_pca9685 import PCA9685
from evdev import InputDevice, ecodes, ff, list_devices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "manual_drive.log")

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

    try:
        _relay_cmd("RELAY_ON")
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

            print(f"{'PAUZA' if is_paused else 'ACTIV '} | viraj {steering.angle:3d} | gas {gas_value:4d} | brake {brake_value:4d}", end="\r")
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
        logging.info("Manual drive stopped")
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
