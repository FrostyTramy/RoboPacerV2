"""
RoboPacerV2 - Cruise Control
==============================
Steering-ul raspunde mereu la stick-ul stang (ca la manual_drive.py) - doar
viteza e automata, tinuta constanta de un regulator PI care ajusteaza
pulsul ESC pe baza vitezei reale citite de la senzorul de pe roata.

Butoane:
    [Dpad SUS] - creste viteza tinta cu 1 km/h
    [Dpad JOS] - scade viteza tinta cu 1 km/h
    [A] - PORNESTE cruise control la viteza tinta curenta - deschide un
          fisier de log nou pentru "cursa" asta
    [Y] - OPRESTE cursa curenta (ESC -> neutru), salveaza logul cu sumar
          (viteza medie, cat de aproape a fost de tinta), si reseteaza
          totul (inclusiv tinta) la 0 - gata pentru o cursa noua cu [A]
    [B] - opreste tot programul (salveaza orice log deschis, oprire
          hardware completa la fel ca la manual_drive.py)

Regulatorul PI nu franeaza/da niciodata inapoi (pulsul nu coboara sub
neutru) - un ESC in modul "Forward/Reverse with Brake" poate interpreta
comenzi variabile de frana ca "a doua apasare" si sa intre in reverse la
throttle complet (vezi avertismentul din data_recorder.py). Daca masina
merge mai repede decat tinta (ex. la vale), pur si simplu nu se mai da
gaz, nu se franeaza activ.

Siguranta suplimentara fata de manual_drive.py: daca senzorul de viteza nu
mai trimite date proaspete (socket picat, estop_listener oprit) mai mult
de ODO_STALE_TIMEOUT_SECONDS, cruise control-ul se opreste automat (ESC ->
neutru) in loc sa continue sa "creada" ca viteza e 0 si sa dea throttle tot
mai mare incercand sa ajunga la tinta - fara feedback real, asta ar putea
accelera necontrolat.
"""

import json
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
LOG_FILE_PATH = os.path.join(BASE_DIR, "cruise_control.log")
SPEED_LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SPEED_LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Steering (servo on PCA9685 channel 0) - identic cu manual_drive.py
# ---------------------------------------------------------------------------
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900
SERVO_MAX_PULSE = 2200
SERVO_MIN_ANGLE = 45
SERVO_MAX_ANGLE = 135
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 4

# ---------------------------------------------------------------------------
# Throttle (ESC on PCA9685 channel 1) - identic cu manual_drive.py
# ---------------------------------------------------------------------------
ESC_CHANNEL = 1
PCA_FREQUENCY_HZ = 50
ESC_ARM_PULSE_US = 1500
ESC_ARM_HOLD_SECONDS = 3.0
ESC_NEUTRAL_US = 1500
ESC_MIN_US = 1000
ESC_MAX_US = 1700

# Cat de aproape de ESC_MAX_US conteaza "puls la maxim" (pulsul scris in log
# are o singura zecimala, deci nu asteptam egalitate exacta) si peste ce
# procent din esantioanele unei curse adaugam nota din _format_summary -
# vezi acolo.
ESC_PULSE_SATURATION_EPSILON_US = 2.0
ESC_PULSE_SATURATION_NOTE_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Regulator = feedforward + PI (nu doar PI pur).
#
# Feedforward: pornim direct de la un puls estimat din viteza maxima reala
# masurata (CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH la puls maxim), presupunand
# o relatie aproximativ liniara puls<->viteza. Asta rezolva exact problema
# observata in loguri: un PI care porneste mereu de la neutru trebuie sa
# "urce" spre puls-ul corect doar din eroare acumulata, ceea ce da o eroare
# de regim stabil mare si constanta procentual (~14-17% sub tinta la 9/10/15
# km/h in cele 9 loguri analizate - inclusiv o cursa de 60.7s cu aceeasi
# abatere ca una de 31s, deci nu era problema de timp, era structurala).
#
# PI-ul ramane, dar acum are un job mult mai mic: doar corecteaza diferenta
# dintre modelul liniar si realitate (frecare, panta, tensiune baterie care
# scade in timp) - integrala creste incet pulsul pe masura ce bateria se
# descarca, exact comportamentul cerut ("motorul e sensored, odata gasit
# pulsul ramane cam acolo, doar bateria il face sa mai creasca usor").
# ---------------------------------------------------------------------------
CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH = 23.0  # masurat de utilizator, la puls maxim (ESC_MAX_US)

# Prioritate schimbata: utilizatorul nu tine la medie, tine la constanta -
# "mai bine 1km constant la 4:00/km decat 500m la 3:00 + 500m la 5:00, chiar
# daca media iese identica". Un log real (114.8s, tinta 10km/h) a aratat
# medie aproape perfecta (10.05) DAR cu un varf de lansare la 12.79 (in
# primele 4s) si mici depasiri (~11-11.5) dupa curbe reale - regulatorul
# "impinge" prea ferm ca sa recupereze media exact. Kp/Ki mai mici + plafon
# de panta mai strans = raspuns mai bland peste tot, cu riscul unei medii
# usor mai imprecise (acceptabil - nu mai e obiectivul).
CRUISE_KP_US_PER_KMH = 10.0     # corectie proportionala pe eroarea RAMASA dupa feedforward
CRUISE_KI_US_PER_KMH_S = 3.0    # component integral - elimina eroarea ramasa in regim stabil
CRUISE_INTEGRAL_MAX_US = 50.0   # anti-windup - limiteaza cat poate acumula integrala (mai strans
                                 # ca inainte, ca o perturbare de 1-2s - o curba - sa nu poata
                                 # "incarca" destula integrala cat sa dea o depasire vizibila dupa)

# Doua plafoane diferite pentru limitarea de panta pe pulsul APLICAT (nu cel
# "cerut" de PI):
#   - LAUNCH (lent) - doar in primele CRUISE_LAUNCH_SECONDS de la [A], cat
#     timp masina sta pe loc si senzorul inca nu are date proaspete (vezi
#     ODO_STALE_GRACE_SECONDS) - fara plafonul asta, la [A] eroarea initiala
#     e tinta intreaga si regulatorul ar cere throttle aproape maxim din
#     prima iteratie, inainte sa existe vreo confirmare reala de viteza.
#   - CRUISE - dupa aceea, cat timp cruise control-ul e activ. Redus fata de
#     versiunea anterioara (era 600, gandit sa reactioneze cat mai rapid) -
#     acum prioritatea e constanta, nu recuperarea rapida, deci plafonul
#     limiteaza direct cat de brusc se poate schimba pulsul, indiferent ce
#     "vrea" regulatorul.
CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S = 120.0
CRUISE_MAX_PULSE_STEP_US_PER_S = 180.0
CRUISE_LAUNCH_SECONDS = 2.0

# Filtru de netezire (low-pass) pe viteza citita, INAINTE sa ajunga la
# regulator. Senzorul are doar 4 magneti pe roata, iar fiecare citire de
# RPM vine dintr-un SINGUR interval intre 2 treceri consecutive - daca
# magnetii nu sunt lipiti perfect echidistant (practic imposibil manual),
# fiecare citire individuala are zgomot real chiar la viteza fizica
# constanta (confirmat empiric: log real la tinta 10km/h arata viteza
# sarind intre ~8.5 si ~12.8 km/h de la o mostra la alta, chiar pe portiuni
# drepte). Un regulator ajustat sa reactioneze rapid (vezi mai sus)
# reactioneaza si la zgomotul asta, nu doar la abateri reale - de-aici
# "ruperile de ritm". Filtrul netezeste zgomotul dar tot urmareste
# schimbari reale sustinute (ex. o curba) in cateva zecimi de secunda.
#
# Crescut fata de prima versiune (0.4) - prioritatea e constanta, nu viteza
# de reactie la o curba, deci accept mai mult lag de filtru in schimbul unui
# semnal mai neted.
CRUISE_SPEED_FILTER_TAU_S = 0.7

TARGET_SPEED_STEP_KMH = 1.0
# Plafonul real vine din puls (ESC_MAX_US, clamp-uit in cruise_pulse_us si
# din nou in ESC.set_pulse_us), nu din tinta - deci aici e doar o limita
# "de bun simt" ca sa nu se poata cere ceva absurd de mare din dpad, nu o
# protectie de siguranta. Daca tinta ceruta e peste ce poate motorul/bateria
# da efectiv, regulatorul tine pulsul la maxim (ESC-ul primeste acelasi
# semnal de "gaz maxim" ca la orice alta tinta neatinsa) si _format_summary
# noteaza in log ca abaterea a fost cauzata de plafonul fizic, nu de un bug.
TARGET_SPEED_MAX_KMH = 100.0

# ---------------------------------------------------------------------------
# Bucla externa (lenta) de recuperare a mediei - deasupra regulatorului
# rapid de mai sus, nu il inlocuieste. Urmareste distanta acumulata fata de
# "cat ar fi trebuit sa parcurga daca mergea constant la tinta" si adauga o
# mica corectie la tinta (nu la puls direct) ca sa recupereze diferenta in
# CRUISE_CATCHUP_WINDOW_S secunde - un deficit mic, recuperat lin, nu o
# smucitura. Regulatorul rapid (Kp/Ki/panta/filtru) ramane neschimbat si
# tine "tinta efectiva" (tinta + corectie) la fel de neted cum tine orice
# alta tinta.
# ---------------------------------------------------------------------------
CRUISE_CATCHUP_WINDOW_S = 10.0     # in cate secunde se recupereaza un deficit de distanta
CRUISE_CATCHUP_MAX_EXTRA_KMH = 3.0  # plafon pe cat de mult poate cere bucla externa peste tinta

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
BTN_ENGAGE = ecodes.BTN_A
BTN_RESET = ecodes.BTN_Y
BTN_STOP = ecodes.BTN_B

AXIS_MAX = 65535
LABEL_DEADZONE = 0.2

# ---------------------------------------------------------------------------
# Odometrie (RPM de la ESP32 via estop_listener) - acelasi socket/formula ca
# manual_drive.py si tools/odometry.py.
# ---------------------------------------------------------------------------
ODO_SOCKET = "/tmp/esp32_odometry.sock"
WHEEL_CIRCUMFERENCE_M = 0.1369  # diametru 43.58mm - vezi manual_drive.py pentru istoricul calibrarii
ODO_RECONNECT_SLEEP_SECONDS = 2.0
ODO_STALE_TIMEOUT_SECONDS = 1.0  # peste atat fara date noi = oprire automata a cruise control-ului
ODO_STALE_GRACE_SECONDS = 2.0    # dupa [A], ignoram staleness-ul atata timp - ESP32-ul nu trimite
                                  # nicio linie "RPM:" cat roata nu s-a miscat deloc (nici "0.00"),
                                  # deci o masina pornita de pe loc pare mereu "stale" in prima clipa

TICK_INTERVAL_SECONDS = 0.1  # cadenta log CSV + afisaj live in terminal
SPLIT_DISTANCE_M = 100.0  # la fiecare atatia metri parcursi, afiseaza + logheaza timpul sutei

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


def steering_axis_to_angle(x_value):
    raw = 1.0 - 2.0 * (x_value / AXIS_MAX)
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


def _format_pace(kmh):
    # Sub acest prag, kmh <= 0 nu prinde toate cazurile: filtrul de netezire
    # poate lasa kmh la o valoare infima dar strict pozitiva (eroare de
    # virgula mobila la coborarea exponentiala spre zero), care ar da un
    # numitor aproape-zero mai jos si un pace absurd (numere astronomice).
    if kmh <= 0.05:
        return "--:--"
    pace_sec = 3600 / kmh
    return f"{int(pace_sec // 60)}:{int(pace_sec % 60):02d}"


def _format_time_ms(total_sec):
    """mm:ss.mmm - folosit pentru timpul de trecere prin fiecare suta (100m),
    unde precizia de-o secunda intreaga (ca la _format_pace) nu e suficienta."""
    total_sec = max(0.0, total_sec)
    minutes = int(total_sec // 60)
    secs = total_sec % 60
    return f"{minutes}:{secs:06.3f}"


def _format_pace_ms(pace_sec_per_km):
    if pace_sec_per_km <= 0:
        return "--:--.---"
    return _format_time_ms(pace_sec_per_km)


def _rpm_to_kmh(rpm):
    return rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000


def cruise_feedforward_offset_us(target_kmh):
    """Puls estimat (peste neutru) pentru target_kmh, dintr-un model liniar
    simplu ancorat in CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH la puls maxim.
    Doar un punct de plecare - PI-ul corecteaza ce lipseste fata de realitate."""
    fraction = target_kmh / CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH
    return fraction * (ESC_MAX_US - ESC_NEUTRAL_US)


def cruise_pulse_us(target_kmh, current_kmh, integral, dt, prev_pulse_us, max_step_us_per_s):
    """Un pas de regulator feedforward + PI + limitare de panta. Intoarce
    (puls_us, integrala_noua). Pulsul nu coboara niciodata sub neutru - vezi
    avertismentul din docstring-ul de la inceputul fisierului despre
    franare/reverse variabile pe acest ESC.

    max_step_us_per_s vine din exterior (CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S
    la lansare, CRUISE_MAX_PULSE_STEP_US_PER_S dupa aceea) ca acelasi
    regulator sa poata porni lin dar sa corecteze rapid in mers."""
    error = target_kmh - current_kmh
    integral += error * dt
    integral = max(-CRUISE_INTEGRAL_MAX_US / CRUISE_KI_US_PER_KMH_S,
                   min(CRUISE_INTEGRAL_MAX_US / CRUISE_KI_US_PER_KMH_S, integral))
    ff_offset = cruise_feedforward_offset_us(target_kmh)
    trim = CRUISE_KP_US_PER_KMH * error + CRUISE_KI_US_PER_KMH_S * integral
    desired_pulse_us = max(ESC_NEUTRAL_US, min(ESC_MAX_US, ESC_NEUTRAL_US + ff_offset + trim))

    max_step = max_step_us_per_s * dt
    step = max(-max_step, min(max_step, desired_pulse_us - prev_pulse_us))
    pulse_us = prev_pulse_us + step

    return pulse_us, integral


_odo_state = {"rpm": 0.0, "last_update": 0.0}
_odo_lock = threading.Lock()


def _set_rpm(value, fresh=True):
    with _odo_lock:
        _odo_state["rpm"] = value
        if fresh:
            _odo_state["last_update"] = time.time()


def _get_rpm():
    with _odo_lock:
        return _odo_state["rpm"]


def _odo_is_stale():
    with _odo_lock:
        return (time.time() - _odo_state["last_update"]) > ODO_STALE_TIMEOUT_SECONDS


def _odo_reader_loop(stop_event):
    """Vezi manual_drive.py - identic. RPM cade la 0 (fara "fresh") daca
    socket-ul pica, asa ca _odo_is_stale() detecteaza corect intreruperea
    chiar daca ultima valoare primita era 0."""
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
                        _set_rpm(float(line[4:]), fresh=True)
                    except ValueError:
                        continue
        finally:
            sock.close()
            _set_rpm(0.0, fresh=False)
            if not stop_event.is_set():
                stop_event.wait(ODO_RECONNECT_SLEEP_SECONDS)


_CONTROL_SOCKET = "/tmp/cruise_control_control.sock"

_live_state = {"target_kmh": 0.0, "kmh": 0.0, "engaged": False}
_live_lock = threading.Lock()


def _update_live(target_kmh, kmh, engaged):
    with _live_lock:
        _live_state["target_kmh"] = target_kmh
        _live_state["kmh"] = kmh
        _live_state["engaged"] = engaged


def _get_live_state():
    with _live_lock:
        return dict(_live_state)


def _control_server_loop(stop_event):
    """Socket local pentru dashboard-ul web - doar STATUS (afisaj), fara
    comenzi de scriere - tinta se schimba exclusiv din D-pad-ul controller-ului."""
    try:
        os.unlink(_CONTROL_SOCKET)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(_CONTROL_SOCKET)
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
                    payload = {
                        "target_kmh": round(live["target_kmh"], 1),
                        "kmh": round(live["kmh"], 2),
                        "pace": _format_pace(live["kmh"]),
                        "engaged": live["engaged"],
                    }
                    try:
                        conn.sendall((json.dumps(payload) + "\n").encode())
                    except OSError:
                        pass
    finally:
        srv.close()
        try:
            os.unlink(_CONTROL_SOCKET)
        except OSError:
            pass


def _format_summary(target_kmh, elapsed_s, speed_samples):
    """speed_samples: lista de (rpm, kmh, pulse_us) - un rand per TICK_INTERVAL_SECONDS."""
    if not speed_samples:
        return "\n--- SUMAR ---\nNiciun esantion inregistrat.\n"
    kmh_vals = [s[1] for s in speed_samples]
    pulse_vals = [s[2] for s in speed_samples]
    moving_kmh_vals = [k for k in kmh_vals if k > 0]
    avg_kmh = sum(moving_kmh_vals) / len(moving_kmh_vals) if moving_kmh_vals else 0.0
    max_kmh = max(kmh_vals)
    deviation = avg_kmh - target_kmh

    # Daca pulsul a stat "lipit" de ESC_MAX_US o parte semnificativa a
    # cursei, regulatorul cerea deja throttle maxim si tot nu a putut atinge
    # tinta - abaterea de mai sus nu e o problema a regulatorului (Kp/Ki),
    # ci plafonul fizic real al motorului/bateriei in acel moment (bateria
    # descarcata da mai putina viteza la acelasi puls - vezi comentariul
    # despre integrala de mai sus in fisier).
    saturated = sum(1 for p in pulse_vals if p >= ESC_MAX_US - ESC_PULSE_SATURATION_EPSILON_US)
    saturation_fraction = saturated / len(pulse_vals)
    note = ""
    if saturation_fraction >= ESC_PULSE_SATURATION_NOTE_FRACTION:
        note = (
            f"NOTA: puls la maxim ({ESC_MAX_US:.0f}us) in {saturation_fraction:.0%} din cursa - "
            f"tinta a fost probabil peste plafonul fizic real al motorului/bateriei in acel moment, "
            f"nu o eroare a regulatorului.\n"
        )

    return (
        "\n--- SUMAR ---\n"
        f"Tinta: {target_kmh:.1f} km/h | Durata: {elapsed_s:.1f}s | "
        f"{len(speed_samples)} esantioane (la {TICK_INTERVAL_SECONDS}s)\n"
        f"km/h  -> mediu {avg_kmh:.2f} (cat timp s-a miscat) | maxim {max_kmh:.2f}\n"
        f"Pace  -> mediu {_format_pace(avg_kmh)}/km | cel mai bun {_format_pace(max_kmh)}/km\n"
        f"Abatere fata de tinta: {deviation:+.2f} km/h\n"
        f"{note}"
    )


def _format_splits(splits):
    """splits: lista de (marcaj_m, elapsed_s, durata_suta_s, pace_sec_per_km),
    una per fiecare SPLIT_DISTANCE_M parcursi. Scrisa in log dupa SUMAR."""
    if not splits:
        return ""
    lines = ["\n--- SUTE (100m) ---"]
    for mark_m, elapsed_s, split_duration, pace_sec in splits:
        lines.append(
            f"{mark_m:5.0f}m | total {_format_time_ms(elapsed_s)} | "
            f"suta {split_duration:6.3f}s | pace {_format_pace_ms(pace_sec)}/km"
        )
    return "\n".join(lines) + "\n"


def _handle_sigterm(signum, frame):
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
    control_thread = None
    control_stop_event = None

    def rumble(duration_ms):
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

    # Definite inaintea oricarui cod care poate arunca exceptie - `finally`
    # le foloseste chiar daca pornirea a picat devreme.
    run_start_time = time.time()
    csv_file = None
    target_kmh = 0.0
    speed_samples = []  # (rpm, kmh, pulse_us) la fiecare TICK_INTERVAL_SECONDS, cat timp csv_file e deschis
    leg_start_time = run_start_time
    splits = []  # (marcaj_m, elapsed_s, durata_suta_s, pace_sec_per_km) - vezi SPLIT_DISTANCE_M
    split_distance_m = 0.0
    next_split_mark_m = SPLIT_DISTANCE_M
    last_split_elapsed_s = 0.0

    try:
        # PCA/ESC configurat SI semnal de neutru trimis inainte de a alimenta
        # ESC-ul prin releu - vezi manual_drive.py.
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = PCA_FREQUENCY_HZ

        esc = ESC(pca)
        esc.neutral()
        steering = SteeringServo(pca)

        controller = find_xbox_controller()
        if controller is None:
            raise ConnectionError("Controller-ul Xbox nu a fost gasit.")
        controller_fd = controller.fd

        _relay_cmd("RELAY_ON")

        esc.arm()

        odo_stop_event = threading.Event()
        odo_thread = threading.Thread(
            target=_odo_reader_loop, args=(odo_stop_event,), daemon=True)
        odo_thread.start()

        control_stop_event = threading.Event()
        control_thread = threading.Thread(
            target=_control_server_loop, args=(control_stop_event,), daemon=True)
        control_thread.start()

        print("\n-----------------------------------------------------")
        print("Cruise control gata. Steering-ul raspunde mereu la stick.")
        print("[Dpad SUS/JOS] schimba tinta cu 1 km/h.")
        print("[A] porneste cruise control la tinta curenta.")
        print("[Y] opreste cursa curenta, salveaza logul, reseteaza tinta.")
        print("[B] opreste tot programul.")
        print("-----------------------------------------------------")

        engaged = False
        engage_time = 0.0
        integral = 0.0
        prev_pulse_us = ESC_NEUTRAL_US
        filtered_kmh = 0.0
        last_hat0y = 0
        last_control_time = time.time()
        last_tick_time = run_start_time

        # Bucla externa de recuperare a mediei - pornesc sa numere abia dupa
        # fereastra de lansare (CRUISE_LAUNCH_SECONDS), ca rampa initiala
        # (inevitabila, oricat de bland pornesti) sa nu fie tratata ca
        # "deficit de recuperat".
        scheduled_distance_m = 0.0
        actual_distance_m = 0.0

        while True:
            ready, _, _ = select.select([controller_fd], [], [], 0.001)

            for _ in ready:
                for event in controller.read():
                    if event.type == ecodes.EV_ABS:
                        abs_name = ecodes.ABS.get(event.code)
                        if abs_name == "ABS_X":
                            angle = steering_axis_to_angle(event.value)
                            steering.set_angle(angle)
                        elif abs_name == "ABS_HAT0Y":
                            if event.value == -1 and last_hat0y == 0:
                                target_kmh = min(TARGET_SPEED_MAX_KMH, target_kmh + TARGET_SPEED_STEP_KMH)
                                print(f"\nTinta viteza: {target_kmh:.1f} km/h")
                            elif event.value == 1 and last_hat0y == 0:
                                target_kmh = max(0.0, target_kmh - TARGET_SPEED_STEP_KMH)
                                print(f"\nTinta viteza: {target_kmh:.1f} km/h")
                            last_hat0y = event.value

                    elif event.type == ecodes.EV_KEY and event.value == 1:
                        if event.code == BTN_ENGAGE:
                            if not engaged:
                                engaged = True
                                engage_time = time.time()
                                integral = 0.0
                                prev_pulse_us = ESC_NEUTRAL_US
                                scheduled_distance_m = 0.0
                                actual_distance_m = 0.0
                                leg_start_time = time.time()
                                speed_samples = []
                                splits = []
                                split_distance_m = 0.0
                                next_split_mark_m = SPLIT_DISTANCE_M
                                last_split_elapsed_s = 0.0
                                csv_path = os.path.join(
                                    SPEED_LOG_DIR, f"cruise_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                                csv_file = open(csv_path, "w")
                                csv_file.write("elapsed_s,target_kmh,effective_target_kmh,rpm,kmh,pace_mmss,pulse_us\n")
                                csv_file.flush()
                                print(f"\nLog: {csv_path}")
                                print(f">>> CRUISE PORNIT la {target_kmh:.1f} km/h (A) <<<")
                                rumble(300)
                        elif event.code == BTN_RESET:
                            if engaged or csv_file is not None:
                                engaged = False
                                esc.neutral()
                                prev_pulse_us = ESC_NEUTRAL_US
                                elapsed = time.time() - leg_start_time
                                summary = _format_summary(target_kmh, elapsed, speed_samples) + _format_splits(splits)
                                if csv_file is not None:
                                    csv_file.write(summary)
                                    csv_file.close()
                                    csv_file = None
                                print(summary)
                                logging.info("Cruise leg resetat\n" + summary)
                                target_kmh = 0.0
                                rumble(600)
                        elif event.code == BTN_STOP:
                            raise KeyboardInterrupt

            rpm = _get_rpm()
            raw_kmh = _rpm_to_kmh(rpm)
            now = time.time()
            dt = now - last_control_time

            # Netezire (vezi CRUISE_SPEED_FILTER_TAU_S) - regulatorul si
            # afisajul/logul folosesc viteza filtrata, nu bruta.
            filter_weight = min(1.0, dt / CRUISE_SPEED_FILTER_TAU_S) if dt > 0 else 0.0
            filtered_kmh += (raw_kmh - filtered_kmh) * filter_weight
            kmh = filtered_kmh
            effective_target_kmh = target_kmh  # implicit - suprascris mai jos daca bucla externa e activa

            if engaged:
                past_grace = (now - engage_time) > ODO_STALE_GRACE_SECONDS
                if past_grace and _odo_is_stale():
                    engaged = False
                    esc.neutral()
                    prev_pulse_us = ESC_NEUTRAL_US
                    print("\n!!! Senzor de viteza indisponibil - cruise control oprit automat !!!")
                    logging.warning("Cruise control oprit automat: date RPM invechite")
                else:
                    in_launch = (now - engage_time) <= CRUISE_LAUNCH_SECONDS
                    max_step = CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S if in_launch else CRUISE_MAX_PULSE_STEP_US_PER_S

                    # Sute (SPLIT_DISTANCE_M) - distanta se acumuleaza din
                    # viteza filtrata pe fiecare iteratie (mult mai fin decat
                    # tick-ul de 0.1s de mai jos), iar momentul exact al
                    # trecerii peste marcaj e interpolat liniar intre
                    # iteratia anterioara (last_control_time) si asta (now),
                    # ca timpul afisat sa aiba precizie sub-tick (milisecunde),
                    # nu doar rotunjit la 0.1s.
                    prev_split_distance_m = split_distance_m
                    split_distance_m += (kmh / 3.6) * dt
                    while split_distance_m >= next_split_mark_m:
                        if split_distance_m > prev_split_distance_m:
                            frac = (next_split_mark_m - prev_split_distance_m) / (split_distance_m - prev_split_distance_m)
                        else:
                            frac = 1.0
                        crossing_time = last_control_time + frac * dt
                        crossing_elapsed_s = crossing_time - leg_start_time
                        split_duration = crossing_elapsed_s - last_split_elapsed_s
                        pace_sec = split_duration * (1000.0 / SPLIT_DISTANCE_M)
                        splits.append((next_split_mark_m, crossing_elapsed_s, split_duration, pace_sec))
                        msg = (
                            f"\n>>> {next_split_mark_m:.0f}m la {_format_time_ms(crossing_elapsed_s)}  "
                            f"(suta: {split_duration:.3f}s, pace {_format_pace_ms(pace_sec)}/km) <<<"
                        )
                        print(msg)
                        logging.info(msg.strip())
                        last_split_elapsed_s = crossing_elapsed_s
                        next_split_mark_m += SPLIT_DISTANCE_M

                    if in_launch:
                        effective_target_kmh = target_kmh
                    else:
                        # Bucla externa: cat ar fi trebuit sa parcurga pana
                        # acum la tinta (target_kmh se poate schimba din
                        # dpad, de-aia se acumuleaza ca integrala, nu
                        # target_kmh * elapsed) vs cat a parcurs efectiv.
                        scheduled_distance_m += (target_kmh / 3.6) * dt
                        actual_distance_m += (kmh / 3.6) * dt
                        deficit_m = scheduled_distance_m - actual_distance_m
                        extra_kmh = (deficit_m / CRUISE_CATCHUP_WINDOW_S) * 3.6
                        extra_kmh = max(-CRUISE_CATCHUP_MAX_EXTRA_KMH,
                                         min(CRUISE_CATCHUP_MAX_EXTRA_KMH, extra_kmh))
                        effective_target_kmh = max(0.0, min(TARGET_SPEED_MAX_KMH, target_kmh + extra_kmh))

                    pulse_us, integral = cruise_pulse_us(effective_target_kmh, kmh, integral, dt, prev_pulse_us, max_step)
                    esc.set_pulse_us(pulse_us)
                    prev_pulse_us = pulse_us
            last_control_time = now

            _update_live(target_kmh, kmh, engaged)

            if now - last_tick_time >= TICK_INTERVAL_SECONDS:
                last_tick_time = now
                pace = _format_pace(kmh)
                status = "CRUISE" if engaged else "gata  "
                print(
                    f"{status} | tinta {target_kmh:5.1f} km/h | curent {kmh:5.2f} km/h | "
                    f"pace {pace}/km   ",
                    end="\r",
                )
                if csv_file is not None:
                    elapsed = now - leg_start_time
                    speed_samples.append((rpm, kmh, prev_pulse_us))
                    csv_file.write(f"{elapsed:.1f},{target_kmh:.1f},{effective_target_kmh:.2f},{rpm:.2f},{kmh:.2f},{pace},{prev_pulse_us:.1f}\n")
                    csv_file.flush()

            time.sleep(0.01)

    except (KeyboardInterrupt, ConnectionError) as e:
        if isinstance(e, ConnectionError):
            print(f"\n{e}")
        print("\nOprire...")
    except Exception as e:
        logging.exception("Eroare majora neasteptata")
        print(f"\nEroare majora neasteptata: {e}")
    finally:
        # Vezi manual_drive.py - ignoram SIGTERM-uri ulterioare din acest
        # punct incolo, ca RELAY_OFF -> !!ESTOP!! -> SIGTERM nou sa nu ne
        # intrerupa curatarea la jumatate.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

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

        if control_stop_event is not None:
            control_stop_event.set()
        if control_thread is not None:
            control_thread.join(timeout=2)

        if csv_file is not None:
            elapsed = time.time() - leg_start_time
            summary = _format_summary(target_kmh, elapsed, speed_samples) + _format_splits(splits)
            csv_file.write(summary)
            csv_file.close()
            print(summary)
            logging.info("Cruise control stopped\n" + summary)
        else:
            logging.info("Cruise control stopped")
        print("\nHardware oprit si curatat. Program inchis.")


if __name__ == "__main__":
    main()
