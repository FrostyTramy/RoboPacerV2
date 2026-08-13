# RoboPacerV2 — Hardware Setup

## Compute

| Item | Value |
|---|---|
| Board | Raspberry Pi 5, 8GB RAM |
| AI accelerator | Hailo AI Kit, 26 TOPS (M.2 HAT, stacked on top of the Pi) |
| Camera | Raspberry Pi Camera Module v2.1 (IMX219 sensor — see `CAMERA_SPEC.md`) |
| Secondary WiFi adapter | ASUS USB-N14, on USB 3.0 — dedicated to the `RoboPacer` hotspot network, separate from the Pi's onboard WiFi |

## Power

| Item | Value |
|---|---|
| Pi power board | Sits under the Pi 5, takes 12V in and steps it down to 5V for the Pi — lets the whole rig charge over Power Delivery fast charge |
| Pi battery | Hama power bank, 10000mAh, PD (feeds the 12V→5V board above) |
| ESC/motor/servo battery | 3S LiPo, 5500mAh, 11.1V — completely separate from the Pi's own power |

The ESC, motor, and servo all share the same power feed from the 3S LiPo, gated by a single relay (see below) — cutting the relay kills all three at once.

## Drivetrain control

| Item | Value |
|---|---|
| PWM controller | PCA9685 (I2C) — steering servo on channel 0, ESC on channel 1 (`SERVO_CHANNEL`/`ESC_CHANNEL` in every driving script) |
| ESC | Hobbywing QuicRun 10BL120, **sensored** mode |
| Motor | Surpass Hobby Rocket 540 Sensored Brushless Motor V3, 13.5T |

## Safety relay + speed sensor (ESP32-CAM)

An ESP32-CAM board (`Esp32/garmin_receiver/garmin_receiver.ino`, advertises over BLE as `GarminPacer|Pacer1`) sits between the 3S LiPo and the ESC, and does two independent jobs:

1. **Relay control** — switches the power line from the 3S LiPo to the ESC on/off. Since the ESC, motor, and servo share that one feed, this is a single point that can cut power to the entire drivetrain — from a script (`RELAY_ON`/`RELAY_OFF` over serial via `estop_listener.py`), from the Garmin watch over BLE, or from the ESP32's own 2-second heartbeat watchdog. See `SAFETY.md` for the full picture.
2. **Speed sensing** — a KY-024 module (49E linear Hall sensor) reads a wheel fitted with 4 magnets, wired to the ESP32. The ESP32 computes RPM from the interval between consecutive magnet passes (`MAGNETS = 4` in the `.ino`) and streams it to the Pi over serial as `RPM:x.xx` lines, relayed to scripts over `/tmp/esp32_odometry.sock` by `estop_listener.py`.

### Wheel calibration

| Item | Value |
|---|---|
| Diameter | 43.58mm |
| Circumference | 0.1369m (`WHEEL_CIRCUMFERENCE_M` in `manual_drive.py`, `cruise_control.py`, `tools/odometry.py`) |

Calibrated empirically with a tape measure against the distance the code computed from RPM (not from the physical wheel spec) — see the constant's inline comment for the exact before/after numbers of the last calibration pass. Re-measure and update all three files together if the wheel or gearing ever changes.
