#!/usr/bin/env bash
#
# esc_watchdog.sh — independent ESC fail-safe for RoboPacerV2.
#
# Problem: the PCA9685 holds whatever PWM duty cycle it was last told to
# output, forever, even if the Raspberry Pi process that set it dies. A
# crash, a `kill -9`, a hang, or a power blip to the Pi will NOT stop the
# ESC on its own — the last throttle command just keeps being sent.
#
# This script runs independently of any driving script (as a systemd
# service) and enforces one rule: if none of the ALLOWED_SCRIPTS are
# currently running, the ESC channel is repeatedly forced to a hard "off"
# state at the PCA9685 register level. While an allowed script IS running,
# this watchdog does nothing and never touches the I2C bus, so it can't
# race or fight with the script that's actually driving.
#
# Resource footprint: a bash loop that sleeps POLL_INTERVAL seconds, then
# does one `pgrep` and (usually) nothing else. No persistent memory growth,
# no busy-waiting, negligible CPU.

set -u

# --- Config -----------------------------------------------------------

I2C_BUS=1
PCA9685_ADDR=0x40

# ESC is on PCA9685 channel 1. Each channel has 4 registers starting at
# 0x06 + 4*channel (ON_L, ON_H, OFF_L, OFF_H). Channel 1 -> 0x0A.. 0x0D.
ESC_ON_L_REG=0x0A
ESC_ON_H_REG=0x0B
ESC_OFF_L_REG=0x0C
ESC_OFF_H_REG=0x0D

# How often to check. Lower = faster reaction to a crash, at the cost of
# (slightly) more wakeups. 0.3s means a dead/killed script leaves the ESC
# running its last command for at most ~0.3s before this forces it off.
POLL_INTERVAL=0.3

# Scripts that are allowed to drive the ESC. While ANY of these is running,
# the watchdog leaves the PCA9685 alone. Match by full path so unrelated
# scripts with a similar name don't count.
ALLOWED_SCRIPTS=(
    "/home/pi/RoboPacerV2/data_recorder/data_recorder.py"
    "/home/pi/RoboPacerV2/main/main.py"
    "/home/pi/RoboPacerV2/model_runner/model_runner.py"
    "/home/pi/RoboPacerV2/tools/servo_calibrate.py"
    "/home/pi/RoboPacerV2/manual_drive/manual_drive.py"
    "/home/pi/RoboPacerV2/cruise_control/cruise_control.py"
)

# --- Functions ----------------------------------------------------------

is_allowed_running() {
    for script in "${ALLOWED_SCRIPTS[@]}"; do
        if pgrep -f -- "$script" >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

force_esc_off() {
    # Full-off per the PCA9685 datasheet: OFF_H bit 4 set, everything else
    # zeroed. This is a hard "no signal" state, not just a neutral pulse -
    # it's the same effect as the PCA9685 never having been programmed.
    i2cset -y "$I2C_BUS" "$PCA9685_ADDR" "$ESC_ON_L_REG" 0x00 2>/dev/null
    i2cset -y "$I2C_BUS" "$PCA9685_ADDR" "$ESC_ON_H_REG" 0x00 2>/dev/null
    i2cset -y "$I2C_BUS" "$PCA9685_ADDR" "$ESC_OFF_L_REG" 0x00 2>/dev/null
    i2cset -y "$I2C_BUS" "$PCA9685_ADDR" "$ESC_OFF_H_REG" 0x10 2>/dev/null
}

# --- Main loop ------------------------------------------------------------

while true; do
    if ! is_allowed_running; then
        force_esc_off
    fi
    sleep "$POLL_INTERVAL"
done
