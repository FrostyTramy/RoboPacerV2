"""
Cruise-control feedforward + PI regulator - byte-identical between
cruise_control/cruise_control.py and main/main.py. Pulse never drops below
neutral (see cruise_control.py's module docstring for why: ESC brake/reverse
ambiguity).
"""
from config.cruise_config import (
    CRUISE_INTEGRAL_MAX_US,
    CRUISE_KI_US_PER_KMH_S,
    CRUISE_KP_US_PER_KMH,
    CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH,
)
from config.hardware_config import ESC_MAX_US, ESC_NEUTRAL_US


def cruise_feedforward_offset_us(target_kmh):
    fraction = target_kmh / CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH
    return fraction * (ESC_MAX_US - ESC_NEUTRAL_US)


def cruise_pulse_us(target_kmh, current_kmh, integral, dt, prev_pulse_us, max_step_us_per_s):
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
