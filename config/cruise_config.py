"""
Cruise-control PI regulator constants - values taken verbatim from
cruise_control/cruise_control.py (the original tuning) and main/main.py
(copied there byte-identically). See config/cruise_pi.py for the regulator
logic that consumes these.
"""

CRUISE_MAX_SPEED_AT_FULL_THROTTLE_KMH = 23.0

CRUISE_KP_US_PER_KMH = 10.0
CRUISE_KI_US_PER_KMH_S = 3.0
CRUISE_INTEGRAL_MAX_US = 50.0

CRUISE_LAUNCH_MAX_PULSE_STEP_US_PER_S = 120.0
CRUISE_MAX_PULSE_STEP_US_PER_S = 180.0
CRUISE_LAUNCH_SECONDS = 2.0

CRUISE_SPEED_FILTER_TAU_S = 0.7

CRUISE_CATCHUP_WINDOW_S = 10.0
CRUISE_CATCHUP_MAX_EXTRA_KMH = 3.0

TARGET_SPEED_STEP_KMH = 1.0

# main.py --speed-mode controller is open-loop PWM (no odometry): each D-pad
# press moves the ESC pulse this many microseconds above/below neutral.
CONTROLLER_PWM_STEP_US = 50.0
TARGET_SPEED_MAX_KMH = 100.0  # real ceiling comes from ESC_MAX_US, see cruise_pi.py
