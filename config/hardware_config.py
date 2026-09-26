"""
Servo (steering), ESC (throttle), and PCA9685 constants - values taken
verbatim from main/main.py, manual_drive/manual_drive.py,
cruise_control/cruise_control.py, data_recorder/data_recorder.py,
model_runner/model_runner.py, and tools/servo_calibrate.py, where they were
duplicated byte-identically across every script.
"""

# Steering (servo on PCA9685 channel 0)
SERVO_CHANNEL = 0
SERVO_MIN_PULSE = 900          # microseconds
SERVO_MAX_PULSE = 2200         # microseconds
# The only three steering numbers to calibrate. Each side's step size is
# derived from them (see servo_esc.steering_label_to_angle): label 0 ->
# STRAIGHT, label +1 (stick right) -> MIN, label -1 (stick left) -> MAX.
# The two sides don't have to be the same distance from STRAIGHT - the
# chassis turns the wheels more per servo degree on one side, so equal
# wheel lock needs unequal servo travel.
SERVO_MIN_ANGLE = 49           # degrees - full lock, label +1
SERVO_MAX_ANGLE = 135          # degrees - full lock, label -1
SERVO_STRAIGHT_ANGLE = 94      # degrees - wheels straight, label 0

if not SERVO_MIN_ANGLE < SERVO_STRAIGHT_ANGLE < SERVO_MAX_ANGLE:
    raise ValueError("hardware_config: need SERVO_MIN_ANGLE < SERVO_STRAIGHT_ANGLE < SERVO_MAX_ANGLE")

# Xbox left stick: this fraction of travel around center counts as straight
# (absorbs stick noise); the rest maps linearly onto label 0..1 per side.
JOYSTICK_DEADZONE = 0.2

# Throttle (ESC on PCA9685 channel 1)
ESC_CHANNEL = 1
PCA_FREQUENCY_HZ = 50          # Hz (20ms period)
ESC_ARM_PULSE_US = 1500        # microseconds
ESC_ARM_HOLD_SECONDS = 3.0     # seconds
ESC_NEUTRAL_US = 1500          # microseconds
ESC_MIN_US = 1000              # microseconds
ESC_MAX_US = 1700              # microseconds

# How close to ESC_MAX_US counts as "pulse maxed out", and above what
# fraction of samples a note gets added to the final run summary.
ESC_PULSE_SATURATION_EPSILON_US = 2.0
ESC_PULSE_SATURATION_NOTE_FRACTION = 0.15

# Errno values that mean "transient I2C glitch, log and keep going" rather
# than a real fault worth crashing on - raised from adafruit_pca9685/
# adafruit_motor.servo when a duty-cycle write hits the bus mid-transfer.
TRANSIENT_I2C_ERRNOS = (121, 19)

# Wheel/odometry
WHEEL_CIRCUMFERENCE_M = 0.1369  # diameter 43.58mm, tape-measure calibrated over 200m
