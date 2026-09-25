"""
Unix domain socket paths and timing constants for the IPC channels shared
between safety/estop_listener.py (server), the driving scripts (clients of
RELAY_SOCKET/ODO_SOCKET), and dashboard/system_stats.py (client of
RELAY_SOCKET, MAIN_CONTROL_SOCKET and MANUAL_DRIVE_CONTROL_SOCKET). Values taken verbatim from
main/main.py, manual_drive/manual_drive.py, and cruise_control/cruise_control.py.
"""

RELAY_SOCKET = "/tmp/esp32_relay.sock"
ODO_SOCKET = "/tmp/esp32_odometry.sock"
MAIN_CONTROL_SOCKET = "/tmp/main_autopilot_control.sock"
MANUAL_DRIVE_CONTROL_SOCKET = "/tmp/manual_drive_control.sock"

ODO_RECONNECT_SLEEP_SECONDS = 2.0
ODO_STALE_TIMEOUT_SECONDS = 1.0     # no fresh RPM past this -> cruise control auto-stops
ODO_STALE_GRACE_SECONDS = 2.0

# How long a driving script gets to exit on its own after SIGTERM before it
# is SIGKILLed - used by safety/estop_listener.py (ESTOP) and the dashboard
# (Stop / Relay OFF). Waiting is safe: by then the relay is already OFF, so
# the ESC and servo have no power. data_recorder gets longer: its shutdown
# drains up to 400 queued frames to disk (writer join timeout 30s) and then
# rewrites driving_log.json - killing it mid-way loses the session's log.
SCRIPT_STOP_GRACE_SECONDS = 5
DATA_RECORDER_STOP_GRACE_SECONDS = 40


def stop_grace_seconds(cmdline_or_path):
    """Grace period for a script, given its path or a process cmdline list."""
    args = [cmdline_or_path] if isinstance(cmdline_or_path, str) else cmdline_or_path
    if any(a.replace("\\", "/").endswith("data_recorder/data_recorder.py") for a in args):
        return DATA_RECORDER_STOP_GRACE_SECONDS
    return SCRIPT_STOP_GRACE_SECONDS
