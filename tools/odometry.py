"""
Odometrie live — citeste RPM de la estop_listener via socket
si afiseaza km/h + pace in terminal.

Rulare: python tools/odometry.py
"""

import socket
import sys

ODO_SOCKET         = "/tmp/esp32_odometry.sock"
WHEEL_CIRCUMFERENCE_M = 0.1257  # diametru 40mm -> π × 0.04

def main():
    print("Conectare la odometry socket...", flush=True)
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(ODO_SOCKET)
    except OSError as e:
        print(f"Eroare: {e}\nVerifica ca estop_listener ruleaza.")
        sys.exit(1)

    print("Conectat. Astept date RPM...\n", flush=True)
    buf = ""
    try:
        while True:
            data = sock.recv(64).decode("utf-8", errors="replace")
            if not data:
                print("\nConexiune inchisa.")
                break
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("RPM:"):
                    continue
                try:
                    rpm = float(line[4:])
                except ValueError:
                    continue
                if rpm > 0:
                    kmh      = rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000
                    pace_sec = 3600 / kmh
                    pace_min = int(pace_sec // 60)
                    pace_s   = int(pace_sec % 60)
                    print(f"\r  {kmh:6.2f} km/h   pace: {pace_min}:{pace_s:02d} /km   ", end="", flush=True)
                else:
                    print(f"\r    0.00 km/h   pace: --:--          ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nOprit.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
