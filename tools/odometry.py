"""
Odometrie live — citeste RPM de la estop_listener via socket
si afiseaza km/h + pace in terminal.

Rulare: python tools/odometry.py
"""

import socket
import sys

ODO_SOCKET         = "/tmp/esp32_odometry.sock"
WHEEL_CIRCUMFERENCE_M = 0.1369  # diametru 43.58mm - calibrat cu ruleta pe 200m reali
MAGNETS = 4  # cate magneti pe roata - vezi Esp32/garmin_receiver/garmin_receiver.ino:
             # fiecare linie "RPM:x" (x>0) = o trecere de magnet = 1/4 din o rotatie completa

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
    pulse_count = 0  # o trecere de magnet = 1/MAGNETS dintr-o rotatie
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
                    pulse_count += 1
                    rotations = pulse_count / MAGNETS
                    kmh      = rpm * WHEEL_CIRCUMFERENCE_M * 60 / 1000
                    pace_sec = 3600 / kmh
                    pace_min = int(pace_sec // 60)
                    pace_s   = int(pace_sec % 60)
                    print(f"\r  {kmh:6.2f} km/h   pace: {pace_min}:{pace_s:02d} /km   "
                          f"rotatii roata: {rotations:7.2f}   ", end="", flush=True)
                else:
                    rotations = pulse_count / MAGNETS
                    print(f"\r    0.00 km/h   pace: --:--           "
                          f"rotatii roata: {rotations:7.2f}   ", end="", flush=True)
    except KeyboardInterrupt:
        rotations = pulse_count / MAGNETS
        distance_m = rotations * WHEEL_CIRCUMFERENCE_M
        print(f"\nOprit. Total: {rotations:.2f} rotatii ale rotii ({distance_m:.2f} m parcursi).")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
