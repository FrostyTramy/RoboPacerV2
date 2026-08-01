# RoboPacerV2 — ESC Safety System

Two independent layers protect against the ESC being left driving after
control is lost. Neither depends on the other, because the failure modes
they cover are different.

## Layer 1 — in-script shutdown handling (`data_recorder/data_recorder.py`)

Covers: Ctrl+C, `systemctl stop`, plain `kill <pid>`, normal exceptions.

- `KeyboardInterrupt` (Ctrl+C / SIGINT) and `ConnectionError` (no controller
  found) are caught and fall through to a `finally` block that sets the ESC
  to neutral, then cuts the PWM signal entirely, deinitializes the PCA9685,
  and releases the servo.
- A `SIGTERM` handler (`_handle_sigterm`) was added because SIGTERM has no
  default Python handler — without it, `systemctl stop` or a plain `kill`
  would terminate the process *without* running the `finally` block,
  leaving the ESC at its last commanded pulse.

**What this cannot cover:** `kill -9` (SIGKILL), a segfault/hard crash, an
infinite hang inside the script, or the Pi losing power. None of those run
any more Python code, so nothing inside the script can react. That's what
layer 2 is for.

## Layer 2 — `esc-watchdog` systemd service (`safety/`)

A completely separate process that doesn't trust the recorder script at
all. Every 0.3s it checks whether any of the scripts in `ALLOWED_SCRIPTS`
(inside `safety/esc_watchdog.sh`) is currently running:

- **If yes** → does nothing, doesn't touch I2C. Full control stays with
  the running script.
- **If no** → writes a hard "full off" state directly to the PCA9685's
  channel-1 (ESC) registers, every cycle, until an allowed script appears
  again.

This is why it survives crashes and `kill -9`: it isn't reacting to the
recorder shutting down cleanly, it's just continuously asking "is anything
supposed to be in control right now?" and defaulting to off whenever the
answer is no. It also survives a Pi reboot after a power loss, since the
service is enabled at boot — the moment the Pi comes back up and the
PCA9685 is powered again, the watchdog starts forcing it off within
~0.3s, before anything else can command it.

**Why it's cheap:** it's a bash loop, not a Python process — no
CircuitPython/Blinka stack to load. Idle footprint is ~3MB RAM and
effectively 0% CPU (one `pgrep` + a sleep per cycle; it only touches I2C
when it actually needs to force something off).

### ⚠️ Important: how you must launch the recorder

The watchdog matches processes by **full path** via `pgrep -f`. Always
start the recorder with its full path:

```bash
python3 /home/pi/RoboPacerV2/data_recorder/data_recorder.py
```

**Not** `cd data_recorder && python3 data_recorder.py` — a relative path
won't contain the string the watchdog is matching against, so it won't
recognize the script as running, and will fight it by forcing the ESC off
every 0.3s while you're trying to drive. If you script/service the
recorder's own launch later, make sure the invocation still contains that
full path somewhere in its command line.

### Adding more ESC-driving scripts

Edit `ALLOWED_SCRIPTS` in `safety/esc_watchdog.sh` and add the new
script's full path, then restart the service:

```bash
sudo systemctl restart esc-watchdog.service
```

### Checking status / testing

```bash
systemctl status esc-watchdog.service      # running? enabled at boot?
journalctl -u esc-watchdog.service -f      # logs (should be near-empty in normal operation)

# confirm the ESC is actually being held off right now (expect 0x10):
i2cget -y 1 0x40 0x0d
```

### Stopping it (e.g. for hardware debugging)

```bash
sudo systemctl stop esc-watchdog.service     # stop for this boot only
sudo systemctl disable esc-watchdog.service  # also stop it starting on boot
```

Remember: with the watchdog stopped, layer 1 (in-script handling) is your
only protection again.
