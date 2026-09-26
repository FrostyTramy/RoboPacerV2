"""
RoboPacerV2 - Controller test (test tool)
==========================================
Live view of any game controller in the terminal (works over SSH) - Xbox,
PlayStation (DualSense / DualShock), Nintendo, or any other gamepad Linux
sees: the raw value of every stick, trigger and D-pad exactly as the
controller sends it (plus the same value scaled to -1..1, triggers 0..1),
which buttons are held, and a log of the last presses. Nothing else is
converted (no deadzone, no steering/throttle math).

Button labels follow the controller type (A/B/X/Y on Xbox, Cross/Circle/
Square/Triangle on PlayStation, ...); the `held:` line always shows the
Linux code name and number too - that's what the robot's scripts compare
against.

Note: the robot's own scripts only look for a controller with "xbox" in its
name (config/input_devices.py) - this tool shows any controller.

Only reads the controller - never touches the motor, servo or relay. Waits
for a controller if none is connected, and reconnects if it drops.

Usage:
    python3 tools/controller_test.py                      first controller found
    python3 tools/controller_test.py --device /dev/input/event6
    python3 tools/controller_test.py --rumble             also test the rumble motors
"""

import argparse
import select
import signal
import sys
import time
from collections import deque

from evdev import InputDevice, ecodes, ff, list_devices

REFRESH_SECONDS = 0.05   # screen redraw rate (20 Hz) - events are read continuously
RECONNECT_SECONDS = 0.5
EVENT_LOG_LINES = 8
BAR = 27
LINE_WIDTH = 78

VENDOR_SONY, VENDOR_MICROSOFT, VENDOR_NINTENDO = 0x054C, 0x045E, 0x057E

# Button labels per controller type, by Linux key code (the code is what
# every driver reports; which physical button sends it depends on the type).
BUTTON_LABELS = {
    "xbox": {
        ecodes.BTN_A: "A", ecodes.BTN_B: "B", ecodes.BTN_X: "X", ecodes.BTN_Y: "Y",
        ecodes.BTN_TL: "LB", ecodes.BTN_TR: "RB", ecodes.BTN_TL2: "LT", ecodes.BTN_TR2: "RT",
        ecodes.BTN_SELECT: "View", ecodes.BTN_START: "Menu", ecodes.BTN_MODE: "Xbox",
        ecodes.BTN_THUMBL: "LS", ecodes.BTN_THUMBR: "RS", ecodes.KEY_RECORD: "Share",
    },
    "playstation": {
        ecodes.BTN_SOUTH: "Cross", ecodes.BTN_EAST: "Circle", ecodes.BTN_NORTH: "Triangle",
        ecodes.BTN_WEST: "Square", ecodes.BTN_TL: "L1", ecodes.BTN_TR: "R1",
        ecodes.BTN_TL2: "L2", ecodes.BTN_TR2: "R2", ecodes.BTN_SELECT: "Create",
        ecodes.BTN_START: "Options", ecodes.BTN_MODE: "PS", ecodes.BTN_THUMBL: "L3",
        ecodes.BTN_THUMBR: "R3", ecodes.BTN_LEFT: "Touchpad",
    },
    "nintendo": {
        ecodes.BTN_SOUTH: "B", ecodes.BTN_EAST: "A", ecodes.BTN_NORTH: "X", ecodes.BTN_WEST: "Y",
        ecodes.BTN_TL: "L", ecodes.BTN_TR: "R", ecodes.BTN_TL2: "ZL", ecodes.BTN_TR2: "ZR",
        ecodes.BTN_SELECT: "-", ecodes.BTN_START: "+", ecodes.BTN_MODE: "Home",
        ecodes.BTN_THUMBL: "LS", ecodes.BTN_THUMBR: "RS", ecodes.BTN_Z: "Capture",
    },
}

HIDE_CURSOR, SHOW_CURSOR, HOME, CLEAR_BELOW, CLEAR_LINE = "\x1b[?25l", "\x1b[?25h", "\x1b[H", "\x1b[J", "\x1b[K"
GREEN, YELLOW, DIM, BOLD, RESET = "\x1b[32m", "\x1b[33m", "\x1b[2m", "\x1b[1m", "\x1b[0m"


def is_gamepad(dev):
    """A gamepad/joystick has sticks (ABS) and gamepad/joystick buttons - which
    leaves out a DualSense's separate motion-sensor and touchpad devices,
    keyboards, mice and the HDMI/power-button inputs."""
    caps = dev.capabilities()
    keys = set(caps.get(ecodes.EV_KEY, []))
    has_pad_buttons = any(ecodes.BTN_JOYSTICK <= k < ecodes.BTN_DIGI for k in keys)
    return has_pad_buttons and ecodes.EV_ABS in caps


def find_gamepads():
    """All connected gamepads (InputDevice objects); everything else opened
    while looking is closed again."""
    pads = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        if is_gamepad(dev):
            pads.append(dev)
        else:
            dev.close()
    return pads


def open_touchpad(pad):
    """The same controller's separate touchpad device (DualSense/DualShock
    report the touchpad click there), matched by the controller's unique ID."""
    if not pad.uniq:
        return None
    for path in list_devices():
        if path == pad.path:
            continue
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        if dev.uniq == pad.uniq and ecodes.BTN_LEFT in dev.capabilities().get(ecodes.EV_KEY, []):
            return dev
        dev.close()
    return None


def controller_type(dev):
    name = dev.name.lower()
    if dev.info.vendor == VENDOR_SONY or "dualsense" in name or "dualshock" in name:
        return "playstation"
    if dev.info.vendor == VENDOR_MICROSOFT or "xbox" in name:
        return "xbox"
    if dev.info.vendor == VENDOR_NINTENDO or "nintendo" in name or "pro controller" in name:
        return "nintendo"
    return "generic"


def axis_layout(absinfo, kind):
    """[(code, name, is_trigger)] for every axis the controller reports.
    Xbox pads over Bluetooth put the right stick on Z/RZ and the triggers on
    GAS/BRAKE; the standard Linux layout (PlayStation, most others) puts the
    right stick on RX/RY and the triggers on Z/RZ."""
    lt, rt = {"playstation": ("L2", "R2"), "nintendo": ("ZL", "ZR")}.get(kind, ("LT", "RT"))
    names = {ecodes.ABS_X: ("Left stick X", False), ecodes.ABS_Y: ("Left stick Y", False),
             ecodes.ABS_HAT0X: ("D-pad left/right", False), ecodes.ABS_HAT0Y: ("D-pad up/down", False)}
    if ecodes.ABS_GAS in absinfo or ecodes.ABS_BRAKE in absinfo:
        names.update({ecodes.ABS_Z: ("Right stick X", False), ecodes.ABS_RZ: ("Right stick Y", False),
                      ecodes.ABS_BRAKE: (f"{lt} trigger", True), ecodes.ABS_GAS: (f"{rt} trigger", True)})
    else:
        names.update({ecodes.ABS_RX: ("Right stick X", False), ecodes.ABS_RY: ("Right stick Y", False),
                      ecodes.ABS_Z: (f"{lt} trigger", True), ecodes.ABS_RZ: (f"{rt} trigger", True)})
    order = [ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX, ecodes.ABS_RY, ecodes.ABS_Z, ecodes.ABS_RZ,
             ecodes.ABS_BRAKE, ecodes.ABS_GAS, ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y]
    codes = [c for c in order if c in absinfo] + sorted(c for c in absinfo if c not in order)
    layout = []
    for code in codes:
        name, trigger = names.get(code, (ecodes.ABS.get(code, str(code)), False))
        layout.append((code, name, trigger and absinfo[code].min >= 0))
    return layout


def code_name(code):
    name = ecodes.BTN.get(code) or ecodes.KEY.get(code) or str(code)
    return "/".join(name) if isinstance(name, list) else name


def short_name(code):
    """Generic label: BTN_SOUTH -> SOUTH, BTN_TL -> TL."""
    name = ecodes.BTN.get(code) or ecodes.KEY.get(code) or str(code)
    name = name[-1] if isinstance(name, list) else name
    return name.replace("BTN_", "").replace("KEY_", "")


def bar(value, lo, hi, centered):
    """Text bar: centered axes grow from the middle, triggers from the left."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0
    pos = round(frac * (BAR - 1))
    cells = [" "] * BAR
    if centered:
        mid = BAR // 2
        for i in range(min(pos, mid), max(pos, mid) + 1):
            cells[i] = "="
        cells[mid] = "|"
        cells[pos] = "O"
    elif value > lo:
        for i in range(pos + 1):
            cells[i] = "="
    return "[" + "".join(cells) + "]"


def rumble(dev, ms=400):
    effect = ff.Effect(
        ecodes.FF_RUMBLE, -1, 0, ff.Trigger(0, 0), ff.Replay(ms, 0),
        ff.EffectType(ff_rumble_effect=ff.Rumble(strong_magnitude=0xFFFF, weak_magnitude=0xFFFF)),
    )
    dev.write(ecodes.EV_FF, dev.upload_effect(effect), 1)


class Session:
    """One connected controller (+ its touchpad device, if it has one)."""

    def __init__(self, pad, others):
        self.pad = pad
        self.touchpad = open_touchpad(pad)
        self.kind = controller_type(pad)
        self.labels = BUTTON_LABELS.get(self.kind, {})
        self.absinfo = dict(pad.capabilities(absinfo=True).get(ecodes.EV_ABS, []))
        self.axes = axis_layout(self.absinfo, self.kind)
        self.values = {code: info.value for code, info in self.absinfo.items()}
        self.buttons = sorted(pad.capabilities().get(ecodes.EV_KEY, []))
        if self.touchpad is not None:
            self.buttons.append(ecodes.BTN_LEFT)
        self.held = set(pad.active_keys())
        self.others = [f"{d.name} ({d.path})" for d in others]
        self.log = deque([f"{time.strftime('%H:%M:%S')}  connected"], maxlen=EVENT_LOG_LINES)
        self.event_count, self.rate_start, self.events_per_s = 0, time.time(), 0.0

    def devices(self):
        return [d for d in (self.pad, self.touchpad) if d is not None]

    def close(self):
        for d in self.devices():
            try:
                d.close()
            except OSError:
                pass

    def label(self, code):
        return self.labels.get(code) or short_name(code)

    def handle(self, dev, event):
        self.event_count += 1
        stamp = time.strftime("%H:%M:%S")
        if dev is self.touchpad:  # only its click - finger position isn't a button
            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_LEFT and event.value in (0, 1):
                self._button(stamp, event.code, event.value)
            return
        if event.type == ecodes.EV_ABS:
            self.values[event.code] = event.value
            if event.code in (ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y):
                self.log.appendleft(f"{stamp}  {ecodes.ABS[event.code]} = {event.value}")
        elif event.type == ecodes.EV_KEY and event.value in (0, 1):
            self._button(stamp, event.code, event.value)

    def _button(self, stamp, code, value):
        (self.held.add if value else self.held.discard)(code)
        self.log.appendleft(f"{stamp}  {self.label(code):8s} {'pressed ' if value else 'released'}  "
                            f"{DIM}{code_name(code)} = {code}{RESET}")

    def render(self):
        now = time.time()
        if now - self.rate_start >= 1.0:
            self.events_per_s = self.event_count / (now - self.rate_start)
            self.event_count, self.rate_start = 0, now
        out = [f"{BOLD}RoboPacerV2 - controller test (raw){RESET}   {DIM}Ctrl+C to quit{RESET}",
               f"{GREEN}connected{RESET}: {self.pad.name}  [{self.kind}]  |  {self.events_per_s:4.0f} events/s"]
        if self.others:
            out.append(f"{DIM}also connected: {'; '.join(self.others)} - pick with --device{RESET}")
        out += ["", f"{BOLD}Axes{RESET}  {DIM}raw value, scaled -1..1 (triggers 0..1), raw range{RESET}"]
        for code, name, trigger in self.axes:
            info = self.absinfo[code]
            v = self.values.get(code, info.value)
            # Straight linear scale of the raw value, nothing else.
            frac = (v - info.min) / (info.max - info.min) if info.max > info.min else 0.0
            scaled = frac if trigger else 2.0 * frac - 1.0
            color = YELLOW if code in (ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y) and v else ""
            out.append(f"  {name:17s} {bar(v, info.min, info.max, not trigger)} {color}{v:6d}{RESET}"
                       f" {scaled:+5.2f}  {DIM}{info.min}..{info.max}{RESET}")

        out += ["", f"{BOLD}Buttons{RESET}  {DIM}(held = highlighted){RESET}"]
        row, row_len = [], 2
        for code in self.buttons:
            text = self.label(code)
            if row_len + len(text) + 4 > LINE_WIDTH:
                out.append("  " + "  ".join(row))
                row, row_len = [], 2
            row.append(f"{GREEN}{BOLD}[{text}]{RESET}" if code in self.held else f"{DIM} {text} {RESET}")
            row_len += len(text) + 4
        if row:
            out.append("  " + "  ".join(row))
        pressed = [f"{self.label(c)} ({code_name(c)} = {c})" for c in sorted(self.held)]
        out.append(f"  held: {', '.join(pressed) if pressed else '-'}")

        out += ["", f"{BOLD}Last events{RESET}"]
        out += [f"  {line}" for line in self.log]
        return out


def main():
    ap = argparse.ArgumentParser(description="Live raw controller test - any gamepad (Xbox, PlayStation, ...).")
    ap.add_argument("--device", metavar="PATH", help="use this controller (e.g. /dev/input/event6) "
                                                      "instead of the first one found")
    ap.add_argument("--rumble", action="store_true", help="rumble the controller when it connects")
    args = ap.parse_args()

    sys.stdout.write(HIDE_CURSOR)
    session = None
    try:
        while True:
            if session is None:
                sys.stdout.write(f"{HOME}{CLEAR_BELOW}Waiting for a controller "
                                 f"(turn it on / pair it - dashboard Bluetooth panel)...  Ctrl+C to quit\n")
                sys.stdout.flush()
                pads = find_gamepads()
                pad = next((p for p in pads if p.path == args.device), None) if args.device else \
                    (pads[0] if pads else None)
                others = [p for p in pads if p is not pad]
                if pad is None:
                    for p in pads:
                        p.close()
                    time.sleep(RECONNECT_SECONDS)
                    continue
                session = Session(pad, others)
                for p in others:
                    p.close()
                sys.stdout.write(f"{HOME}{CLEAR_BELOW}")
                if args.rumble:
                    try:
                        rumble(pad)
                        session.log.appendleft(f"{time.strftime('%H:%M:%S')}  rumble sent - you should feel it")
                    except OSError as e:
                        session.log.appendleft(f"{time.strftime('%H:%M:%S')}  rumble failed: {e}")
                last_draw = 0.0

            try:
                by_fd = {d.fd: d for d in session.devices()}
                ready, _, _ = select.select(list(by_fd), [], [], REFRESH_SECONDS)
                for fd in ready:
                    dev = by_fd[fd]
                    for event in dev.read():
                        session.handle(dev, event)
            except OSError as e:  # Bluetooth dropped (errno 19) - wait for it to come back
                session.close()
                session = None
                sys.stdout.write(f"{HOME}{CLEAR_BELOW}Controller lost ({e}) - reconnecting...\n")
                sys.stdout.flush()
                time.sleep(RECONNECT_SECONDS)
                continue

            now = time.time()
            if now - last_draw >= REFRESH_SECONDS:
                last_draw = now
                lines = session.render()
                sys.stdout.write(HOME + "".join(line + CLEAR_LINE + "\n" for line in lines) + CLEAR_BELOW)
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a 2nd Ctrl+C mustn't interrupt the cleanup
        if session is not None:
            session.close()
        sys.stdout.write(SHOW_CURSOR + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
