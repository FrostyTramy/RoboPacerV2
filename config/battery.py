"""
3S LiPo battery monitor - INA219 on I2C bus 1, address 0x41 (A0 soldered).

Register access, config word, shunt value and the resting-voltage -> percent
table are taken from the standalone example (baterie_ina219.py). The
dashboard reads this for the voltage/percent shown on every page.

The percentage is approximate: the table is for a battery at rest, so it
reads a few percent low while the motor is drawing current (voltage sag).
The voltage is smoothed a little (EMA) so the number doesn't jitter with
every throttle blip.

Never raises: read_battery() returns None if the chip doesn't answer, so a
missing/unpowered INA219 can't break the status page.
"""

import threading
import time

from smbus2 import SMBus

I2C_BUS = 1
ADDRESS = 0x41
R_SHUNT = 0.1  # ohms (the "R100" resistor on the module)
CELLS = 3

REG_CONFIG = 0x00
REG_SHUNT = 0x01
REG_BUS = 0x02
# 32V, PGA /8 (+-320mV), 12-bit ADC, continuous (shunt + bus)
CONFIG = 0x399F

# Per-cell resting voltage -> percent
LIPO_TABLE = [
    (4.20, 100), (4.15, 95), (4.11, 90), (4.08, 85), (4.02, 80),
    (3.98, 75), (3.95, 70), (3.91, 65), (3.87, 60), (3.85, 55),
    (3.84, 50), (3.82, 45), (3.80, 40), (3.79, 35), (3.77, 30),
    (3.75, 25), (3.73, 20), (3.71, 15), (3.69, 10), (3.61, 5),
    (3.27, 0),
]

CELL_V_WARN = 3.7    # below: "weak battery"
CELL_V_EMPTY = 3.5   # below: "discharged - charge it"
NO_BATTERY_V = 1.0   # bus voltage under this = nothing connected to the sensor

CACHE_SECONDS = 0.5  # several open tabs must not multiply the I2C traffic
EMA_ALPHA = 0.3

_lock = threading.Lock()
_cache = {"at": 0.0, "value": None}
_smoothed_v = None
_configured = False


def _read_reg(bus, reg):
    """INA219 sends the MSB first; smbus reads little-endian -> swap."""
    v = bus.read_word_data(ADDRESS, reg)
    return ((v & 0xFF) << 8) | (v >> 8)


def _write_reg(bus, reg, val):
    bus.write_word_data(ADDRESS, reg, ((val & 0xFF) << 8) | (val >> 8))


def percent(v_total):
    v_cell = v_total / CELLS
    if v_cell >= LIPO_TABLE[0][0]:
        return 100.0
    if v_cell <= LIPO_TABLE[-1][0]:
        return 0.0
    for (v_hi, p_hi), (v_lo, p_lo) in zip(LIPO_TABLE, LIPO_TABLE[1:]):
        if v_lo <= v_cell <= v_hi:
            return p_lo + (v_cell - v_lo) / (v_hi - v_lo) * (p_hi - p_lo)
    return 0.0


def _level(v_cell):
    if v_cell < CELL_V_EMPTY:
        return "empty"
    if v_cell < CELL_V_WARN:
        return "low"
    return "ok"


def read_battery():
    """{"voltage_v", "cell_v", "current_a", "power_w", "percent", "level"} or
    None. level is "ok" | "low" | "empty" | "none" (no battery on the sensor)."""
    global _smoothed_v, _configured
    with _lock:
        now = time.time()
        if now - _cache["at"] < CACHE_SECONDS:
            return _cache["value"]
        value = None
        try:
            with SMBus(I2C_BUS) as bus:
                if not _configured:
                    _write_reg(bus, REG_CONFIG, CONFIG)
                    _configured = True
                v_bus = (_read_reg(bus, REG_BUS) >> 3) * 0.004  # LSB = 4 mV
                raw_shunt = _read_reg(bus, REG_SHUNT)
                if raw_shunt & 0x8000:
                    raw_shunt -= 1 << 16
                v_shunt = raw_shunt * 10e-6  # LSB = 10 uV
            v_battery = v_bus + v_shunt
            current = v_shunt / R_SHUNT
            if v_battery < NO_BATTERY_V:
                _smoothed_v = None
                value = {"voltage_v": round(v_battery, 2), "cell_v": None, "current_a": None,
                         "power_w": None, "percent": None, "level": "none"}
            else:
                _smoothed_v = v_battery if _smoothed_v is None else (
                    EMA_ALPHA * v_battery + (1 - EMA_ALPHA) * _smoothed_v)
                value = {
                    "voltage_v": round(_smoothed_v, 2),
                    "cell_v": round(_smoothed_v / CELLS, 2),
                    "current_a": round(current, 2),
                    "power_w": round(v_bus * current, 1),
                    "percent": round(percent(_smoothed_v)),
                    "level": _level(_smoothed_v / CELLS),
                }
        except Exception:
            _configured = False  # chip may have been power-cycled - re-init next time
        _cache.update(at=now, value=value)
        return value
