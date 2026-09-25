import Toybox.Lang;
import Toybox.Timer;
import Toybox.WatchUi;
using Toybox.BluetoothLowEnergy as Ble;

class BleManager extends Ble.BleDelegate {
    var _device       as Ble.Device?         = null;
    var _writeChar    as Ble.Characteristic? = null;
    var _statusChar   as Ble.Characteristic? = null;
    var _foundDevices as Lang.Array          = [];
    var _targetName   as String              = "";
    var _wasConnected as Boolean             = false;
    var _rescanTimer  as Timer.Timer?        = null;

    function initialize() { BleDelegate.initialize(); }

    function start() as Void {
        Ble.setDelegate(self);
        Ble.registerProfile({
            :uuid => SERVICE_UUID,
            :characteristics => [
                { :uuid => WRITE_CHAR_UUID  },
                { :uuid => STATUS_CHAR_UUID }
            ]
        });
    }

    function stop() as Void {
        stopScan();
        if (_rescanTimer != null) { _rescanTimer.stop(); _rescanTimer = null; }
        if (_device != null) { Ble.unpairDevice(_device); _device = null; }
        _writeChar  = null;
        _statusChar = null;
    }

    function startScan() as Void {
        Ble.setScanState(Ble.SCAN_STATE_SCANNING);
    }

    function stopScan() as Void {
        Ble.setScanState(Ble.SCAN_STATE_OFF);
    }

    function startRescan() as Void {
        if (_rescanTimer != null) { _rescanTimer.stop(); }
        _rescanTimer = new Timer.Timer();
        _rescanTimer.start(method(:onRescanTimeout), 15000, false);
        Ble.setScanState(Ble.SCAN_STATE_SCANNING);
    }

    function onRescanTimeout() as Void {
        stopScan();
        _rescanTimer = null;
        getApp().onRescanFinished();
    }

    function isRescanning() as Boolean {
        return _rescanTimer != null;
    }

    function getFoundDevices() as Lang.Array { return _foundDevices; }

    function resetForNewScan() as Void {
        _foundDevices = [];
        _targetName   = "";
        _wasConnected = false;
        if (_device != null) { Ble.unpairDevice(_device); _device = null; }
        _writeChar  = null;
        _statusChar = null;
        Ble.setScanState(Ble.SCAN_STATE_SCANNING);
    }

    function connectTo(scanResult as Ble.ScanResult, displayName as String) as Void {
        _targetName = displayName;
        stopScan();
        if (_rescanTimer != null) { _rescanTimer.stop(); _rescanTimer = null; }
        Ble.pairDevice(scanResult);
    }

    function sendRelay(on as Boolean) as Boolean {
        if (_writeChar == null) { return false; }
        var cmd = on ? [0x01]b : [0x02]b;
        try { _writeChar.requestWrite(cmd, {}); return true; } catch (e) { return false; }
    }

    function requestStatus() as Void {
        if (_statusChar == null) { return; }
        try { _statusChar.requestRead(); } catch (e) {}
    }

    // ── BleDelegate ──────────────────────────────────────────────

    function onProfileRegister(uuid as Ble.Uuid, status as Ble.Status) as Void {
        if (status == 0) { startScan(); }
    }

    function onScanResults(scanResults as Ble.Iterator) as Void {
        var wasEmpty = (_foundDevices.size() == 0);
        var updated  = false;

        for (var r = scanResults.next(); r != null; r = scanResults.next()) {
            if (!(r instanceof Ble.ScanResult)) { continue; }
            var sr   = r as Ble.ScanResult;
            var name = sr.getDeviceName();
            if (name == null || name.length() <= 12) { continue; }
            if (!name.substring(0, 12).equals("GarminPacer|")) { continue; }
            var displayName = name.substring(12, name.length());

            var exists = false;
            for (var i = 0; i < _foundDevices.size(); i++) {
                if ((_foundDevices[i] as Lang.Dictionary)[:name].equals(displayName)) {
                    exists = true; break;
                }
            }
            if (!exists) {
                _foundDevices.add({ :name => displayName, :scanResult => sr });
                updated = true;
            }
        }

        if (updated) {
            if (wasEmpty) {
                getApp().onFirstDeviceFound();
            } else {
                getApp().onMoreDevicesFound();
            }
        }
    }

    function onConnectedStateChanged(device as Ble.Device, state as Ble.ConnectionState) as Void {
        if (state == Ble.CONNECTION_STATE_CONNECTED) {
            _device = device;
            var service = device.getService(SERVICE_UUID);
            if (service != null) {
                _writeChar  = service.getCharacteristic(WRITE_CHAR_UUID);
                _statusChar = service.getCharacteristic(STATUS_CHAR_UUID);
                try { _writeChar.requestWrite(APP_SECRET, {}); } catch (e) {}
                _wasConnected = true;
                getApp().onConnected();
            }
        } else {
            _device     = null;
            _writeChar  = null;
            _statusChar = null;
            if (_wasConnected) {
                _wasConnected = false;
                getApp().onDisconnected();
            }
        }
    }

    function onCharacteristicRead(char as Ble.Characteristic, status as Ble.Status, value as Lang.ByteArray) as Void {
        if (status != 0 || value.size() == 0) { return; }
        if (char.getUuid().equals(STATUS_CHAR_UUID)) {
            getApp().onRelayStateConfirmed(value[0] == 0x01);
        }
    }
}
