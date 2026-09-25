import Toybox.Application;
import Toybox.Lang;
import Toybox.WatchUi;
using Toybox.BluetoothLowEnergy as Ble;

const SERVICE_UUID     = Ble.stringToUuid("a0b0c0d0-e0f0-1234-5678-9abcdef01234");
const WRITE_CHAR_UUID  = Ble.stringToUuid("a0b0c0d0-e0f0-1234-5678-9abcdef05678");
const STATUS_CHAR_UUID = Ble.stringToUuid("a0b0c0d0-e0f0-1234-5678-9abcdef09abc");

// APP_SECRET este definit in Secrets.mc (ignorat de git — vezi Secrets.mc.example)

class GarminPacerApp extends Application.AppBase {
    var bleManager      as BleManager;
    var scanView        as ScanView;
    var _relayView      as RelayView?      = null;
    var _relayDelegate  as RelayDelegate?  = null;
    var _deviceListOpen as Boolean         = false;

    function initialize() {
        AppBase.initialize();
        bleManager = new BleManager();
        scanView   = new ScanView();
    }

    function onStart(state as Lang.Dictionary?) as Void {
        bleManager.start();
    }

    function onStop(state as Lang.Dictionary?) as Void {
        bleManager.stop();
    }

    function getInitialView() as [WatchUi.Views] or [WatchUi.Views, WatchUi.InputDelegates] {
        return [scanView, new ScanDelegate()];
    }

    // Apelat de BleManager cand primul device e gasit
    function onFirstDeviceFound() as Void {
        bleManager.stopScan();
        pushDeviceList();
    }

    // Apelat de BleManager cand un device nou e gasit in timpul rescan-ului
    function onMoreDevicesFound() as Void {
        refreshDeviceList();
    }

    // Apelat de BleManager cand rescan-ul de 15s se incheie
    function onRescanFinished() as Void {
        refreshDeviceList();
    }

    function pushDeviceList() as Void {
        _deviceListOpen = true;
        var menu = new WatchUi.Menu2({ :title => "Dispozitive" });
        var devices = bleManager.getFoundDevices();
        for (var i = 0; i < devices.size(); i++) {
            var d = devices[i] as Lang.Dictionary;
            menu.addItem(new WatchUi.MenuItem(d[:name] as String, null, i, {}));
        }
        var rescanLabel = bleManager.isRescanning() ? "Rescaneaza..." : "Rescan";
        menu.addItem(new WatchUi.MenuItem(rescanLabel, null, :rescan, {}));
        WatchUi.pushView(menu, new DeviceListDelegate(), WatchUi.SLIDE_LEFT);
    }

    function refreshDeviceList() as Void {
        if (!_deviceListOpen) { return; }
        WatchUi.popView(WatchUi.SLIDE_LEFT);
        _deviceListOpen = false;
        pushDeviceList();
    }

    // Apelat de BleManager cand conexiunea BLE e gata
    function onConnected() as Void {
        _deviceListOpen = false;
        var view     = new RelayView();
        var delegate = new RelayDelegate(view);
        _relayView     = view;
        _relayDelegate = delegate;
        WatchUi.switchToView(view, delegate, WatchUi.SLIDE_LEFT);
    }

    // Apelat de BleManager la deconectare neasteptata
    function onDisconnected() as Void {
        _relayView     = null;
        _relayDelegate = null;
        _deviceListOpen = false;
        bleManager.resetForNewScan();
        WatchUi.switchToView(new ScanView(), new ScanDelegate(), WatchUi.SLIDE_RIGHT);
    }

    // Apelat cand STATUS_CHAR e citit (starea confirmata de ESP32)
    function onRelayStateConfirmed(on as Boolean) as Void {
        if (_relayDelegate != null) {
            (_relayDelegate as RelayDelegate).onRelayConfirmed(on);
        }
    }
}

function getApp() as GarminPacerApp {
    return Application.getApp() as GarminPacerApp;
}
