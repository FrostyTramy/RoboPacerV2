import Toybox.Lang;
import Toybox.WatchUi;
using Toybox.BluetoothLowEnergy as Ble;

class DeviceListDelegate extends WatchUi.Menu2InputDelegate {
    function initialize() {
        Menu2InputDelegate.initialize();
    }

    function onSelect(item as WatchUi.MenuItem) as Void {
        var id = item.getId();

        // Butonul Rescan
        if (id.equals(:rescan)) {
            getApp().bleManager.startRescan();
            getApp().refreshDeviceList();
            return;
        }

        // Device selectat
        var idx     = id as Number;
        var devices = getApp().bleManager.getFoundDevices();
        if (idx >= 0 && idx < devices.size()) {
            var d    = devices[idx] as Lang.Dictionary;
            var name = d[:name] as String;
            getApp()._deviceListOpen = false;
            // Pop device list inapoi la ScanView si arata "Conectez..."
            WatchUi.popView(WatchUi.SLIDE_RIGHT);
            getApp().scanView.showConnecting(name);
            getApp().bleManager.connectTo(
                d[:scanResult] as Ble.ScanResult,
                name
            );
        }
    }

    function onBack() as Void {
        getApp()._deviceListOpen = false;
        // Sistemul face pop automat; ScanView.onShow() va reporni scan-ul
    }
}
