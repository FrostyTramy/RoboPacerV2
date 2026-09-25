import Toybox.Lang;
import Toybox.WatchUi;

class ScanDelegate extends WatchUi.BehaviorDelegate {
    function initialize() { BehaviorDelegate.initialize(); }

    function onBack() as Boolean {
        return false; // iesire din aplicatie
    }
}
