import Toybox.Graphics;
import Toybox.Lang;
import Toybox.Timer;
import Toybox.WatchUi;

class ScanView extends WatchUi.View {
    var _timer      as Timer.Timer;
    var _dotCount   as Number  = 0;
    var _connecting as Boolean = false;
    var _deviceName as String  = "";

    function initialize() {
        View.initialize();
        _timer = new Timer.Timer();
    }

    function showConnecting(name as String) as Void {
        _connecting = true;
        _deviceName = name;
        WatchUi.requestUpdate();
    }

    function onShow() as Void {
        if (!_connecting) {
            getApp().bleManager.startScan();
        }
        _timer.start(method(:onTick), 600, true);
    }

    function onHide() as Void {
        _timer.stop();
    }

    function onTick() as Void {
        _dotCount = (_dotCount + 1) % 4;
        WatchUi.requestUpdate();
    }

    function onUpdate(dc as Graphics.Dc) as Void {
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_BLACK);
        dc.clear();

        var cx = dc.getWidth()  / 2;
        var cy = dc.getHeight() / 2;

        var dots = "";
        for (var i = 0; i < _dotCount; i++) { dots += "."; }

        if (_connecting) {
            dc.setColor(Graphics.COLOR_BLUE, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, cy - 25, Graphics.FONT_MEDIUM, "Conectez" + dots, Graphics.TEXT_JUSTIFY_CENTER);
            dc.setColor(Graphics.COLOR_LT_GRAY, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, cy + 20, Graphics.FONT_XTINY, _deviceName, Graphics.TEXT_JUSTIFY_CENTER);
        } else {
            dc.setColor(Graphics.COLOR_YELLOW, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, cy - 25, Graphics.FONT_MEDIUM, "Scanez" + dots, Graphics.TEXT_JUSTIFY_CENTER);
            dc.setColor(Graphics.COLOR_LT_GRAY, Graphics.COLOR_TRANSPARENT);
            dc.drawText(cx, cy + 20, Graphics.FONT_XTINY, "GarminPacer|...", Graphics.TEXT_JUSTIFY_CENTER);
        }
    }
}
