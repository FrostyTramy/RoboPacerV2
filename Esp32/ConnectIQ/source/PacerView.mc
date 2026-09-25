import Toybox.Graphics;
import Toybox.Lang;
import Toybox.WatchUi;

// Stari de afisare
const RELAY_PENDING = 0;  // gri — asteptam confirmarea de la ESP32
const RELAY_ON      = 1;  // verde — confirmat ON
const RELAY_OFF     = 2;  // rosu  — confirmat OFF

class RelayView extends WatchUi.View {
    var _state as Number = RELAY_PENDING; // initial gri pana la primul poll

    function initialize() { View.initialize(); }

    function setPending() as Void {
        _state = RELAY_PENDING;
        WatchUi.requestUpdate();
    }

    function setConfirmed(on as Boolean) as Void {
        _state = on ? RELAY_ON : RELAY_OFF;
        WatchUi.requestUpdate();
    }

    function getState() as Number { return _state; }

    function onUpdate(dc as Graphics.Dc) as Void {
        var bgColor;
        var text;

        if (_state == RELAY_ON) {
            bgColor = Graphics.COLOR_GREEN;
            text    = "ON";
        } else if (_state == RELAY_OFF) {
            bgColor = Graphics.COLOR_RED;
            text    = "OFF";
        } else {
            bgColor = Graphics.COLOR_DK_GRAY;
            text    = "...";
        }

        dc.setColor(Graphics.COLOR_TRANSPARENT, bgColor);
        dc.clear();

        var cx = dc.getWidth()  / 2;
        var cy = dc.getHeight() / 2;

        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        dc.drawText(cx, cy - 40, Graphics.FONT_NUMBER_HOT, text, Graphics.TEXT_JUSTIFY_CENTER);

        dc.drawText(cx, cy + 50, Graphics.FONT_XTINY,
                    "UP=ON  DOWN=OFF",
                    Graphics.TEXT_JUSTIFY_CENTER);
    }
}
