import Toybox.Lang;
import Toybox.Timer;
import Toybox.WatchUi;

class RelayDelegate extends WatchUi.BehaviorDelegate {
    var _view       as RelayView;
    var _pollTimer  as Timer.Timer;
    var _spamTimer  as Timer.Timer;
    var _desiredOn  as Boolean = false;
    var _spamActive as Boolean = false;

    function initialize(view as RelayView) {
        BehaviorDelegate.initialize();
        _view      = view;
        _pollTimer = new Timer.Timer();
        _spamTimer = new Timer.Timer();
        // Poll STATUS_CHAR la 500ms pentru a actualiza starea afisata
        _pollTimer.start(method(:onPollTimer), 500, true);
    }

    function onPollTimer() as Void {
        getApp().bleManager.requestStatus();
    }

    // Apelat de App cand ESP32 confirma starea
    function onRelayConfirmed(on as Boolean) as Void {
        if (_spamActive) {
            if (on == _desiredOn) {
                // Starea dorita confirmata — opreste spam si actualizeaza view
                _spamTimer.stop();
                _spamActive = false;
                _view.setConfirmed(on);
            }
            // Altfel: spam inca activ, ignoram confirmarea intermediara (ramane gri)
        } else {
            _view.setConfirmed(on);
        }
    }

    function onSpamTimer() as Void {
        if (!_spamActive) { _spamTimer.stop(); return; }
        getApp().bleManager.sendRelay(_desiredOn);
    }

    function onKey(keyEvent as WatchUi.KeyEvent) as Boolean {
        var key = keyEvent.getKey();

        if (key == WatchUi.KEY_UP) {
            _desiredOn  = true;
            _spamActive = true;
            _view.setPending();
            _spamTimer.stop();
            getApp().bleManager.sendRelay(true);
            _spamTimer.start(method(:onSpamTimer), 300, true);
            return true;
        }

        if (key == WatchUi.KEY_DOWN) {
            _desiredOn  = false;
            _spamActive = true;
            _view.setPending();
            _spamTimer.stop();
            getApp().bleManager.sendRelay(false);
            _spamTimer.start(method(:onSpamTimer), 300, true);
            return true;
        }

        return false;
    }

    function onBack() as Boolean {
        _pollTimer.stop();
        _spamTimer.stop();
        return false;
    }
}
