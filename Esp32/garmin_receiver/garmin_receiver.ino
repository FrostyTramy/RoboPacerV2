#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── Configuratie ──────────────────────────────────────────────
#define DEVICE_NAME          "Pacer1"   // numele afisat pe ceas
#define RELAY_PIN            2          // LED built-in ESP32 (inlocuieste releul la test; schimba la 26 cand ai releu)
#define WATCHDOG_TIMEOUT_MS  2000       // ms fara heartbeat → relay off
#define SCRIPTS_MAX_LEN      200        // caractere maxime pentru lista de scripturi

#include "secrets.h"  // APP_SECRET, APP_SECRET_LEN — nu e pe GitHub, vezi secrets.h.example
// ─────────────────────────────────────────────────────────────

#define SERVICE_UUID       "a0b0c0d0-e0f0-1234-5678-9abcdef01234"
#define CHAR_UUID          "a0b0c0d0-e0f0-1234-5678-9abcdef05678"  // WRITE  (comenzi Garmin)
#define STATUS_CHAR_UUID   "a0b0c0d0-e0f0-1234-5678-9abcdef09abc"  // READ   (stare releu → Garmin)
#define SCRIPTS_CHAR_UUID  "a0b0c0d0-e0f0-1234-5678-9abcdef0ef01"  // READ   (lista scripturi → Garmin)

#define CMD_AUTH  0xAA
#define BTN_UP    0x01
#define BTN_DOWN  0x02
#define BTN_LAP   0x03
#define BTN_ENTER 0x04
#define CMD_RUN   0x10  // + 1 byte index script
#define CMD_STOP  0x11

BLECharacteristic* pWriteChar   = nullptr;
BLECharacteristic* pStatusChar  = nullptr;
BLECharacteristic* pScriptsChar = nullptr;
bool authenticated = false;
bool relayOn       = false;

// Watchdog
unsigned long lastHeartbeatMs = 0;
bool watchdogArmed = false;
String serialBuf = "";

void onButtonPress(uint8_t btn);
void onRunScript(uint8_t idx);

// Seteaza starea releului si actualizeaza caracteristica de status BLE
void updateRelayState(bool on) {
    relayOn = on;
    digitalWrite(RELAY_PIN, on ? HIGH : LOW);
    if (pStatusChar != nullptr) {
        uint8_t val = on ? 0x01 : 0x00;
        pStatusChar->setValue(&val, 1);
    }
}

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        authenticated = false;
        Serial.println("[BLE] Conectat, astept autentificare...");
    }
    void onDisconnect(BLEServer* pServer) override {
        authenticated = false;
        Serial.println("[BLE] Deconectat, reincep advertising...");
        pServer->getAdvertising()->start();
    }
};

class CharCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pChar) override {
        String val = pChar->getValue();
        if (val.length() == 0) return;
        uint8_t cmd = (uint8_t)val[0];

        if (cmd == CMD_AUTH) {
            if ((uint8_t)val.length() < 1 + APP_SECRET_LEN) {
                Serial.println("[AUTH] Pachet prea scurt");
                return;
            }
            bool ok = true;
            for (int i = 0; i < APP_SECRET_LEN; i++) {
                if ((uint8_t)val[i + 1] != APP_SECRET[i]) { ok = false; break; }
            }
            authenticated = ok;
            Serial.println(ok ? "[AUTH] Autentificat OK" : "[AUTH] Cod gresit!");
            return;
        }

        if (!authenticated) {
            Serial.println("[AUTH] Comanda ignorata — neautentificat");
            return;
        }

        switch (cmd) {
            case BTN_UP:    Serial.println("BUTON: UP");    onButtonPress(BTN_UP);    break;
            case BTN_DOWN:  Serial.println("BUTON: DOWN");  onButtonPress(BTN_DOWN);  break;
            case BTN_LAP:   Serial.println("BUTON: LAP");                             break;
            case BTN_ENTER: Serial.println("BUTON: ENTER");                           break;
            case CMD_STOP:
                Serial.println("[SCRIPTS] STOP");
                Serial.println("!!STOP!!");
                break;
            case CMD_RUN:
                if (val.length() >= 2) {
                    uint8_t idx = (uint8_t)val[1];
                    Serial.printf("[SCRIPTS] RUN %d\n", idx);
                    onRunScript(idx);
                }
                break;
            default:
                Serial.printf("CMD: 0x%02X\n", cmd);
                break;
        }
    }
};

void onButtonPress(uint8_t btn) {
    if (btn == BTN_UP) {
        updateRelayState(true);
        Serial.println("[RELAY] ON");
    } else if (btn == BTN_DOWN) {
        updateRelayState(false);
        Serial.println("[RELAY] OFF");
        Serial.println("!!ESTOP!!");
    }
}

void onRunScript(uint8_t idx) {
    Serial.print("!!RUN!!");
    Serial.println(idx);
}

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    updateRelayState(false);

    String advName = String("GarminPacer|") + DEVICE_NAME;
    BLEDevice::init(advName.c_str());

    BLEServer* pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(BLEUUID(SERVICE_UUID), 16);

    pWriteChar = pService->createCharacteristic(
        CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    pWriteChar->setCallbacks(new CharCallbacks());

    pStatusChar = pService->createCharacteristic(
        STATUS_CHAR_UUID,
        BLECharacteristic::PROPERTY_READ
    );
    uint8_t initVal = 0x00;
    pStatusChar->setValue(&initVal, 1);

    pScriptsChar = pService->createCharacteristic(
        SCRIPTS_CHAR_UUID,
        BLECharacteristic::PROPERTY_READ
    );
    pScriptsChar->setValue("");

    pService->start();

    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    BLEAdvertisementData advData;
    advData.setName(advName.c_str());
    pAdv->setAdvertisementData(advData);
    BLEDevice::startAdvertising();

    Serial.println("Advertising ca '" + advName + "'...");
}

void loop() {
    // Citeste comenzi de pe serial (non-blocking, char cu char)
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            serialBuf.trim();

            if (serialBuf.equals("!!HB!!")) {
                lastHeartbeatMs = millis();
                if (!watchdogArmed) {
                    watchdogArmed = true;
                    Serial.println("[WD] Armat");
                }
            } else if (serialBuf.equals("!!ON!!")) {
                updateRelayState(true);
                Serial.println("[RELAY] ON via serial");
            } else if (serialBuf.equals("!!OFF!!")) {
                updateRelayState(false);
                Serial.println("[RELAY] OFF via serial");
            } else if (serialBuf.startsWith("!!SCRIPTS!!")) {
                String data = serialBuf.substring(11); // dupa "!!SCRIPTS!!"
                if (data.length() > SCRIPTS_MAX_LEN) {
                    data = data.substring(0, SCRIPTS_MAX_LEN);
                }
                if (pScriptsChar != nullptr) {
                    pScriptsChar->setValue(data.c_str());
                }
                Serial.println("[SCRIPTS] Lista actualizata: " + data);
            }

            serialBuf = "";
        } else {
            serialBuf += c;
        }
    }

    // Watchdog: daca heartbeat-ul s-a oprit, opreste releul
    if (watchdogArmed && (millis() - lastHeartbeatMs > WATCHDOG_TIMEOUT_MS)) {
        watchdogArmed = false;
        updateRelayState(false);
        Serial.println("[WD] Relay OFF — heartbeat pierdut");
    }

    delay(10);
}
