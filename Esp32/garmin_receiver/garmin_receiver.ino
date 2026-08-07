#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── Configuratie ──────────────────────────────────────────────
#define DEVICE_NAME          "Pacer1"   // numele afisat pe ceas
#define RELAY_PIN            2          // LED built-in ESP32 (inlocuieste releul la test; schimba la 26 cand ai releu)
#define WATCHDOG_TIMEOUT_MS  2000       // ms fara heartbeat → relay off

#include "secrets.h"  // APP_SECRET, APP_SECRET_LEN — nu e pe GitHub
// ─────────────────────────────────────────────────────────────

#define SERVICE_UUID      "a0b0c0d0-e0f0-1234-5678-9abcdef01234"
#define CHAR_UUID         "a0b0c0d0-e0f0-1234-5678-9abcdef05678"  // WRITE (comenzi Garmin)
#define STATUS_CHAR_UUID  "a0b0c0d0-e0f0-1234-5678-9abcdef09abc"  // READ  (stare releu → Garmin)

#define CMD_AUTH  0xAA
#define BTN_UP    0x01
#define BTN_DOWN  0x02
#define BTN_LAP   0x03
#define BTN_ENTER 0x04

BLECharacteristic* pWriteChar  = nullptr;
BLECharacteristic* pStatusChar = nullptr;
bool authenticated = false;
bool relayOn       = false;

// Watchdog
unsigned long lastHeartbeatMs = 0;
bool watchdogArmed = false;
String serialBuf = "";

void onButtonPress(uint8_t btn);

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
            case BTN_UP:    Serial.println("BUTON: UP");    break;
            case BTN_DOWN:  Serial.println("BUTON: DOWN");  break;
            case BTN_LAP:   Serial.println("BUTON: LAP");   break;
            case BTN_ENTER: Serial.println("BUTON: ENTER"); break;
            default:        Serial.printf("CMD: 0x%02X\n", cmd); break;
        }
        onButtonPress(cmd);
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

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    updateRelayState(false);

    String advName = String("GarminPacer|") + DEVICE_NAME;
    BLEDevice::init(advName.c_str());

    BLEServer* pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(BLEUUID(SERVICE_UUID), 10);

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
