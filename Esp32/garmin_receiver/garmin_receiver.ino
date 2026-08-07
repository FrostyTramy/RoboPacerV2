#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── Configuratie ──────────────────────────────────────────────
#define DEVICE_NAME          "Pacer1"   // numele afisat pe ceas
#define RELAY_PIN            2          // LED built-in ESP32 (inlocuieste releul la test; schimba la 26 cand ai releu)
#define WATCHDOG_TIMEOUT_MS  2000       // ms fara heartbeat → relay off

// Codul secret — trebuie sa fie identic cu APP_SECRET din App.mc
const uint8_t APP_SECRET[] = { 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6 };
const uint8_t APP_SECRET_LEN = 6;
// ─────────────────────────────────────────────────────────────

#define SERVICE_UUID "a0b0c0d0-e0f0-1234-5678-9abcdef01234"
#define CHAR_UUID    "a0b0c0d0-e0f0-1234-5678-9abcdef05678"

#define CMD_AUTH  0xAA
#define BTN_UP    0x01
#define BTN_DOWN  0x02
#define BTN_LAP   0x03
#define BTN_ENTER 0x04

BLECharacteristic* pWriteChar = nullptr;
bool authenticated = false;

// Watchdog
unsigned long lastHeartbeatMs = 0;
bool watchdogArmed = false;
String serialBuf = "";

void onButtonPress(uint8_t btn);

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
        digitalWrite(RELAY_PIN, HIGH);
        Serial.println("[RELAY] ON");
    } else if (btn == BTN_DOWN) {
        digitalWrite(RELAY_PIN, LOW);
        Serial.println("[RELAY] OFF");
        Serial.println("!!ESTOP!!");
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);

    String advName = String("GarminPacer|") + DEVICE_NAME;
    BLEDevice::init(advName.c_str());

    BLEServer* pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(SERVICE_UUID);
    pWriteChar = pService->createCharacteristic(
        CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    pWriteChar->setCallbacks(new CharCallbacks());
    pService->start();

    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    BLEAdvertisementData advData;
    advData.setName(advName.c_str());
    pAdv->setAdvertisementData(advData);
    BLEDevice::startAdvertising();

    Serial.println("Advertising ca '" + advName + "'...");
}

void loop() {
    // Citeste heartbeat de pe serial (non-blocking, char cu char)
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
            }
            serialBuf = "";
        } else {
            serialBuf += c;
        }
    }

    // Watchdog: daca heartbeat-ul s-a oprit, opreste releul
    if (watchdogArmed && (millis() - lastHeartbeatMs > WATCHDOG_TIMEOUT_MS)) {
        watchdogArmed = false;
        digitalWrite(RELAY_PIN, LOW);
        Serial.println("[WD] Relay OFF — heartbeat pierdut");
    }

    delay(10);
}
