#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#define DEVICE_NAME         "Pacer1"
#define RELAY_PIN           13
#define LED_PIN             4
#define WATCHDOG_TIMEOUT_MS 2000

#include "secrets.h"  // APP_SECRET[], APP_SECRET_LEN

#define SERVICE_UUID     "a0b0c0d0-e0f0-1234-5678-9abcdef01234"
#define WRITE_CHAR_UUID  "a0b0c0d0-e0f0-1234-5678-9abcdef05678"
#define STATUS_CHAR_UUID "a0b0c0d0-e0f0-1234-5678-9abcdef09abc"

#define CMD_AUTH 0xAA
#define CMD_ON   0x01
#define CMD_OFF  0x02

BLECharacteristic* pWriteChar  = nullptr;
BLECharacteristic* pStatusChar = nullptr;

bool authenticated = false;
bool relayOn       = false;

unsigned long lastHeartbeatMs = 0;
bool watchdogArmed = false;
String serialBuf = "";

// Seteaza ambele iesiri + actualizeaza STATUS_CHAR + trimite ESTOP pe serial la OFF
void setRelay(bool on) {
    if (on == relayOn) {
        Serial.printf("[RELAY] Ignorat — deja %s\n", on ? "ON" : "OFF");
        return;
    }
    relayOn = on;
    digitalWrite(RELAY_PIN, on ? HIGH : LOW);
    digitalWrite(LED_PIN,   on ? HIGH : LOW);
    if (pStatusChar != nullptr) {
        uint8_t val = on ? 0x01 : 0x00;
        pStatusChar->setValue(&val, 1);
    }
    Serial.printf("[RELAY] %s\n", on ? "ON" : "OFF");
    if (!on) {
        Serial.println("!!ESTOP!!");
    }
}

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer*) override {
        authenticated = false;
        Serial.println("[BLE] Conectat");
    }
    void onDisconnect(BLEServer* pServer) override {
        authenticated = false;
        setRelay(false);
        Serial.println("[BLE] Deconectat, restart advertising");
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
            Serial.println(ok ? "[AUTH] OK" : "[AUTH] Gresit!");
            return;
        }

        if (!authenticated) {
            Serial.println("[AUTH] Comanda ignorata — neautentificat");
            return;
        }

        if      (cmd == CMD_ON)  { setRelay(true);  }
        else if (cmd == CMD_OFF) { setRelay(false); }
        else { Serial.printf("[BLE] CMD necunoscut: 0x%02X\n", cmd); }
    }
};

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(LED_PIN,   OUTPUT);
    digitalWrite(RELAY_PIN, LOW);
    digitalWrite(LED_PIN,   LOW);

    String advName = String("GarminPacer|") + DEVICE_NAME;
    BLEDevice::init(advName.c_str());

    BLEServer* pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(BLEUUID(SERVICE_UUID), 8);

    pWriteChar = pService->createCharacteristic(
        WRITE_CHAR_UUID,
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
                setRelay(true);
            } else if (serialBuf.equals("!!OFF!!")) {
                setRelay(false);
            }
            serialBuf = "";
        } else {
            serialBuf += c;
        }
    }

    if (watchdogArmed && (millis() - lastHeartbeatMs > WATCHDOG_TIMEOUT_MS)) {
        watchdogArmed = false;
        Serial.println("[WD] Heartbeat pierdut");
        setRelay(false);
    }

    delay(10);
}
