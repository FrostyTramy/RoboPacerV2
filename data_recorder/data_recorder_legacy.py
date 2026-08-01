# ============================================================================
# DEPRECATED / UNUSED — kept only for reference.
# Replaced by data_recorder.py in this same folder (cleaner structure,
# correct ESC arming sequence, matches camera.py's capture settings).
# Do not run this file for new recordings.
# ============================================================================

import board
import busio
import time
import json
import os
import cv2
import select
from picamera2 import Picamera2
from evdev import InputDevice, ecodes, list_devices
import numpy as np 

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# ===================================================================
# 📌 VARIABILE VIZIBILE DE CONFIGURARE LOGGING
# ===================================================================
BASE_DIR = "/home/pi/Documents/neuronal"
# 1. Calea completă către fișierul JSON de log
JSON_FILE_PATH = os.path.join(BASE_DIR, "driving_log.json")
# 2. Calea completă către folderul unde se vor salva cadrele
FRAMES_DIR_PATH = os.path.join(BASE_DIR, "frames")
# ===================================================================

SERVO_CHANNEL = 0
SERVO_MIN_ANGLE = 45 
SERVO_MAX_ANGLE = 135 
SERVO_NEUTRAL_ANGLE = 90
SERVO_OFFSET = 1  

STEERING_BIAS_CORRECTION = 0.04 
SCALING_FACTOR = 1.0 / 0.95 # Factorul de scalare: 0.95 -> 1.0
STEERING_DEADZONE = 0.06 # Deadzone: -0.06 la 0.06 devine 0.0

# NOU: Inversarea pentru a alinia la standardul ML (-1.0 = Stânga, +1.0 = Dreapta)
STEERING_INVERT_FACTOR = -1.0 

ESC_CHANNEL = 1
ARM_PULSE = 1000
NEUTRAL = 1500
MIN_SPEED = 1000
MAX_SPEED = 1700

# CONSTANTE BUTOANE MODIFICATE
BTN_START_RECORDING = ecodes.BTN_A  # Start Înregistrare / Resume
BTN_PAUSE = ecodes.BTN_Y       # Pauză
BTN_SAVE_AND_STOP = ecodes.BTN_B   # Salvare și Oprire
# ------------------------------------------------------------------

CENTER_X = 32767
DEADZONE = 1000
X_MIN = 0  
X_MAX = 65535 

current_fps = 0
frame_count_fps = 0
start_time_fps = time.time()

gas_value = 0
brake_value = 0
current_x_value = CENTER_X 
driving_log = None 
pca = None
my_servo = None
picam2 = None
controller = None
controller_fd = -1


def set_esc_pulse(pulse_us):
    if pca is None:
        return
        
    duty_cycle_16bit = int((pulse_us / 20000) * 0xFFFF)
    
    # 🛑 GESTIONARE EROARE: Încapsulează operațiunea I2C
    try:
        pca.channels[ESC_CHANNEL].duty_cycle = duty_cycle_16bit
    except OSError as e:
        if e.errno == 121: # Remote I/O error
            print(f"⚠️ AVERTISMENT I2C (ESC): Remote I/O error la setarea pulsului {pulse_us}µs.", end='\r')
            time.sleep(0.01) # Pauză scurtă după eroare pentru a permite resetarea magistralei
        else:
            raise e


def calc_pulse(gas_value, brake_value):
 gas_offset = (gas_value / 1023.0) * (MAX_SPEED - NEUTRAL)
 brake_offset = -(brake_value / 1023.0) * (NEUTRAL - MIN_SPEED)

 pulse = NEUTRAL + gas_offset + brake_offset
 pulse = max(MIN_SPEED, min(MAX_SPEED, int(pulse)))
 return pulse

def map_angle_to_steering(angle):
 steering_value = np.interp(angle, [SERVO_MIN_ANGLE, SERVO_MAX_ANGLE], [-1.0, 1.0])
 return float(steering_value)

def get_steering_label(servo_angle):
 
 steering_label = map_angle_to_steering(servo_angle)
 steering_label += STEERING_BIAS_CORRECTION
 
 # 1. Aplică scalarea liniară: 0.95 -> 1.0
 steering_label *= SCALING_FACTOR
 
 # 2. INVERSARE: Aplică factorul de inversare
 steering_label *= STEERING_INVERT_FACTOR
 
 # 3. APLICARE DEADZONE: Dacă este între -0.06 și 0.06, setează la 0.0
 if abs(steering_label) < STEERING_DEADZONE:
   steering_label = 0.0
 
 # 4. Limitare finală (pentru a garanta exact -1.0 sau 1.0 la capete)
 steering_label = max(-1.0, min(1.0, steering_label))
   
 # CORECȚIE: Rotunjire la 2 zecimale pentru a coincide cu afișajul în JSON și pe ecran
 return round(steering_label, 2)

def cleanup_and_save(pca_obj, servo_obj, picam2_obj, driving_log_list):
 
 if driving_log_list is not None and len(driving_log_list) > 0:
  print("\nSalvez log-ul de conducere...")
  try:
   # Scriem tot log-ul, inclusiv intrările vechi și cele noi
   with open(JSON_FILE_PATH, 'w') as f:
    json.dump(driving_log_list, f, indent=4)
   print(f"Log salvat în {JSON_FILE_PATH} cu {len(driving_log_list)} intrări (noi + vechi).")
  except Exception as e:
   print(f"Eroare la salvarea log-ului JSON: {e}")
 
 if picam2_obj is not None and hasattr(picam2_obj, 'started') and picam2_obj.started:
  try:
   picam2_obj.stop()
  except Exception as e:
   print(f"Eroare la oprirea camerei: {e}")
   
 cv2.destroyAllWindows()
 print("\nOprire program (Ctrl+C).")


def initialize_logging_data():
    """
    Verifică fișierul JSON și folderul de cadre.
    Încarcă log-ul existent și determină indexul de la care trebuie să înceapă.
    """
    log = []
    start_frame_index = 0

    # 1. Încărcare log JSON existent
    if os.path.exists(JSON_FILE_PATH) and os.path.getsize(JSON_FILE_PATH) > 0:
        try:
            with open(JSON_FILE_PATH, 'r') as f:
                log = json.load(f)
            print(f"Log existent încărcat: {len(log)} intrări.")
        except json.JSONDecodeError:
            print(f"Atenție: Fișierul JSON este corupt sau gol. Încep un log nou.")
            log = []
        except Exception as e:
            print(f"Eroare la citirea log-ului JSON: {e}. Încep un log nou.")
            log = []
    else:
        print(f"Fișier JSON nou creat la: {JSON_FILE_PATH}")


    # 2. Determinare index de start pentru cadre (cea mai mare valoare)
    os.makedirs(FRAMES_DIR_PATH, exist_ok=True)
    
    # Găsește toate fișierele care seamănă cu 'frame_XXXXX.jpg'
    existing_frames = [f for f in os.listdir(FRAMES_DIR_PATH) if f.startswith("frame_") and f.endswith(".jpg")]
    
    if existing_frames:
        # Extrage numărul (indexul) din numele fișierului și găsește maximul
        indices = []
        for frame_name in existing_frames:
            try:
                # Extrage 'XXXXX' din 'frame_XXXXX.jpg'
                index_str = frame_name.split('_')[1].split('.')[0]
                indices.append(int(index_str))
            except Exception:
                # Ignoră fișierele cu nume incorecte
                continue
        
        if indices:
            # Începe de la indexul imediat următor celui mai mare index găsit
            start_frame_index = max(indices) + 1
            print(f"Folderul de cadre conține imagini. Încep înregistrarea de la indexul: {start_frame_index:05d}.")
    else:
        print(f"Folderul de cadre este gol. Încep înregistrarea de la indexul: 00000.")

    return log, start_frame_index


try:
 
 i2c = busio.I2C(board.SCL, board.SDA) 
 pca = PCA9685(i2c)
 pca.frequency = 50
 my_servo = servo.Servo(pca.channels[SERVO_CHANNEL], min_pulse=600, max_pulse=2700)

 picam2 = Picamera2()
 config = picam2.create_video_configuration(main={"size": (320, 240)}) 
 picam2.configure(config)
 
 devices = [InputDevice(path) for path in list_devices()]
 for dev in devices:
  if 'xbox' in dev.name.lower():
   controller = dev
   break
 if not controller:
  raise ConnectionError("Controller-ul Xbox nu a fost găsit. Oprind procesul...")
  
 controller_fd = controller.fd

 print(f"Armez ESC-ul la {ARM_PULSE}µs. Așteaptă 3 secunde...")
 set_esc_pulse(ARM_PULSE)
 time.sleep(3)
 print(f"ESC armat. Setez la neutru {NEUTRAL}µs.")

 picam2.start()
 set_esc_pulse(NEUTRAL) 
 
 # 🛑 GESTIONARE EROARE la setarea inițială a servo-ului
 try:
     my_servo.angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET
 except OSError as e:
     if e.errno == 121:
         print("⚠️ AVERTISMENT I2C (SERVO): Remote I/O error la setarea unghiului inițial.")
     else:
         raise e
 
 # 🔄 AICI SE INIȚIALIZEAZĂ LOG-UL ȘI INDEXUL DE START
 driving_log, frame_index = initialize_logging_data()
 print(f"Structura de salvare pregătită. Imaginile se duc în: {FRAMES_DIR_PATH}")
 # -----------------------------------------------------------------------------

 is_recording = False
 is_paused = False # Variabilă de stare pentru pauză
 
 print("\n-----------------------------------------------------")
 print(f"Apasă [A] pentru a ÎNCEPE ÎNREGISTRAREA/RESUME.")
 print(f"Apasă [Y] pentru a PAUZA înregistrarea.")
 print(f"Apasă [B] pentru a OPRI și SALVA log-ul.")
 print("-----------------------------------------------------")


 while True:
  
  # Folosim select cu un timeout foarte mic (0.001)
  r, w, x = select.select([controller_fd], [], [], 0.001) 
  
  # 🛑 GESTIONARE EROARE: Citirea unghiului servo (poate eșua, dar nu îl ignorăm complet, ci îl citim înainte de update)
  # current_servo_angle = my_servo.angle # Lăsăm această linie deoparte, deoarece citirea poate eșua, și ne bazăm pe valoarea calculată
  
  # PROCESARE EVENIMENTE CONTROLLER
  for fd in r:
    
    # Citim evenimentele disponibile pentru a evita "blocarea" lor în buffer
    events = controller.read()
    
    for event in events:
     
     if event.type == ecodes.EV_ABS:
      
      try:
       abs_name = ecodes.ABS[event.code]
      except KeyError:
       abs_name = None 

      # CONTROL ESC (GAS/BRAKE) - Utilizează funcția set_esc_pulse cu gestionarea erorilor
      if abs_name == 'ABS_GAS':
       gas_value = event.value
       pulse = calc_pulse(gas_value, brake_value)
       
       if not is_paused: 
        set_esc_pulse(pulse)
       else:
        set_esc_pulse(NEUTRAL) 
        
       if is_recording: print(f"GAS={gas_value} BRAKE={brake_value} -> ESC={pulse} ", end='\r')

      elif abs_name == 'ABS_BRAKE':
       brake_value = event.value
       pulse = calc_pulse(gas_value, brake_value)
       
       if not is_paused: 
        set_esc_pulse(pulse)
       else:
        set_esc_pulse(NEUTRAL) 
        
       if is_recording: print(f"GAS={gas_value} BRAKE={brake_value} -> ESC={pulse} ", end='\r')
       
      # CONTROL SERVO (STEERING)
      elif abs_name == 'ABS_X':
       current_x_value = event.value
       CENTER = 32767
       DEADZONE = 1000

       if abs(current_x_value - CENTER) < DEADZONE:
        current_servo_angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET
       else:
        current_servo_angle = SERVO_MAX_ANGLE - (current_x_value / 65535) * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
        current_servo_angle += SERVO_OFFSET
        current_servo_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, int(current_servo_angle)))

       # Setează unghiul servo doar dacă NU este în pauză
       if not is_paused:
        # 🛑 GESTIONARE EROARE: Încapsulează operațiunea I2C
        try:
         my_servo.angle = current_servo_angle
        except OSError as e:
         if e.errno == 121: 
          print("⚠️ AVERTISMENT I2C (SERVO): Remote I/O error la setarea unghiului.", end='\r')
         else:
          raise e
       else:
        # 🛑 GESTIONARE EROARE: Menține centrul
        try:
         my_servo.angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET 
        except OSError as e:
         if e.errno == 121:
          pass # Ignorăm eroarea și continuăm
         else:
          raise e

     elif event.type == ecodes.EV_KEY and event.value == 1:
      
      # TASTA A (START/RESUME)
      if event.code == BTN_START_RECORDING:
       if is_paused:
        is_paused = False
        print("\n>>> RESUME (A) - Înregistrare continuă. <<<")
       elif not is_recording:
        is_recording = True
        start_time_fps = time.time() 
        frame_count_fps = 0
        print("\n>>> START ÎNREGISTRARE DATĂ! <<<")

      # TASTA Y (PAUZĂ)
      elif event.code == BTN_PAUSE:
       if is_recording and not is_paused:
        is_paused = True
        set_esc_pulse(NEUTRAL) # Oprește mașina
        # Setează servo la centru cu gestionarea erorilor
        try:
         my_servo.angle = SERVO_NEUTRAL_ANGLE + SERVO_OFFSET 
        except OSError as e:
         if e.errno == 121:
          pass # Ignorăm eroarea și continuăm
         else:
          raise e
        print("\n>>> PAUZĂ (Y) - Înregistrare întreruptă. <<<")

      # TASTA B (OPRIRE/SALVARE)
      elif event.code == BTN_SAVE_AND_STOP:
       raise KeyboardInterrupt # Declanșează secvența de salvare
  
  # CAPTURĂ ȘI ÎNREGISTRARE DATE
  # Folosim valoarea calculată a unghiului servo, nu citirea hardware directă.
  current_servo_angle = current_servo_angle if 'current_servo_angle' in locals() else SERVO_NEUTRAL_ANGLE + SERVO_OFFSET
  steering_label = get_steering_label(current_servo_angle) 
  
  # 🛑 GESTIONARE EROARE: Încapsulează operațiunea camerei
  try:
   frame = picam2.capture_array()
  except RuntimeError:
   print("❌ EROARE CAMERĂ: Nu pot captura un cadru. Sărind peste înregistrare...", end='\r')
   time.sleep(0.05)
   continue # Treci la următoarea iterație a buclei

  
  if is_recording and not is_paused: # Înregistrarea se face doar dacă NU e în pauză
   
   frame_count_fps += 1
   time_elapsed = time.time() - start_time_fps
   if time_elapsed > 0:
    current_fps = frame_count_fps / time_elapsed 
   
   # 🔄 UTILIZEAZĂ INDEXUL DE CADRU CONTINUU
   image_filename = f"frame_{frame_index:05d}.jpg"
   image_path_full = os.path.join(FRAMES_DIR_PATH, image_filename)
   cv2.imwrite(image_path_full, cv2.cvtColor(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))

   log_entry = {
    "image_path": f"{os.path.basename(FRAMES_DIR_PATH)}/{image_filename}", 
    "steering_angle": steering_label 
   }
   driving_log.append(log_entry)

   # CORECȚIE: Afișăm în consolă cu formatarea .2f
   print(f"Înregistrez... {len(driving_log)} total, cadru nou: {frame_index:05d} | Viraj: {steering_label:.2f} | FPS: {current_fps:.1f} ", end='\r')
   frame_index += 1 # Incrementează indexul pentru următorul cadru

  # AFIȘARE IMAGINE
  frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  
  # Afișează starea corectă
  status_str = "OFF"
  color = (0, 0, 255) # Roșu (Oprit)
  
  if is_recording and is_paused:
   status_str = "PAUZĂ (Y)"
   color = (255, 255, 0) # Galben (Pauză)
  elif is_recording and not is_paused:
   status_str = "REC (A)"
   color = (0, 255, 0) # Verde (Înregistrare)
  
  # Afișează virajul final (folosim :.2f)
  status_text = f"Stare: {status_str} | Viraj: {steering_label:.2f}"
  
  cv2.putText(frame_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
  
  if current_fps > 0:
   fps_text = f"FPS: {current_fps:.1f} | Total cadre: {len(driving_log)}"
   cv2.putText(frame_display, fps_text, (10, frame_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
   
  cv2.imshow("Data Collector", frame_display)

  if cv2.waitKey(1) & 0xFF == ord('q'):
   raise KeyboardInterrupt 
  
  time.sleep(0.005) # Pauză scurtă pentru a stabiliza I/O și CPU

except (KeyboardInterrupt, ConnectionError) as e:
 cleanup_and_save(pca, my_servo, picam2, driving_log)
 
except Exception as e:
 print(f"\nEroare majoră neașteptată: {e}")
 
finally:
 if pca is not None:
  try:
   set_esc_pulse(0)
   if my_servo is not None:
    # Setează unghiul servo la None (neutralizare) cu gestionarea erorilor
    try:
     my_servo.angle = None
    except OSError as e:
     if e.errno == 121:
      pass
     else:
      raise e
   pca.deinit()
   print("ESC și servomotor oprite, PCA9685 de-initializat.")
  except Exception as e:
   print(f"Eroare la de-inițializarea hardware-ului în final: {e}")