"""
PCA9685 I2C init boilerplate - byte-identical across main/main.py,
manual_drive/manual_drive.py, cruise_control/cruise_control.py,
data_recorder/data_recorder.py, and model_runner/model_runner.py.
"""
import board
import busio
from adafruit_pca9685 import PCA9685

from config.hardware_config import PCA_FREQUENCY_HZ


def init_pca9685():
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = PCA_FREQUENCY_HZ
    return pca
