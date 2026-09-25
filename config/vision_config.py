"""
Model input / frame-stacking constants - values taken verbatim from
main/main.py and model_runner/model_runner.py, where preprocess() and
build_frame_stack() must stay pixel-for-pixel identical to the trainer's
load_and_preprocess() (see trainer/engine/train_core.py).
"""
import numpy as np

MODEL_SIZE = 224  # ResNet-18 trained at 224x224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FRAME_STACK_N = 3
FRAME_STACK_GAP_SECONDS = 0.1

SMOOTH_ALPHA = 0.5       # EMA smoothing on the predicted steering label
STEERING_DEADZONE = 0.06  # |smoothed label| below this is treated as 0
