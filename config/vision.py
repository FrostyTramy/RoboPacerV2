"""
Frame preprocessing + stacking - byte-identical between main/main.py and
model_runner/model_runner.py. preprocess() must stay pixel-for-pixel
identical to the trainer's load_and_preprocess() (see
trainer/engine/train_core.py) - inference has to see exactly what the model
was trained on.
"""
import cv2
import numpy as np

from config.vision_config import FRAME_STACK_GAP_SECONDS, FRAME_STACK_N, IMAGENET_MEAN, IMAGENET_STD, MODEL_SIZE


def preprocess(frame_bgr):
    frame_rgb = frame_bgr[:, :, ::-1]
    img = cv2.resize(frame_rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def build_frame_stack(history, now, frame_stack_n=FRAME_STACK_N):
    frames = [history[-1][1]]
    for k in range(1, frame_stack_n):
        target_ts = now - k * FRAME_STACK_GAP_SECONDS
        best = history[0][1]
        for ts, img in reversed(history):
            if ts <= target_ts:
                best = img
                break
        frames.append(best)
    return np.concatenate(frames, axis=-1)


def quantize_input(img_float_nhwc, scale, zero_point):
    return np.clip(np.round(img_float_nhwc / scale + zero_point), 0, 255).astype(np.uint8)
