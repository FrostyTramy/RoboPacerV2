"""
Frame preprocessing + stacking for main/main.py. Inference has to see
exactly what the model was trained on, so everything here must stay
pixel-for-pixel equivalent to the trainer's load_and_preprocess() (see
trainer/engine/train_core.py).

Two paths produce the Hailo's uint8 input:
  - preprocess() + quantize_input(): the reference, written like the
    trainer (float32 normalize, then quantize with the HEF's scale/zp).
  - preprocess_quantized(): what main.py runs every frame. Normalize +
    quantize is a fixed function of each 8-bit pixel value per channel, so
    it is precomputed once into a 256-entry table (make_quant_lut) built
    FROM the reference functions - same bytes out, no float math per frame
    (~0.8 ms instead of ~6 ms). main.py checks the two agree on its first
    frame and refuses to run if they don't.
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


def quantize_input(img_float_nhwc, scale, zero_point):
    return np.clip(np.round(img_float_nhwc / scale + zero_point), 0, 255).astype(np.uint8)


def make_quant_lut(scale, zero_point):
    """256x1x3 uint8 table for cv2.LUT, indexed in the camera's BGR channel
    order: entry [v, 0, c] = the quantized value preprocess() +
    quantize_input() give pixel value v in channel c. Built with those same
    two functions, so the float math can't drift apart."""
    values = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2)  # RGB order
    normalized = (values.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    lut_rgb = quantize_input(normalized, scale, zero_point)
    return np.ascontiguousarray(lut_rgb[:, :, ::-1])


def preprocess_quantized(frame_bgr, lut_bgr):
    """Camera frame -> the Hailo's uint8 RGB input (MODEL_SIZE x MODEL_SIZE
    x 3, contiguous). Byte-identical to quantize_input(preprocess(frame)):
    resize and a per-channel table don't depend on channel order, so they
    run on the camera's contiguous BGR array (resizing the reversed view,
    as preprocess() does, forces a full-frame copy first) and the swap to
    RGB comes last."""
    small = cv2.resize(frame_bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.LUT(small, lut_bgr), cv2.COLOR_BGR2RGB)


def select_stack_frames(history, now, frame_stack_n=FRAME_STACK_N):
    """The frames of a stack, newest first: history[-1], then for each k the
    newest one captured at or before now - k * FRAME_STACK_GAP_SECONDS
    (the oldest available if none is that old yet). history is a
    time-ordered sequence of (timestamp, frame)."""
    frames = [history[-1][1]]
    for k in range(1, frame_stack_n):
        target_ts = now - k * FRAME_STACK_GAP_SECONDS
        best = history[0][1]
        for ts, img in reversed(history):
            if ts <= target_ts:
                best = img
                break
        frames.append(best)
    return frames


def build_frame_stack(history, now, frame_stack_n=FRAME_STACK_N):
    return np.concatenate(select_stack_frames(history, now, frame_stack_n), axis=-1)
