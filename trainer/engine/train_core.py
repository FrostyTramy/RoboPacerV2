"""
RoboPacerV2 Trainer - training engine core (dataset, model, training loop,
ONNX export, calibration data). Deliberately Docker-free: this module runs
unmodified on a headless RunPod pod (via train_a100.sh) as well as on
Windows (via server.py), so it must never import flask, tkinter, or touch
Docker - see compile_pipeline.py for all of that.

Design goal: the .hef eventually compiled from this module's output must
match, pixel-for-pixel, what feeds the model at inference time on the Pi.
There are two Pi-side scripts that run inference -
RoboPacerV2/model_runner/model_runner.py (steering only) and
RoboPacerV2/main/main.py (steering + cruise-control speed) - but main.py's
preprocessing is a verbatim copy of model_runner.py's, so both stay in
lockstep with load_and_preprocess() below automatically as long as that
copy is kept exact (same decode-to-RGB, same cv2.resize/INTER_LINEAR with
no anti-aliasing, same normalize math). If you ever change the
preprocessing on either side, change it in load_and_preprocess() here AND
in both Pi scripts - a silent mismatch here doesn't error, it just quietly
caps the model's real-world accuracy.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models

from defaults import CORE_RECIPE_DEFAULTS, A100_THROUGHPUT_PRESET

IMG_SIZE = 224  # must match main.py's MODEL_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fallback constants for function default args - mirror defaults.py's
# CORE_RECIPE_DEFAULTS so a call site that doesn't go through a full config
# dict (e.g. gen_calib.py, direct unit tests) still gets the validated
# recipe rather than some other arbitrary number.
FRAME_STACK_N_DEFAULT = CORE_RECIPE_DEFAULTS["frame_stack_n"]
FRAME_STACK_GAP_SECONDS_DEFAULT = CORE_RECIPE_DEFAULTS["frame_stack_gap_seconds"]
SPLIT_BLOCK_SECONDS_DEFAULT = CORE_RECIPE_DEFAULTS["split_block_seconds"]
CLASSIC_SPLIT_BLOCK_FRAMES_DEFAULT = CORE_RECIPE_DEFAULTS["classic_split_block_frames"]
CALIB_N_DEFAULT = CORE_RECIPE_DEFAULTS["calib_n"]

ENGINE_DIR = Path(__file__).parent
ROOT_DIR = ENGINE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"


# ── Preprocessing - must match main.py's preprocess() exactly ─────────────

def load_and_preprocess(path, flip=False, brightness=1.0, cache=None):
    """
    data_recorder.py on the Pi saves frames straight from the camera's
    capture_array() with no channel conversion; that array is BGR-ordered
    in memory (picamera2's "RGB888" format is actually BGR - a naming
    quirk confirmed against its source). cv2.imwrite treats input as BGR
    and writes a normal, correctly-colored JPEG. So cv2.imread here gives
    back that same BGR order - flip it to RGB exactly like main.py flips
    its own raw capture array, so both sides land on the same channel
    order before resize/normalize.

    cache (optional): dict of {str(path): decoded+resized RGB uint8 array},
    pre-populated by _preload_frames_to_ram(). When given and it already
    has this path, skips disk I/O and JPEG decode entirely - flip/
    brightness/normalize (cheap, per-sample augmentation) still run fresh
    below since those vary per __getitem__ call and can't be cached.
    """
    key = str(path)
    if cache is not None and key in cache:
        img_rgb = cache[key]
    else:
        img_bgr = cv2.imread(key)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = img_bgr[:, :, ::-1]
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        if cache is not None:
            cache[key] = img_rgb
    if flip:
        img_rgb = img_rgb[:, ::-1, :]
    if brightness != 1.0:
        img_rgb = np.clip(img_rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img  # HWC float32, RGB, normalized


# ── Dataset format detection ─────────────────────────────────────────────

def _detect_format(records):
    """Returns True if this is a timestamped (frame-stackable) dataset,
    False if it's classic (single-frame, no timestamps)."""
    has_ts = [("timestamp" in r) for r in records]
    if all(has_ts):
        return True
    if not any(has_ts):
        return False
    raise ValueError(
        "This dataset mixes records with and without a 'timestamp' field - "
        "can't tell whether it's classic (single-frame) or timestamped "
        "(frame-stacked). Don't combine recordings made with and without "
        "data_recorder.py's --legacy flag in the same driving_log.json."
    )


def resolve_dataset(json_path):
    """
    Loads driving_log.json, filters to records whose image actually exists
    on disk, and detects classic vs timestamped format. Shared by run(),
    compile_pipeline.run_smoke_test(), and gen_calib.py so this logic can't
    drift between them. Raises FileNotFoundError / ValueError on bad input
    - callers decide how to report that (push an event, HTTP 400, etc).
    """
    json_path = Path(json_path)
    if json_path.is_dir():
        json_path = json_path / "driving_log.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path) as f:
        records = json.load(f)
    data_root = json_path.parent
    records = [r for r in records if (data_root / r["image_path"]).exists()]
    if not records:
        raise ValueError(f"No valid images found under {data_root}")
    is_stacked = _detect_format(records)
    return records, data_root, is_stacked


# ── Frame-stack assembly - shared by SteeringDataset and
# save_calibration_data, so the temporal lookback logic can't drift between
# the two ────────────────────────────────────────────────────────────────

def _history_indices(records_sorted, idx, n=FRAME_STACK_N_DEFAULT, gap=FRAME_STACK_GAP_SECONDS_DEFAULT):
    """
    Returns n indices into records_sorted (time-ordered): [idx, idx ~gap
    seconds earlier, idx ~2*gap seconds earlier, ...]. Walks backward frame
    by frame accumulating real elapsed time rather than assuming a fixed
    recording fps. If history runs out, or two adjacent recorded frames are
    implausibly far apart in time (a pause or a new recording session
    appended to the same driving_log.json - not actually one continuous
    drive), stops there and pads by repeating the last valid frame found -
    same fallback main.py uses live when it hasn't captured enough history
    yet (e.g. right after startup).

    n=1 (classic, single-frame datasets) short-circuits before touching
    "timestamp" at all - classic records don't have that field.
    """
    if n == 1:
        return [idx]
    cur_ts = records_sorted[idx]["timestamp"]
    result = [idx]
    for k in range(1, n):
        target_ts = cur_ts - k * gap
        found = result[-1]
        jj = result[-1]
        while jj > 0:
            step = records_sorted[jj]["timestamp"] - records_sorted[jj - 1]["timestamp"]
            if step > gap * 5:  # pause / session boundary
                break
            jj -= 1
            if records_sorted[jj]["timestamp"] <= target_ts:
                found = jj
                break
        result.append(found)
    return result


def _chunk_indices(records_sorted, block_seconds=SPLIT_BLOCK_SECONDS_DEFAULT, gap_seconds=FRAME_STACK_GAP_SECONDS_DEFAULT):
    """
    Groups time-ordered record indices into contiguous chunks spanning
    roughly block_seconds each (also cut at session boundaries, so a chunk
    never silently straddles a pause/new-recording gap). Used to split
    train/val by whole chunks instead of individual frames, so near-
    duplicate frames a few ms apart don't get scattered across both sides.
    """
    chunks = []
    current = []
    chunk_start_ts = None
    for i, r in enumerate(records_sorted):
        ts = r["timestamp"]
        if current and (ts - records_sorted[current[-1]]["timestamp"] > gap_seconds * 5
                         or ts - chunk_start_ts >= block_seconds):
            chunks.append(current)
            current = []
            chunk_start_ts = None
        if chunk_start_ts is None:
            chunk_start_ts = ts
        current.append(i)
    if current:
        chunks.append(current)
    return chunks


def _chunk_indices_by_count(records_sorted, block_frames=CLASSIC_SPLIT_BLOCK_FRAMES_DEFAULT):
    """Classic-format datasets have no timestamps, so unlike
    _chunk_indices above, near-duplicate frames are grouped into
    train/val blocks by a fixed frame count (assuming driving_log.json
    stays in capture order) instead of wall-clock time."""
    return [list(range(i, min(i + block_frames, len(records_sorted))))
            for i in range(0, len(records_sorted), block_frames)]


def _load_stack(records_sorted, idx, data_root, flip=False, brightness=1.0,
                 n=FRAME_STACK_N_DEFAULT, cache=None, gap=FRAME_STACK_GAP_SECONDS_DEFAULT):
    indices = _history_indices(records_sorted, idx, n=n, gap=gap)
    frames = [
        load_and_preprocess(data_root / records_sorted[i]["image_path"], flip=flip, brightness=brightness, cache=cache)
        for i in indices
    ]
    return np.concatenate(frames, axis=-1)  # H, W, 3 * n


# ── RAM frame cache (use_ram_cache) ────────────────────────────────────
# Frame-stacking means each image is read again as "history" by several
# nearby samples, and WeightedRandomSampler re-reads rare-angle frames many
# times per epoch on top of that - on a 151k-image dataset this makes
# training disk-I/O-bound rather than GPU-bound (observed: A100 sitting at
# 0-4% util while CPU wasn't even saturated). Decoding every unique frame
# once into RAM up front removes disk I/O from the hot loop entirely.
# Threaded (not multiprocessed) because cv2.imread/resize are C++ calls
# that release the GIL, so threads still get real parallelism here; and
# because this dict is built in the main process *before* DataLoader
# workers fork, so on Linux (RunPod) copy-on-write means all workers share
# the one in-memory copy instead of duplicating it per worker.
def _preload_frames_to_ram(records, data_root, push, max_workers=16):
    from concurrent.futures import ThreadPoolExecutor

    paths = sorted({r["image_path"] for r in records})
    push({"type": "log", "level": "info",
          "text": f"Preloading {len(paths)} unique frames into RAM..."})

    def _decode(rel_path):
        full = str(data_root / rel_path)
        img_bgr = cv2.imread(full)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {full}")
        img_rgb = img_bgr[:, :, ::-1]
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return full, np.ascontiguousarray(img_rgb)

    cache = {}
    done = 0
    log_step = max(1, len(paths) // 20)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for full, img in ex.map(_decode, paths):
            cache[full] = img
            done += 1
            if done % log_step == 0 or done == len(paths):
                push({"type": "log", "level": "info", "text": f"  preloaded {done}/{len(paths)} frames"})

    mb = sum(a.nbytes for a in cache.values()) / (1024 * 1024)
    push({"type": "log", "level": "success",
          "text": f"Preload done: {len(cache)} frames cached in RAM (~{mb/1024:.1f} GB)."})
    return cache


# ── Dataset ─────────────────────────────────────────────────────────────

class SteeringDataset(Dataset):
    """
    records_sorted must be the FULL time-ordered dataset (not just this
    split) - a sample near the start of the val split still needs to look
    back into frames that may only exist in records_sorted, regardless of
    which split they'd individually belong to. sample_indices selects which
    of those records are actually this split's targets.
    """

    def __init__(self, records_sorted, sample_indices, data_root, augment=False,
                 frame_stack_n=FRAME_STACK_N_DEFAULT, cache=None,
                 flip_prob=CORE_RECIPE_DEFAULTS["flip_prob"],
                 brightness_range=(CORE_RECIPE_DEFAULTS["brightness_min"], CORE_RECIPE_DEFAULTS["brightness_max"]),
                 gap_seconds=FRAME_STACK_GAP_SECONDS_DEFAULT):
        self.records_sorted = records_sorted
        self.sample_indices = sample_indices
        self.data_root = Path(data_root)
        self.augment = augment
        self.frame_stack_n = frame_stack_n
        self.cache = cache
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.gap_seconds = gap_seconds

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        idx = self.sample_indices[i]
        rec = self.records_sorted[idx]
        angle = float(rec["steering_angle"])

        flip = self.augment and random.random() < self.flip_prob
        brightness = random.uniform(*self.brightness_range) if self.augment else 1.0
        if flip:
            angle = -angle

        stack = _load_stack(self.records_sorted, idx, self.data_root, flip=flip, brightness=brightness,
                             n=self.frame_stack_n, cache=self.cache, gap=self.gap_seconds)
        tensor = torch.from_numpy(stack.transpose(2, 0, 1).copy())  # HWC -> CHW
        return tensor, torch.tensor(angle, dtype=torch.float32)


# ── Class-imbalance handling ────────────────────────────────────────────
# Real driving logs are dominated by near-zero steering (most of a drive is
# straight road) - e.g. one recorded set here is 74% exactly angle==0.0 and
# only ~2% sharp turns (|angle|>0.6). Plain MSE weights every frame equally,
# so the loss is minimized almost entirely by getting the abundant straight
# frames right; the rare turn frames barely move the gradient and the model
# converges to predicting near-zero for everything. Fix: oversample rare
# steering magnitudes during training so each bucket contributes roughly
# equally to what the model sees per epoch. Only applied to the train split -
# val keeps the true distribution so val MSE stays a meaningful, comparable
# metric across runs.
_BALANCE_BIN_EDGES = [0.0, 0.001, 0.1, 0.3, 0.6, 1.0 + 1e-6]


def _balanced_sample_weights(records_sorted, sample_indices):
    angles = np.abs([records_sorted[i]["steering_angle"] for i in sample_indices])
    bin_idx = np.digitize(angles, _BALANCE_BIN_EDGES[1:-1])
    counts = np.bincount(bin_idx, minlength=len(_BALANCE_BIN_EDGES) - 1)
    weights = 1.0 / counts[bin_idx]
    return torch.as_tensor(weights, dtype=torch.double)


# ── Model - proven to compile cleanly with the Hailo DFC and match what
# main.py expects (single input, single float output) ─────────────────

def build_model(frame_stack_n=FRAME_STACK_N_DEFAULT):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_ch = 3 * frame_stack_n
    if in_ch != m.conv1.in_channels:
        # Widen the first conv to accept the stacked input. Tile the
        # pretrained 3-channel filters across the extra copies and divide by
        # frame_stack_n, so a stack of similar-looking frames produces a
        # first-conv output at roughly the scale the pretrained BatchNorm
        # right after it was calibrated for - otherwise each extra copy adds
        # fully to the sum and the activation statistics start out badly off
        # from what the rest of the pretrained network expects. Classic
        # (frame_stack_n=1) datasets skip this entirely - in_ch is already 3.
        old_conv1 = m.conv1
        new_conv1 = nn.Conv2d(in_ch, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.repeat(1, frame_stack_n, 1, 1) / frame_stack_n)
        m.conv1 = new_conv1

    m.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    return m


# ── Train / eval ────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device, push, epoch, epochs, use_amp=False, non_blocking=False):
    model.train()
    total, samples = 0.0, 0
    n = len(loader)
    log_step = max(1, n // 10)
    for i, (imgs, angles) in enumerate(loader):
        imgs, angles = imgs.to(device, non_blocking=non_blocking), angles.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(imgs).squeeze(1), angles)
        else:
            loss = nn.functional.mse_loss(model(imgs).squeeze(1), angles)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(imgs)
        samples += len(imgs)
        if (i + 1) % log_step == 0:
            push({"type": "log", "level": "info",
                  "text": f"  epoch {epoch}/{epochs}  batch {i+1}/{n}  loss={total/samples:.5f}"})
    return total / samples


@torch.no_grad()
def _eval_epoch(model, loader, device, use_amp=False, non_blocking=False):
    model.eval()
    total = 0.0
    for imgs, angles in loader:
        imgs, angles = imgs.to(device, non_blocking=non_blocking), angles.to(device, non_blocking=non_blocking)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += nn.functional.mse_loss(model(imgs).squeeze(1), angles).item() * len(imgs)
        else:
            total += nn.functional.mse_loss(model(imgs).squeeze(1), angles).item() * len(imgs)
    return total / len(loader.dataset)


# ── ONNX export ─────────────────────────────────────────────────────────

def export_onnx(model, device, path, frame_stack_n=FRAME_STACK_N_DEFAULT):
    """
    dynamo=False forces the old TorchScript-based exporter, which honors
    opset_version directly. Without it, newer PyTorch (2.5+) defaults to
    the torch.export-based "dynamo" exporter, which only emits opset 18
    and then tries to auto-downgrade to our requested opset 13 - that
    downgrade path has a known bug in onnx's version converter that fails
    on ResNet's Resize/pooling ops ("No initializer or constant input to
    node found"). The Hailo DFC (3.34.0) needs opset 13, so we go through
    the old exporter instead of fighting the downgrade converter.
    """
    model.eval()
    dummy = torch.zeros(1, 3 * frame_stack_n, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["steering"],
        dynamic_axes={"input": {0: "batch"}, "steering": {0: "batch"}},
        dynamo=False,
    )


# ── Calibration data for the Hailo INT8 quantizer - reuses the exact same
# preprocessing (and frame-stack assembly) as training, saved directly in
# NHWC (what the DFC wants), so compile.sh doesn't need a separate
# transpose step ───────────────────────────────────────────────────────

def save_calibration_data(records, data_root, out_path, n=CALIB_N_DEFAULT,
                           frame_stack_n=FRAME_STACK_N_DEFAULT, gap_seconds=FRAME_STACK_GAP_SECONDS_DEFAULT):
    if frame_stack_n > 1:
        records_sorted = sorted(records, key=lambda r: r["timestamp"])
    else:
        records_sorted = records  # no timestamps to sort by - order doesn't matter, single-frame
    sample_idx = random.sample(range(len(records_sorted)), min(n, len(records_sorted)))
    arrays = [_load_stack(records_sorted, i, Path(data_root), n=frame_stack_n, gap=gap_seconds) for i in sample_idx]
    calib = np.stack(arrays).astype(np.float32)  # (N, H, W, 3 * frame_stack_n)
    np.save(str(out_path), calib)


# ── Config resolution - merges defaults.py's CORE_RECIPE_DEFAULTS (and,
# when a100=True, A100_THROUGHPUT_PRESET on top) with whatever explicit
# values the caller passed, so every run() call sees a complete, concrete
# config regardless of how many fields the caller actually specified
# (Standard mode: 4 fields; Custom mode: all of them; train_a100.sh: a
# handful of CLI flags) ─────────────────────────────────────────────────

def _resolve_run_config(config):
    resolved = dict(CORE_RECIPE_DEFAULTS)
    if config.get("a100"):
        resolved.update(A100_THROUGHPUT_PRESET)
    for key in CORE_RECIPE_DEFAULTS:
        if config.get(key) is not None:
            resolved[key] = config[key]
    resolved["json_path"] = config.get("json_path")
    resolved["model_name"] = (config.get("model_name") or "model").strip() or "model"
    resolved["pth_only"] = bool(config.get("pth_only"))
    resolved["a100"] = bool(config.get("a100"))
    return resolved


# ── Main entry point ────────────────────────────────────────────────────

def run(config, push=None, should_stop=None, emit_done=True):
    """
    Trains a model and always stops after saving models/<name>.pth and
    models/<name>_calib_data_nhwc.npy - it never touches ONNX export or
    the Hailo/Docker compile step (see compile_pipeline.compile_from_pth,
    which re-exports ONNX straight from the saved .pth). Callers that want
    the full train+compile pipeline chain run() with compile_from_pth()
    themselves (see server.py's /api/run/full route) - pass emit_done=False
    here so the SSE stream's terminal "done" event comes from the chained
    compile step instead.

    config keys (all optional except json_path - see defaults.py for the
    full list of tunables and their defaults):
      json_path, model_name, epochs, batch_size, learning_rate,
      weight_decay, frame_stack_n, frame_stack_gap_seconds, train_split,
      val_split, split_block_seconds, classic_split_block_frames, calib_n,
      seed, flip_prob, brightness_min, brightness_max, num_workers,
      use_ram_cache, use_torch_compile, use_amp, drop_last, pin_memory,
      pth_only, a100
    a100=True layers defaults.A100_THROUGHPUT_PRESET under any explicit
    overrides above it - this is what train_a100.sh's --a100 flag sets.

    Returns True if a checkpoint was saved, False otherwise - so a caller
    chaining into compile_from_pth() (server.py's /api/run/full route)
    knows whether it's safe to proceed.
    """
    if push is None:
        push = lambda e: print(e.get("text", e))

    def _finish():
        if emit_done:
            push({"type": "done"})

    cfg = _resolve_run_config(config)
    model_name = cfg["model_name"]
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    MODELS_DIR.mkdir(exist_ok=True)

    if not cfg["json_path"]:
        push({"type": "log", "level": "error", "text": "json_path is required."})
        _finish()
        return False

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["use_amp"]) and device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    push({"type": "log", "level": "info", "text": f"Device: {device}"})

    try:
        records, data_root, is_stacked = resolve_dataset(cfg["json_path"])
    except (FileNotFoundError, ValueError) as e:
        push({"type": "log", "level": "error", "text": str(e)})
        _finish()
        return False
    push({"type": "log", "level": "info", "text": f"Valid samples: {len(records)}"})
    if len(records) < 10:
        push({"type": "log", "level": "error", "text": "Not enough valid samples to train."})
        _finish()
        return False

    frame_stack_n = int(cfg["frame_stack_n"]) if is_stacked else 1
    push({"type": "log", "level": "info",
          "text": f"Dataset format: {'timestamped, ' + str(frame_stack_n) + '-frame stack' if is_stacked else 'classic, single-frame'}."})

    gap_seconds = float(cfg["frame_stack_gap_seconds"])
    if is_stacked:
        # records_sorted stays intact (full, time-ordered) so any sample can
        # look back for stack history regardless of which split it landed in
        # - only whole time-blocks get shuffled and split into train/val.
        records_sorted = sorted(records, key=lambda r: r["timestamp"])
        chunks = _chunk_indices(records_sorted, block_seconds=float(cfg["split_block_seconds"]), gap_seconds=gap_seconds)
    else:
        # No timestamps to sort/group by - driving_log.json is already in
        # capture order, so fixed-size index blocks stand in for time blocks.
        records_sorted = records
        chunks = _chunk_indices_by_count(records_sorted, block_frames=int(cfg["classic_split_block_frames"]))
    random.shuffle(chunks)
    total = len(records_sorted)
    train_target = int(float(cfg["train_split"]) * total)
    val_target = int(float(cfg["val_split"]) * total)
    train_idx, val_idx = [], []
    for chunk in chunks:
        if len(train_idx) < train_target:
            train_idx.extend(chunk)
        elif len(val_idx) < val_target:
            val_idx.extend(chunk)
        # else: leftover chunks held out, unused
    push({"type": "split", "train": len(train_idx), "val": len(val_idx), "total": len(records_sorted)})

    cache = _preload_frames_to_ram(records, data_root, push) if cfg["use_ram_cache"] else None

    train_weights = _balanced_sample_weights(records_sorted, train_idx)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_idx), replacement=True)
    num_workers = int(cfg["num_workers"])
    pin_memory = bool(cfg["pin_memory"])
    extra_kwargs = {"prefetch_factor": 4} if num_workers > 0 else {}
    flip_prob = float(cfg["flip_prob"])
    brightness_range = (float(cfg["brightness_min"]), float(cfg["brightness_max"]))
    train_ds = SteeringDataset(records_sorted, train_idx, data_root, augment=True, frame_stack_n=frame_stack_n,
                               cache=cache, flip_prob=flip_prob, brightness_range=brightness_range,
                               gap_seconds=gap_seconds)
    val_ds = SteeringDataset(records_sorted, val_idx, data_root, augment=False, frame_stack_n=frame_stack_n,
                             cache=cache, gap_seconds=gap_seconds)
    train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                          persistent_workers=num_workers > 0, pin_memory=pin_memory,
                          # drop_last: keeps every training batch the same shape, so
                          # torch.compile doesn't have to also handle/recompile for
                          # one odd-sized tail batch per epoch, and BatchNorm never
                          # sees a tiny, noisier tail batch. Only worth it at large
                          # batch sizes (A100 preset) - negligible loss either way.
                          drop_last=bool(cfg["drop_last"]), **extra_kwargs)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        persistent_workers=num_workers > 0, pin_memory=pin_memory, **extra_kwargs)

    model = build_model(frame_stack_n).to(device)
    train_model = model
    if cfg["use_torch_compile"] and device.type == "cuda":
        # Fuses/compiles the training graph for this fixed input shape - a
        # meaningful speedup for a model this small, where kernel-launch
        # overhead otherwise limits how much of the GPU actually gets used.
        # `model` (uncompiled) stays the checkpoint/export target below, so
        # this can't affect the saved .pth's state_dict keys.
        try:
            train_model = torch.compile(model)
            push({"type": "log", "level": "info", "text": "torch.compile enabled for training."})
        except Exception as e:
            push({"type": "log", "level": "warning", "text": f"torch.compile unavailable, continuing without it: {e}"})
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")
    ckpt_path = MODELS_DIR / f"{model_name}.pth"

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(train_model, train_ld, optimizer, device, push, epoch, epochs,
                                   use_amp=use_amp, non_blocking=pin_memory)
        val_loss = _eval_epoch(train_model, val_ld, device, use_amp=use_amp, non_blocking=pin_memory)
        scheduler.step()
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        push({"type": "epoch", "epoch": epoch, "total": epochs,
              "train": round(train_loss, 6), "val": round(val_loss, 6), "best": is_best})
        if should_stop and should_stop():
            push({"type": "log", "level": "warning", "text": f"Stopped at epoch {epoch}."})
            break

    if not ckpt_path.exists():
        push({"type": "log", "level": "error", "text": "Training stopped before any checkpoint was saved."})
        _finish()
        return False

    push({"type": "log", "level": "success", "text": f"Best val MSE: {best_val:.6f}"})
    push({"type": "file", "name": ckpt_path.name})

    calib_path = MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
    push({"type": "log", "level": "info", "text": "Saving calibration data..."})
    save_calibration_data(records, data_root, calib_path, n=int(cfg["calib_n"]), frame_stack_n=frame_stack_n,
                           gap_seconds=gap_seconds)
    push({"type": "file", "name": calib_path.name})

    if cfg["pth_only"]:
        push({"type": "log", "level": "success",
              "text": f"Training complete: models/{ckpt_path.name} + {calib_path.name} saved - "
                      f"compile it separately (Compile-only action, or compile_pipeline.compile_from_pth)."})
    else:
        push({"type": "log", "level": "success",
              "text": f"Training complete: models/{ckpt_path.name} + {calib_path.name} saved."})
    _finish()
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to driving_log.json")
    p.add_argument("--name", default="model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--pth-only", action="store_true",
                   help="Informational only - run() always stops after .pth + calibration data; "
                        "compiling to HEF is a separate step.")
    p.add_argument("--a100", action="store_true",
                   help="Layer defaults.A100_THROUGHPUT_PRESET on top of the core recipe: "
                        "RAM frame cache, BF16 AMP, torch.compile, num_workers=12, batch 256, "
                        "pin_memory, cudnn.benchmark, async H2D transfer.")
    args = p.parse_args()
    run({"json_path": args.json, "model_name": args.name,
         "epochs": args.epochs, "batch_size": args.batch_size,
         "pth_only": args.pth_only, "a100": args.a100})
