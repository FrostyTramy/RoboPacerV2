"""
RoboPacerV2 Trainer - training + export module.
Runs on Windows (or any machine with a GPU/CPU + Docker). Called by
server.py, or standalone via `python train.py --json ... --name mycar`.

Design goal: the .hef this produces must match, pixel-for-pixel, what
RoboPacerV2/main/main.py feeds the model at inference time on the Pi. See
load_and_preprocess() below - it must stay in lockstep with main.py's
preprocess() function (same decode-to-RGB, same cv2.resize/INTER_LINEAR
with no anti-aliasing, same normalize math). If you ever change one, change
the other - a silent mismatch here doesn't error, it just quietly caps the
model's real-world accuracy.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

IMG_SIZE = 224  # must match main.py's MODEL_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CALIB_N = 200

# ── Frame stacking (temporal memory) ───────────────────────────────────────
# The model sees the current frame plus FRAME_STACK_N - 1 frames from its
# recent past, concatenated as extra input channels, so it can tell "I'm
# already correcting left" apart from "I've always been going straight" -
# info a single memoryless frame can't carry. Consecutive camera captures
# (~10ms apart at 100fps) look nearly identical, so we don't stack raw
# consecutive frames - we pick frames roughly FRAME_STACK_GAP_SECONDS apart
# in wall-clock time. Both constants must match main.py's copies exactly:
# main.py builds the same kind of stack live, from a rolling buffer, and the
# HEF's input shape (3 * FRAME_STACK_N channels) is baked in at compile time.
FRAME_STACK_N = 3
FRAME_STACK_GAP_SECONDS = 0.1

ENGINE_DIR = Path(__file__).parent
ROOT_DIR = ENGINE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"


# ── Preprocessing - must match main.py's preprocess() exactly ─────────────

def load_and_preprocess(path, flip=False, brightness=1.0):
    """
    data_recorder.py on the Pi saves frames straight from the camera's
    capture_array() with no channel conversion; that array is BGR-ordered
    in memory (picamera2's "RGB888" format is actually BGR - a naming
    quirk confirmed against its source). cv2.imwrite treats input as BGR
    and writes a normal, correctly-colored JPEG. So cv2.imread here gives
    back that same BGR order - flip it to RGB exactly like main.py flips
    its own raw capture array, so both sides land on the same channel
    order before resize/normalize.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = img_bgr[:, :, ::-1]
    if flip:
        img_rgb = img_rgb[:, ::-1, :]
    if brightness != 1.0:
        img_rgb = np.clip(img_rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img  # HWC float32, RGB, normalized


# ── Frame-stack assembly - shared by SteeringDataset and
# save_calibration_data, so the temporal lookback logic can't drift between
# the two ────────────────────────────────────────────────────────────────

def _require_timestamps(records):
    if any("timestamp" not in r for r in records):
        raise ValueError(
            "This dataset has records without a 'timestamp' field - frame "
            "stacking needs it to know which frames are close in time. "
            "Re-record with the current data_recorder.py (it now stamps "
            "every frame), or use an older trainer build for this dataset."
        )


def _history_indices(records_sorted, idx, n=FRAME_STACK_N, gap=FRAME_STACK_GAP_SECONDS):
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
    """
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


def _load_stack(records_sorted, idx, data_root, flip=False, brightness=1.0):
    indices = _history_indices(records_sorted, idx)
    frames = [
        load_and_preprocess(data_root / records_sorted[i]["image_path"], flip=flip, brightness=brightness)
        for i in indices
    ]
    return np.concatenate(frames, axis=-1)  # H, W, 3 * FRAME_STACK_N


# ── Dataset ─────────────────────────────────────────────────────────────

class SteeringDataset(Dataset):
    """
    records_sorted must be the FULL time-ordered dataset (not just this
    split) - a sample near the start of the val split still needs to look
    back into frames that may only exist in records_sorted, regardless of
    which split they'd individually belong to. sample_indices selects which
    of those records are actually this split's targets.
    """

    def __init__(self, records_sorted, sample_indices, data_root, augment=False):
        self.records_sorted = records_sorted
        self.sample_indices = sample_indices
        self.data_root = Path(data_root)
        self.augment = augment

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        idx = self.sample_indices[i]
        rec = self.records_sorted[idx]
        angle = float(rec["steering_angle"])

        flip = self.augment and random.random() < 0.5
        brightness = random.uniform(0.7, 1.3) if self.augment else 1.0
        if flip:
            angle = -angle

        stack = _load_stack(self.records_sorted, idx, self.data_root, flip=flip, brightness=brightness)
        tensor = torch.from_numpy(stack.transpose(2, 0, 1).copy())  # HWC -> CHW
        return tensor, torch.tensor(angle, dtype=torch.float32)


# ── Model - proven to compile cleanly with the Hailo DFC and match what
# main.py expects (single input, single float output) ─────────────────

def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_ch = 3 * FRAME_STACK_N
    if in_ch != m.conv1.in_channels:
        # Widen the first conv to accept the stacked input. Tile the
        # pretrained 3-channel filters across the extra copies and divide by
        # FRAME_STACK_N, so a stack of similar-looking frames produces a
        # first-conv output at roughly the scale the pretrained BatchNorm
        # right after it was calibrated for - otherwise each extra copy adds
        # fully to the sum and the activation statistics start out badly off
        # from what the rest of the pretrained network expects.
        old_conv1 = m.conv1
        new_conv1 = nn.Conv2d(in_ch, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.repeat(1, FRAME_STACK_N, 1, 1) / FRAME_STACK_N)
        m.conv1 = new_conv1

    m.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    return m


# ── Train / eval ────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device, push, epoch, epochs):
    model.train()
    total, samples = 0.0, 0
    n = len(loader)
    log_step = max(1, n // 10)
    for i, (imgs, angles) in enumerate(loader):
        imgs, angles = imgs.to(device), angles.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(imgs).squeeze(1), angles)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(imgs)
        samples += len(imgs)
        if (i + 1) % log_step == 0:
            push({"type": "log", "level": "info",
                  "text": f"  epoch {epoch}/{epochs}  batch {i+1}/{n}  loss={total/samples:.5f}"})
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for imgs, angles in loader:
        imgs, angles = imgs.to(device), angles.to(device)
        total += nn.functional.mse_loss(model(imgs).squeeze(1), angles).item() * len(imgs)
    return total / len(loader.dataset)


# ── ONNX export ─────────────────────────────────────────────────────────

def export_onnx(model, device, path):
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
    dummy = torch.zeros(1, 3 * FRAME_STACK_N, IMG_SIZE, IMG_SIZE, device=device)
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

def save_calibration_data(records, data_root, out_path, n=CALIB_N):
    _require_timestamps(records)
    records_sorted = sorted(records, key=lambda r: r["timestamp"])
    sample_idx = random.sample(range(len(records_sorted)), min(n, len(records_sorted)))
    arrays = [_load_stack(records_sorted, i, Path(data_root)) for i in sample_idx]
    calib = np.stack(arrays).astype(np.float32)  # (N, H, W, 3 * FRAME_STACK_N)
    np.save(str(out_path), calib)


# ── Docker HEF compilation ──────────────────────────────────────────────

DFC_WHEEL_PATH = ENGINE_DIR / "compile" / "resources" / "hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"


def check_compile_prereqs(push):
    """Checked BEFORE training starts, not after - a missing wheel or a
    stopped Docker Desktop should fail in a second, not after a training
    run that can take hours."""
    ok = True
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        push({"type": "log", "level": "error", "text": "Docker is not running - start Docker Desktop first."})
        ok = False
    if not DFC_WHEEL_PATH.exists():
        push({"type": "log", "level": "error",
              "text": f"Hailo DFC wheel not found at {DFC_WHEEL_PATH}. "
                      f"Download it from the Hailo Developer Zone and place it there (see README.md)."})
        ok = False
    return ok


def _ensure_lf_line_endings(path):
    """compile.sh runs as bash inside the Linux container - CRLF line
    endings (which a Windows checkout can introduce despite
    .gitattributes, e.g. if the file was checked out before that rule
    existed) break it in confusing ways (stray \\r breaks `set -e`, every
    argument, etc.). Fix it in place here rather than depending on git/
    checkout behavior on whatever machine this runs on."""
    raw = path.read_bytes()
    fixed = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if fixed != raw:
        path.write_bytes(fixed)


def compile_hef(model_name, push):
    _ensure_lf_line_endings(ENGINE_DIR / "compile" / "compile.sh")

    push({"type": "log", "level": "info", "text": "Building hailo-dfc image (cached after first run)..."})
    build = subprocess.Popen(
        ["docker", "build", "--progress=plain", "-t", "hailo-dfc",
         "-f", str(ENGINE_DIR / "compile" / "Dockerfile"), str(ENGINE_DIR)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    for line in build.stdout:
        if line.rstrip():
            push({"type": "log", "level": "docker", "text": line.rstrip()})
    build.wait()
    if build.returncode != 0:
        push({"type": "log", "level": "error", "text": "Docker build failed."})
        return False

    push({"type": "log", "level": "info", "text": f"Compiling {model_name} to HEF..."})
    run = subprocess.Popen(
        ["docker", "run", "--rm", "-v", f"{ROOT_DIR}:/workspace",
         "hailo-dfc", "bash", "/workspace/engine/compile/compile.sh", model_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    for line in run.stdout:
        if line.rstrip():
            push({"type": "log", "level": "docker", "text": line.rstrip()})
    run.wait()
    if run.returncode != 0:
        push({"type": "log", "level": "error", "text": "DFC compilation failed."})
        return False

    hef_path = MODELS_DIR / f"{model_name}.hef"
    if not hef_path.exists():
        # compile.sh exiting 0 doesn't guarantee it worked - `set -e` only
        # stops the script if the shell actually parses it correctly (e.g.
        # CRLF line endings from a Windows checkout can silently break
        # that, letting every step after a failure keep running and still
        # print "SUCCESS"). Trust the actual file, not the exit code.
        push({"type": "log", "level": "error",
              "text": f"compile.sh exited 0 but {hef_path} was never created - "
                      f"check the docker log above for the real error (often a CRLF "
                      f"line-ending issue in compile.sh on Windows checkouts)."})
        return False

    push({"type": "log", "level": "success", "text": f"HEF ready: models/{model_name}.hef"})
    return True


# ── Smoke test - exercises the full export+compile pipeline (ONNX export,
# Docker, Hailo DFC) in seconds instead of hours, with an untrained model.
# Doesn't touch or overwrite any of your real named models. ────────────

SMOKE_TEST_NAME = "smoketest"


def run_smoke_test(config, push=None):
    if push is None:
        push = lambda e: print(e.get("text", e))

    json_path = Path(config["json_path"])
    if json_path.is_dir():
        json_path = json_path / "driving_log.json"
    if not json_path.exists():
        push({"type": "log", "level": "error", "text": f"Not found: {json_path}"})
        push({"type": "done"})
        return

    if not check_compile_prereqs(push):
        push({"type": "done"})
        return

    with open(json_path) as f:
        records = json.load(f)
    data_root = json_path.parent
    records = [r for r in records if (data_root / r["image_path"]).exists()]
    if not records:
        push({"type": "log", "level": "error", "text": "No valid images found in the dataset - need at least 1."})
        push({"type": "done"})
        return
    push({"type": "log", "level": "info", "text": f"Using {min(5, len(records))} of {len(records)} images for calibration (untrained model - this only tests the pipeline, not accuracy)."})

    MODELS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)  # random/pretrained-backbone weights, 0 epochs of fine-tuning
    model.eval()

    ckpt_path = MODELS_DIR / f"{SMOKE_TEST_NAME}.pth"
    torch.save(model.state_dict(), ckpt_path)

    onnx_path = MODELS_DIR / f"{SMOKE_TEST_NAME}.onnx"
    push({"type": "log", "level": "info", "text": "Exporting ONNX..."})
    export_onnx(model, device, onnx_path)
    push({"type": "log", "level": "success", "text": "ONNX export OK."})

    calib_path = MODELS_DIR / f"{SMOKE_TEST_NAME}_calib_data_nhwc.npy"
    save_calibration_data(records, data_root, calib_path, n=5)

    ok = compile_hef(SMOKE_TEST_NAME, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [MODELS_DIR / f"{SMOKE_TEST_NAME}.har", MODELS_DIR / f"{SMOKE_TEST_NAME}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "log", "level": "success",
              "text": f"Pipeline OK end-to-end - models/{SMOKE_TEST_NAME}.hef compiled successfully. "
                      f"(It's an untrained model - don't put it on the Pi, this was just to test the pipeline.)"})
        push({"type": "file", "name": f"{SMOKE_TEST_NAME}.hef"})
    push({"type": "done"})


# ── Main entry point ────────────────────────────────────────────────────

def run(config, push=None, should_stop=None):
    """
    config keys:
      json_path  : path to driving_log.json (frames resolved relative to its folder)
      model_name : output name (models/<name>.pth, .onnx, .hef)
      epochs     : int (default 20)
      batch_size : int (default 32)
    """
    if push is None:
        push = lambda e: print(e.get("text", e))

    json_path = Path(config["json_path"])
    if json_path.is_dir():
        json_path = json_path / "driving_log.json"
    model_name = config.get("model_name", "model").strip() or "model"
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 32))
    MODELS_DIR.mkdir(exist_ok=True)

    if not json_path.exists():
        push({"type": "log", "level": "error", "text": f"Not found: {json_path}"})
        return

    if not check_compile_prereqs(push):
        push({"type": "log", "level": "error",
              "text": "Fix the above before starting - training can take hours and "
                      "the HEF compile step needs these to be in place."})
        return

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    push({"type": "log", "level": "info", "text": f"Device: {device}"})

    with open(json_path) as f:
        records = json.load(f)
    data_root = json_path.parent
    records = [r for r in records if (data_root / r["image_path"]).exists()]
    push({"type": "log", "level": "info", "text": f"Valid samples: {len(records)}"})
    if len(records) < 10:
        push({"type": "log", "level": "error", "text": "Not enough valid samples to train."})
        return

    _require_timestamps(records)
    # records_sorted stays intact (full, time-ordered) so any sample can look
    # back for stack history regardless of which split it landed in - only
    # the shuffled *indices* get split into train/val.
    records_sorted = sorted(records, key=lambda r: r["timestamp"])
    all_idx = list(range(len(records_sorted)))
    random.shuffle(all_idx)
    n_train = int(0.8 * len(all_idx))
    n_val = int(0.1 * len(all_idx))
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    push({"type": "split", "train": len(train_idx), "val": len(val_idx), "total": len(records_sorted)})

    train_ld = DataLoader(SteeringDataset(records_sorted, train_idx, data_root, augment=True),
                           batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(SteeringDataset(records_sorted, val_idx, data_root, augment=False),
                         batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")
    ckpt_path = MODELS_DIR / f"{model_name}.pth"

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_ld, optimizer, device, push, epoch, epochs)
        val_loss = _eval_epoch(model, val_ld, device)
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

    push({"type": "log", "level": "success", "text": f"Best val MSE: {best_val:.6f}"})
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    push({"type": "file", "name": f"{model_name}.pth"})

    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    push({"type": "log", "level": "info", "text": "Exporting ONNX..."})
    export_onnx(model, device, onnx_path)

    calib_path = MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
    push({"type": "log", "level": "info", "text": "Saving calibration data..."})
    save_calibration_data(records, data_root, calib_path)

    ok = compile_hef(model_name, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [MODELS_DIR / f"{model_name}.har", MODELS_DIR / f"{model_name}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "file", "name": f"{model_name}.hef"})
    else:
        push({"type": "log", "level": "warning",
              "text": f"Training succeeded (models/{model_name}.pth, .onnx, {calib_path.name} all saved) - "
                      f"fix the compile issue above, then use 'Retry compile' with the same model name "
                      f"to finish without retraining."})
    push({"type": "done"})


def retry_compile(config, push=None):
    """Recompiles an already-trained model to HEF, reusing the .onnx and
    calibration data saved by a previous run() call - skips training
    entirely. Use this after fixing a Docker/DFC problem so a failed
    compile doesn't mean redoing hours of training."""
    if push is None:
        push = lambda e: print(e.get("text", e))

    model_name = config.get("model_name", "model").strip() or "model"
    json_path_str = (config.get("json_path") or "").strip()
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    calib_path = MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
    ckpt_path = MODELS_DIR / f"{model_name}.pth"

    if not onnx_path.exists():
        if not ckpt_path.exists():
            push({"type": "log", "level": "error",
                  "text": f"Neither {onnx_path} nor {ckpt_path} exist - nothing to compile, run training first."})
            push({"type": "done"})
            return
        push({"type": "log", "level": "warning", "text": f"{onnx_path} missing - re-exporting from {ckpt_path.name}..."})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        export_onnx(model, device, onnx_path)
        push({"type": "log", "level": "success", "text": "Re-exported ONNX."})

    if not calib_path.exists():
        if not json_path_str:
            push({"type": "log", "level": "error",
                  "text": f"{calib_path} missing and no dataset path given - set the dataset "
                          f"folder field and retry to regenerate calibration data."})
            push({"type": "done"})
            return
        json_path = Path(json_path_str)
        if json_path.is_dir():
            json_path = json_path / "driving_log.json"
        if not json_path.exists():
            push({"type": "log", "level": "error", "text": f"Not found: {json_path}"})
            push({"type": "done"})
            return
        push({"type": "log", "level": "warning", "text": "Calibration data missing - regenerating..."})
        with open(json_path) as f:
            records = json.load(f)
        data_root = json_path.parent
        records = [r for r in records if (data_root / r["image_path"]).exists()]
        save_calibration_data(records, data_root, calib_path)
        push({"type": "log", "level": "success", "text": "Regenerated calibration data."})

    if not check_compile_prereqs(push):
        push({"type": "done"})
        return

    ok = compile_hef(model_name, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [MODELS_DIR / f"{model_name}.har", MODELS_DIR / f"{model_name}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "file", "name": f"{model_name}.hef"})
    push({"type": "done"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to driving_log.json")
    p.add_argument("--name", default="model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    run({"json_path": args.json, "model_name": args.name,
         "epochs": args.epochs, "batch_size": args.batch_size})
