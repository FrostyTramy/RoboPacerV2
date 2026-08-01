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

ENGINE_DIR = Path(__file__).parent
ROOT_DIR = ENGINE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"


# ── Preprocessing - must match main.py's preprocess() exactly ─────────────

def load_and_preprocess(path):
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
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img  # HWC float32, RGB, normalized


# ── Dataset ─────────────────────────────────────────────────────────────

class SteeringDataset(Dataset):
    def __init__(self, records, data_root, augment=False):
        self.records = records
        self.data_root = Path(data_root)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        angle = float(rec["steering_angle"])

        img_bgr = cv2.imread(str(self.data_root / rec["image_path"]))
        img_rgb = img_bgr[:, :, ::-1]

        if self.augment:
            if random.random() < 0.5:
                img_rgb = img_rgb[:, ::-1, :]  # horizontal flip
                angle = -angle
            brightness = random.uniform(0.7, 1.3)
            img_rgb = np.clip(img_rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())  # HWC -> CHW
        return tensor, torch.tensor(angle, dtype=torch.float32)


# ── Model - proven to compile cleanly with the Hailo DFC and match what
# main.py expects (single input, single float output) ─────────────────

def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["steering"],
        dynamic_axes={"input": {0: "batch"}, "steering": {0: "batch"}},
    )


# ── Calibration data for the Hailo INT8 quantizer - reuses the exact same
# preprocessing as training, saved directly in NHWC (what the DFC wants),
# so compile.sh doesn't need a separate transpose step ────────────────

def save_calibration_data(records, data_root, out_path, n=CALIB_N):
    sample = random.sample(records, min(n, len(records)))
    arrays = [load_and_preprocess(Path(data_root) / r["image_path"]) for r in sample]
    calib = np.stack(arrays).astype(np.float32)  # (N, H, W, C)
    np.save(str(out_path), calib)


# ── Docker HEF compilation ──────────────────────────────────────────────

def compile_hef(model_name, push):
    push({"type": "log", "level": "info", "text": "Checking Docker..."})
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        push({"type": "log", "level": "error", "text": "Docker is not running - start Docker Desktop first."})
        return False

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

    push({"type": "log", "level": "success", "text": f"HEF ready: models/{model_name}.hef"})
    return True


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
    model_name = config.get("model_name", "model").strip() or "model"
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 32))
    MODELS_DIR.mkdir(exist_ok=True)

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

    random.shuffle(records)
    n_train = int(0.8 * len(records))
    n_val = int(0.1 * len(records))
    train_r = records[:n_train]
    val_r = records[n_train:n_train + n_val]
    push({"type": "split", "train": len(train_r), "val": len(val_r), "total": len(records)})

    train_ld = DataLoader(SteeringDataset(train_r, data_root, augment=True),
                           batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(SteeringDataset(val_r, data_root, augment=False),
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

    calib_path = MODELS_DIR / "calib_data_nhwc.npy"
    push({"type": "log", "level": "info", "text": "Saving calibration data..."})
    save_calibration_data(records, data_root, calib_path)

    ok = compile_hef(model_name, push)
    onnx_path.unlink(missing_ok=True)
    for tmp in [MODELS_DIR / f"{model_name}.har", MODELS_DIR / f"{model_name}_optimized.har"]:
        tmp.unlink(missing_ok=True)
    if ok:
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
