"""
One-off helper: generate real Hailo calibration data from a local dataset,
for use with the "Compile from PTH" flow (compile_from_pth in train.py) -
avoids it falling back to synthetic random calibration data, which gives
worse INT8 quantization accuracy than real driving frames.

Usage:
    python gen_calib.py <path to driving_log.json or its folder> <model_name>

Writes models/<model_name>_calib_data_nhwc.npy - matching the exact filename
compile_from_pth() already looks for, so if you use the same model_name in
the "Compile from PTH" UI, it picks this file up automatically (no need to
fill in the optional calib .npy field).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import _detect_format, save_calibration_data, MODELS_DIR, FRAME_STACK_N

if len(sys.argv) < 3:
    print("Usage: python gen_calib.py <driving_log.json or dataset folder> <model_name>")
    sys.exit(1)

json_path = Path(sys.argv[1])
if json_path.is_dir():
    json_path = json_path / "driving_log.json"
model_name = sys.argv[2]

with open(json_path) as f:
    records = json.load(f)
data_root = json_path.parent
records = [r for r in records if (data_root / r["image_path"]).exists()]
if not records:
    print(f"No valid images found under {data_root}")
    sys.exit(1)

is_stacked = _detect_format(records)
frame_stack_n = FRAME_STACK_N if is_stacked else 1

MODELS_DIR.mkdir(exist_ok=True)
out_path = MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
save_calibration_data(records, data_root, out_path, frame_stack_n=frame_stack_n)
print(f"Saved calibration data ({len(records)} candidate frames, frame_stack_n={frame_stack_n}): {out_path}")
