"""
One-off helper: generate real Hailo calibration data from a local dataset,
for the "Compile-only" flow (compile_from_pth in compile_pipeline.py) when
the .npy that training saved next to the .pth is lost.

Usage:
    python gen_calib.py <path to driving_log.json or its folder> <model_name>

Writes models/<model_name>_calib_data_nhwc.npy - pick that file in the
Compile-only panel's calibration field (picking models/<model_name>.pth
fills it in automatically).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_core import resolve_dataset, save_calibration_data, MODELS_DIR, FRAME_STACK_N_DEFAULT

if len(sys.argv) < 3:
    print("Usage: python gen_calib.py <driving_log.json or dataset folder> <model_name>")
    sys.exit(1)

json_path = sys.argv[1]
model_name = sys.argv[2]

try:
    records, data_root, is_stacked = resolve_dataset(json_path)
except (FileNotFoundError, ValueError) as e:
    print(str(e))
    sys.exit(1)

frame_stack_n = FRAME_STACK_N_DEFAULT if is_stacked else 1

MODELS_DIR.mkdir(exist_ok=True)
out_path = MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
save_calibration_data(records, data_root, out_path, frame_stack_n=frame_stack_n)
print(f"Saved calibration data ({len(records)} candidate frames, frame_stack_n={frame_stack_n}): {out_path}")
