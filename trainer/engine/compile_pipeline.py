"""
RoboPacerV2 Trainer - Hailo DFC compile pipeline (Docker orchestration).
Runs on Windows only (needs Docker Desktop) - never imported by
train_a100.sh, which only needs train_core.py. Everything here that touches
a .pth/.onnx/model math (build_model, export_onnx, save_calibration_data,
resolve_dataset) is re-used from train_core, not reimplemented, so training
and compiling never drift out of sync on what a checkpoint actually is.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import train_core

DFC_WHEEL_PATH = train_core.ENGINE_DIR / "compile" / "resources" / "hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"
SMOKE_TEST_NAME = "smoketest"


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
                      f"Download it from the Hailo Developer Zone and place it there (see INSTALL.md)."})
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
    _ensure_lf_line_endings(train_core.ENGINE_DIR / "compile" / "compile.sh")

    push({"type": "log", "level": "info", "text": "Building hailo-dfc image (cached after first run)..."})
    build = subprocess.Popen(
        ["docker", "build", "--progress=plain", "-t", "hailo-dfc",
         "-f", str(train_core.ENGINE_DIR / "compile" / "Dockerfile"), str(train_core.ENGINE_DIR)],
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
        ["docker", "run", "--rm", "-v", f"{train_core.ROOT_DIR}:/workspace",
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

    hef_path = train_core.MODELS_DIR / f"{model_name}.hef"
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


def retry_compile(config, push=None):
    """Recompiles an already-trained model to HEF, reusing the .onnx and
    calibration data saved by a previous run() call - skips training
    entirely. Use this after fixing a Docker/DFC problem so a failed
    compile doesn't mean redoing hours of training."""
    if push is None:
        push = lambda e: print(e.get("text", e))

    model_name = (config.get("model_name") or "model").strip() or "model"
    json_path_str = (config.get("json_path") or "").strip()
    onnx_path = train_core.MODELS_DIR / f"{model_name}.onnx"
    calib_path = train_core.MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"
    ckpt_path = train_core.MODELS_DIR / f"{model_name}.pth"

    # frame_stack_n isn't stored anywhere separately - the checkpoint's own
    # conv1 weight shape is the authoritative record of what it was actually
    # trained/exported with (classic vs timestamped dataset), so read it back
    # from there instead of re-deriving it and risking it drifting out of
    # sync with what's actually in the .pth.
    frame_stack_n = None
    if ckpt_path.exists():
        frame_stack_n = torch.load(ckpt_path, map_location="cpu")["conv1.weight"].shape[1] // 3

    if not onnx_path.exists():
        if not ckpt_path.exists():
            push({"type": "log", "level": "error",
                  "text": f"Neither {onnx_path} nor {ckpt_path} exist - nothing to compile, run training first."})
            push({"type": "done"})
            return
        push({"type": "log", "level": "warning", "text": f"{onnx_path} missing - re-exporting from {ckpt_path.name}..."})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train_core.build_model(frame_stack_n).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        train_core.export_onnx(model, device, onnx_path, frame_stack_n)
        push({"type": "log", "level": "success", "text": "Re-exported ONNX."})

    if not calib_path.exists():
        if not json_path_str:
            push({"type": "log", "level": "error",
                  "text": f"{calib_path} missing and no dataset path given - set the dataset "
                          f"folder field and retry to regenerate calibration data."})
            push({"type": "done"})
            return
        push({"type": "log", "level": "warning", "text": "Calibration data missing - regenerating..."})
        try:
            records, data_root, is_stacked = train_core.resolve_dataset(json_path_str)
        except (FileNotFoundError, ValueError) as e:
            push({"type": "log", "level": "error", "text": str(e)})
            push({"type": "done"})
            return
        if frame_stack_n is None:  # no checkpoint to read it from - fall back to the dataset itself
            frame_stack_n = train_core.FRAME_STACK_N_DEFAULT if is_stacked else 1
        train_core.save_calibration_data(records, data_root, calib_path, frame_stack_n=frame_stack_n)
        push({"type": "log", "level": "success", "text": "Regenerated calibration data."})

    if not check_compile_prereqs(push):
        push({"type": "done"})
        return

    ok = compile_hef(model_name, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [train_core.MODELS_DIR / f"{model_name}.har", train_core.MODELS_DIR / f"{model_name}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "file", "name": f"{model_name}.hef"})
    push({"type": "done"})


def compile_from_pth(config, push=None):
    """Loads an external .pth, exports it to ONNX, and compiles it to HEF.

    config keys: pth_path, model_name, json_path (dataset folder - used to
    generate real calibration data if none exists yet), calib_npy (optional
    explicit calibration .npy path, takes priority over json_path),
    allow_synthetic_calib (opt-in bool, only consulted if neither of the
    above yields calibration data - see the priority order below)."""
    if push is None:
        push = lambda e: print(e.get("text", e))

    pth_path = Path((config.get("pth_path") or "").strip())
    model_name = (config.get("model_name") or "model").strip() or "model"
    train_core.MODELS_DIR.mkdir(exist_ok=True)
    onnx_path = train_core.MODELS_DIR / f"{model_name}.onnx"
    calib_path = train_core.MODELS_DIR / f"{model_name}_calib_data_nhwc.npy"

    if not pth_path.exists():
        push({"type": "log", "level": "error", "text": f"PTH file not found: {pth_path}"})
        push({"type": "done"})
        return

    push({"type": "log", "level": "info", "text": f"Loading checkpoint: {pth_path.name}"})
    state = torch.load(str(pth_path), map_location="cpu")
    frame_stack_n = state["conv1.weight"].shape[1] // 3
    push({"type": "log", "level": "info", "text": f"Detected frame_stack_n: {frame_stack_n}"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_core.build_model(frame_stack_n).to(device)
    model.load_state_dict(state)
    model.eval()
    push({"type": "log", "level": "info", "text": "Model loaded. Exporting ONNX..."})
    train_core.export_onnx(model, device, onnx_path, frame_stack_n)
    push({"type": "log", "level": "success", "text": f"ONNX exported: models/{model_name}.onnx"})

    # Calibration data priority: (1) already exists for this model name,
    # (2) an explicitly supplied .npy, (3) real data generated from a
    # supplied dataset folder, (4) synthetic data, but only if the caller
    # explicitly opts in - this is a deliberately weaker fallback, not the
    # silent default it used to be.
    if not calib_path.exists():
        calib_npy_str = (config.get("calib_npy") or "").strip()
        json_path_str = (config.get("json_path") or "").strip()
        if calib_npy_str and Path(calib_npy_str).exists():
            shutil.copy(calib_npy_str, calib_path)
            push({"type": "log", "level": "info", "text": "Calibration data copied."})
        elif json_path_str:
            try:
                records, data_root, _ = train_core.resolve_dataset(json_path_str)
            except (FileNotFoundError, ValueError) as e:
                push({"type": "log", "level": "error", "text": str(e)})
                push({"type": "done"})
                return
            push({"type": "log", "level": "info", "text": "Generating calibration data from dataset..."})
            train_core.save_calibration_data(records, data_root, calib_path, frame_stack_n=frame_stack_n)
            push({"type": "log", "level": "success", "text": "Calibration data generated from real frames."})
        elif config.get("allow_synthetic_calib"):
            push({"type": "log", "level": "warning",
                  "text": "No dataset or calibration file given - using synthetic calibration data "
                          "(lower quantization accuracy), as explicitly requested."})
            calib = np.random.randn(train_core.CALIB_N_DEFAULT, train_core.IMG_SIZE, train_core.IMG_SIZE,
                                     3 * frame_stack_n).astype(np.float32)
            np.save(str(calib_path), calib)
        else:
            push({"type": "log", "level": "error",
                  "text": "No calibration data available - provide a dataset folder, a calibration "
                          ".npy file, or explicitly opt into synthetic calibration data."})
            push({"type": "done"})
            return

    if not check_compile_prereqs(push):
        push({"type": "done"})
        return

    ok = compile_hef(model_name, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [train_core.MODELS_DIR / f"{model_name}.har", train_core.MODELS_DIR / f"{model_name}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "file", "name": f"{model_name}.hef"})
    push({"type": "done"})


def run_smoke_test(config, push=None):
    """0-epoch run through the full pipeline (ONNX export + Docker + Hailo
    compile) with an untrained model - takes seconds, not hours, to catch
    export/Docker/DFC problems. Never touches your real named models.
    config keys: json_path."""
    if push is None:
        push = lambda e: print(e.get("text", e))

    json_path_str = config.get("json_path")
    if not json_path_str:
        push({"type": "log", "level": "error", "text": "json_path is required."})
        push({"type": "done"})
        return

    if not check_compile_prereqs(push):
        push({"type": "done"})
        return

    try:
        records, data_root, is_stacked = train_core.resolve_dataset(json_path_str)
    except (FileNotFoundError, ValueError) as e:
        push({"type": "log", "level": "error", "text": str(e)})
        push({"type": "done"})
        return

    frame_stack_n = train_core.FRAME_STACK_N_DEFAULT if is_stacked else 1
    push({"type": "log", "level": "info",
          "text": f"Dataset format: {'timestamped, ' + str(frame_stack_n) + '-frame stack' if is_stacked else 'classic, single-frame'}."})
    push({"type": "log", "level": "info",
          "text": f"Using {min(5, len(records))} of {len(records)} images for calibration "
                  f"(untrained model - this only tests the pipeline, not accuracy)."})

    train_core.MODELS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_core.build_model(frame_stack_n).to(device)  # untrained - 0 epochs of fine-tuning
    model.eval()

    ckpt_path = train_core.MODELS_DIR / f"{SMOKE_TEST_NAME}.pth"
    torch.save(model.state_dict(), ckpt_path)

    onnx_path = train_core.MODELS_DIR / f"{SMOKE_TEST_NAME}.onnx"
    push({"type": "log", "level": "info", "text": "Exporting ONNX..."})
    train_core.export_onnx(model, device, onnx_path, frame_stack_n)
    push({"type": "log", "level": "success", "text": "ONNX export OK."})

    calib_path = train_core.MODELS_DIR / f"{SMOKE_TEST_NAME}_calib_data_nhwc.npy"
    train_core.save_calibration_data(records, data_root, calib_path, n=5, frame_stack_n=frame_stack_n)

    ok = compile_hef(SMOKE_TEST_NAME, push)
    if ok:
        onnx_path.unlink(missing_ok=True)
        for tmp in [train_core.MODELS_DIR / f"{SMOKE_TEST_NAME}.har", train_core.MODELS_DIR / f"{SMOKE_TEST_NAME}_optimized.har"]:
            tmp.unlink(missing_ok=True)
        push({"type": "log", "level": "success",
              "text": f"Pipeline OK end-to-end - models/{SMOKE_TEST_NAME}.hef compiled successfully. "
                      f"(It's an untrained model - don't put it on the Pi, this was just to test the pipeline.)"})
        push({"type": "file", "name": f"{SMOKE_TEST_NAME}.hef"})
    push({"type": "done"})


def run_smoke_test_no_dataset(config=None, push=None):
    """Test-compiles an untrained model using synthetic calibration data -
    no dataset needed at all. Backs the Compile-only panel's smoke test
    (today's smoketest_pth route) - moved here out of server.py so the
    Docker/Hailo logic lives in exactly one place."""
    if push is None:
        push = lambda e: print(e.get("text", e))

    train_core.MODELS_DIR.mkdir(exist_ok=True)
    name = "smoketest_pth"
    push({"type": "log", "level": "info", "text": "Generating untrained test model (no training)..."})
    device = torch.device("cpu")
    frame_stack_n = train_core.FRAME_STACK_N_DEFAULT
    model = train_core.build_model(frame_stack_n).to(device)
    onnx_path = train_core.MODELS_DIR / f"{name}.onnx"
    calib_path = train_core.MODELS_DIR / f"{name}_calib_data_nhwc.npy"
    train_core.export_onnx(model, device, onnx_path, frame_stack_n)
    calib = np.random.randn(train_core.CALIB_N_DEFAULT, train_core.IMG_SIZE, train_core.IMG_SIZE,
                             3 * frame_stack_n).astype(np.float32)
    np.save(str(calib_path), calib)
    push({"type": "log", "level": "info", "text": "ONNX exported. Starting Docker compile..."})
    if not check_compile_prereqs(push):
        push({"type": "done"})
        return
    ok = compile_hef(name, push)
    for f in [onnx_path, calib_path,
              train_core.MODELS_DIR / f"{name}.har",
              train_core.MODELS_DIR / f"{name}_optimized.har",
              train_core.MODELS_DIR / f"{name}.hef"]:
        Path(f).unlink(missing_ok=True)
    if ok:
        push({"type": "log", "level": "success", "text": "Test compile succeeded!"})
    push({"type": "done"})
