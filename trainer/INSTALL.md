# RoboPacerV2 Trainer — Install & Run

## Overview

There are two ways to train, and you'll likely use both on different runs:

- **Windows, local CPU/GPU** — a web UI (`engine/start.bat`) for the full
  loop: train → compile to `.hef`. Good for quick iteration on small
  datasets.
- **RunPod A100 (cloud)** — a standalone script (`engine/train_a100.sh`)
  for fast training on a full dataset. It only trains — it hands back a
  `.pth` checkpoint + a calibration `.npy`, which you then feed into the
  **Compile-only** action of the Windows web UI. The Hailo DFC compile
  step runs inside a Docker container and only works on Windows/local
  Docker — it never runs on the pod.

So a cloud-trained model always finishes its journey on Windows, in
Compile-only mode, before it's a usable `.hef`.

---

## Part A — Windows setup (train and/or compile)

1. **Install Python** (3.10+) and **Docker Desktop**, and make sure Docker
   Desktop is actually running before you try to compile anything.

2. **Install Python dependencies**:
   ```
   pip install -r trainer/requirements.txt
   ```
   (`engine/start.bat` also does this automatically every time it launches,
   so you can skip this step if you're always launching via the batch
   file.)

3. **Get the Hailo DFC wheel.** Download
   `hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl` from the
   [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account
   required) and place it at:
   ```
   trainer/engine/compile/resources/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
   ```
   This folder is gitignored and empty on a fresh clone — you have to create
   it and drop the wheel in yourself. You only need this if you intend to
   compile on this machine (Full or Compile-only actions). Training-only
   runs don't need it.

4. **Get a dataset folder.** Either record one with
   `data_recorder/data_recorder.py` on the Pi and copy it over (e.g.
   `data_recorder/set1/`, containing `driving_log.json` + `frames/`), or
   point at wherever you already have one. You'll select this folder from
   the web UI later — either the folder itself or the `driving_log.json`
   inside it works.

5. **Launch the trainer:**
   ```
   trainer\engine\start.bat
   ```
   Open `http://localhost:5000`.

### Using the web UI

- **Standard/Custom toggle** (top right) — Standard shows just dataset
  path, model name, epochs, batch size. Custom reveals every advanced
  hyperparameter (learning rate, frame stack size, split ratios, DataLoader
  workers, the A100 throughput toggles, etc.), all prefilled with the same
  defaults the validated recipe uses.
- **Three action tabs:**
  - **Full (Train + Compile)** — trains, then immediately compiles to
    `.hef` via Docker. This is what you want for an end-to-end local run.
    Also has **"Retry compile"** (recompile from an already-trained model
    without retraining — use this if the compile step fails after training
    already finished) and **"Smoke test (with dataset)"** (runs an
    untrained model through the full export + compile pipeline in seconds,
    to confirm your Docker/wheel setup works before committing to a real
    multi-hour run — writes to `models/smoketest.*`, never touches your
    real named models).
  - **Train only** — trains and saves a `.pth` + calibration `.npy`, no
    compile step. This is what a RunPod run produces; use this tab if
    you're training locally but want to compile later/elsewhere.
  - **Compile only** — takes an existing `.pth` (from a previous local
    Train-only run, or copied back from RunPod) and compiles it to `.hef`.
    Also has **"Test compile (no dataset)"**, the no-dataset equivalent of
    the Full tab's smoke test.
- Path fields validate on blur — you'll see a green `✓ 1,532 frames,
  timestamped (3-frame stack)` or a red `✗ error` before you even click
  Start.
- The status bar at the bottom shows Idle/Running/Done, an elapsed timer,
  and the latest epoch's train/val loss. **Stop** cancels a running job
  (it finishes the current epoch, saves a checkpoint if it's the best one
  so far, then stops cleanly rather than leaving things half-written).
- Refreshing the page mid-run reattaches to the running job instead of
  losing your log.

Output lands in `trainer/models/<name>.pth` (+ `.onnx` intermediate, `.hef`
if compiled, `_calib_data_nhwc.npy`). Copy the `.hef` onto the Pi, into
`main/` and/or `model_runner/` — exactly one `.hef` file must be present in
whichever folder you're running from.

---

## Part B — RunPod / A100 setup (cloud training)

1. **Create a pod** with an A100 GPU. Attach or create a **persistent
   network volume mounted at `/workspace`** — this is RunPod's only
   storage that survives a pod stop/terminate, so everything below has to
   live under it.

2. **Clone the repo *inside* `/workspace`:**
   ```
   cd /workspace
   git clone <this repo's URL> RoboPacerV2
   ```
   Cloning anywhere else means your work vanishes the moment the pod
   stops.

3. **Get your dataset onto the pod**, either:
   - upload it manually into a folder under `/workspace`, e.g.
     `/workspace/PistaAtletism/` containing `driving_log.json` + `frames/`, or
   - zip it (`driving_log.json` + `frames/` in one `.zip`), upload the zip to
     Google Drive, share it as "Anyone with the link", and paste that link when
     `train_a100.sh` asks for it in step 5 (or pass `--drive-link "<url>"`) — it downloads and
     unzips it into `/workspace` for you (via `gdown`, installed
     automatically). A single zip is far more reliable than a Drive
     *folder* link when there are thousands of individual frame files.

4. **Start a tmux session** so training survives a dropped SSH connection:
   ```
   tmux new -s train
   ```

5. **Run the training script:**
   ```
   cd /workspace/RoboPacerV2/trainer/engine
   ./train_a100.sh
   ```
   It auto-detects your dataset folder by searching `/workspace` (up to 2
   levels deep) for a `driving_log.json`, skipping `models/` and `.git/`.
   - If it finds exactly one candidate, it uses it automatically.
   - If it finds more than one, it lists them and asks you to re-run with
     `--dataset <FolderName>` to disambiguate.
   - Other flags: `--drive-link URL` (fetch the dataset zip from Google
     Drive first, see step 3), `--name MODEL_NAME` (override the
     auto-derived name, which otherwise comes from the dataset folder's
     basename), `--epochs N`, `--batch-size N`, `--search-root PATH`
     (default `/workspace`).

   This applies the full validated `--a100` recipe (batch 256, RAM frame
   cache, `torch.compile`, BF16 autocast, pinned/non-blocking transfers,
   12 DataLoader workers) — the same throughput toggles exposed
   individually (and off by default) in the Windows Custom UI.

6. **Detach** with `Ctrl+B D` and reattach later with `tmux attach -t
   train` to check progress.

7. **When it finishes**, it prints the two files you need:
   ```
   models/<name>.pth
   models/<name>_calib_data_nhwc.npy
   ```
   Copy both back to your Windows machine (`scp`, RunPod's file browser,
   whatever's convenient).

8. **Finish on Windows:** launch the web UI, go to **Compile-only**, point
   it at the `.pth` you copied back, and at its `_calib_data_nhwc.npy`
   (both required - if the `.npy` sits next to the `.pth`, picking the
   `.pth` fills it in for you).

9. **Tear down the pod** once you've retrieved your files — A100 time
   costs money even when idle.

---

## Troubleshooting

- **"Docker is not running - start Docker Desktop first."** — exactly what
  it says; the compile step checks this before doing anything else so you
  find out in a second, not after hours of training.
- **"Hailo DFC wheel not found at ..."** — you skipped or mis-placed step 3
  of Part A. The path is case- and filename-sensitive:
  `engine/compile/resources/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`.
- **CRLF / line-ending errors from inside the Docker container** —
  shouldn't happen (`.gitattributes` forces LF on `.sh` files, and
  `compile_hef()` also fixes line endings defensively before running), but
  if you ever hand-edit `compile.sh` on Windows and see `bash: $'\r':
  command not found` or similar, that's what happened.
- **"This dataset mixes records with and without a 'timestamp' field..."**
  — your `driving_log.json` combines recordings made with and without
  `data_recorder.py`'s `--legacy` flag. Don't concatenate logs recorded in
  different formats; keep them as separate dataset folders instead.
- **"No dataset folder found under /workspace"** / **"Multiple dataset
  folders found..."** (RunPod) — either upload a dataset before running
  `train_a100.sh`, or pass `--dataset <FolderName>` to disambiguate when
  more than one `driving_log.json` is found.
- **A refreshed browser tab shows a blank log for a job you know is
  running** — shouldn't happen; `/api/status` is polled on load
  specifically to reattach. If it does, check the server console for
  errors — the Flask process may have restarted and lost in-memory job
  state (job progress is not persisted to disk).
