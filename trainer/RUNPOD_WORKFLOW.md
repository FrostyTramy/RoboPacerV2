# RunPod Training — Quick Start

## Step 0 — Upload your dataset to Google Drive

Zip your dataset (`driving_log.json` + `frames/` in one `.zip`), upload it
to Google Drive, share it as **"Anyone with the link"** (public), and copy
the link.

## Step 1 — Configure the pod

- **GPU:** 1x A100 (40GB is enough for this recipe's batch size; the 80GB
  variant works too, it just costs more for no real benefit here).
- **Template:** any RunPod PyTorch/CUDA template — comes with `torch`
  preinstalled, so `train_a100.sh` doesn't have to redownload it.
- **Container Disk:** ~20GB (OS + CUDA + Python deps).
- **Volume Disk:** 50GB+, mounted at `/workspace`. This must be a
  **persistent** network volume — it's the only thing that survives a pod
  stop/terminate, and the cloned repo, dataset, and model outputs all live
  here. Bump it up if your dataset zip is bigger than a few GB.

## Step 2 — Clone + train (on the pod)

Start tmux first so training survives a dropped connection:

```bash
tmux new -s train
```

Then just 2 commands:

```bash
git clone -b clean-trainer https://github.com/FrostyTramy/RoboPacerV2 /workspace/RoboPacerV2 && cd /workspace/RoboPacerV2/trainer/engine
bash train_a100.sh --drive-link "https://drive.google.com/file/d/XXXXXXXX/view?usp=sharing"
```

This clones just the trainer branch into `/workspace` (RunPod's persistent
volume), downloads + unzips your dataset from the Drive link automatically,
and trains with the full validated `--a100` recipe.

`Ctrl+B D` to detach, `tmux attach -t train` to check back in. When it's
done, it prints:

```
/workspace/RoboPacerV2/trainer/models/<name>.pth
/workspace/RoboPacerV2/trainer/models/<name>_calib_data_nhwc.npy
```

## Step 3 — Pull the `.pth` off the pod with WinSCP

Pods use key auth only — no password.

1. Open **WinSCP** → New Site.
2. File protocol **SFTP**. Enter the host/IP and port shown in RunPod's
   **Connect** panel. User name: `root`. Leave password blank.
3. **Advanced → SSH → Authentication → Private key file** → browse to the
   key you already made (e.g. `C:\Users\<you>\.ssh\id_ed25519`). If WinSCP
   offers to convert it to `.ppk`, accept.
4. Login (accept the host key prompt on first connect). Browse to
   `/workspace/RoboPacerV2/trainer/models/` on the remote (right) side.
5. Drag `<name>.pth` and `<name>_calib_data_nhwc.npy` into `trainer/models/`
   on the local (left) side.

## Step 4 — Compile

Run `trainer\engine\start.bat`, open `http://localhost:5000` → **Compile-only**,
pick the `.pth` and its `_calib_data_nhwc.npy` (both required - picking the
`.pth` fills in the `.npy` when it sits next to it). Out comes your `.hef`.

---

Other `train_a100.sh` flags if needed: `--dataset <FolderName>`
(disambiguate if multiple datasets exist), `--name <ModelName>`,
`--epochs N`, `--batch-size N`.
