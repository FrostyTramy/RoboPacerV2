# RunPod Training — Quick Start

## Step 0 — Upload your dataset to Google Drive

Zip your dataset (`driving_log.json` + `frames/` in one `.zip`), upload it
to Google Drive, share it as **"Anyone with the link"** (public), and copy
the link.

## Step 1 — Clone + train (on the pod)

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

## Step 2 — Pull the `.pth` off the pod with WinSCP

1. Open **WinSCP** → New Session.
2. File protocol **SFTP**, host/port/username + password (or private key)
   from RunPod's **Connect** panel.
3. Login, browse to `/workspace/RoboPacerV2/trainer/models/` on the remote
   (right) side.
4. Drag `<name>.pth` and `<name>_calib_data_nhwc.npy` into `trainer/models/`
   on the local (left) side.

## Step 3 — Compile

Run `trainer\engine\start.bat`, open `http://localhost:5000` → **Compile-only**,
point it at the `.pth`. It auto-picks up the `.npy` sitting next to it. Out
comes your `.hef`.

---

Other `train_a100.sh` flags if needed: `--dataset <FolderName>`
(disambiguate if multiple datasets exist), `--name <ModelName>`,
`--epochs N`, `--batch-size N`.
