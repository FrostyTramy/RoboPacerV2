# RunPod Training — Quick Start

Minimal path: clone the repo on the pod, run one command with your Google
Drive dataset link, get a trained model back. (Compile to `.hef` still
happens on Windows afterward — Docker/Hailo DFC isn't available on the pod.)

## Prerequisite (one-time, on Windows)

`git clone` on the pod only pulls what's already pushed to GitHub, so push
this trainer code first:

```bash
git add trainer && git commit -m "trainer rewrite" && git push
```

Skip this if it's already pushed.

## On the pod

Zip your dataset (`driving_log.json` + `frames/` in one `.zip`), upload it
to Google Drive, share it as **"Anyone with the link"**, then run:

```bash
git clone https://github.com/FrostyTramy/RoboPacerV2 && cd RoboPacerV2/trainer/engine
bash train_a100.sh --drive-link "https://drive.google.com/file/d/XXXXXXXX/view?usp=sharing"
```

That's it. The script installs its own dependencies, downloads + unzips the
dataset from Drive, auto-detects the dataset folder, and trains with the
full validated `--a100` recipe (batch 256, RAM frame cache, `torch.compile`,
BF16 autocast, 12 workers).

Make sure the pod's persistent volume is mounted at `/workspace` and you
`cd` there before cloning — that's RunPod's only storage that survives a
pod stop. If your connection might drop mid-run, wrap it in tmux:

```bash
tmux new -s train
# run the two commands above inside it
# Ctrl+B D to detach, tmux attach -t train to check back later
```

## When it finishes

It prints:

```
models/<name>.pth
models/<name>_calib_data_nhwc.npy
```

Copy both back to Windows (RunPod's file browser, `scp`, whatever's
convenient) into `trainer/models/`, then run `trainer\engine\start.bat`,
open `http://localhost:5000`, and use **Compile-only** with that `.pth` to
get your `.hef`. It auto-picks up the `.npy` sitting next to it — no need
to re-supply the dataset.

Other `train_a100.sh` flags if you need them: `--dataset <FolderName>`
(disambiguate if multiple datasets exist), `--name <ModelName>`,
`--epochs N`, `--batch-size N`.
