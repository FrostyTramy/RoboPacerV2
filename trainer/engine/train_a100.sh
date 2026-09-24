#!/usr/bin/env bash
# RoboPacerV2 Trainer - standalone RunPod/A100 entrypoint.
#
# Never invoked by the web UI (that's Windows-only). Run this manually on a
# RunPod pod, inside a tmux session so it survives a dropped SSH connection:
#
#   tmux new -s train
#   ./train_a100.sh
#   Ctrl+B D to detach, tmux attach -t train to reattach later.
#
# This repo must be cloned INSIDE /workspace (RunPod's only persistent
# mount) - anything written outside it is lost when the pod stops.
#
# Usage:
#   train_a100.sh [--drive-link URL] [--dataset FOLDER_NAME] [--name MODEL_NAME]
#                 [--epochs N] [--batch-size N] [--search-root PATH]
#
# --drive-link fetches the dataset for you instead of uploading it by hand:
# zip your dataset folder (driving_log.json + frames/) into one .zip, share
# it from Google Drive as "Anyone with the link", and pass that link here.
# Uses gdown, which handles Drive's large-file confirm token; a single zip
# is far more reliable than pointing at a Drive *folder* full of thousands
# of individual frame files (per-file rate limits make that slow/flaky).
set -e

SEARCH_ROOT="/workspace"
DATASET_NAME=""
OVERRIDE_NAME=""
EPOCHS=""
BATCH_SIZE=""
DRIVE_LINK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET_NAME="$2"; shift 2 ;;
        --name) OVERRIDE_NAME="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --search-root) SEARCH_ROOT="$2"; shift 2 ;;
        --drive-link) DRIVE_LINK="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$DRIVE_LINK" ]]; then
    echo "Downloading dataset from Google Drive..."
    pip install -q gdown
    ZIP_PATH="$SEARCH_ROOT/_drive_dataset.zip"
    gdown --fuzzy "$DRIVE_LINK" -O "$ZIP_PATH"
    echo "Extracting..."
    unzip -q -o "$ZIP_PATH" -d "$SEARCH_ROOT"
    rm -f "$ZIP_PATH"
fi

# Candidate dataset folders = every directory under $SEARCH_ROOT (up to 2
# levels deep, to cover one extra nesting level from a zip extraction) that
# contains a driving_log.json, excluding models/ and .git/.
mapfile -t HITS < <(find "$SEARCH_ROOT" -maxdepth 3 -type f -name driving_log.json \
                     -not -path '*/models/*' -not -path '*/.git/*' 2>/dev/null)

CANDIDATES=()
for hit in "${HITS[@]}"; do
    dir=$(dirname "$hit")
    already=false
    for c in "${CANDIDATES[@]}"; do
        [[ "$c" == "$dir" ]] && already=true && break
    done
    $already || CANDIDATES+=("$dir")
done

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "Error: No dataset folder found under $SEARCH_ROOT - upload a folder containing driving_log.json + frames/ first."
    exit 1
elif [[ ${#CANDIDATES[@]} -eq 1 ]]; then
    DATASET_DIR="${CANDIDATES[0]}"
    echo "Found dataset: $DATASET_DIR"
elif [[ -n "$DATASET_NAME" ]]; then
    DATASET_DIR=""
    for c in "${CANDIDATES[@]}"; do
        [[ "$(basename "$c")" == "$DATASET_NAME" ]] && DATASET_DIR="$c" && break
    done
    if [[ -z "$DATASET_DIR" ]]; then
        echo "Error: No dataset folder named '$DATASET_NAME' found. Candidates:"
        for c in "${CANDIDATES[@]}"; do echo "  - $(basename "$c")"; done
        exit 1
    fi
    echo "Using dataset: $DATASET_DIR"
else
    echo "Error: Multiple dataset folders found under $SEARCH_ROOT - re-run with --dataset <FolderName>. Candidates:"
    for c in "${CANDIDATES[@]}"; do echo "  - $(basename "$c")"; done
    exit 1
fi

MODEL_NAME="${OVERRIDE_NAME:-$(basename "$DATASET_DIR")}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Model name: $MODEL_NAME"
echo "Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/../requirements-runpod.txt"

echo "Starting training (--a100 recipe: batch 256, RAM cache, torch.compile, bf16 autocast)..."
python3 "$SCRIPT_DIR/train_core.py" \
    --json "$DATASET_DIR/driving_log.json" \
    --name "$MODEL_NAME" \
    --epochs "${EPOCHS:-20}" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    --pth-only --a100

echo ""
echo "Done. Copy these two files back to Windows:"
echo "  models/${MODEL_NAME}.pth"
echo "  models/${MODEL_NAME}_calib_data_nhwc.npy"
echo "Next step: run Compile-only in the Windows web UI with these files."
