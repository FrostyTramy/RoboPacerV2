# RoboPacerV2 Trainer

Runs on Windows (or any machine with a GPU/CPU + Docker Desktop) - trains
the steering model on a dataset collected by `data_recorder/data_recorder.py`
on the Pi, then compiles it to a `.hef` for the Hailo-8 via Docker.
`model_runner/model_runner.py` (steering-only) or `main/main.py` (steering +
cruise-control speed) on the Pi run the resulting `.hef` - both run the
exact same inference/preprocessing code, just drop the `.hef` next to
whichever one you're using.

For setup and how to run it (Windows web UI, RunPod/A100 cloud training,
troubleshooting), see **[INSTALL.md](INSTALL.md)**. The rest of this file
covers the *why* behind the training approach - dataset format detection,
class imbalance handling, and preprocessing parity - which doesn't change
regardless of where or how you run it.

## Dataset format: classic vs timestamped (frame-stacked)

`data_recorder.py` can record two formats:

- **Default** - each frame also gets a `timestamp`. This lets the trainer
  build a *frame-stacked* input: the current frame plus the 2 preceding
  ones (~0.1s apart), concatenated as extra channels, so the model has
  some short-term memory ("I'm already correcting left" vs "I've always
  gone straight") instead of reacting to each frame in isolation.
- **`data_recorder.py --legacy`** - classic format, just `image_path` +
  `steering_angle`, no timestamp. Trains a plain single-frame model - no
  temporal memory, but a smaller/simpler network and no dependency on
  recording fps being reasonably steady.

You don't need to tell the trainer or the Pi-side scripts which one you
used - they all detect it automatically:

- `engine/train_core.py` checks whether the dataset's records have a
  `timestamp` field and picks single-frame vs frame-stacked training
  accordingly (adjusting the model's input channels, export shape, and
  calibration data to match). A dataset that mixes both formats (e.g.
  from concatenating two recording sessions made with different flags)
  fails fast with a clear error instead of training on ambiguous data.
- `model_runner/model_runner.py` and `main/main.py` both read the
  compiled `.hef`'s own input shape at startup and infer the same thing
  from its channel count (3 = classic, 9 = the default 3-frame stack) -
  so whichever `.hef` you drop next to either script, it drives the car
  correctly either way.

If a `driving_log.json` already exists, `data_recorder.py` keeps
recording in whatever format is already in that file (ignoring
`--legacy` if it doesn't match) rather than mixing formats in one
dataset.

## Class imbalance (steering distribution)

Real driving logs are dominated by near-zero steering - most of a drive
is straight road. On one recorded set here, 74% of frames had
`steering_angle == 0.0` exactly, and only ~2% were sharp turns
(`|angle| > 0.6`). Plain MSE loss weighs every frame equally, so with a
distribution like that the loss is minimized almost entirely by getting
the abundant straight frames right - the rare turn frames barely move
the gradient, and the model converges to predicting near-zero for
almost everything (visibly: tiny, hesitant steering that won't commit
to a real turn).

`engine/train_core.py` counters this with a `WeightedRandomSampler` on the
training split only (validation keeps the true distribution, so val MSE
stays a meaningful, comparable metric across runs): samples are bucketed
by `|steering_angle|` and reweighted so each bucket contributes roughly
equally per epoch, regardless of how rare it actually is in the raw
dataset. This can't invent recovery/turning examples that were never
recorded - it just stops the ones that *do* exist from being drowned out
by the straight-driving majority.

## Why the preprocessing looks the way it does

`engine/train_core.py`'s `load_and_preprocess()` / `SteeringDataset` intentionally
use `cv2.resize(..., INTER_LINEAR)` and manual normalization instead of
`torchvision.transforms`, because that's exactly what the Pi-side scripts
(`model_runner/model_runner.py`, `main/main.py`) do at inference time. If
these ever drift apart, the model sees
different pixel values during training than during real driving - a bug
that doesn't throw an error, it just quietly caps accuracy. If you change
one side, change the other.
