# RoboPacerV2 Trainer

Runs on Windows (or any machine with a GPU/CPU + Docker Desktop) - trains
the steering model on a dataset collected by `data_recorder/data_recorder.py`
on the Pi, then compiles it to a `.hef` for the Hailo-8 via Docker.
`main/main.py` on the Pi runs the resulting `.hef`.

## Setup

1. `pip install -r requirements.txt` (or just run `start.bat`, it does this for you)
2. Get `hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl` from the
   [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account
   required) and place it at `engine/compile/resources/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`
3. Install Docker Desktop, make sure it's running
4. Copy the dataset folder from the Pi (`data_recorder/set1/`, containing
   `driving_log.json` + `frames/`) onto this machine

## Run

```
start.bat
```

Open `http://localhost:5000`, point it at your dataset folder (e.g.
`set1`, or the `driving_log.json` inside it directly - both work), set a
model name / epochs / batch size, click Start. Training runs natively
(GPU if available); the final HEF compile step runs inside the
`hailo-dfc` Docker container. Output lands in `models/<name>.hef` (plus
`.pth` checkpoint).

Copy the `.hef` onto the Pi, into `main/`, next to `main.py` (exactly one
`.hef` file must be in that folder).

Training checks Docker + the DFC wheel are in place *before* starting -
not after - so a missing wheel or a stopped Docker Desktop fails in a
second instead of after a multi-hour training run.

## Before a real (multi-hour) training run

Click **"Test export (0 epochs)"** first - it runs an untrained model
through the exact same ONNX export + Docker + Hailo DFC compile pipeline,
in seconds. If your Docker/wheel/DFC setup is broken, this is where
you'll find out - not after hours of training. It writes to
`models/smoketest.*` and never touches your real named models.

## If the HEF compile step fails after training already finished

Training already saved `models/<name>.pth` and `.onnx` - you do not need
to retrain. Fix whatever Docker/DFC issue caused the failure, then click
**"Retry compile (no retrain)"** with the same model name - it recompiles
straight from the existing `.onnx` and calibration data.

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

You don't need to tell the trainer or `main.py` which one you used -
they both detect it automatically:

- `engine/train.py` checks whether the dataset's records have a
  `timestamp` field and picks single-frame vs frame-stacked training
  accordingly (adjusting the model's input channels, export shape, and
  calibration data to match). A dataset that mixes both formats (e.g.
  from concatenating two recording sessions made with different flags)
  fails fast with a clear error instead of training on ambiguous data.
- `main/main.py` reads the compiled `.hef`'s own input shape at startup
  and infers the same thing from its channel count (3 = classic, 9 = the
  default 3-frame stack) - so whichever `.hef` you drop into `main/`,
  next to `main.py`, it drives the car correctly either way.

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

`engine/train.py` counters this with a `WeightedRandomSampler` on the
training split only (validation keeps the true distribution, so val MSE
stays a meaningful, comparable metric across runs): samples are bucketed
by `|steering_angle|` and reweighted so each bucket contributes roughly
equally per epoch, regardless of how rare it actually is in the raw
dataset. This can't invent recovery/turning examples that were never
recorded - it just stops the ones that *do* exist from being drowned out
by the straight-driving majority.

## Why the preprocessing looks the way it does

`engine/train.py`'s `load_and_preprocess()` / `SteeringDataset` intentionally
use `cv2.resize(..., INTER_LINEAR)` and manual normalization instead of
`torchvision.transforms`, because that's exactly what `main/main.py` does
on the Pi at inference time. If these ever drift apart, the model sees
different pixel values during training than during real driving - a bug
that doesn't throw an error, it just quietly caps accuracy. If you change
one side, change the other.
