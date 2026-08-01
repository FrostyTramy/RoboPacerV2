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

## Why the preprocessing looks the way it does

`engine/train.py`'s `load_and_preprocess()` / `SteeringDataset` intentionally
use `cv2.resize(..., INTER_LINEAR)` and manual normalization instead of
`torchvision.transforms`, because that's exactly what `main/main.py` does
on the Pi at inference time. If these ever drift apart, the model sees
different pixel values during training than during real driving - a bug
that doesn't throw an error, it just quietly caps accuracy. If you change
one side, change the other.
