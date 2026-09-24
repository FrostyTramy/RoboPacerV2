"""
RoboPacerV2 Trainer - single source of truth for training hyperparameter
defaults. Both the web UI (Standard/Custom prefill via GET /api/defaults)
and the standalone A100 script (train_core.py's --a100 CLI flag) read from
here, so a number never has to be kept in sync by hand in two places.

CORE_RECIPE_DEFAULTS is the validated recipe that produced the best result
so far (30 epochs, best val MSE 0.008018) - every value here is a real
knob train_core.py actually reads, not a placeholder.

A100_THROUGHPUT_PRESET is layered on top of CORE_RECIPE_DEFAULTS only when
a100=True is set in the run config (train_a100.sh always sets this; the
Windows web UI never bundles these into one button - see trainer/INSTALL.md
and the Custom-mode field list for why they stay individually exposed
there). None of these change what the model learns, only how fast a GPU
gets to the same place - see train_core.py's batch_size comment for the
one exception (batch size is treated as recipe-relevant, not pure
throughput).
"""

CORE_RECIPE_DEFAULTS = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "frame_stack_n": 3,
    "frame_stack_gap_seconds": 0.1,
    "train_split": 0.8,
    "val_split": 0.1,
    "split_block_seconds": 3.0,
    "classic_split_block_frames": 90,
    "calib_n": 200,
    "seed": 42,
    "flip_prob": 0.5,
    "brightness_min": 0.7,
    "brightness_max": 1.3,
    "num_workers": 0,
    "use_ram_cache": False,
    "use_torch_compile": False,
    "use_amp": False,
    "drop_last": False,
    "pin_memory": False,
}

A100_THROUGHPUT_PRESET = {
    "batch_size": 256,
    "num_workers": 12,
    "use_ram_cache": True,
    "use_torch_compile": True,
    "use_amp": True,
    "drop_last": True,
    "pin_memory": True,
}
