# RoboPacerV2 — Camera Specification

## Hardware

| Item | Value |
|---|---|
| Sensor | IMX219 (Raspberry Pi Camera Module 2, NoIR variant) |
| Tuning file | `imx219_noir.json` (`data_recorder.py`) / `imx219.json` (`camera.py`, `tools/camera_calibrate.py` — non-NoIR tuning is being trialed there only, see Notes) |
| Field of view | 62.2° horizontal × 48.8° vertical (~72.4° diagonal) |
| Inference accelerator | Hailo AI Kit, 26 TOPS — tested at up to 400 fps for the trained model |

## Capture configuration

| Setting | Value | Used in |
|---|---|---|
| Resolution | 640×480 | `camera/camera.py`, `data_recorder/data_recorder.py` |
| Pixel format | RGB888 | same |
| Frame rate | 120 fps | same |
| Analogue gain | 16.0 (`camera.py`) / 12.0 (`data_recorder.py`, reduced for outdoor daylight) | — |

Resolution, format, and frame rate are set identically in both scripts via `create_video_configuration(main={"size": (640, 480), "format": "RGB888"}, controls={...})`. Tuning file and analogue gain currently differ between the two (see table) — reconcile once the `camera.py` color calibration work below is finalized.

## Training dataset frame size

| Setting | Value |
|---|---|
| Saved frame size | 640×480 (same as capture — no resize; model is trained and run at this resolution) |
| Saved format | JPEG, saved directly from the captured frame (no color conversion) |
| Label | Steering angle only, range −1.0 to +1.0, logged in `driving_log.json` |

## Physical mount & calibration

| Setting | Value |
|---|---|
| **Calibrated look-ahead distance** | **4.00 m** |
| Calibration method | Wall + measuring tape placed 4m from the camera; tilt adjusted downward until the wall exits the top edge of the frame |
| Mount height | TBD — fill in once measured |
| Near-field blind zone (closest visible ground) | TBD — fill in once measured |

### Why 4m

Operating speed range is 10–30 km/h (2.78–8.33 m/s). Total system latency (camera capture + Hailo inference + I2C write to the PCA9685) is on the order of 15–25ms — negligible next to the servo's own mechanical response time (~90–150ms for a full steering throw). Combined system latency ≈ 150–200ms.

- Minimum "not stale" distance at 30 km/h: 8.33 m/s × 0.2s ≈ 1.7m
- Target look-ahead for smooth (non-reactive) curve tracking: ~0.5–0.7s at 30 km/h ≈ 4–6m

4m sits at the low end of that smooth-tracking range — enough margin over the reflex floor for stable control, while keeping the frame tight on the track surface (avoiding horizon/background clutter) and preserving a small near-field blind spot for tight lateral tracking at speed.

## Notes

- Hailo inference throughput (400 fps tested) is well above what the pipeline needs — the camera is capped at 120 fps and the servo's mechanical slew rate is the actual bottleneck. Faster inference will not meaningfully change the required look-ahead distance; a faster-slewing servo would.
- If the mount height or near-field blind distance change (different chassis, camera bracket, etc.), recheck the 4m calibration with the wall test described above.
- The NoIR sensor produces a purple/pink cast outdoors in daylight (vegetation reflects strongly in near-IR, and NoIR has no filter to block it before the sensor — confirmed even blacks were tinted, which auto-white-balance calibration cannot correct since it's an additive IR contamination, not a gain mismatch). Trialing a swap to the standard (IR-cut) IMX219 tuning in `camera.py` and `tools/camera_calibrate.py` only, to work out the right settings before touching `data_recorder.py` (still on NoIR tuning). If night operation with IR illumination is needed later, that requires a second NoIR module or a switchable IR-cut filter.
