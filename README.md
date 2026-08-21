# vehicle_classification

Vehicle detection, tracking, and size classification (SV / MV / HV) on traffic
video. A vehicle is detected per frame, tracked across frames with a
persistent ID, and — the first time it crosses a calibrated lane line by
more than a configurable margin — classified as a **Small (SV)**, **Medium
(MV)**, or **Heavy (HV)** vehicle based on its on-screen size relative to
that line's length. Counts and an annotated video are the end product.

This repo is a structured rewrite of a set of one-off scripts
(`vehicle_detection_template_v2_3.py` and friends, still sitting one level
up in `../`) into tested, reusable modules — same core logic, minus the
bugs and dead code that came from everything living in a single file.

## How it works

```
frame ─▶ VehicleDetector ─▶ VehicleTracker ─▶ VehicleClassifier ─▶ annotator ─▶ output video
         (YOLO)             (boxmot ByteTrack)  (lane-crossing +      (draws boxes,
                                                  size vs. line)        labels, counts)
```

- **Detection** — YOLO (Ultralytics) restricted to COCO's car/bus/truck
  classes, with a minimum-bbox-area filter to drop noise.
- **Tracking** — [boxmot](https://github.com/mikel-brostrom/boxmot)'s
  ByteTrack gives each vehicle a persistent ID across frames (BotSort/
  StrongSort are wired in as config options but untested — no ReID
  weights shipped here yet).
- **Classification** — each calibrated lane line doubles as (a) a
  crossing trigger and (b) a size reference: a vehicle wider than 80% of
  the line's length is MV/HV depending on whether it's also that tall;
  otherwise SV. Classified once per track, first crossing wins.
- **Annotation** — draws the calibration lines, each tracked box with its
  label, and a running SV/MV/HV tally onto the frame.

## Repo structure

```
vehicle_classification/
├── src/
│   ├── pipeline/
│   │   └── data_types.py        # Shared dataclasses: BoundingBox, TrackedObject, LaneLine
│   ├── detection/
│   │   └── vehicle_detector.py  # VehicleDetector — YOLO wrapper
│   ├── tracking/
│   │   └── tracker.py           # VehicleTracker — boxmot wrapper (ByteTrack/BotSort/StrongSort)
│   ├── classification/
│   │   └── vehicle_classifier.py # VehicleClassifier — lane-crossing + size classification
│   └── utils/
│       └── annotator.py         # Drawing functions: lines, boxes+labels, running count
├── tests/
│   ├── unit/                    # Fast, no model/video/GPU needed (mocked dependencies)
│   └── integration/             # Real model + real video frames — marked `integration`,
│                                 # skips itself if road_traffic.mp4 / yolov8n.pt aren't present
├── models/detection/             # Model weights go here (gitignored — see Setup)
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Test tooling (pytest)
└── pyproject.toml                # pytest config
```

```
main.py                           # CLI entry point — see Running inference below
```

Every module under `src/` was ported deliberately, not blindly copied — see
each file's docstring for what changed vs. the original scripts and why
(e.g. `vehicle_classifier.py` fixes a width/height mix-up in the original's
size-classification check).

## Setup

Requires **Python 3.11** and, for the pinned `torch`/`torchvision` build,
an NVIDIA GPU with CUDA 12.4 (`requirements.txt` pins `+cu124` wheels —
swap those two lines for CPU-only or a different CUDA version if your
hardware differs).

```bash
# 1. Create and activate the environment (any tool works — conda shown here)
conda create -n vehicle_cls python=3.11
conda activate vehicle_cls

# 2. Install runtime + test dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

If you're on a machine without a pre-existing `pip`/`ensurepip` in a fresh
conda env, bootstrap it first: `python -m ensurepip --upgrade`.

**Model weights**: place a YOLO checkpoint at `models/detection/yolov8n.pt`
(the default `VehicleDetector` path — override via the `model_path`
constructor argument to use a different one, e.g. `yolov8x.pt` for higher
accuracy at more compute cost). Weights are gitignored; you supply your own.

**Video for integration tests**: the integration suite expects
`road_traffic.mp4` one directory above the repo root (i.e. `../road_traffic.mp4`
relative to this README). Without it, those tests skip themselves rather
than fail — see below.

## Running the tests

```bash
# Everything
pytest

# Unit tests only — fast, no model weights or video required
pytest -m "not integration"

# Integration tests only — real model + real video frames
pytest -m integration
```

Integration tests skip themselves (not fail) if `models/detection/yolov8n.pt`
or `../road_traffic.mp4` aren't present.

## Running inference

```bash
python main.py --video road_traffic.mp4 --model models/detection/yolov8n.pt
```

1. The first frame opens in a **"Calibrate lanes"** window. Click-drag-release
   to draw one line across each lane you want classified against, then press
   `c` to continue. If you draw 4 or more lines, the 4th one drawn is treated
   as the curved/merge lane (wider crossing tolerance) — draw that one last.
2. Processing then runs frame-by-frame: a **"Processed Frames"** preview
   window shows live detection/tracking/classification (press `q` to stop
   early), while every frame — including ones after an early quit — already
   written is saved to `--output`.
3. On completion, the final SV/MV/HV counts print to the console.

All CLI options (`python main.py --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--video` | `../road_traffic.mp4` | Input video path |
| `--model` | `models/detection/yolov8n.pt` | YOLO weights path |
| `--tracker` | `ByteTrack` | `ByteTrack` \| `BotSort` \| `StrongSort` (the latter two need ReID weights this repo doesn't ship) |
| `--output` | `vehicle_track.mp4` | Where to write the annotated video |
| `--intersection-threshold` | `50.0` | Minimum % of a calibration line a bbox must cross to trigger classification |
| `--no-display` | off | Skip the live preview windows (still writes `--output`) — use for headless/batch runs |

**Programmatic use** — `main.process_video()` is the per-frame loop
(detect → track → classify → annotate → write) factored out from the CLI
and calibration UI, so it's directly callable with your own pre-built
components and lines:

```python
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.classification.vehicle_classifier import VehicleClassifier
from src.pipeline.data_types import LaneLine
from main import process_video

detector = VehicleDetector(model_path="models/detection/yolov8n.pt")
tracker = VehicleTracker(tracker_class="ByteTrack")
lines = [LaneLine(start=(0, 400), end=(640, 400))]  # skip interactive calibration
classifier = VehicleClassifier(lines=lines)

counts = process_video(
    video_path="road_traffic.mp4",
    lines=lines,
    detector=detector,
    tracker=tracker,
    classifier=classifier,
    output_path="output.mp4",
    display=False,
)
print(counts)  # {'SV': ..., 'MV': ..., 'HV': ...}
```

**Not covered by the automated test suite:** `calibrate_lanes()` (the
interactive click-drag UI) and `main()` (CLI parsing + calibration + the
loop, wired together) — both need a display and a human drawing lines, so
they're manually tested. `process_video()` itself has a real integration
test (`tests/integration/test_main_smoke.py`) covering the detect → track
→ classify → annotate → write loop end-to-end.
