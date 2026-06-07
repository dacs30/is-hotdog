# SeeFood 🌭 — Android Hot Dog Detector

A fully offline Android app that points your camera at the world and tells you one thing:
**HOTDOG** or **NOT HOTDOG**. Inspired by Silicon Valley's "SeeFood".

It runs an on-device **YOLOv8n** TFLite model (COCO class 52 = `hot dog`) — no network,
no server, all inference on the phone.

And yes, this is from the Silicon Valley TV show.

## Stack

- **Kotlin** + **CameraX** (preview + image analysis)
- **TensorFlow Lite** 2.14 (GPU delegate with CPU fallback)
- Min SDK **26** (Android 8.0), compile/target SDK **36**
- AGP 8.9.1, Kotlin 2.0.21, Gradle 8.13

## Project layout

```
app/src/main/
├── assets/yolov8n.tflite                 ← exported model (NHWC float32, [1,84,8400])
├── java/com/seefood/hotdog/
│   ├── MainActivity.kt                    ← CameraX + UI
│   ├── HotdogDetector.kt                  ← TFLite inference wrapper
│   └── ImageUtils.kt                      ← ImageProxy → upright Bitmap
└── res/
    ├── layout/activity_main.xml
    └── values/{colors,strings,themes}.xml
```

## How it works

1. CameraX `ImageAnalysis` (RGBA_8888, KEEP_ONLY_LATEST) hands each frame to a background thread.
2. The frame is rotated upright and scaled to **640×640**, normalized to `[0,1]` RGB in **NHWC** order.
3. The interpreter runs YOLOv8n; output is `[1, 84, 8400]` — 4 box coords + 80 class scores × 8400 candidate boxes.
4. We take the max score across all boxes for **channel 56** (`4 + 52`, the "hot dog" class).
5. If that score `> 0.5`, it's a HOTDOG.

`HotdogDetector` reads the model's actual input/output tensor shapes at load time, so it stays
correct even if you re-export with a different layout.

## Build

```bash
# JBR from Android Studio works as the JDK:
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug
# APK -> app/build/outputs/apk/debug/app-debug.apk
```

Install on a connected device:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or just open the folder in Android Studio and hit Run. A device with a **real camera** is the
meaningful test (emulator cameras show a synthetic scene with no hot dogs).

## Re-exporting the model

The model was produced with Ultralytics (`export_model.py`). The export toolchain
(`onnx2tf`, TensorFlow) requires **Python 3.10–3.12** — 3.13/3.14 are not yet supported, and
this machine's pyenv 3.13.0 was also built without `_lzma`.

```bash
~/.pyenv/versions/3.12.8/bin/python -m venv .venv-export
. .venv-export/bin/activate
pip install ultralytics onnx onnxslim onnx2tf sng4onnx onnx_graphsurgeon tensorflow tf_keras onnxruntime ai-edge-litert
python export_model.py
cp yolov8n_saved_model/yolov8n_float32.tflite app/src/main/assets/yolov8n.tflite
```

For a smaller/faster model, export with `int8=True` (or use the `float16` variant the export
also produces) — but note an int8 model changes the input dtype, which `HotdogDetector` would
need to handle.

## Tuning

- **Confidence threshold** lives in `HotdogDetector.CONFIDENCE_THRESHOLD` (default `0.5`).
  The on-screen confidence read-out helps you calibrate.
- Validated offline: a hot dog photo scores ~0.88; banana / pizza / people score 0.00.

## Notes / deviations from the original spec

- Input is **NHWC** `[1,640,640,3]` — the TFLite export is channels-last, not NCHW.
- The "hot dog" score is at output channel **56** (`4 + 52`), not `52 - 4`.
- The `.venv-export/` and `yolov8n_saved_model/` directories are local export scratch space
  (git-ignored); delete them to reclaim disk once you're happy with `app/src/main/assets/yolov8n.tflite`.
