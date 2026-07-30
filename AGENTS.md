# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
.venv\Scripts\pip.exe install -r requirements.txt

# Launch inference GUI
.venv\Scripts\python.exe app.py --infer

# Launch training GUI
.venv\Scripts\python.exe app.py --train

# Syntax-check one file
.venv\Scripts\python.exe -m py_compile path/to/file.py

# Syntax-check the app packages
.venv\Scripts\python.exe -m compileall -q app.py rvc gui
```

There is no automated test suite. Runtime verification is manual: start the relevant GUI, check console logs, and exercise the changed path.

## Original RVC Source

- **Location:** `Retrieval-based-Voice-Conversion-WebUI/` (separate git repo, fetch with `git -C "Retrieval-based-Voice-Conversion-WebUI" fetch origin && git -C "Retrieval-based-Voice-Conversion-WebUI" reset --hard origin/main`)
- **Remote:** `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git`
- The local upstream clone is in sync with `origin/main` (verified 2026-07-30). Upstream's recent updates vs. this project's actual status:
  - **CUDA Graph inference acceleration** — already integrated (`rvc/tools/cuda_graph.py`; used in f0_extractor/synthesis/rmvpe/model_session; runtime-enabled via `configure_cuda_graph`, `Config.use_cuda_graph` defaults to False)
  - **ASIO glitch fix + multi-API device support** — N/A; realtime engine uses sounddevice, host API from portaudio, no ASIO-specific code
  - **GPU processing for UVR5 / audio loading & resampling** — N/A; UVR5 is WebUI-only, audio loading uses CPU librosa + ffmpeg (`rvc/audio/loader.py`)
  - **NSF inference optimization** — NSF implemented (`rvc/synthesizer/decoder.py` `GeneratorNSF`); specific upstream optimizations not separately verified
  - **TensorRT & ONNX export warnings** — N/A (WebUI export feature, not used)
  - **PyMSS backend replacing UVR5 separation** — N/A (vocal separation, WebUI-only)
  - **DirectML support for PyMSS** — N/A (non-NVIDIA GPU; this project requires CUDA)
- Only import relevant changes — our codebase is heavily refactored and structurally different from upstream.
- When syncing from upstream, check: `rvc/audio/`, `rvc/inference/`, `rvc/models/`, `rvc/nn/`, `rvc/runtime/`

## Runtime Requirements

- Windows + NVIDIA CUDA GPU are required; `rvc.runtime.Config` exits if CUDA is unavailable.
- Python dependencies are in `requirements.txt`; core libraries are PyTorch, PySide6, sounddevice, librosa, FAISS, transformers, and torchfcpe.
- Required model assets live under `assets/`: HuBERT, RMVPE, user weights, indices, pretrained training weights, and ffmpeg.

## Architecture

The app has one entrypoint, `app.py`, with two GUI modes:

- `--infer` loads `gui.infer.window.MainWindow`
- `--train` loads `gui.train.window.TrainWindow`

Layering rule:

- `rvc/` is core runtime, inference, audio, model, and training code.
- `gui/` owns PySide6 windows, widgets, managers, and QThread workers.
- Core code must not import `gui` or `PySide6`.
- Runtime device/path config comes from `rvc.runtime`; GUI state persistence comes from `gui.configs`.

### Inference GUI — 5 Tab Layout

The inference GUI uses a compact 5-tab interface:

| Tab | Path | Description |
|-----|------|-------------|
| **驱动** (Audio Driver) | `gui/infer/tabs/audio_driver_tab.py` | Audio device selection (input/output/output2), host API, sample rate mode (model/device), refresh button |
| **参数** (Global Params) | `gui/infer/tabs/global_params_tab.py` | Block time, crossfade time, extra context time, RMS mix, consonant protect (0~1), F0 method (RMVPE/FCPE) |
| **模型** (Models) | `gui/infer/tabs/models_tab.py` | Model management — add `.pth` models, activate/remove, index file association |
| **处理** (Processing) | `gui/infer/tabs/audio_tab.py` | 5-band EQ (60Hz/200Hz/1kHz/3kHz/8kHz), reverb, effect presets, enable/disable |
| **离线** (Offline) | `gui/infer/tabs/offline_tab.py` | Offline batch conversion — select input/output WAV files, start conversion |

### Inference flow

Realtime path:

```text
Mic → RealtimeEngine → VCPipeline → Synthesizer → SOLA/effects/output routing → Speaker
```

Main pieces:

- `rvc/audio/realtime_engine.py` is the realtime facade for sounddevice streams, rolling buffers, resampling, model calls, SOLA, BGM, effects, and secondary output.
- Pure realtime helpers live beside it:
  - `sola.py` for SOLA alignment/crossfade
  - `realtime_mix.py` for RMS envelope mix
  - `realtime_effects.py` for EQ/reverb parameter sync
  - `output_router.py` for BGM and main/secondary output routing
- `rvc/inference/pipeline.py` is a facade over:
  - `model_session.py` for HuBERT/Synthesizer/index loading
  - `feature_processing.py` for HuBERT features, padding masks, protect blend, upsampling
  - `pitch_tracker.py` for F0 extraction windows and realtime pitch cache
  - `index_retrieval.py` for FAISS loading/blending
  - `synthesis.py` for `net_g.infer()` calls and formant resampling
  - `model_loader.py` for PyTorch Synthesizer loading
- Offline conversion uses `gui/infer/workers.py::OfflineWorker` and `rvc.inference.OfflineConfig`, then calls `VCPipeline.infer_offline()`.

### Training flow

Training GUI uses `gui/train/window.py` and `gui/train/workers.py::TrainWorker`. Core stages live in `rvc/train/`:

```text
preprocess → extract_f0 → extract_feature → trainer → export .pth
```

Training config path is resolved through `rvc.runtime.train_config_path()` and currently points to `assets/configs/48ktrain_config.json`.

### State and config

- GUI state file: `assets/configs/save_state.json`
- GUI state APIs: `gui.configs.load_config()`, `save_config()`, `InferGuiState`, `TrainGuiState`
- `InferGuiState` includes fields for all tab parameters: block_time, crossfade_time, extra_time, protect, f0method, sr_mode, eq_enabled, EQ bands, reverb, rms_mix, preset, hostapi, input/output devices, active_model
- Runtime device config: `from rvc.runtime import Config`
- Do not put runtime CUDA/device logic in `gui.configs`.

## Critical Implementation Rules

### Sampling rate

In realtime audio code, frame math must use the active runtime sample rate:

```python
self.sr = self.sr_model if sr_type == "sr_model" else self.sr_dev
zc = self.sr // 100
self.block_samples = int(np.round(block_t * self.sr / zc)) * zc
```

Do not hardcode `48000` for realtime frame sizes. `sr_mode="model"` and `sr_mode="device"` must both keep working.

### Model precision

Models may be half or float. After feature/index/protect operations, restore the model dtype before synthesis. Pitch coarse tensors must be `long`; continuous pitch tensors must match the model path's expected dtype.

### Realtime callback safety

Avoid blocking operations, file I/O, model loading, large allocations, and unnecessary CPU/GPU sync inside the sounddevice callback. Stop loading threads before starting a new load; the stop button must cancel loading as well as stop a running stream.

### Audio effects

Use the modular effect chain in `rvc/audio/effects.py`. Realtime EQ must stay FFT-based; do not use stateful IIR filters in the realtime callback because they can create block-boundary artifacts.

### GUI style

GUI code should use `gui.styles` (`ButtonStyles`, `LabelStyles`, `CardStyles`, `Layout`, `Colors`) instead of ad-hoc inline sizing/colors.

### Caching

- `rvc.models.default_inference_cache` caches HuBERT, F0 extractors, Synthesizer models, and FAISS indices.
- `Config.use_cuda_graph` defaults to `False` but is runtime-enabled by `configure_cuda_graph` when the GPU supports it. Do not force it on globally without explicit benchmarking and fallback handling.

## Verification

For code changes, at minimum run:

```bash
.venv\Scripts\python.exe -m py_compile <changed-file.py>
```

For broader refactors:

```bash
.venv\Scripts\python.exe -m compileall -q app.py rvc gui
```

Manual checks depend on the touched area:

- Inference GUI: launch, load a `.pth`, start/stop realtime, check FCPE/RMVPE if F0 code changed, verify all 5 tabs (驱动/参数/模型/处理/离线) display correctly and parameters are saved/restored
- Audio engine: test both model/device sample-rate modes, SOLA stability, EQ/reverb, main and secondary output
- Offline inference: convert one file with the active model
- Training GUI: launch and verify the changed stage can start
