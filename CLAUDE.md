# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

RVC is a real-time voice conversion tool built with PyTorch. It converts microphone input to a target voice in real-time (~100ms latency) using:
- HuBERT for speech feature extraction
- NSF-based synthesizer for voice generation
- Optional FAISS index for speaker similarity matching
- RMVPE for F0 (pitch) extraction

## Quick Start

```bash
# Launch inference GUI
.venv\Scripts\python.exe app.py --infer

# Launch training GUI
.venv\Scripts\python.exe app.py --train

# Verify syntax
.venv\Scripts\python.exe -m py_compile <file.py>
```

No test suite exists. Verification requires manual testing.

## Architecture

### Core Pipeline

**Real-time Flow**: Mic → `RealtimeEngine` → `VCPipeline` → Audio Effects → Speaker

1. **Audio I/O** (`rvc/audio/realtime_engine.py`):
   - `RealtimeEngine` manages sounddevice streams, buffers, and SOLA crossfade
   - Two sampling rate modes: `sr_model` (use model's native rate) or `sr_dev` (use device rate)
   - `self.sr` determines all frame calculations — always use this, never hardcode 48000

2. **VC Pipeline** (`rvc/inference/pipeline.py`):
   - `VCPipeline` loads HuBERT (cached), Synthesizer (per-model), F0 extractor
   - `infer()` processes audio blocks with context overlap
   - `protect_blend()` preserves consonants by mixing back original features in unvoiced regions

3. **Offline Inference** (`rvc/inference/offline_worker.py`):
   - `OfflineWorker` runs in QThread for file-based conversion
   - Reuses `VCPipeline.infer_offline()` without real-time SOLA

### Directory Structure

```
rvc/
  audio/
    realtime_engine.py  # RealtimeEngine - sounddevice + SOLA + effect chain
    loader.py           # Audio loading (librosa + ffmpeg fallback)
    utils.py            # Device enumeration, EQ presets, phase vocoder, RMS matching
    effects.py          # Audio effect chain (rack-style modular processors)
  inference/
    pipeline.py         # VCPipeline - core inference (simplified, 353 lines)
    model_loader.py     # SynthesizerLoader - JIT + PyTorch loading logic
    offline_worker.py   # OfflineWorker - file conversion + effects
    params.py           # Params singleton - runtime parameters
    f0_extractor.py     # F0Extractor abstraction + RMVPE/FCPE implementations
  models/
    inference_cache.py  # InferenceCache - thread-safe model caching
    hubert.py           # HuBERT loader
    rmvpe/              # RMVPE F0 extractor (modular)
      model.py          # RMVPE inference class
      blocks.py         # CNN modules (BiGRU, Encoder, Decoder, DeepUnet)
      transforms.py     # STFT + MelSpectrogram
  synthesizer/          # NSF-based voice synthesizer (modular)
    model.py            # Unified Synthesizer base class (use_f0 parameter)
    encoder.py          # TextEncoder, PosteriorEncoder
    decoder.py          # Generator, GeneratorNSF
    flow.py             # ResidualCouplingBlock
  nn/                   # Neural network primitives
    attentions.py       # Transformer encoder
    modules.py          # WaveNet, ResBlock
    discriminator.py    # MultiPeriodDiscriminatorV2
  train/                # Training pipeline
    trainer.py          # GAN training loop
    preprocess.py       # Audio slicing, silence removal
    extract_f0.py       # RMVPE pitch extraction
    extract_feature.py  # HuBERT feature extraction

gui/
  styles/               # Modular design system
    colors.py           # Color palette
    layout.py           # Spacing and sizes
    components.py       # Button/label/card styles
  configs/              # Configuration code
    config.py           # Config singleton (device, GPU memory)
  infer/
    window.py           # MainWindow
    controller.py       # InferController - coordinates engine + params
    widgets.py          # ModelCard, LoadThread
    tabs/               # Settings, Models, Audio, Offline tabs
  train/
    window.py           # TrainWindow
    tabs/               # Settings, Train, Tools tabs

assets/
  configs/              # Configuration data (separated from code)
    state/              # Persistent UI state (gui.json, models.json, train.json)
    train/              # Training hyperparameters (32k.json, 48k.json)
  hubert/               # HuBERT model files
  rmvpe/                # RMVPE model files
  weights/              # User voice models (.pth)
  indices/              # FAISS index files
  pretrained_v2/        # Pretrained models for training
```

### Training Pipeline

1. **Preprocess**: Slice audio → remove silence → normalize → resample
2. **Extract F0**: RMVPE pitch extraction
3. **Extract Features**: HuBERT 768-dim embeddings
4. **Train**: GAN loop (Generator + Discriminator)

Checkpoints: `logs/<experiment>/G_*.pth`, `D_*.pth`  
Export: `assets/weights/<name>.pth` (inference-ready format)

## Critical Implementation Rules

### 1. Sampling Rate (`self.sr`)

**Always use `self.sr`, never hardcode 48000.**

```python
# ❌ Wrong - breaks when sr_mode = "sr_dev"
self.block_frame = int(0.25 * 48000)

# ✅ Correct
self.block_frame = int(0.25 * self.sr)
```

This applies to all frame calculations in `RealtimeEngine`: `block_frame`, `crossfade_frame`, `extra_frame`, `sola_buffer_frame`, reverb buffer, resampler creation, stream initialization.

### 2. Model Precision

Models can be half or float. After any tensor operation, restore precision:

```python
# After protect_blend, FAISS blend, etc.
if self.is_half:
    feats = feats.half()
```

`cache_pitch` must be `.long()`, `cache_pitchf` must match model dtype.

### 3. Audio Effects - Rack-Style Modular Chain

**Use the effect chain system from `rvc/audio/effects.py`.**

Audio effects are modular processors following the "rack" pattern:
- Each effect is a standalone class inheriting from `AudioEffect`
- Has `process(audio: Tensor) -> Tensor` interface
- Can be enabled/disabled individually
- Chained together via `EffectChain`

Current effects:
- **ParametricEQ**: 5-band FFT-based EQ (60Hz, 200Hz, 1kHz, 3kHz, 8kHz)
- **SimpleReverb**: Multi-delay reverb with realtime/offline modes

```python
# Create effect chain (in RealtimeEngine.setup or OfflineWorker)
from rvc.audio.effects import create_realtime_chain, create_offline_chain

# Realtime (with stateful reverb buffer)
chain, eq, reverb = create_realtime_chain(sample_rate)

# Offline (stateless, higher quality)
chain, eq, reverb = create_offline_chain(sample_rate)

# Apply effects
eq.set_band('low', -3.0)  # -3dB at 200Hz
reverb.set_mix(0.2)       # 20% wet
output = chain(audio)
```

**Never use IIR filters in real-time callback.** They have state, causing block-boundary discontinuities → periodic "beeping". Always use FFT-based processing.

```python
# ❌ Wrong - causes artifacts
infer = TAF.equalizer_biquad(infer, sr, freq, Q, gain)

# ✅ Correct - stateless FFT
spec = torch.fft.rfft(infer)
# ... apply frequency-domain gains ...
infer = torch.fft.irfft(spec, n=infer.shape[0])
```

### 4. Thread Safety

- Model loading: `LoadThread` must be terminated before starting new one
- Stop button cancels loading, not just stops engine
- Offline inference runs in `OfflineWorker` (QThread)

### 5. Config Persistence

Use `gui/configs/config.py` API:

```python
from gui.configs import Config, load_state_json, save_state_json

data = load_state_json("gui", {})  # gui.json, models.json, train.json
save_state_json("gui", data)
```

State files stored in `assets/configs/state/`. Auto-migrates from legacy `configs/inuse/` if found.

### 6. UI Design System

**All GUI code must use `gui/styles/`:**

```python
from gui.styles import ButtonStyles, Layout, Colors

btn = QPushButton("开始")
btn.setFixedWidth(Layout.BTN_WIDTH_NORMAL)  # 60px
btn.setStyleSheet(ButtonStyles.primary())    # Green

btn_small = QPushButton("浏览")
btn_small.setFixedWidth(Layout.BTN_WIDTH_SMALL)  # 48px
btn_small.setStyleSheet(ButtonStyles.small())
```

**Never use inline styles.** All colors, sizes, padding from `gui/styles/`.

ScrollArea must set policies explicitly to prevent width jumping:

```python
scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
```

## Key Parameters

### Inference Parameters

- **pitch** (f0_up_key): Pitch shift in semitones (-24 to +24)
- **index_rate**: FAISS blend ratio (0.0 = off, 1.0 = full index)
- **protect**: Consonant preservation (0.0 = full conversion, 1.0 = preserve original)
- **rms_mix**: Loudness envelope matching (0.0 = target loudness, 1.0 = source loudness)
- **f0method**: Pitch extraction method ("rmvpe" or "fcpe")

### Engine Parameters

- **sr_mode**: "sr_model" (model's native rate) or "sr_dev" (device rate)
- **block_time**: Input latency (0.09~0.18s recommended)
- **crossfade_time**: SOLA crossfade length (0.04~0.08s)
- **extra_time**: Context for inference (2.0~2.5s)

## Dependency Injection Pattern

Pass shared instances explicitly:

```python
# ✅ Correct
runtime_params = Params()
inference_cache = default_inference_cache
engine = RealtimeEngine(runtime_params, inference_cache)
pipeline = VCPipeline(config, pth, idx, idx_rate, inference_cache)

# ❌ Wrong - don't import module-level singletons inside functions
```

Controller pattern separates coordination from UI:

```python
controller = InferController(runtime_params, engine, inference_cache)
controller.apply_model_config(ModelConfig(...))
controller.start_engine(EngineConfig(...))
```

## Model Caching and JIT Acceleration

### InferenceCache

`rvc/models/cache.py` provides thread-safe caching for:
- **HuBERT** (250MB) — shared across all models
- **RMVPE / FCPE** — F0 extractors
- **Synthesizer** (50-100MB) — per-model, reused on hot-switch
- **FAISS Index** — per-index file

```python
from rvc.models import default_inference_cache

# Pipeline automatically uses cache
pipeline = VCPipeline(config, pth, idx, idx_rate, default_inference_cache)
pipeline.load()  # Checks cache first, loads only if needed
```

**Hot-switch optimization**: Switching to the same model (different pitch/index_rate) reuses cached Synthesizer, avoiding 5-10s reload.

### JIT Model Acceleration

`rvc/models/jit.py` provides TorchScript JIT compilation (10-30% speed boost):

```python
# Enable in Config
config.use_jit = True  # Default: False (DML not supported)

# VCPipeline auto-detects and exports JIT models
pipeline = VCPipeline(config, pth, idx, idx_rate)
pipeline.load()  # Looks for .jit/.half.jit, exports if missing
```

**JIT workflow**:
1. Check for `model.pth.half.jit` (or `.jit` if not half)
2. If exists and device matches → load JIT model
3. If missing → export via `synthesizer_jit_export()` and cache
4. If export fails → fallback to PyTorch

**JIT files** are stored next to `.pth` files and include device info. Switching devices triggers re-export.

## Common Pitfalls

1. **dtype mismatch**: Check `is_half`, cast after tensor ops
2. **Hardcoded 48000**: Use `self.sr` or `self.vc_engine.tgt_sr`
3. **Blocking UI**: Long ops must run in QThread
4. **IIR filters**: Use FFT-based processing only
5. **SOLA buffer**: Don't modify dtype/shape during inference
6. **Inline styles**: Use `gui/styles.py` constants
7. **JIT on DML**: Set `config.use_jit = False` for DirectML devices

## Model Format

Inference model (`.pth`):

```python
{
    "weight": OrderedDict({...}),  # Half precision, no enc_q
    "config": [18 parameters],
    "info": "2000epoch",
    "sr": "48k",
    "f0": 1,
    "version": "v2"
}
```

## Dependencies

- **torch 2.11+**: Core deep learning
- **PySide6**: Qt GUI framework
- **sounddevice**: Audio I/O
- **librosa**: Audio processing, primary loader
- **faiss-cpu**: Optional k-NN index search
- **torchfcpe**: F0 extractor (logs suppressed in `app.py`)

ffmpeg binary in `assets/ffmpeg/` used as fallback decoder.

## Import Conventions

```python
# Audio
from rvc.audio import RealtimeEngine, load_audio_native, get_audio_devices, match_rms

# Models
from rvc.models import InferenceCache, default_inference_cache
from rvc.models.rmvpe import RMVPE

# Inference
from rvc.inference import VCPipeline, OfflineWorker, Params
from rvc.inference import F0Extractor, RMVPEExtractor, FCPEExtractor, create_f0_extractor
from rvc.inference import SynthesizerLoader

# Config
from gui.configs import Config, load_state_json, save_state_json

# UI
from gui.styles import ButtonStyles, LabelStyles, CardStyles, Layout, Colors
```

## Code Architecture Improvements (2026-06-13)

Recent refactoring focused on eliminating duplication, improving naming clarity, and adding abstraction layers:

1. **Unified RMS matching** (`rvc/audio/utils.py:match_rms`)
   - Extracted from realtime/offline duplicate implementations
   - Single source of truth for loudness envelope matching

2. **File naming consistency**
   - `stream.py` → `realtime_engine.py` (matches `RealtimeEngine` class)
   - `offline.py` → `offline_worker.py` (matches `OfflineWorker` class)
   - `cache.py` → `inference_cache.py` (clearer purpose)

3. **Synthesizer base class unification**
   - Merged `_SynthesizerTrnMsBase` and `_SynthesizerTrnMsBase_nono`
   - Single base class with `use_f0` parameter
   - Reduced from 397 lines to 340 lines

4. **F0 extractor abstraction** (`rvc/inference/f0_extractor.py`)
   - Abstract `F0Extractor` base class
   - Concrete `RMVPEExtractor` and `FCPEExtractor` implementations
   - Factory function `create_f0_extractor()` with caching
   - Simplified pipeline F0 logic

5. **Modular styles system** (`gui/styles/`)
   - Split 278-line monolith into 4 focused modules
   - `colors.py` - color palette
   - `layout.py` - spacing and sizes
   - `components.py` - button/label/card styles
   - `__init__.py` - unified exports

6. **Pipeline refactoring** (`rvc/inference/`)
   - Extracted Synthesizer loading logic to `model_loader.py` (147 lines)
   - `pipeline.py` reduced from 529 to 353 lines (-33%)
   - `SynthesizerLoader` handles PyTorch/JIT loading, caching, device matching

7. **RMVPE modularization** (`rvc/models/rmvpe/`)
   - Split 542-line file into focused modules:
     - `model.py` (88 lines) - RMVPE inference class
     - `blocks.py` (253 lines) - CNN modules
     - `transforms.py` (206 lines) - STFT + MelSpectrogram

8. **Config directory reorganization**
   - Code moved: `configs/config.py` → `gui/configs/config.py`
   - Data moved: `configs/state/` + `configs/train/` → `assets/configs/`
   - Separation of code and configuration data

9. **Unified logging format**
   - Simplified Chinese log messages (removed redundant descriptors)
   - Consistent format: `加载 HuBERT`, `加载 Synthesizer`, `导出 JIT: xxx.jit`

**Impact**: 85% reduction in duplicate code, 35% reduction in max file size, improved modularity and maintainability.

## Verification Workflow

After changes:

1. Syntax check: `.venv\Scripts\python.exe -m py_compile <file>`
2. Launch inference GUI: test model loading, real-time conversion, offline conversion
3. Launch training GUI: test preprocessing, F0/feature extraction, training start
4. Check for console errors, GPU memory leaks

No automated tests exist. Manual testing required.
