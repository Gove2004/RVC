# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Launch inference GUI
.venv\Scripts\python.exe app.py --infer

# Launch training GUI
.venv\Scripts\python.exe app.py --train

# Syntax check (no pytest/unittest available)
.venv\Scripts\python.exe -m py_compile <file.py>
```

## Architecture Overview

### Directory Structure (Post-Refactor)

```
RVC/
├── gui/                    # GUI 模块（推理 + 训练）
│   ├── infer/             # 推理 GUI
│   │   ├── controller.py  # 推理控制器（依赖注入）
│   │   ├── window.py      # 主窗口
│   │   ├── widgets.py     # ModelCard, ModelListData
│   │   └── tabs/          # 各功能 Tab
│   └── train/             # 训练 GUI
├── rvc/                   # 核心推理与训练模块
│   ├── audio/             # 音频处理
│   │   ├── loader.py      # 音频加载（librosa + ffmpeg fallback）
│   │   ├── stream.py      # 实时音频流管理（RealtimeEngine）
│   │   └── utils.py       # 工具函数（设备枚举、预设）
│   ├── models/            # 模型加载
│   │   ├── hubert.py      # HuBERT 加载
│   │   ├── rmvpe.py       # RMVPE F0 提取器
│   │   └── cache.py       # InferenceCache（线程安全）
│   ├── inference/         # 推理逻辑
│   │   ├── pipeline.py    # VCPipeline（核心推理管线）
│   │   ├── offline.py     # OfflineWorker（离线推理线程）
│   │   └── params.py      # Params（运行时参数单例）
│   ├── synthesizer/       # 合成器（已模块化）
│   ├── nn/                # 神经网络基础层
│   └── train/             # 训练管线
├── configs/               # 配置管理
│   ├── state/             # 可变状态（gui.json, models.json, train.json）
│   ├── train/             # 训练配置（32k.json, 48k.json）
│   └── config.py          # 配置 API
└── assets/                # 资源文件
    ├── weights/           # 模型权重
    ├── indices/           # FAISS 索引
    ├── pretrained_v2/     # 预训练模型
    ├── hubert/            # HuBERT 权重
    ├── rmvpe/             # RMVPE 权重
    └── ffmpeg/            # ffmpeg 二进制
```

### Real-time Voice Conversion Pipeline

**Flow**: Microphone → Audio I/O → HuBERT → FAISS → Synthesizer → Audio Effects → Speaker

1. **Audio I/O Layer** (`rvc/audio/stream.py` - `RealtimeEngine`)
   - Manages sounddevice streams, buffers, SOLA crossfade
   - Critical: `self.sr` determines all frame calculations (block, crossfade, reverb)
   - Two sampling rate modes: `sr_model` (model's native rate) or `sr_dev` (device rate)
   - SOLA algorithm handles pitch-shift without glitches at block boundaries

2. **VC Pipeline** (`rvc/inference/pipeline.py` - `VCPipeline`)
   - Loads HuBERT (cached), Synthesizer (per-model), F0 extractor (FCPE/RMVPE)
   - `infer()` method processes one audio block with overlap for context
   - **Protect mechanism**: blends unvoiced consonants back from original features to prevent distortion
   - FAISS index (optional): k=8 weighted blending for speaker similarity matching

3. **Model Precision** (critical for stability)
   - Models can be half or float precision
   - After any tensor operation (protect_blend, FAISS blend), **must restore dtype**:
     ```python
     if self.is_half:
         feats = feats.half()
     ```
   - `cache_pitch` must be `.long()`, `cache_pitchf` must match model precision

### Offline Inference

**Flow**: Audio File → Preprocessing → VC Pipeline → Post-processing → Output File

- `rvc/inference/offline.py` (`OfflineWorker`) runs in background thread
- Reuses `VCPipeline.infer()` but without SOLA (processes full audio at once)
- Progress reported via signals

### Audio Effects (声学效果)

Located in `rvc/audio/stream.py` callback, applied **after** VC inference:

- **5-band EQ**: Uses **FFT-based** frequency domain processing (not IIR biquad filters)
  - Why: IIR filters have state, causing discontinuities at block boundaries → periodic "beeping" artifacts
  - Implementation: `torch.fft.rfft` → apply Gaussian band gains → `torch.fft.irfft`
- **Reverb**: Simple multi-tap delay with decay
- **RMS Mix**: Envelope-based loudness matching

### GUI Architecture

- **Entry**: `app.py` dispatches to `gui/infer/window.py` or `gui/train/window.py`
- **Controller**: `gui/infer/controller.py` manages inference coordination (model config, runtime params, engine lifecycle)
- **Threading**:
  - Model loading: `LoadThread` (avoids blocking UI)
  - Training: `ToolThread` for each pipeline stage
  - Offline inference: `OfflineWorker`
- **Model Management**: `widgets.py` contains `ModelCard` (expandable card with sliders) and `ModelListData` (JSON persistence via `configs/state/models.json`)

### Training Pipeline

Located in `rvc/train/`:

1. **Preprocess** (`preprocess.py`): Slice audio → remove silence → normalize → resample
2. **Extract F0** (`extract_f0.py`): RMVPE pitch extraction
3. **Extract Features** (`extract_feature.py`): HuBERT 768-dim embeddings
4. **Train** (`trainer.py`): GAN training loop (Generator + Discriminator)

Checkpoints saved to `logs/<experiment>/`, exportable to `assets/weights/<name>.pth`.

## Critical Implementation Details

### Sampling Rate (`sr_type`) Bug Pattern

When adding new audio processing, always use `self.sr` (not hardcoded):

```python
# ❌ Wrong
self.block_frame = int(0.25 * 48000)  # breaks when sr != 48k

# ✅ Correct
self.block_frame = int(0.25 * self.sr)
```

### Protect (辅音保护) Mechanism

- Range: 0.0 (full conversion) to 1.0 (preserve original consonants)
- Implementation: `protect_blend()` uses `pitchf` as mask (pitchf < 1 = unvoiced)
- User preference in this codebase: **protect = 0.9** (high preservation, index_rate = 0)

### EQ Implementation Constraint

**Never use IIR filters in real-time callback**. They cause block-boundary artifacts because state is not preserved between calls. Always use FFT-based processing:

```python
# ❌ Wrong - causes periodic beeping
infer = TAF.equalizer_biquad(infer.unsqueeze(0), sr, 1000.0, 1.0, gain).squeeze(0)

# ✅ Correct - stateless frequency domain
spec = torch.fft.rfft(infer)
# ... apply gain curve ...
infer = torch.fft.irfft(spec, n=infer.shape[0])
```

### Thread Safety for Model Loading

When switching models rapidly:
1. **Terminate** old `LoadThread` before starting new one
2. Stop button must also cancel loading (not just stop engine)

```python
if self._loading and self._lt and self._lt.isRunning():
    self._lt.terminate()
    self._lt.wait()
```

### Config Persistence

- **State files**: `configs/state/*.json` (GUI state, models list, training state)
  - `gui.json` - Inference GUI state (devices, sliders, sr_mode)
  - `models.json` - Model list and per-model settings
  - `train.json` - Training GUI state (exp_name, paths, hyperparams)
- **Train configs**: `configs/train/{32k,48k}.json` (model architecture configs)
- **API**: Use `load_state_json(name, default)` and `save_state_json(name, data)` from `configs/config.py`
- **Legacy migration**: Old `configs/inuse/*.json` files automatically migrated to `configs/state/` on first access
- Missing keys use default values (backward compatible)

## Common Pitfalls

1. **dtype mismatch**: Always check `is_half` and cast tensors before passing to model
2. **Hardcoded 48000**: Use `self.sr` / `self.vc_engine.tgt_sr` instead
3. **Blocking UI**: Long operations must run in `QThread` (see `LoadThread`, `OfflineWorker`)
4. **副输出 (secondary output)**: Must use callback mode, not `stream.write()` (MME driver issue)
5. **SOLA buffer**: Don't modify `sola_buffer` dtype/shape during inference
6. **Global state**: Use dependency injection - pass `runtime_params` and `inference_cache` explicitly instead of importing module-level globals

## Architecture Patterns

- **Dependency injection**: `RealtimeEngine` and `VCPipeline` accept `runtime_params` and `inference_cache` in constructors
- **Controller pattern**: `gui/infer/controller.py` separates coordination logic from UI (see `InferController`)
- **Explicit caching**: ML models cached in `rvc/models/cache.py` (`InferenceCache` class) with thread-safe operations
- **Dataclass configs**: Use `ModelConfig`, `RuntimeConfig`, `EngineConfig` for typed parameter groups
- **Modular organization**: Code organized by function (audio/, models/, inference/) not file type

## User Preferences (from conversation)

- Index rate: **0** (never uses FAISS index)
- Protect: **0.9** (preserves 90% of original consonants for clarity)
- Dialect: 四川话 (z/zh, c/ch, s/sh merged), so high protect prevents over-correction
- EQ: Rarely used, prefers original sound

## Dependencies

- **torchfcpe**: F0 extractor, logs [INFO]/[WARN] suppressed in `app.py`
- **sounddevice**: Audio I/O, requires matching HostAPI for input/output devices
- **faiss-cpu**: Optional, for FAISS index search (k-NN speaker matching)
- **librosa**: Primary audio loader, ffmpeg fallback in `rvc/audio/loader.py`
- **ffmpeg**: Located in `assets/ffmpeg/`, used as fallback decoder

## Import Conventions

After restructuring, use these import patterns:

```python
# Audio processing
from rvc.audio import RealtimeEngine, load_audio, get_audio_devices, PRESETS

# Model loading
from rvc.models import InferenceCache, default_inference_cache
from rvc.models.hubert import load_hubert
from rvc.models.rmvpe import RMVPE

# Inference
from rvc.inference import VCPipeline, OfflineWorker, Params

# Config
from configs.config import load_state_json, save_state_json, train_config_path
```

## File Conventions

- Model files: `.pth` (PyTorch checkpoint with `weight`, `config`, `info`, `sr`, `f0`, `version`)
- Index files: `.index` (FAISS index, optional)
- Logs use `logging` module, configured in `app.py`
