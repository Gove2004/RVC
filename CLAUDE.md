# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVC 实时语音转换工具 — 基于深度学习的实时变声器。PySide6 桌面GUI，CUDA-only，Python 3.13 + RTX 5060。

## Commands

```bash
# 启动实时变声 GUI
.venv/Scripts/python.exe app.py

# 安装依赖
.venv/Scripts/pip.exe install -r requirements.txt
```

## Architecture

**实时推理流程**: 用户选择模型 → 点击"开始" → `LoadThread` 加载模型 → `RealtimeEngine.setup()` 打开音频流 → `sounddevice` 回调 `_cb()` → `RealtimeVC.infer()` → SOLA 交叉淡化 → 扬声器输出

### GUI 结构 (`app.py`)

三个 Tab + 底部控制栏，统一 Grid 布局（间距 6px、边距 10px、标签最小宽度 35px）：

- **设置** (`_build_settings_tab`) — 音频设备路由 + 引擎参数（阈值、采样长度、淡入淡出、额外上下文、F0 算法）
- **模型** (`_build_models_tab`) — 可滚动的 `ModelCard` 列表，单选选中，展开显示参数（音调/性别/索引/响度），删除按钮在展开底部
- **声学** (`_build_audio_tab`) — 降噪 + 音效（EQ/饱和/压限/混响 + 预设系统）

底部控制栏：`开始`（成功后显示"运行中"）| `停止` | `当前: 模型名` | `延迟: X` | `推理: X`

### 核心组件

- **`RealtimeEngine`** — 管理 `sounddevice.Stream`、缓冲区、重采样器、SOLA。`_cb()` 是音频回调（实时线程），读取全局 `p` Params 对象获取参数。
- **`RealtimeVC`** — 推理引擎。加载 HuBERT + 合成器 + 可选 FAISS 索引。`infer()` 接收 16kHz 滚动缓冲区，返回模型采样率音频。
- **`ModelCard`** — 可折叠的模型配置卡片。`load_requested` 信号通知 MainWindow 选中模型。
- **`ModelListData`** — `configs/models.json` 的读写。
- **`Params`** — 全局运行时参数单例（`p = Params()`），回调线程读、主线程写（依赖 GIL 保证原子性）。

### 模型缓存

模块级缓存，跨 `RealtimeVC` 实例复用：
- `rvc/hubert.py` → `_hubert_cache` — HuBERT 模型（设备级缓存）
- `rvc/realtime_engine.py` → `_rmvpe_cache` / `_fcpe_cache` — F0 提取器（设备级缓存）

合成器不缓存（每个 .pth 文件权重不同，每次重新加载）。

### 配置持久化

- `configs/inuse/gui_config.json` — 全局配置（设备、引擎参数、EQ、选中的模型路径）
- `configs/models.json` — 模型列表（每个模型的 name/pth/idx/pitch/index_rate/rms_mix/gender）

### 关键模式

- **滚动缓冲区**: 音频以 ~250ms 块流入，维护 2.5s 上下文窗口。HuBERT 在整个缓冲区上运行，但只解码最新块。
- **SOLA 交叉淡化**: 使用互相关找到最佳拼接点，正弦平方窗口淡入淡出。
- **F0 方法**: `fcpe`（快速）和 `rmvpe`（精确），全局选择，不绑定单个模型。
- **索引**: `index_rate > 0` 时才加载 FAISS 索引（惰性加载）。
- **声学效果**: EQ (低/中/高频 biquad) + tanh 饱和 + 动态压限 + 4-tap 延迟混响。5 个内置预设。
- **日志**: 统一中文格式，每个组件一行（`加载 HuBERT` / `加载 Synthesizer` / `加载完成`）。

## Environment

- **Python**: 3.13+
- **GPU**: NVIDIA only (CUDA required). Target: RTX 5060
- **fairseq-fixed**: 社区修复版 fairseq，用于加载 HuBERT 模型
- **OS**: Windows 11
