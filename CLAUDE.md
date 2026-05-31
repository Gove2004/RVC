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

**实时推理流程**: `app.py` (PySide6 GUI) → `RealtimeEngine` (sounddevice 音频回调) → `RealtimeVC.infer()` (滚动缓冲区 + HuBERT + FAISS + 合成器) → SOLA 交叉淡化 → 扬声器输出

### 核心组件

- **`app.py`** — PySide6 紧凑 GUI，三个标签页（音频设置/变声引擎/声学母带）。`RealtimeEngine` 管理实时 mic→VC→speaker 管道。

- **`rvc/realtime_engine.py`** — `RealtimeVC` 类。实时推理引擎，滚动缓冲区 + skip_head/return_length 模式。加载 HuBERT、合成器、可选 FAISS 索引。

- **`rvc/hubert.py`** — `HuBERT` 模型架构（独立实现，无需 fairseq/transformers）。从 `assets/hubert/hubert_base.pt` 加载 fairseq 权重。

- **`rvc/synthesizer.py`** — `SynthesizerTrnMsNSFsid` (带F0) / `SynthesizerTrnMsNSFsid_nono` (无F0)。VAE + Normalizing Flow + HiFi-GAN 解码器。

- **`rvc/rmvpe.py`** — RMVPE 音高提取器。CNN 从梅尔频谱图预测 F0。

- **`rvc/nn/`** — 神经网络组件：attentions.py (多头注意力)、modules.py (WaveNet, ResBlock)、commons.py (工具函数)、transforms.py (标准化流)。

- **`rvc/denoise/`** — TorchGate 频谱门控降噪。

### 关键模式

- **滚动缓冲区**: 音频以 ~250ms 块流入，维护 2.5s 上下文窗口。HuBERT 在整个缓冲区上运行，但只解码最新块。
- **SOLA 交叉淡化**: 使用互相关找到最佳拼接点，正弦平方窗口淡入淡出，消除块边界咔嗒声。
- **声学母带**: EQ (低/中/高频) + 电子管饱和 + 动态压限 + 空间混响，带预设系统。
- **F0 方法**: 仅支持 `fcpe`（快速）和 `rmvpe`（精确）。
- **HuBERT layer 12**: 始终使用第12层编码器输出（768维）。
- **配置持久化**: 设置自动保存到 `configs/inuse/gui_config.json`。

## 目录结构

```
app.py                    # PySide6 桌面 GUI 入口
requirements.txt          # 依赖
.env                      # 路径配置

configs/
  config.py               # Config 单例 (CUDA-only)
  v2/32k.json, v2/48k.json

rvc/
  realtime_engine.py      # RealtimeVC 实时推理引擎
  hubert.py               # HuBERT 模型架构 + 加载器
  rmvpe.py                # RMVPE 音高提取
  synthesizer.py          # 合成器模型定义
  nn/                     # 神经网络组件
    attentions.py         # 多头注意力
    commons.py            # 工具函数
    modules.py            # WaveNet, ResBlock, Flow
    transforms.py         # 标准化流变换
  denoise/                # 降噪
    torchgate.py          # TorchGate 频谱门控
    utils.py              # 辅助函数

assets/
  weights/                # 用户训练的 .pth 模型
  pretrained_v2/          # 预训练 v2 模型
  hubert/                 # HuBERT 权重 (hubert_base.pt)
  rmvpe/                  # RMVPE 权重 (rmvpe.pt)
  indices/                # FAISS .index 文件
```

## Environment

- **Python**: 3.13+
- **GPU**: NVIDIA only (CUDA required). Target: RTX 5060
- **fairseq-fixed**: 社区修复版 fairseq，用于加载 HuBERT 模型
- **OS**: Windows 11
