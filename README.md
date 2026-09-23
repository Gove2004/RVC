# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 彻底重写的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟实时实测显示（硬件时间戳、瞬时值）
- **快速启动** — 惰性导入架构，窗口秒开（~0.3s）；torch 后台预热，首次点「开始」不卡顿
- **CUDA Graph 加速** — HuBERT/RMVPE/合成器三大模型全链路图捕获，开流前静音预热完成
- **系统托盘** — 关闭窗口最小化到托盘（变声不中断），托盘可开始/停止变声
- **离线推理** — 音频文件流式转换（复用实时全链路，显存封顶，任意长不 OOM）
- **模型训练** — 从人声音频训练自定义模型，支持 40k/48k、中断续训、云训练向导
- **音域映射** — 半音尺度线性映射（原声音域 → 目标音域），保持音程不变不跑调
- **双输出** — 主输出 + 可选副输出（虚拟音频设备）
- **参数结构统一** — `InferenceParams` 分组对象（voice/f0/buffer/audio），实时/离线共用

## 系统要求

- **推理/训练 GUI**：Windows 10/11（ffmpeg 按硬编码 Windows 路径解析、托盘依赖 Win32）
- **云训练向导（autodl/）**：Linux 可用——ffmpeg 按环境变量 `RVC_FFMPEG` → 系统 PATH → `assets/ffmpeg/ffmpeg.exe` 顺序定位
- Python 3.13+
- NVIDIA GPU（CUDA 支持；无 GPU 时启动即报中文错误，无 CPU 回退——刻意行为）

## 安装

```bash
git clone <repo-url>
cd RVC
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

训练功能需要以下预训练权重（放在 `assets/` 对应目录）：

- `assets/rmvpe/rmvpe.pt` — RMVPE F0 提取器
- `assets/hubert/base/` 和 `assets/hubert/chinese/` — HuBERT 特征提取器
- `assets/pretrained/f0G48k.pth` / `f0D48k.pth` — 预训练底模（可选，加速收敛）

## 使用

```bash
# 推理 GUI
.venv\Scripts\python.exe app.py --infer

# 训练 GUI
.venv\Scripts\python.exe app.py --train

# 云训练向导（交互式，AutoDL 等云 GPU）
.venv\Scripts\python.exe autodl_train.py
# 等价于: python -m autodl
```

或使用启动脚本：`start-infer.bat`（推理）、`start-train.bat`（训练）。

## 架构概览

```
app.py                    # 入口：CLI 参数解析 → 启动推理/训练 GUI
├── rvc/                  # 核心引擎（无 GUI 依赖，可独立使用）
│   ├── core/             # 基础层：配置 dataclass、常量、异常定义
│   ├── runtime/          # 运行时：设备管理、CUDA Graph（合并捕获/重放）、路径配置、通用 LRU
│   ├── dsp/              # 纯信号处理（无 torch/IO 依赖）：SOLA 拼接、RMS 混合、mel 频谱、Hz↔MIDI
│   ├── io/               # 音频 IO：文件加载/写入（ffmpeg 路径显式传参）、设备查询
│   ├── models/           # 神经网络模型：HuBERT、合成器（encoder/decoder/flow）、RMVPE
│   ├── nn/               # 通用神经网络组件（注意力、判别器等）
│   ├── pipeline/         # 推理管线：F0 提取/后处理、特征提取、合成、模型会话与缓存
│   │   └── pitch/        # 音高子模块：提取器、后处理（音域映射/中值滤波）、跟踪器
│   ├── streaming/        # 流处理：VoiceEngine 门面、运行器、音频流、输出路由、RMS/SOLA 效果处理
│   └── train/            # 训练管线：预处理、F0/特征提取、训练器、checkpoint、损失
├── gui/                  # PySide6 GUI 层（View 不得 import rvc——依赖单向约束）
│   ├── infer/            # 推理 GUI
│   │   ├── view/         # 视图：主窗口（window.py 装配 + lifecycle.py 生命周期）、控件、托盘、页签
│   │   ├── controller/   # 控制器：引擎控制、遥测快照、设备目录、离线转换
│   │   └── state/        # 状态：参数绑定、配置持久化
│   ├── train/            # 训练 GUI
│   │   ├── view/         # 视图：主窗口、控件、训练/设置/工具页签
│   │   ├── controller/   # 控制器：训练控制器、训练工作线程
│   │   └── state/        # 状态：训练 GUI 状态
│   ├── configs/          # 通用配置读写
│   └── styles/           # QSS 主题与样式
├── autodl/               # 云 GPU 训练向导（交互式 CLI，chdir 在 main() 内执行）
│   ├── logger.py         # 控制台+文件双写日志
│   ├── prompts.py        # 交互式提问工具
│   ├── envcheck.py       # 环境体检（依赖/GPU/ffmpeg/权重），ffmpeg 定位结果显式返回
│   ├── wizard.py         # 参数向导
│   ├── steps.py          # 训练步骤执行
│   └── __main__.py       # 主流程编排
├── autodl_train.py       # 云训练向导薄包装入口（等价 python -m autodl）
```

### 依赖方向（八层单向，无环）

```
gui/  autodl/                    # 最上层（互相独立）
  ↓
rvc/streaming   rvc/train        # 引擎门面 / 训练管线
  ↓
rvc/pipeline                     # 推理管线
  ↓
rvc/models        rvc/io         # 网络模型 / 音频 IO
  ↓
rvc/nn                           # 通用网络组件
  ↓
rvc/dsp                          # 纯信号处理
  ↓
rvc/runtime                      # 设备/图/路径/缓存
  ↓
rvc/core                         # 配置/常量/异常（最底层）
```

- `rvc/core` 无内部依赖；`rvc/` 不依赖 `gui/`，可作为库独立使用
- `gui/` 只通过 `rvc/streaming.VoiceEngine` 与 `rvc/pipeline` 与核心交互，View 层经 Controller 取数（`snapshot()` 遥测快照），禁止直接 import rvc
- `autodl/` 只通过 `rvc/train` 与训练管线交互
- 依赖方向为设计约定：单向无环，新增代码必须遵守（View 层禁 import rvc 由代码评审把关）

### 关键数据流（实时推理）

```
麦克风 → AudioStream(输入) → Runner(分块) → Pipeline
  ├─ F0Extractor → pitch/postprocess(音域映射+中值滤波) → coarse F0
  ├─ HuBERT → 语义特征
  └─ Synthesizer(F0+特征) → 合成音频
→ dsp/sola(SOLA 拼接) → dsp/rms(RMS 音量混合) → AudioStream(输出) → 扬声器
```

## 目录结构说明

| 目录 | 职责 |
|---|---|
| `rvc/core/` | `config.py`（InferenceParams/OfflineParams/TrainConfig 等 dataclass）、`constants.py`（采样率/阈值等常量）、`errors.py`（异常类型） |
| `rvc/runtime/` | `device.py`（设备管理）、`graph.py`（CUDA Graph 捕获/重放）、`caches.py`（通用 LRU）、`paths.py`（路径配置） |
| `rvc/dsp/` | `sola.py`（SOLA 时长对齐拼接）、`rms.py`（RMS 音量混合）、`mel.py`（mel 频谱）、`hz_midi.py`（Hz↔MIDI 换算）——纯函数、无 torch/IO 依赖 |
| `rvc/io/` | `audio_file.py`（ffmpeg 解码加载，`ffmpeg_exe` 显式传参）、`wav_file.py`（WAV 读写/元信息）、`devices.py`（音频设备查询） |
| `rvc/pipeline/` | `pipeline.py`（推理管线主体）、`state.py`（引擎状态 dataclass）、`context.py`（上下文字段）、`features.py`（HuBERT 特征）、`synthesis.py`（合成器调用）、`sessions.py`（模型会话管理）、`cache.py`（模型缓存）、`loader.py`（模型加载） |
| `rvc/pipeline/pitch/` | `extractor.py`（F0 提取器抽象层，RMVPE/FCPE）、`postprocess.py`（音域映射/中值滤波/归一化）、`tracker.py`（音高跟踪） |
| `rvc/streaming/` | `engine.py`（`VoiceEngine` 门面，实时/离线共用）、`runner.py`（推理循环）、`stream.py`（音频流管理）、`output.py`（输出路由）、`effects.py`（RMS 响度混合 / SOLA） |
| `rvc/nn/` | `attentions.py`/`modules.py`/`discriminator.py`（原版拷贝）、`vits_blocks.py`（VITS 通用数值块，原 commons.py） |
| `rvc/train/` | `trainer.py`（训练主循环，`export_dir` 显式参数）、`preprocess.py`（切片/重采样）、`extract_f0.py`/`extract_feature.py`（批量提取）、`checkpoint.py`（保存/加载/淘汰/合并）、`losses.py`、`dataset.py`（TrainSample/TrainBatch 数据集与 collate）、`mel_processing.py` |
| `gui/infer/` | 推理 GUI：window.py（装配）/lifecycle.py（生命周期）/页签 + 引擎/设备/离线控制器 + 遥测快照 + 参数绑定 |
| `gui/train/` | 训练 GUI：窗口/页签 + TrainController + 训练工作线程 + 训练状态 |

## 设计决策（为什么）

- **训练数据集用 `NamedTuple` 而非 dataclass** — DataLoader 的 `pin_memory=True` 只对 namedtuple 逐元素重建并钉页（torch 内部按 `_fields` 识别）；dataclass 会被原样返回导致钉页**静默失效**，`non_blocking` 拷贝语义随之漂移。
- **`load_model` 入口无条件清 F0 提取器的 CUDA Graph 缓存**（含命中缓存快路径）— 图内编译的是旧张量地址；且离线场景 `inference_cache=None` 时管线内部回退到全局 `default_inference_cache`，因此对「显式 cache 或全局单例」统一清理，避免离线切模型残留旧图。
- **CUDA Graph：张量走参数、标量走闭包** — 捕获/回放要求张量地址静态，标量（`skip_head` 等）若作为参数传入会用外部变量地址而非静态地址，导致声音沙哑/失真；`return_length2` 随 formant 变化、经闭包编译进图，故必须进 `graph_key`，否则错误命中旧图（输出长度错误、爆音）。
- **F0 每块从完整缓冲区重提取（取前 p_len 帧）** — 旧滚动缓存方案的 F0 窗口与 HuBERT 输入窗口长度不同会错位；现在两路输入完全一致、帧天然对齐，取前 p_len 帧是特征侧（upsample 后取前 p_len 帧）的对齐要求。
- **惰性导入** — GUI 启动路径不 import torch，窗口秒开；torch 后台预热，首次点「开始」不卡顿。
- **无 GPU 直接报中文错误、不做 CPU 回退** — 刻意行为：实时链路的数值路径按 CUDA 设计，CPU 回退既慢又会掩盖配置问题。
- **`default_inference_cache` 保持全局单例** — 实时/离线共用一份图缓存是有意设计，改成实例归属会破坏跨管线复用。
- **八层单向依赖（`gui/autodl → rvc/streaming|train → … → rvc/core`）** — `rvc/` 不依赖 GUI，可作为库独立使用；View 层经 Controller 取数、禁止 import rvc，保证引擎可脱离 GUI 测试与复用。
- **魔法数具名化但值一律不变**（`>128`/`>=64`/预热 30） — 提升可读性而不改数值行为；`crossfade_time=0.04` 是 dataclass 字段默认值而非算法常数，故不具名化。

## 云训练

详见 `autodl/` 包。交互式向导自动完成环境检查、参数配置、预处理、F0/特征提取、训练全流程，支持断点续训。ffmpeg 定位结果显式贯穿全部步骤（无进程内全局重定向）。
