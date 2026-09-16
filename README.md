# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 重度重构的实时变声工具，支持实时推理、离线转换和模型训练。

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

- Windows 11 / Linux
- Python 3.13+
- NVIDIA GPU（CUDA 支持）

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
│   ├── io/               # 音频 IO：文件加载/写入、设备查询
│   ├── pipeline/         # 推理管线：F0 提取/后处理、特征提取、合成、CUDA Graph、模型缓存
│   │   └── pitch/        # 音高子模块：提取器、后处理（音域映射/中值滤波）、跟踪器
│   ├── streaming/        # 实时流处理：引擎、运行器、音频流、输出、对齐、音量混合
│   ├── models/           # 神经网络模型：HuBERT、合成器（encoder/decoder/flow）、RMVPE
│   ├── nn/               # 通用神经网络组件（注意力、判别器等）
│   ├── runtime/          # 运行时：设备管理、CUDA Graph、路径配置
│   └── train/            # 训练管线：预处理、F0/特征提取、训练器、checkpoint、损失
├── gui/                  # PySide6 GUI 层
│   ├── infer/            # 推理 GUI
│   │   ├── view/         # 视图：主窗口、控件、托盘、参数页签
│   │   ├── controller/   # 控制器：引擎控制、设备管理、离线转换
│   │   └── state/        # 状态：参数绑定、配置持久化
│   ├── train/            # 训练 GUI
│   │   ├── view/         # 视图：主窗口、控件、训练/设置/工具页签
│   │   ├── controller/   # 控制器：训练工作线程
│   │   └── state/        # 状态：训练 GUI 状态
│   ├── configs/          # 通用配置读写
│   └── styles/           # QSS 主题与样式
├── autodl/               # 云 GPU 训练向导（交互式 CLI）
│   ├── logger.py         # 控制台+文件双写日志
│   ├── prompts.py        # 交互式提问工具
│   ├── envcheck.py       # 环境体检（依赖/GPU/ffmpeg/权重）
│   ├── wizard.py         # 参数向导
│   ├── steps.py          # 训练步骤执行
│   └── __main__.py       # 主流程编排
├── autodl_train.py       # 云训练向导薄包装入口（等价 python -m autodl）
└── tests/                # 纯函数单元测试（unittest，无需 pytest）
    └── run_tests.py      # 测试运行器
```

### 依赖方向

```
gui/ → rvc/streaming → rvc/pipeline → rvc/models → rvc/nn
                  ↘ rvc/io → rvc/core
                  ↘ rvc/runtime
autodl/ → rvc/train → rvc/pipeline → rvc/models
```

- `rvc/core` 无内部依赖，是最底层
- `rvc/` 不依赖 `gui/`，可作为库独立使用
- `gui/` 只通过 `rvc/streaming.RealtimeEngine` 和 `rvc/pipeline` 与核心交互
- `autodl/` 只通过 `rvc/train` 与训练管线交互

### 关键数据流（实时推理）

```
麦克风 → AudioStream(输入) → Runner(分块) → Pipeline
  ├─ F0Extractor → pitch/postprocess(音域映射+中值滤波) → coarse F0
  ├─ HuBERT → 语义特征
  └─ Synthesizer(F0+特征) → 合成音频
→ SOLA 拼接 → RMS 音量混合 → AudioStream(输出) → 扬声器
```

## 目录结构说明

| 目录 | 职责 |
|---|---|
| `rvc/core/` | `config.py`（InferenceParams/OfflineParams/TrainConfig 等 dataclass）、`constants.py`（采样率/阈值等常量）、`errors.py`（异常类型） |
| `rvc/io/` | `audio_file.py`（ffmpeg 解码加载）、`wav_file.py`（WAV 读写/元信息）、`devices.py`（音频设备查询） |
| `rvc/pipeline/` | `pipeline.py`（推理管线主体）、`state.py`（引擎状态 dataclass）、`features.py`（HuBERT 特征）、`synthesis.py`（合成器调用）、`cache.py`（模型缓存）、`loader.py`（模型加载）、`cuda_graph.py`（CUDA Graph 捕获） |
| `rvc/pipeline/pitch/` | `extractor.py`（F0 提取器抽象层，RMVPE/FCPE）、`postprocess.py`（音域映射/中值滤波/归一化）、`tracker.py`（音高跟踪） |
| `rvc/streaming/` | `engine.py`（RealtimeEngine 门面）、`runner.py`（推理循环）、`stream.py`（音频流管理）、`output.py`（输出路由）、`alignment.py`（时间戳对齐）、`loudness.py`（响度测量）、`mix.py`（SOLA/RMS 混合） |
| `rvc/models/` | `hubert.py`（HuBERT 封装）、`synthesizer_*.py`（合成器四件套）、`rmvpe/`（RMVPE 模型实现） |
| `rvc/train/` | `trainer.py`（训练主循环）、`preprocess.py`（切片/重采样）、`extract_f0.py`/`extract_feature.py`（批量提取）、`checkpoint.py`（保存/加载/淘汰）、`losses.py`、`data_utils.py`、`mel.py`/`mel_processing.py` |
| `gui/infer/` | 推理 GUI：窗口/控件/页签 + 引擎/设备/离线控制器 + 参数绑定 |
| `gui/train/` | 训练 GUI：窗口/控件/页签 + 训练工作线程 + 训练状态 |

## 测试

```bash
.venv\Scripts\python.exe -m tests.run_tests
```

覆盖音高后处理（音域映射/中值滤波/归一化）、RMS 音量混合、配置 dataclass 等纯函数，共 28 个测试用例。

## 云训练

详见 `autodl/` 包。交互式向导自动完成环境检查、参数配置、预处理、F0/特征提取、训练全流程，支持断点续训。
