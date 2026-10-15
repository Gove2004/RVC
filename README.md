# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 彻底重写的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟实时显示（块时长 + 推理耗时，瞬时值；与麦克风对拍实测口径一致）
- **F0 双后端** — FCPE（实时首选，mel 前端为仓库内无同步复刻）与 RMVPE 可切换
- **快速启动** — 惰性导入架构，窗口秒开（~0.3s）；torch 后台预热，首次点「开始」不卡顿
- **CUDA Graph 加速** — HuBERT / F0 网络 / 合成器全链路图捕获（FCPE 的 mel 前端刻意 eager，见 pipeline 注释），开流前静音预热完成
- **系统托盘** — 关闭窗口最小化到托盘（变声不中断），托盘可开始/停止变声
- **离线推理** — 音频文件流式转换（复用实时全链路，显存封顶，任意长不 OOM）
- **模型训练** — 命令行向导（`app.py --train`）：环境体检 → 预处理 → F0/特征提取 → 训练 → 导出，支持 40k/48k、中断续训、按指纹跳过已完成阶段；Windows / Linux 同一套流程
- **模型工具** — 推理 GUI「融合」/「详情」页签：模型合并、模型信息查看、改 zip 原名
- **音域映射** — 半音尺度线性映射（原声音域 → 目标音域），保持音程不变不跑调；超出模型音域的帧钳制到最近边界（保持有声，不变气声）
- **双输出** — 主输出 + 可选副输出（虚拟音频设备）
- **参数结构统一** — `InferenceParams` 分组对象（voice/f0/buffer/audio/texture），实时/离线共用

## 系统要求

- **推理 GUI**：Windows 10/11（托盘依赖 Win32；ffmpeg 优先取项目内 `assets/ffmpeg/ffmpeg.exe`）
- **训练命令行向导**：Windows / Linux 通用——ffmpeg 按环境变量 `RVC_FFMPEG` → 系统 PATH → `assets/ffmpeg/ffmpeg.exe`（仅 Windows）顺序定位
- Python 3.13+
- NVIDIA GPU（CUDA 支持；推理端无 GPU 时启动即报中文错误，无 CPU 回退——刻意行为）

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

# 训练命令行向导（Windows / Linux 通用；数据集路径可作第一个参数直接传入）
.venv\Scripts\python.exe app.py --train
.venv\Scripts\python.exe app.py --train D:\my_dataset
# 等价于: python -m rvc.train.cli [数据集目录]
```

向导会依次询问 11 项参数（直接回车 = 默认值），每项都在问题旁用括号标出**可选项与默认值**，
例如 `采样率（40k 更省显存、更快；48k 高频细节更多；可选 40k / 48k；默认 48k）: `：
数据集路径、产物根目录、实验名、采样率、HuBERT 特征器、batch size、学习率、训练轮次、
保存间隔、保留 checkpoint 组数、是否用预训练底模。

向导会自动跳过已完成的阶段（切片指纹一致则跳过预处理，F0/特征齐全则跳过提取，检测到
checkpoint 则自动续训）。重新预处理会连同旧 checkpoint 一起清除——这种情况会在
「配置确认」之前先明确告警。

**向导有存档**：首次运行各项默认值即内置默认值，问完后本次答案会写进
`assets/configs/train_cli_defaults.json`，作为下次运行的默认值（删掉该文件即恢复内置默认值；
命令行传入的数据集路径始终优先于存档；存档里越界/类型不符的项与当前机器上不存在的目录
一律丢弃回退默认）。

产物根目录默认按平台自适应：

| 平台 | 默认产物根目录 |
|---|---|
| 云实例（存在 `/root/autodl-tmp`） | `/root/autodl-tmp` |
| 其他（Windows / 本地 Linux） | 项目根 |

实验目录 = `<产物根目录>/logs/<实验名>`，导出模型 = `<产物根目录>/models/<实验名>`；
环境变量 `RVC_OUT_ROOT` 可覆盖默认根。训练中 Ctrl+C = 本轮跑完保存后退出，再按一次强退。

或使用启动脚本：`start-infer.bat`（推理）、`start-train.bat`（Windows 训练）、`start-train.sh`（Linux/macOS 训练）。

## 架构概览

```
app.py                    # 入口：--infer 启动推理 GUI；--train 启动训练命令行向导
├── rvc/                  # 核心引擎（无 GUI 依赖，可独立使用）
│   ├── core/             # 基础层：配置 dataclass、常量、异常定义
│   ├── runtime/          # 运行时：设备管理、CUDA Graph（合并捕获/重放）、路径配置、通用 LRU
│   ├── dsp/              # 纯信号处理（无 torch/IO 依赖）：SOLA 拼接、RMS 混合、mel 频谱、Hz↔MIDI
│   ├── io/               # 音频 IO：文件加载/写入（ffmpeg 路径显式传参）、设备查询
│   ├── models/           # 神经网络模型：HuBERT、合成器（encoder/decoder/flow）、RMVPE
│   ├── nn/               # 通用神经网络组件（注意力、判别器等）
│   ├── pipeline/         # 推理管线：F0 提取/后处理、特征提取、合成、模型会话与缓存
│   │   └── pitch/        # 音高子模块：提取器、后处理（音域映射/中值滤波）
│   ├── streaming/        # 流处理：VoiceEngine 门面、运行器、音频流、输出路由、RMS/SOLA 效果处理
│   └── train/            # 训练管线：预处理、F0/特征提取、训练器、checkpoint、损失
│       └── cli/          # 训练命令行向导（Windows/Linux 共用）：体检、参数向导、步骤执行
├── gui/                  # PySide6 GUI 层（View 不得 import rvc——依赖单向约束）
│   ├── view/             # 视图：主窗口（window.py 装配 + lifecycle.py 生命周期）、控件、托盘、页签
│   ├── controller/       # 控制器：引擎控制、遥测快照、设备目录、离线转换、模型工具
│   ├── state/            # 状态：参数绑定、配置持久化
│   ├── configs/          # 通用配置读写
│   └── styles/           # QSS 主题与样式
```

### 依赖方向（八层单向，无环）

```
gui/                             # 最上层（推理 GUI）
rvc/train/cli/                   # 最上层（训练命令行，与 gui 独立）
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
- `rvc/train/cli/` 只通过 `rvc/train` 与训练管线交互，不依赖 `gui/`
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
| `rvc/core/` | `config.py`（InferenceParams/OfflineParams/TrainConfig 等 dataclass）、`constants.py`（采样率、F0 阈值、训练预处理默认值等跨层常量）、`errors.py`（异常类型） |
| `rvc/runtime/` | `device.py`（设备管理）、`graph.py`（CUDA Graph 捕获/重放）、`caches.py`（通用 LRU）、`paths.py`（路径配置） |
| `rvc/dsp/` | `sola.py`（SOLA 时长对齐拼接）、`rms.py`（RMS 音量混合）、`mel.py`（mel 频谱）、`hz_midi.py`（Hz↔MIDI 换算）——纯函数、无 torch/IO 依赖 |
| `rvc/io/` | `audio_file.py`（ffmpeg 解码加载，`ffmpeg_exe` 显式传参）、`wav_file.py`（WAV 读写/元信息）、`devices.py`（音频设备查询） |
| `rvc/pipeline/` | `pipeline.py`（推理管线主体）、`state.py`（引擎状态 dataclass）、`context.py`（上下文字段）、`features.py`（HuBERT 特征）、`synthesis.py`（合成器调用）、`sessions.py`（模型会话管理）、`cache.py`（模型缓存）、`loader.py`（模型加载） |
| `rvc/pipeline/pitch/` | `extractor.py`（F0 提取器抽象层，RMVPE/FCPE）、`postprocess.py`（音域映射/中值滤波/归一化） |
| `rvc/streaming/` | `engine.py`（`VoiceEngine` 门面，实时/离线共用）、`runner.py`（推理循环）、`stream.py`（音频流管理）、`output.py`（输出路由）、`effects.py`（RMS 响度混合 / SOLA） |
| `rvc/nn/` | `attentions.py`/`modules.py`/`discriminator.py`（原版拷贝）、`vits_blocks.py`（VITS 通用数值块，原 commons.py） |
| `rvc/train/` | `trainer.py`（训练主循环，`export_dir` 显式参数）、`preprocess.py`（切片/重采样）、`extract_f0.py`/`extract_feature.py`（批量提取）、`checkpoint.py`（保存/加载/淘汰/合并/检视）、`losses.py`、`dataset.py`（TrainSample/TrainBatch 数据集与 collate）、`mel_processing.py` |
| `rvc/train/cli/` | `run.py`（主流程编排）、`envcheck.py`（环境体检：依赖/GPU/ffmpeg/权重/素材）、`wizard.py`（参数向导）、`defaults.py`（上次参数存档 → 下次默认值）、`steps.py`（四步执行）、`logger.py`（控制台+文件双写）、`prompts.py`（交互提问） |
| `gui/` | 推理 GUI：`view/`（window.py 装配、lifecycle.py 生命周期、tabs/ 六页签：设备/音色/参数/离线/融合/详情）+ `controller/`（引擎控制/设备目录/离线转换/模型工具）+ `state/`（参数绑定）+ `configs/` + `styles/` |

## 设计决策（为什么）

- **训练 = 纯命令行，推理 = GUI** — 训练是长时无人值守任务，问答式向导 + 逐步跳过已完成阶段，天然适合脚本化与远程（SSH）运行；需要图形交互的只有推理与模型工具，故留在 GUI。
- **训练数据集用 `NamedTuple` 而非 dataclass** — DataLoader 的 `pin_memory=True` 只对 namedtuple 逐元素重建并钉页（torch 内部按 `_fields` 识别）；dataclass 会被原样返回导致钉页**静默失效**，`non_blocking` 拷贝语义随之漂移。
- **`load_model` 入口无条件清 F0 提取器的 CUDA Graph 缓存**（含命中缓存快路径）— 图内编译的是旧张量地址；且离线场景 `inference_cache=None` 时管线内部回退到全局 `default_inference_cache`，因此对「显式 cache 或全局单例」统一清理，避免离线切模型残留旧图。
- **CUDA Graph：张量走参数、标量走闭包** — 捕获/回放要求张量地址静态，标量（`skip_head` 等）若作为参数传入会用外部变量地址而非静态地址，导致声音沙哑/失真；`return_length2` 随 formant 变化、经闭包编译进图，故必须进 `graph_key`，否则错误命中旧图（输出长度错误、爆音）。
- **F0 每块从完整缓冲区重提取（取前 p_len 帧）** — 旧滚动缓存方案的 F0 窗口与 HuBERT 输入窗口长度不同会错位；现在两路输入完全一致、帧天然对齐，取前 p_len 帧是特征侧（upsample 后取前 p_len 帧）的对齐要求。
- **惰性导入** — GUI 启动路径不 import torch，窗口秒开；torch 后台预热，首次点「开始」不卡顿。
- **无 GPU 直接报中文错误、不做 CPU 回退** — 刻意行为：实时链路的数值路径按 CUDA 设计，CPU 回退既慢又会掩盖配置问题。
- **`default_inference_cache` 保持全局单例** — 实时/离线共用一份图缓存是有意设计，改成实例归属会破坏跨管线复用。
- **八层单向依赖（`gui | rvc/train/cli → rvc/streaming|train → … → rvc/core`）** — `rvc/` 不依赖 GUI，可作为库独立使用；View 层经 Controller 取数、禁止 import rvc，保证引擎可脱离 GUI 测试与复用。
- **魔法数具名化但值一律不变**（`>128`/`>=64`/预热 30） — 提升可读性而不改数值行为；`crossfade_time=0.04` 是 dataclass 字段默认值而非算法常数，故不具名化。

## 训练

```bash
.venv\Scripts\python.exe app.py --train          # Windows
./start-train.sh                                 # Linux / macOS
```

交互式向导自动完成环境检查、素材统计、参数配置、预处理、F0/特征提取、训练与导出，
支持断点续训；每一步都按指纹/产物完整性决定是否跳过，重复运行不会重做已完成的工作。
ffmpeg 定位结果显式贯穿全部步骤（无进程内全局重定向），Windows 与 Linux 共用同一实现。
