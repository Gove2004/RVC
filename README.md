# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 彻底重写的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟实时显示（算法延迟 = 2×块时长 + HuBERT 未来上下文 + 推理耗时，默认不含驱动层；有实测校准值时经 `driver_latency_compensation` 写入才计入）
- **F0 双后端** — FCPE（实时首选，mel 前端为仓库内无同步复刻）与 RMVPE 可切换
- **快速启动** — 惰性导入架构，窗口秒开（~0.3s）；torch 后台预热，首次点「开始」不卡顿
- **CUDA Graph 加速** — HuBERT / F0 网络 / 合成器全链路图捕获（FCPE 的 mel 前端刻意 eager，见 pipeline 注释），开流前静音预热完成
- **系统托盘** — 关闭窗口最小化到托盘（变声不中断），托盘可开始/停止变声
- **离线推理** — 音频文件流式转换（复用实时全链路，**显存**封顶不随音频变长；输出块累积在内存里，内存占用与音频时长成正比）
- **模型训练** — 命令行向导（`app.py --train`）：环境体检 → 预处理 → F0/特征提取 → 训练 → 导出，支持 40k/48k、中断续训、按指纹跳过已完成阶段；Windows / Linux 同一套流程
- **模型工具** — 推理 GUI「融合」/「详情」页签：模型合并、模型信息查看（选定即自动解析）
- **音域映射** — 半音尺度线性映射（原声音域 → 目标音域），音程按恒定斜率等比缩放（k<1 时压缩音高动态，刻意行为）；超出模型音域的帧钳制到最近边界（保持有声，不变气声）
- **双输出** — 主输出 + 可选副输出（虚拟音频设备）
- **参数结构统一** — `InferenceParams` 分组对象（voice/f0/buffer/audio），实时/离线共用

## 系统要求

- **推理 GUI**：Windows 10/11（Windows 限定来自回收站 Shell API `SHFileOperationW` 与 WASAPI 虚拟声卡用法；托盘是纯 Qt `QSystemTrayIcon`）
- **ffmpeg 定位**（单源 `rvc/io/audio_file.py:locate_ffmpeg`）：环境变量 `RVC_FFMPEG` → 系统 PATH → 项目内 `assets/ffmpeg/ffmpeg.exe`（第三级仅 Windows）
- **训练命令行向导**：Windows / Linux 通用（同上查找链；非 Windows 平台请走系统 PATH 或 `RVC_FFMPEG`）
- **Python 3.13** — 开发与验证环境（3.13.7 + torch 2.11.0+cu128）。仓库没有 `pyproject.toml`
  / `python_requires`，解释器版本不被强制；代码本身没有 3.13 专属语法，但更低版本未经实测，
  安装依赖时若不满足 wheel 要求再回退即可
- NVIDIA GPU（CUDA 支持；推理端无 GPU 时启动即报中文错误，无 CPU 回退——刻意行为）

## 安装

```bash
git clone <repo-url>
cd RVC
python -m venv .venv
# torch 必须先按 CUDA 12.8 索引安装：PyPI 上的默认 wheel 是 CPU 版，
# 直接装 requirements.txt 会得到无 CUDA 的 torch（详见 requirements.txt 顶部）
.venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip.exe install -r requirements.txt
# 校验：应输出 12.8，若为 None 说明装成了 CPU 版
.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda)"
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

## 测试

```bash
.venv\Scripts\python.exe -m unittest discover tests -v
```

快、稳、无需 GPU/模型（dsp 换算、音域映射双后端等价性、F0 量化与分桶边界、
推理缓存与 CUDA Graph 缓存的淘汰时机、训练链的静默失败点、离线块循环、GUI 离屏
构建与工具队列），秒级跑完；需要真实模型/设备的端到端验证不走这里，靠实机确认。

## 架构概览

```
app.py                    # 入口：--infer 启动推理 GUI；--train 启动训练命令行向导
├── rvc/                  # 核心引擎（无 GUI 依赖，可独立使用）
│   ├── core/             # 基础层：配置 dataclass、常量、异常定义
│   ├── runtime/          # 运行时：设备管理、CUDA Graph（合并捕获/重放）、路径配置、通用 LRU
│   ├── dsp/              # 纯信号处理（无 IO、无模型依赖；sola/rms/hz_midi 直接吃 torch 张量，mel 为 numpy）
│   ├── io/               # 音频 IO：文件加载/写入（ffmpeg 路径显式传参）、设备查询
│   ├── models/           # 神经网络模型：HuBERT、合成器（encoder/decoder/flow）、RMVPE
│   ├── nn/               # 通用神经网络组件（注意力、判别器等）
│   ├── pipeline/         # 推理管线：F0 提取/后处理、特征提取、合成、模型会话与缓存
│   │   └── pitch/        # 音高子模块：提取器、后处理（音域映射/中值滤波）
│   ├── streaming/        # 流处理：VoiceEngine 门面、运行器、音频流、输出路由、RMS/SOLA 效果处理
│   │   └── offline.py    # 离线文件的纯步骤：读音频 / 逐块喂实时管线 / 写 wav
│   └── train/            # 训练管线：预处理、F0/特征提取、训练器、checkpoint、损失
│       └── cli/          # 训练命令行向导（Windows/Linux 共用）：体检、参数向导、步骤执行
├── gui/                  # PySide6 GUI 层（View 不得 import rvc——依赖单向约束）
│   ├── view/             # 视图：主窗口（window.py 装配 + lifecycle.py 生命周期）、控件、托盘、页签
│   ├── controller/       # 控制器：引擎控制、遥测快照、设备目录、离线转换、模型工具
│   ├── state/            # 状态：控件 ↔ InferenceParams 参数绑定（持久化在 gui/configs）
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
- `gui/` 的 rvc 依赖集中在 Controller / State / Configs 三层（`VoiceEngine`、`rvc.pipeline.cache`、`rvc.io.devices`、`rvc.core.config`、`rvc.train.checkpoint` 等）；View 层只用 `TYPE_CHECKING` 引用 rvc 类型，运行时不直接 import，取数一律经 Controller（如 `snapshot()` 遥测快照）
- `rvc/train/cli/` 只通过 `rvc/train` 与训练管线交互，不依赖 `gui/`
- 依赖方向为设计约定：单向无环，新增代码必须遵守——由 `tests/test_layering.py` 的 AST 静态检查自动守卫（rvc 不 import gui、View 运行时不 import rvc、包级依赖必须向下且无环、新包必须先登记 `RANKS`）

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
| `rvc/runtime/` | `device.py`（设备管理）、`graph.py`（CUDA Graph 捕获/重放）、`lru.py`（通用 LRU）、`paths.py`（路径配置） |
| `rvc/dsp/` | `sola.py`（SOLA 时长对齐拼接）、`rms.py`（RMS 音量混合）、`mel.py`（mel 频谱，numpy）、`hz_midi.py`（Hz↔MIDI，torch/numpy 双后端）——纯函数、无 IO 与模型依赖 |
| `rvc/io/` | `audio_file.py`（ffmpeg 解码加载，`ffmpeg_exe` 显式传参）、`wav_file.py`（WAV 读写/元信息）、`devices.py`（音频设备查询）、`atomic.py`（原子写 + 坏 JSON 备份回退，全仓单源） |
| `rvc/pipeline/` | `pipeline.py`（推理管线主体）、`state.py`（引擎状态 dataclass）、`context.py`（上下文字段）、`features.py`（HuBERT 特征）、`synthesis.py`（合成器调用）、`sessions.py`（模型会话管理）、`cache.py`（模型缓存）、`loader.py`（模型加载） |
| `rvc/pipeline/pitch/` | `extractor.py`（F0 提取器抽象层，RMVPE/FCPE）、`postprocess.py`（音域映射/中值滤波/归一化；torch 与 numpy 共用**一份**数学实现，后端差异经 `_Ops` 注入） |
| `rvc/models/` | `synthesizer.py`（合成器基类 + v2 变体：带 F0 / 无 F0）、`synthesizer_encoder.py`/`synthesizer_decoder.py`/`synthesizer_flow.py`（子模块）、`hubert.py`（特征器，transformers 惰性导入）、`rmvpe/`（F0 模型子包） |
| `rvc/streaming/` | `engine.py`（`VoiceEngine` 门面，实时/离线共用）、`runner.py`（推理循环）、`stream.py`（音频流管理）、`output.py`（主输出写入 + 副输出块队列 `Out2Queue`）、`effects.py`（RMS 响度混合 / SOLA）、`passthrough.py`（原声直通流，不经模型）、`offline.py`（离线文件的读/pad 切块/写，纯步骤可单测） |
| `rvc/nn/` | `attentions.py`/`modules.py`/`discriminator.py`（原版拷贝）、`vits_blocks.py`（VITS 通用数值块，原 commons.py） |
| `rvc/train/` | `trainer.py`（训练主循环，`export_dir` 显式参数）、`preprocess.py`（切片/重采样）、`extract_f0.py`/`extract_feature.py`（批量提取）、`checkpoint.py`（保存/加载/淘汰/合并/检视）、`losses.py`、`dataset.py`（三级缓存 + 分桶采样 + collate）、`mel_processing.py`、`atomic_save.py`（中间产物原子写，避免半截文件被当成有效缓存） |
| `rvc/train/cli/` | `run.py`（主流程编排）、`envcheck.py`（环境体检：依赖/GPU/ffmpeg/权重/素材）、`wizard.py`（参数向导）、`defaults.py`（上次参数存档 → 下次默认值）、`steps.py`（四步执行）、`logger.py`（控制台+文件双写）、`prompts.py`（交互提问） |
| `gui/` | 推理 GUI：`view/`（window.py 装配、lifecycle.py 生命周期、host_impl.py 控制器契约的控件实现、tabs/ 六页签：设备/音色/参数/离线/融合/详情）+ `controller/`（引擎控制/设备目录/离线转换/模型工具/回收站）+ `state/`（参数绑定）+ `configs/` + `styles/` |
| `tests/` | 标准库 unittest（不用 pytest）。覆盖纯函数契约（Hz↔MIDI、音域映射、RMS、SOLA、F0 量化/分桶）、推理缓存与图缓存生命周期、训练链静默失败点、离线块循环与副输出队列、GUI 离屏构建与工具队列，以及**分层契约**（AST 单向无环）；全部无需 GPU/模型，秒级跑完 |
| `tools/` | 离线调试小工具（`pitch_map_demo.html` 音域映射演示页），不参与构建、不被任何代码 import |

## 设计决策（为什么）

- **训练 = 纯命令行，推理 = GUI** — 训练是长时无人值守任务，问答式向导 + 逐步跳过已完成阶段，天然适合脚本化与远程（SSH）运行；需要图形交互的只有推理与模型工具，故留在 GUI。
- **训练数据集用 `NamedTuple` 而非 dataclass** — DataLoader 的 `pin_memory=True` 只对 namedtuple 逐元素重建并钉页（torch 内部按 `_fields` 识别）；dataclass 会被原样返回导致钉页**静默失效**，`non_blocking` 拷贝语义随之漂移。
- **`load_model` 刻意不清 F0 提取器的 CUDA Graph** — F0 提取器与 pth 无关，其图跨重启复用是安全的；各场景兜底：换 F0 方法走 `switch_f0_method→release` 自清，换模型走淘汰侧回收，改阈值/块时长经签名 key 自动重捕获。
- **CUDA Graph：张量走参数、标量走闭包** — 捕获/回放要求张量地址静态，标量（`skip_head` 等）若作为参数传入会用外部变量地址而非静态地址，导致声音沙哑/失真；`return_length2` 随 formant 变化、经闭包编译进图，故必须进 `graph_key`，否则错误命中旧图（输出长度错误、爆音）。
- **性别因子（formant）运行中可调，代价挪到启动** — `return_length2` 决定合成器的输出形状，一个整数帧数就是一张图。量程内的档位只有 10 个左右（250ms 块 / 40kHz），启动时整段预捕获约 2.5s、显存 +79MB（RTX 5060 Laptop 实测），之后拖动滑条只是切换回放的图，音频回调不产生捕获。离线转换只用一个档位，不预捕获。
- **调 formant 不改音高** — 合成器按 `return_length2` 帧合成时会把谐波源 `interpolate` 到 `n*upp`，这次时长拉伸恰好抵掉重采样回 `return_length` 帧的升频；若再在 F0 上乘一次比例，音高会被抬一遍（实测 +2.5 档：396Hz → 460Hz）。所以 `pitchf` 原样进合成器，实测三档输出基频一致（396.04Hz）、频谱质心随档位移动（1173 → 1378 / 1031Hz）。
- **GUI 侧不留长时主线程等待** — 模型工具（合并/检视）是入队 + 队列泵：正在跑就排队、调用立刻返回，由 `finished` 拉起下一项；只有退出路径允许 `wait()`（否则进程挂死）。
- **F0 每块重提取，但两条路径的窗口不同** — RMVPE 走整段缓冲区，取前 p_len 帧；FCPE 只算尾部 0.6s 窗口（前段 F0 永远用不上），帧数不足 p_len 时**右对齐补零**到 p_len。对齐规则写在 `_sync_f0_parallel`：整段窗口取前 p_len 帧、尾窗补零到 p_len，两者都保证末帧落在下标 `p_len-1`。
- **F0 后处理：双后端一份实现** — 训练（numpy）与推理（torch）必须逐值一致，所以 `apply_pitch_map` / `normalize_f0_to_coarse` 只保留一份数学表达式，`clip/where/nan_to_num/取整` 经 `_Ops` 注入；等价性由单测锁定。
- **惰性导入** — GUI 启动路径不 import torch，窗口秒开；torch 后台预热，首次点「开始」不卡顿。同一条理由下 `rvc/models/hubert.py` 对 transformers 也做惰性导入：惰性化后 `import rvc.pipeline.pipeline` 为 0.11s，而一次性拉起 transformers（→sklearn→torch._dynamo）要 21s，训练 CLI 与单测都直接受益。
- **无 GPU 直接报中文错误、不做 CPU 回退** — 刻意行为：实时链路的数值路径按 CUDA 设计，CPU 回退既慢又会掩盖配置问题。
- **`default_inference_cache` 保持全局单例** — 实时/离线共用一份图缓存是有意设计，若改成实例归属，跨管线复用就没了。
- **模型缓存的回收点在「淘汰」，不在「加载」** — 命中缓存时拿到的就是上次那个对象，它的图仍然有效；在那里 purge 等于每次切回旧模型都重捕获一遍。`LRUCache.set` 把被淘汰的值回报给调用方，由 `InferenceCache` 显式回收（模型清图、提取器 `release()`）。
- **训练分桶采样不丢桶尾** — 语料量不是 `batch_size` 整数倍时，按 `range(0, len - len % bs, bs)` 切会把桶尾样本每轮静默跳过且无提示；正确做法是把各桶尾部合并成尾批，`__len__` 与批次数严格相等（`ceil(total/bs)`）。
- **样本读盘三级缓存** — ffmpeg 解码 → `wav.npy`；STFT → `.spec.pt`；每轮重复读取 → 进程内按字节预算的样本 LRU（训练总预算默认 256MB，按 worker 数均分，`0` 关闭）。单样本省约 4ms，代价是显存之外的内存占用需显式预算。
- **训练 DataLoader 内存策略** — 默认保持 2 个持久 worker、每 worker 只预取 1 个 batch；样本 LRU 的总预算默认 256MB，按 worker 数均分。Windows 或低内存环境可设置 `RVC_TRAIN_WORKERS=0` 关闭 worker，`RVC_TRAIN_CACHE_MB=0` 关闭样本内存缓存，`RVC_TRAIN_PREFETCH` 可调整预取数。
- **八层单向依赖（`gui | rvc/train/cli → rvc/streaming|train → … → rvc/core`）** — `rvc/` 不依赖 GUI，可作为库独立使用；View 层经 Controller 取数、禁止 import rvc，保证引擎可脱离 GUI 测试与复用。
- **魔法数具名化但值一律不变**（`>128`/`>=64`/预热 10） — 提升可读性而不改数值行为；`crossfade_time=0.05` 是 dataclass 字段默认值而非算法常数，故不具名化（SOLA 缓冲实际取 `min(crossfade_samples, 50ms)`）。

## 训练

```bash
.venv\Scripts\python.exe app.py --train          # Windows
./start-train.sh                                 # Linux / macOS
```

交互式向导自动完成环境检查、素材统计、参数配置、预处理、F0/特征提取、训练与导出，
支持断点续训；每一步都按指纹/产物完整性决定是否跳过，重复运行不会重做已完成的工作。
ffmpeg 定位结果显式贯穿全部步骤（无进程内全局重定向），Windows 与 Linux 共用同一实现。
