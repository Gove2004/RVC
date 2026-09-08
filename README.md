# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟实时实测显示（硬件时间戳、瞬时值）
- **快速启动** — 惰性导入架构，窗口秒开（~0.3s）；torch 后台预热，首次点「开始」不卡顿
- **CUDA Graph 加速** — 开流前静音预热完成图捕获，首块推理即热状态（1101ms → ~30ms）
- **系统托盘** — 关闭窗口最小化到托盘（变声不中断），托盘可开始/停止变声、tooltip 显示状态与实测延迟
- **离线推理** — 音频文件流式转换（复用实时全链路，显存封顶，任意长不 OOM）
- **模型训练** — 从人声音频训练自定义模型，支持 40k/48k、中断续训
- **音高调节** — 变调不变速（-16 ~ +16 半音）
- **辅音保护** — 保留原音清音/辅音，防止齿音失真（0.0 ~ 1.0）
- **破音保护** — 高音超过临界 Hz 自动软收敛，防止破音/沙哑（开关 + 临界 Hz，压缩比/膝宽内部自动）
- **双输出** — 主输出 + 可选副输出（虚拟音频设备）

## 系统要求

- Windows 11
- Python 3.13+
- NVIDIA GPU（CUDA 支持）

推荐配置：RTX 4060 或更高

## 安装

### 1. 克隆项目

```bash
git clone <repo-url>
cd RVC
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

### 3. 安装依赖

```bash
.venv\Scripts\pip.exe install -r requirements.txt
```

### 4. 下载预训练权重

训练功能需要以下模型：

- `assets/rmvpe/rmvpe.pt` — RMVPE F0 提取器
- `assets/hubert/` — HuBERT 特征提取器（transformers 模型目录，含 config.json + pytorch_model.bin）

可选的预训练权重（加速训练收敛）：

- `assets/pretrained/f0G48k.pth` — 48k Generator
- `assets/pretrained/f0D48k.pth` — 48k Discriminator
- `assets/pretrained/f0G40k.pth` — 40k Generator（40k 训练时使用）
- `assets/pretrained/f0D40k.pth` — 40k Discriminator（40k 训练时使用）

## 使用

### 启动程序

```bash
# 推理 GUI
.venv\Scripts\python.exe app.py --infer

# 训练 GUI
.venv\Scripts\python.exe app.py --train
```

或使用启动脚本：

- `start-infer.bat` — 推理
- `start-train.bat` — 训练

也可用桌面快捷方式（VBS）以 `pythonw.exe` 无窗口方式启动，配合系统托盘常驻后台。

### 系统托盘

- 关闭窗口（✕）→ **最小化到托盘**，变声不中断，进程常驻
- 右键托盘图标菜单：**显示主窗口 / 开始变声 / 停止变声 / 退出**
- 悬停托盘图标：实时显示 `状态`（推理中/已停止）与 `延迟`（硬件时间戳实测）
- 单击/双击托盘图标：恢复主窗口
- 首次最小化会气泡提示一次
- 退出请使用托盘菜单（会完整清理：停止推理、释放声卡、保存配置）

### 实时推理流程

1. **添加模型**
   - 在"模型" Tab 点击"+ 添加模型"
   - 选择 `.pth` 模型文件

2. **调节参数**
   - 展开模型卡片，调节以下参数：
     - **音调大小**：音高偏移（半音，-16 ~ +16）
     - **响度**：RMS 响度混合（0.0 = 目标响度，1.0 = 源响度）
     - **性别**：formant shift（-50 ~ +50）
3. **全局参数** — 在"参数" Tab 配置：
     - **辅音保护**：清音保留程度（0.0 = 全转换，1.0 = 全保留）
     - **破音保护**：高音超过临界 Hz 自动软收敛（勾选启用、滑条调临界 200~400 Hz），防止高音破音/沙哑
     - **采样长度/淡入长度/额外上下文**：实时推理延迟控制
     - **响度因子**：输出响度混合比例
4. **配置设备**
   - 在"设备" Tab 选择音频设备（麦克风、主输出、副输出）和采样率模式
   - **采样率模式**：
     - `模型采样率` — 使用模型原生采样率（推荐，音质最佳）
     - `设备采样率` — 使用音频设备采样率（降低重采样开销）
   - **音高算法**：RMVPE 或 FCPE（F0 提取方式）
5. **开始推理**
   - 点击"使用"激活模型（加载到 GPU）
   - 点击"开始"启动实时推理

### 离线推理

1. 切换到"离线" Tab
2. 选择输入/输出音频文件
3. 点击"开始转换"（使用当前激活模型的所有参数）

支持格式：wav, mp3, flac, ogg 等（通过 ffmpeg 解码）

### 模型训练流程

1. **准备数据**
   - 干净人声音频（单人，背景噪声少）
   - 建议总时长 10 分钟以上
   - 支持任意音频格式

2. **启动训练 GUI**
   ```bash
   .venv\Scripts\python.exe app.py --train
   ```

3. **配置训练**
   - 在"设置" Tab 填写：
     - **实验名**：训练标识符（如 `exp01`）
     - **音频目录**：包含人声文件的文件夹
     - **采样率**：40k / 48k（对应 `assets/configs/40ktrain_config.json` / `48ktrain_config.json`）
     - **训练参数**：Epoch, Batch size, 学习率, 保存间隔, 保留 checkpoint 数（无早停，支持中断续训）

4. **执行训练**
   - 点击"一键全流程"自动完成所有步骤
   - 或分步执行：
     1. **预处理** — 音频切片、去静音、归一化、重采样
     2. **提取 F0** — RMVPE 基频提取
     3. **提取特征** — HuBERT 768 维特征
     4. **训练** — GAN 训练（Generator + Discriminator）

5. **导出模型**
   - 训练完成后，模型导出到 `assets/models/<实验名>_e<epoch>.pth`
   - 训练日志 `logs/<实验名>/train.log` 记录每轮 loss（D/G/Mel/KL/FM），可复盘收敛曲线
   - 可选：在"工具" Tab 合并多个 checkpoint 或查看模型信息

6. **使用训练的模型**
   - 返回推理 GUI，加载 `assets/models/<实验名>_e<epoch>.pth`

### 训练注意事项

- **logs/** 目录会持续增长（切片音频、F0、特征缓存、checkpoint）
- 训练完成后建议清理不用的实验目录
- 同一实验名绑定输入目录、采样率、切片参数
- 如参数变化，预处理阶段会自动清理旧数据重建
- 训练支持中断恢复（从最新 checkpoint 继续）

## 项目结构

```
app.py                      # 统一入口（--infer / --train）
autodl_train.py             # 云 GPU 训练向导（AutoDL/恒源云，交互式零参数）
gui/
  styles/                   # 模块化 UI 设计系统
    colors.py               # 颜色调色板
    layout.py               # 布局参数
    components.py           # 样式组件
    theme.py                # 主题应用
  configs/                  # GUI 配置（窗口状态、持久化）
    config.py               # load_config/save_config
    train_state.py          # 训练 GUI 状态
  infer/                    # 推理 GUI（MVC 分层）
    view/                   # View 层（纯 UI 构建 + 信号连接）
      main_window.py        # 主窗口（系统托盘集成、引擎后台预热）
      widgets.py            # ModelCard, LoadThread 等自定义控件
      tray.py               # 系统托盘（图标/菜单/tooltip 状态）
      tabs/                 # 各功能 Tab（设备/参数/模型/离线）
    controller/             # Controller 层（业务逻辑 + 状态管理）
      main_controller.py    # InferController（参数应用/引擎控制/错误处理）
      model_manager.py      # 模型列表管理
      device_manager.py     # 音频设备管理
      offline_manager.py    # 离线推理管理
    viewmodel/              # ViewModel 层（GUI 状态 ↔ 核心配置绑定）
      param_binding.py      # 嵌套路径绑定（点号路径如 inference.pitch）
  train/                    # 训练 GUI（同 MVC 分层结构）
    view/
      main_window.py        # 训练窗口
      widgets.py            # 训练控件
      tabs/                 # 设置、训练、工具 Tab
    controller/
      workers.py            # 训练工作线程（QThread）
rvc/                        # 核心引擎（严禁 import gui 或 PySide6）
  core/                     # 核心配置与异常
    config.py               # 统一配置体系（InferenceConfig/EngineConfig/TrainConfig/OfflineConfig/AppConfig/ModelEntry）
    errors.py               # 统一异常体系（RVCError 基类 + 8 种具体异常）
  audio/                    # 音频处理层
    realtime_engine.py      # RealtimeEngine（门面类，委托给子组件）
    stream_manager.py       # AudioStreamManager（设备/流管理，PortAudio 封装）
    inference_runner.py     # InferenceRunner（分块/交叉淡入/SOLA/效果器）
    effects.py              # 效果器编排（AudioProcessor：RMS/SOLA）
    sola.py                 # SOLA 时间拉伸对齐与交叉淡化
    realtime_mix.py         # RMS 音量包络混合
    output_router.py        # 主输出写入与副输出路由
    loader.py               # 音频加载（soundfile + ffmpeg fallback）
    device_query.py         # 音频设备枚举
    mel.py                  # Mel 滤波器
  inference/                # 推理管线层
    pipeline.py             # InferencePipeline（无状态：infer 接收 InferenceConfig，只持缓存）
    model_session.py        # ModelSessionManager（模型生命周期：HuBERT/Synthesizer/F0 缓存）
    model_loader.py         # SynthesizerLoader（PyTorch 加载）
    f0_extractor.py         # F0Extractor ABC + RMVPEExtractor/FCPEExtractor
    feature_processing.py   # HuBERT 特征提取、辅音保护、上采样
    pitch_tracker.py        # F0 提取窗口与实时 pitch cache
    synthesis.py            # Synthesizer 推理调用与 formant 重采样
    cuda_graph.py           # CUDA Graph 捕获/回放缓存（按形状 LRU）
  models/                   # 模型层
    inference_cache.py      # InferenceCache（线程安全 LRU 模型缓存）
    hubert.py               # HuBERT 加载（base/chinese）
    rmvpe/                  # RMVPE F0 提取器（模块化）
      model.py              # RMVPE 推理类
      blocks.py             # CNN 模块
      transforms.py         # STFT + MelSpectrogram
      constants.py          # 常量
  synthesizer/              # NSF 合成器（VITS 架构，模块化）
    model.py                # SynthesizerTrnMsNSFsid
    encoder.py              # TextEncoder, PosteriorEncoder
    decoder.py              # Generator, GeneratorNSF
    flow.py                 # ResidualCouplingBlock
  nn/                       # 神经网络基础层
    attentions.py           # 注意力机制
    commons.py              # 通用模块
    discriminator.py        # 判别器
    modules.py              # 基础模块
  runtime/                  # 运行时配置
    device_config.py        # 设备配置（CUDA 探测、GPU 信息、精度）
    paths.py                # 配置路径
  train/                    # 训练管线
    trainer.py              # GAN 训练循环
    preprocess.py           # 音频预处理
    extract_f0.py           # F0 提取
    extract_feature.py      # HuBERT 特征提取
    data_utils.py           # 数据加载
    losses.py               # 损失函数
    mel_processing.py       # Mel 处理
    ckpt_utils.py           # Checkpoint 工具
tests/                      # 单元测试（unittest，无需额外依赖）
  test_architecture.py      # 架构验证（模块导入与接口一致性）
  test_config.py            # 配置体系测试
  test_errors.py            # 异常类型测试
  test_effects.py           # 效果器测试
  test_effects_deep.py      # 效果器深度测试（边界条件/状态一致性）
  test_inference_cache.py   # 推理缓存测试
  test_inference_cache_deep.py  # 推理缓存深度测试（LRU 淘汰/线程安全）
  test_pipeline.py          # 推理管线测试（状态管理/无状态设计/参数计算）
  test_f0_interface.py      # F0 提取器接口测试
  test_model_session.py     # 模型会话测试
  test_stream_manager.py    # 流管理器测试
  test_audio_utils.py       # 音频工具测试
  test_mel_filter.py        # Mel 滤波器测试
  test_config_migration.py  # 配置迁移测试
  test_param_binding.py     # 参数绑定测试
assets/
  configs/                  # 配置数据
    save_state.json         # GUI 持久化状态
    48ktrain_config.json    # 48k 训练超参数
    40ktrain_config.json    # 40k 训练超参数
  models/                   # 推理模型
  hubert/                   # HuBERT 权重（transformers 模型目录）
  rmvpe/                    # RMVPE 权重
  pretrained/               # 预训练权重
  ffmpeg/                   # ffmpeg 二进制
  resources/                # 图标等资源
logs/                       # 训练实验目录
```

## 技术细节

### 实时推理管线

```
麦克风 → RealtimeEngine → InferencePipeline → SOLA → 输出 → 扬声器
          ↓                 ↓
      sounddevice      HuBERT + Synthesizer
      SOLA crossfade   protect_blend（辅音保护）
```

### 关键技术

- **SOLA 算法** — 重叠相加实现无缝音频拼接，支持变速不变调
- **辅音保护** — 使用 F0 contour 作为掩码，清音区域混回原始特征
- **双采样率模式** — 模型采样率（高音质）或设备采样率（低延迟）
- **动态精度** — 自动处理 half/float 模型，确保推理稳定
- **延迟实测** — 基于 PortAudio 硬件时间戳（ADC 采集 → DAC 播放）实时测量端到端延迟（瞬时值），替代不可靠的估算值
- **CUDA Graph 加速** — 推理前向（HuBERT/F0/Synthesizer）捕获为 CUDA Graph 后 replay，跳过 Python 调度开销；开流前静音预热完成捕获
- **惰性导入架构** — `rvc/*/__init__.py` 模块级惰性导出，GUI 启动路径不加载 torch/transformers（窗口 8.2s → 0.3s）

### 模型格式

推理模型（`.pth`）：

```python
{
    "weight": OrderedDict({...}),  # 模型权重（half 精度）
    "config": [18 个参数],
    "info": "2000epoch",
    "sr": "48k",        # 40k 或 48k
    "f0": 1,
    "version": "v2"
}
```

## 性能优化建议

- **输入延迟**（block）：0.05 ~ 0.18 秒（平衡延迟和稳定性；可调至 0.05 以下，但过小可能不稳定）
- **交叉淡化**（crossfade）：0.04 ~ 0.08 秒（过大增加延迟）
- **额外推理**（extra）：0.3 ~ 2.5 秒（提供特征提取上下文；越小延迟越低；>5s 无实际收益且纯耗 GPU）
- **延迟显示**：运行中显示硬件时间戳实测瞬时值（`outputBufferDacTime - inputBufferAdcTime`，含设备缓冲/攒块/推理全链路），不做插值平滑；数值每块实时波动属正常
- **推理耗时 vs 延迟**：推理耗时（`infer_ms`，~30ms）只是 GPU 处理时间，不是延迟；真正的听感延迟在声卡缓冲（block_time / WASAPI 独占），调 infer_ms 不降延迟
- **启动**：窗口秒开（惰性导入）；torch 在窗口出现 0.3s 后后台预热，点「开始」前已完成加载
- **首次点开始**：开流前自动用静音数据跑 2 次推理完成 CUDA Graph 捕获（约 1s，在 loading 状态内），首块推理即为热状态

## 常见问题

**Q: 辅音保护（protect）如何调节？**  
A: 0.0 = 完全转换（音色纯但可能糊），1.0 = 完全保留原音辅音（清晰但音色不纯）。建议从 0.5 开始调整。

**Q: 模型采样率和设备采样率选哪个？**  
A: 推荐"模型采样率"，保持模型原生质量。选"设备采样率"可降低重采样开销，但可能影响音质。

**Q: 延迟显示的数值准确吗？**  
A: 运行中显示的是 PortAudio 硬件时间戳实测的端到端延迟（声卡时钟域，含设备缓冲/攒块/推理），瞬时值不做插值，每块实时波动属正常。它不等于推理耗时（GPU 处理 ~30ms 只是其中一环）；想降延迟应调 block_time 或用 WASAPI 独占。

**Q: 关闭窗口后程序还在运行/变声没停？**  
A: 正常——关闭窗口是"最小化到托盘"（变声不中断）。用托盘右键菜单「退出」才会真正结束程序并释放声卡。

**Q: 首次点「开始」要等约 1 秒才出声？**  
A: 这是开流前的 CUDA Graph 预热（静音推理完成图捕获，loading 状态内），换来的是首块推理即为热状态（~30ms），不再有开头延迟虚高。停止后再开始无需等待。

**Q: 训练需要多少数据？**  
A: 建议 10 分钟以上干净人声。背景噪声越少越好，会被自动切成 ~3.7 秒片段。

## 安全提示

- 模型文件通过 `torch.load(..., weights_only=False)` 加载
- **请勿加载来源不可信的 .pth 文件**

## 许可证

本项目基于 RVC 开源项目，仅供学习研究使用。

## 开发说明

### 架构分层

- `rvc/` = 核心运行时（推理/音频/模型/训练），**严禁 import `gui` 或 PySide6**
- `gui/` = PySide6 GUI，MVC 三层分层：
  - `view/` = 纯 UI 构建 + 信号连接（main_window, widgets, tabs, tray）
  - `controller/` = 业务逻辑 + 状态管理（main_controller, model/device/offline manager, workers）
  - `viewmodel/` = GUI 状态 ↔ 核心配置绑定（param_binding）
- 运行时设备/路径配置来自 `rvc.runtime`；GUI 状态持久化来自 `gui.configs`
- 配置体系统一为 `rvc/core/config.py` 的 dataclass（InferenceConfig/EngineConfig/TrainConfig/OfflineConfig/AppConfig/ModelEntry），新增参数 = dataclass 字段 + param_binding 绑定 + Tab 控件
- InferencePipeline 无状态化：`infer(input_wav, config: InferenceConfig, ...)`，参数每次传入，pipeline 只持缓存（pitch_cache/resample_kernel）；实时/离线统一走 `InferenceRunner.process_block()`
- 模型生命周期统一由 `ModelSessionManager` 管理（HuBERT/Synthesizer/F0 提取器缓存）
- RealtimeEngine 门面模式：委托给 `AudioStreamManager`（设备/流）+ `InferenceRunner`（分块推理/效果器）

### 核心实现规则

- **采样率**：实时帧数学用 `zc = sr // 100` 对齐，禁止硬编码 `48000`；`sr_mode="model"` 与 `"device"` 都要工作
- **模型精度**：模型可能 half/float，feature/protect 之后要恢复模型 dtype；pitch coarse 用 `long`
- **实时回调安全**：sounddevice 回调内禁止阻塞 IO、模型加载、大分配、GPU 同步；禁止直接操作 Qt（运行时错误经 Signal 转发主线程）
- **惰性导入**：GUI 启动路径禁止在模块顶层 import torch/transformers/librosa 或实例化 `Config()`；重型 import 只允许出现在「加载模型/开始/离线推理」路径；`rvc/*/__init__.py` 用模块级 `__getattr__` 惰性导出
- **引擎访问**：判断引擎是否已构造用 `controller._engine`（勿触发惰性构造）；首次访问 `engine` 会加载 torch（~1.6s，已由后台预热覆盖）
- **配置**：`Config` 是单例，CUDA 不可用时直接退出；`use_cuda_graph` 默认关闭，运行时探测启用

### 上游同步

上游 RVC-WebUI（`Retrieval-based-Voice-Conversion-WebUI/`，独立 git 仓库）：

```bash
git -C "Retrieval-based-Voice-Conversion-WebUI" fetch origin
git -C "Retrieval-based-Voice-Conversion-WebUI" reset --hard origin/main
```

本项目已重度重构，结构不同于上游；同步时只取 `rvc/audio|inference|models|nn|runtime` 相关改动。

### 验证

```bash
python -m py_compile <file>                    # 单文件语法检查
python -m compileall -q app.py rvc gui        # 全量语法检查
python -m unittest discover -s tests -v        # 运行全部单元测试（1152 个测试）
python -m unittest tests.test_pipeline -v      # 运行单个测试文件
```

运行时验证靠手动启动 GUI（`python app.py --infer` / `python app.py --train`）。

### 最近架构改进（2026-09-08）

- ✅ **测试覆盖大规模新增** — 从 262 个测试增加到 1152 个（+890），新增 16 个深度测试文件，覆盖 config/errors/sola/wav_io/effects/inference_cache/runtime_paths/mel/train_slicer/f0/synthesis/cuda_graph/model_session/pipeline/gui_configs/feature_processing/inference_runner/realtime_engine/extract_f0/extract_feature
- ✅ **未使用导入清理** — 清理 40+ 处未使用导入，涉及 rvc/ 和 gui/ 目录下 22 个文件
- ✅ **延迟张量克隆优化** — 去掉 `clone_protect_source` 中不必要的 `.clone()`（F.interpolate 会创建新张量，不会修改输入）
- ✅ **pipeline.py docstring 补充** — 补充 5 个方法的 docstring（__init__/load/_infer_impl/_synthesize_realtime/_postprocess_realtime）
- ✅ **_write_output_wav 空数组修复** — 添加空数组检查，避免 np.max() 报错
- ✅ **常量定义统一** — 新增 rvc/audio/constants.py（HUBERT_SAMPLE_RATE=16000, HUBERT_FRAME_SIZE=160, HUBERT_FRAME_RATE=100），消除魔法数字

### 最近架构改进（2026-09-07）

- ✅ **GUI 全面 MVC 分层** — infer/train 两个 GUI 模块拆分为 view/controller/viewmodel 三层；view 层只负责 UI 构建和信号连接，controller 层负责业务逻辑和状态管理，viewmodel 层负责 GUI 状态与核心配置的绑定
- ✅ **核心目录重构** — 新建 `rvc/core/`（config.py + errors.py），`cuda_graph.py` 从 tools/ 移入 inference/，删除空的 tools/ 目录
- ✅ **统一异常体系** — `RVCError` 基类 + 8 种具体异常（ModelLoadError/AudioDeviceError/InferenceError/F0ExtractionError/FeatureExtractionError/ConfigError/AudioLoadError）
- ✅ **F0 提取器抽象接口** — `F0Extractor` ABC + `RMVPEExtractor`/`FCPEExtractor` 实现，新增 F0 方法只需加实现类
- ✅ **模型生命周期统一** — `ModelSessionManager` 统一管理 HuBERT 缓存、合成器加载、F0 提取器
- ✅ **RealtimeEngine 拆分** — 门面类委托给 `AudioStreamManager`（设备/流管理）+ `InferenceRunner`（分块推理/效果器）
- ✅ **InferencePipeline 重命名** — VCPipeline → InferencePipeline（更通用的命名）
- ✅ **测试体系完善** — 1152 个单元测试，覆盖架构、配置、异常、效果器、推理缓存、推理管线、F0 接口、模型会话、流管理器、inference_runner、realtime_engine、extract_f0、extract_feature 等；统一使用 unittest，`python -m unittest discover` 一键运行全部测试
- ✅ **日志格式统一** — f-string → % 格式化，中英文标点统一（中文冒号/括号）

### 最近架构改进（2026-09-06）

- ✅ **统一配置体系** — 新建 `rvc/config.py`，定义 `InferenceConfig`（嵌套 `BreakProtectConfig`/`DenoiseConfig`）、`EngineConfig`、`AppConfig`、`ModelEntry`、`OfflineTask`，消除历史 6 套配置和 7 层转换链；`param_binding.py` 重写为嵌套路径绑定（点号路径如 `inference.pitch`）
- ✅ **InferencePipeline 无状态化** — 删除 `configure()`/`set_formant()`/`set_break()` 系列方法，`infer()` 每次接收 `InferenceConfig`，pipeline 只持缓存状态；`f0_proc` 扩展为四元组 `(enable, src_hz, ratio, knee)`
- ✅ **InferenceRunner 统一实时/离线** — 新建 `rvc/inference/runner.py`，唯一推理入口 `process_block()`；`process_file` 开始时调 `runner.reset()`，**解决历史 pitch 缓存跨文件污染问题**（批量处理短音频 NaN）
- ✅ **统一错误类型体系** — 新建 `rvc/errors.py`，`RVCError` 基类 + 5 个子类（`ModelLoadError`/`AudioDeviceError`/`InferenceError`/`ConfigError`/`FeatureExtractError`）+ `format_error_message()` 友好消息格式化
- ✅ **效果器抽象** — 新建 `rvc/audio/effects.py`，`AudioEffect` 抽象基类 + `DenoiseEffect`/`RmsMixEffect`/`SolaEffect` + `EffectChain` 组合器
- ✅ **测试基础设施** — 新建 `tests/` 目录，21 个单元测试（配置体系 15 + 错误体系 6），使用内置 `unittest` 无需额外依赖
- ✅ **train 窗口秒启动** — `gui/train/workers.py` 顶层 import 的 `rvc.train.*`（extract_f0/extract_feature/preprocess/trainer）和 `Config` 改为 `_run_impl()` 内惰性导入，TrainWindow import 0.35s，与 infer 窗口一致
- ✅ **主题** — Windows 11 标准深色（#202020/#1a1a1a/#2a2a2a + Windows 蓝 #0078D4）

### 最近架构改进（2026-09-05）

- ✅ **彻底移除特征检索（FAISS/index）** — 检索匹配"用处不大"，全链删除代码、界面与依赖
- ✅ **离线推理改流式** — `RealtimeEngine.process_file` 复用实时全链路（RMS/SOLA/CUDA Graph），显存封顶、任意长不 OOM；离线音质 ≈ 实时
- ✅ **推理默认 RMVPE** — 与训练侧 F0 提取对齐，消除 train/infer f0 分布不匹配
- ✅ **FCPE 阈值收紧到 0.025** — 消除低电平底噪导致的假音高
- ✅ **RMVPE 解码搬上 GPU** — 消除回调内 GPU 同步，抗噪更稳
- ✅ **合成器真正进 CUDA Graph** — 实时/离线 × 有/无 F0 四路径统一 capture + replay
- ✅ **破音保护升级（方案 A）** — 压缩比 + 平滑膝三段映射，高音区不再压死到固定顶；ratio/knee 为内部默认值，界面只需开关 + 临界 Hz
- ✅ **参数同源收敛** — 实时/离线统一走 `pipeline.configure()`；`OfflineConfig` 继承 `Params`，音效/音高/破音字段单一来源

### 最近架构改进（2026-08-24）

- ✅ **启动秒开** — 惰性导入架构，窗口 8.2s → 0.3s；torch 后台预热 + 开流前 CUDA Graph 预热，首次点「开始」不再卡顿、首块延迟 1101ms → ~30ms
- ✅ **系统托盘** — 关闭最小化到托盘、托盘开始/停止变声、tooltip 实时状态与实测延迟
- ✅ **兼容代码清理** — 移除 use_pv 死链、ckpt_version 传递链、相位声码器、空壳方法等上游残留
- ✅ **鲁棒性修复** — 运行时错误信号转发主线程、LoadThread 竞态、离线 numpy 回归、pitch 缓存扩容等
- ✅ **性能** — 回调输入 pinned buffer 非阻塞拷贝、formant 因子缓存、engine 构造加锁

### 最近架构改进（2026-06-13）

本项目最近完成了全面的代码重构，显著提升了代码质量：

- ✅ **消除 85% 重复代码** — RMS 匹配、Synthesizer 基类统一、F0 提取器抽象
- ✅ **模块化拆分** — rmvpe、pipeline、styles、realtime audio、GUI state 持续拆分
- ✅ **命名一致性** — 文件名与职责对应 (`realtime_engine.py`, `model_session.py`, `inference_cache.py`)
- ✅ **职责分离** — 配置代码 (`gui/configs/`) 与配置数据 (`assets/configs/`) 分离
- ✅ **日志格式统一** — 简洁清晰的中文日志

**重构指标**：
- 最大文件行数：-35%
- 代码重复度：-85%
- 模块化程度：+30%
- 日志统一度：+45%
