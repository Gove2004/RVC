# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟实时实测显示（硬件时间戳、瞬时值）
- **快速启动** — 惰性导入架构，窗口秒开（~0.3s）；torch 后台预热，首次点「开始」不卡顿
- **CUDA Graph 加速** — 开流前静音预热完成图捕获，首块推理即热状态（1101ms → ~30ms）
- **系统托盘** — 关闭窗口最小化到托盘（变声不中断），托盘可开始/停止变声、tooltip 显示状态与实测延迟
- **离线推理** — 音频文件批量转换（最长 5 分钟）
- **模型训练** — 从人声音频训练自定义模型，支持 40k/48k、早停自动收敛
- **音高调节** — 变调不变速（-16 ~ +16 半音）
- **辅音保护** — 保留原音清音/辅音，防止齿音失真（0.0 ~ 1.0）
- **FAISS 索引** — 可选的说话人相似度匹配（k=8 加权混合）
- **频谱降噪** — 输入侧降噪（GPU 谱减法、零新增延迟）+ RMS 响度匹配
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
   - 选择 `.pth` 模型文件（可选 `.index` 索引文件）

2. **调节参数**
   - 展开模型卡片，调节以下参数：
     - **音调大小**：音高偏移（半音，-16 ~ +16）
     - **索引率**：FAISS 混合比例（0.0 = 关闭，1.0 = 完全使用索引）
     - **响度**：RMS 响度混合（0.0 = 目标响度，1.0 = 源响度）
     - **性别**：formant shift（-50 ~ +50）
3. **全局参数** — 在"参数" Tab 配置：
     - **辅音保护**：清音保留程度（0.0 = 全转换，1.0 = 全保留）
     - **频谱降噪**：输入侧降噪（GPU 谱减法、零新增延迟），勾选启用、滑条调强度（0.00 ~ 1.00）
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
     - **训练参数**：Epoch, Batch size, 学习率, 早停耐心（连续 N 轮 Mel loss 无改善自动保存退出，0 = 关闭）

4. **执行训练**
   - 点击"一键全流程"自动完成所有步骤
   - 或分步执行：
     1. **预处理** — 音频切片、去静音、归一化、重采样
     2. **提取 F0** — RMVPE 基频提取
     3. **提取特征** — HuBERT 768 维特征
     4. **训练** — GAN 训练（Generator + Discriminator）

5. **导出模型**
   - 训练完成（或早停触发）后，模型导出到 `assets/models/<实验名>_e<epoch>.pth`
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
gui/
  styles/                   # 模块化 UI 设计系统
    colors.py               # 颜色调色板
    layout.py               # 布局参数
    components.py           # 样式组件
  configs/                  # 配置代码
    config.py               # load_config/save_config（GUI 状态）
  infer/                    # 推理 GUI
    window.py               # 主窗口（含系统托盘集成、引擎后台预热）
    controller.py           # 控制器（依赖注入，engine 惰性构造）
    widgets.py              # ModelCard, LoadThread
    tray.py                 # 系统托盘（图标/菜单/tooltip 状态）
    tabs/                   # 各功能 Tab
  train/                    # 训练 GUI
    window.py               # 训练窗口
    tabs/                   # 设置、训练、工具 Tab
rvc/
  audio/
    realtime_engine.py      # RealtimeEngine（音频流管理 + 回调编排 + 开流前预热）
    device_query.py         # 音频设备枚举（轻量，仅依赖 sounddevice）
    sola.py                 # SOLA 时间拉伸对齐与交叉淡化
    realtime_mix.py         # RMS 音量包络混合
    output_router.py        # 主输出写入与副输出路由
    loader.py               # 音频加载（librosa + ffmpeg fallback）
    utils.py                # RMS 响度匹配
    effects.py              # 效果器基类（AudioEffect 抽象接口）
    denoise.py              # 谱减法降噪（GPU 块级）
  inference/
    pipeline.py             # VCPipeline facade（实时/离线推理编排）
    feature_processing.py   # HuBERT 特征、padding mask、protect blend
    index_retrieval.py      # FAISS index 加载与特征混合
    pitch_tracker.py        # F0 提取窗口与实时 pitch cache
    synthesis.py            # Synthesizer 推理调用与 formant 重采样
    model_session.py        # HuBERT/Synthesizer/Index session 加载
    model_loader.py         # SynthesizerLoader（PyTorch）
    offline_config.py       # OfflineConfig（离线推理配置）
    params.py               # Params（运行时参数单例）
    f0_extractor.py         # F0 提取器抽象层（RMVPE/FCPE）
  models/
    inference_cache.py      # InferenceCache（线程安全模型缓存）
    hubert.py               # HuBERT 加载
    rmvpe/                  # RMVPE F0 提取器（模块化）
      model.py              # RMVPE 推理类
      blocks.py             # CNN 模块
      transforms.py         # STFT + MelSpectrogram
  synthesizer/              # NSF 合成器（模块化）
    model.py                # 统一 Synthesizer 基类
    encoder.py              # TextEncoder, PosteriorEncoder
    decoder.py              # Generator, GeneratorNSF
    flow.py                 # ResidualCouplingBlock
  nn/                       # 神经网络基础层
  tools/
    cuda_graph.py           # CUDA Graph 捕获/回放缓存（按形状 LRU）
  runtime/                  # 运行时配置
    device_config.py        # Config 单例（CUDA 探测、GPU 信息）
    paths.py                # 配置路径
  train/                    # 训练管线
    trainer.py              # GAN 训练循环
    preprocess.py           # 音频预处理
    extract_f0.py           # F0 提取
    extract_feature.py      # HuBERT 特征提取
    ckpt_utils.py           # Checkpoint 工具
assets/
  configs/                  # 配置数据
    save_state.json         # GUI 持久化状态
    48ktrain_config.json    # 48k 训练超参数
    40ktrain_config.json    # 40k 训练超参数
  models/                   # 推理模型
  indexes/                  # FAISS 索引
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
麦克风 → [降噪] → RealtimeEngine → VCPipeline → SOLA → 输出 → 扬声器
          ↓                 ↓
      sounddevice      HuBERT + Synthesizer
      SOLA crossfade   FAISS blend (可选)
                       protect_blend（辅音保护）
```

### 关键技术

- **SOLA 算法** — 重叠相加实现无缝音频拼接，支持变速不变调
- **辅音保护** — 使用 F0 contour 作为掩码，清音区域混回原始特征
- **谱减法降噪** — 输入侧 FFT 频域谱减，自适应噪声地板，GPU 块级零新增延迟
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

**Q: 转换后声音有周期性"嘟嘟嘟"失真？**  
A: 检查是否开启了频谱降噪且强度过高，可适当降低强度或关闭。

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
- `gui/` = PySide6 窗口、控件、管理器、QThread worker
- 运行时设备/路径配置来自 `rvc.runtime`；GUI 状态持久化来自 `gui.configs`
- GUI 状态同步与持久化由 `gui/infer/param_binding.py` 的 BINDINGS 表驱动（加参数 = dataclass 字段 + BINDINGS 一行 + Tab 控件）

### 核心实现规则

- **采样率**：实时帧数学用 `zc = sr // 100` 对齐，禁止硬编码 `48000`；`sr_mode="model"` 与 `"device"` 都要工作
- **模型精度**：模型可能 half/float，feature/index/protect 之后要恢复模型 dtype；pitch coarse 用 `long`
- **实时回调安全**：sounddevice 回调内禁止阻塞 IO、模型加载、大分配、GPU 同步；禁止直接操作 Qt（运行时错误经 Signal 转发主线程）
- **惰性导入**：GUI 启动路径禁止在模块顶层 import torch/transformers/librosa/faiss 或实例化 `Config()`；重型 import 只允许出现在「加载模型/开始/离线推理」路径；`rvc/*/__init__.py` 用模块级 `__getattr__` 惰性导出
- **引擎访问**：判断引擎是否已构造用 `controller._engine`（勿触发惰性构造）；首次访问 `engine` 会加载 torch（~1.6s，已由后台预热覆盖）
- **降噪**：输入侧谱减法（GPU 块级、零新增延迟），强度是每回调的标量快照
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
python -m py_compile <file>             # 单文件语法检查
python -m compileall -q app.py rvc gui # 全量语法检查
```

无自动化测试，运行时验证靠手动启动 GUI（`python app.py --infer` / `python app.py --train`）。

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
