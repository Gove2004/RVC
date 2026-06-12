# RVC 实时语音转换工具

基于 RVC (Retrieval-based Voice Conversion) 的实时变声器，支持实时变声、离线推理和模型训练。

## ✨ 特色功能

- **实时变声** — 麦克风输入实时变声输出，低延迟（~100ms）
- **离线推理** — 音频文件批量转换
- **模型训练** — 从干净音频训练自定义 voice 模型
- **音高调节** — 变调不变速（-16~+16 半音）
- **性别因子** — 声音性别特征调节
- **辅音保护** — 可调节清音/辅音保留程度（0.0~1.0）
- **FAISS 索引** — 说话人相似度匹配（可选）
- **声学效果** — 5 段 EQ / 混响，支持预设（人声增强、温暖厚实、明亮清脆等）
- **双输出支持** — 主输出 + 副输出（虚拟音频设备）

## 🎨 UI 设计系统 (2026-06-13 更新)

本项目拥有完整的统一 UI 设计系统：

- **语义化颜色** — 绿色（主要）、蓝色（次要）、红色（危险）
- **统一布局** — 8px 边距、6px 间距、3px 圆角
- **标准化组件** — 所有按钮、标签、卡片使用统一样式
- **集中管理** — `gui/styles.py` 统一样式系统，零内联样式
- **易维护** — 一处修改全局生效

详见 `gui/styles.py` 和 `CLAUDE.md`。

## 功能

- **实时变声** — 麦克风输入实时变声输出，低延迟
- **离线推理** — 音频文件转换
- **模型训练** — 从干净音频训练自定义 voice 模型
- **音高调节** — 变调不变速
- **性别因子** — 声音性别特征调节
- **辅音保护** — 可调节清音/辅音保留程度（0.0~1.0）
- **FAISS 索引** — 说话人相似度匹配
- **声学效果** — 5 段 EQ / 混响，支持预设

## 环境要求

- Windows 11
- Python 3.13+
- NVIDIA GPU（CUDA required），推荐 RTX 5060 或更高

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd RVC

# 创建虚拟环境
python -m venv .venv

# 安装依赖
.venv\Scripts\pip.exe install -r requirements.txt
```

### 额外依赖（训练需要）

训练使用 RMVPE 提取 F0，需要下载预训练权重：

- `assets/rmvpe/rmvpe.pt` — RMVPE 模型
- `assets/hubert/hubert_base.pt` — HuBERT 模型

预训练权重（可选，加速训练收敛）：

- `assets/pretrained_v2/f0G48k.pth` — 48k Generator 预训练
- `assets/pretrained_v2/f0D48k.pth` — 48k Discriminator 预训练

## 使用

```bash
# 启动推理 GUI
.venv\Scripts\python.exe app.py --infer

# 启动训练 GUI
.venv\Scripts\python.exe app.py --train
```

或双击启动脚本：

- `start-infer.bat` / `start-infer.vbs` — 推理（bat 有控制台，vbs 静默）
- `start-train.bat` / `start-train.vbs` — 训练

### 推理流程

1. 在"模型" Tab 添加模型（.pth 文件）
2. 展开模型卡片，调节参数：
   - **音调**：音高偏移（半音）
   - **索引率**：FAISS 相似度混合比例（0.0~1.0）
   - **响度**：RMS 响度混合（0.0~1.0）
   - **性别**：性别因子调节（-50~+50）
   - **辅音保护**：清音/辅音保留程度（0.0=全转换，1.0=全保留原音）
3. 在"设置" Tab 选择音频设备和采样率模式
   - **模型采样率**：使用模型原生采样率（推荐）
   - **设备采样率**：使用音频设备采样率
4. 点击"使用"激活模型，点击"开始"启动实时变声
5. 在"声学" Tab 调节音效：
   - **5 段 EQ**：超低频(60Hz) / 低频(200Hz) / 中频(1kHz) / 中高频(3kHz) / 高频(8kHz)
   - **混响**：空间混响效果（0.0~0.5）
   - **预设**：一键应用预设（人声增强、温暖厚实、明亮清脆、唱歌混响等）

### 离线推理

1. 切换到"离线" Tab
2. 选择输入/输出文件
3. 点击"开始转换"（自动使用当前激活模型的所有参数）

### 训练流程

1. 准备干净人声音频目录（支持 wav/mp3/flac/ogg 等格式）
2. 启动训练 GUI，选择音频目录
3. 设置实验名、采样率（48k/32k）、训练参数
4. 点击"一键全流程"或分步执行：
   - **1. 预处理** — 音频切片、去静音、归一化、重采样
   - **2. 提取F0** — RMVPE 基频提取
   - **3. 提取特征** — HuBERT 768 维语音特征
   - **4. 训练** — GAN 训练 RVC 模型
5. 训练完成后模型导出到 `assets/weights/<实验名>.pth`
6. 回到推理 GUI 加载导出的模型

训练支持中断恢复，checkpoint 保存在 `logs/<实验名>/`。

### 运行资产与实验目录

- `logs/` 会持续增长，包含切片音频、F0、特征、spec 缓存和 checkpoint，训练完成后建议定期清理不用的实验目录。
- 同一个实验名会绑定输入目录、采样率和切片参数；如果这些条件变化，预处理阶段会自动清理旧运行产物并重建。
- 如果你跳过预处理直接执行 `提取F0` / `提取特征` / `训练`，而实验目录与当前输入不一致，程序会直接阻止继续，避免混入旧实验数据。
- `.pth` 模型文件通过 `torch.load(..., weights_only=False)` 加载，请不要加载来源不可信的模型文件。

## 项目结构

```
app.py                  统一入口 (--infer / --train)
gui/
  styles.py             ⭐ 统一 UI 设计系统 (2026-06-13 新增)
  infer/
    window.py           推理主窗口 (MainWindow)
    controller.py       推理控制器（依赖注入）
    widgets.py          ModelCard, ModelListData, LoadThread
    tabs/               推理 GUI Tab 构建器
      models_tab.py     模型管理（可滚动列表 + 展开卡片）
      settings_tab.py   设备和参数设置
      audio_tab.py      声学效果（5段EQ + 混响 + 预设）
      offline_tab.py    离线推理
  train/
    window.py           训练主窗口 (TrainWindow)
    widgets.py          ToolThread, browse helpers
    tabs/               训练 GUI Tab 构建器
rvc/
  audio/
    stream.py           实时音频引擎 (RealtimeEngine, SOLA, 音效)
    loader.py           统一音频加载（librosa + ffmpeg fallback）
    utils.py            音频工具（设备枚举、EQ 预设）
  inference/
    pipeline.py         推理管线 (VCPipeline, protect_blend)
    offline.py          离线推理线程 (OfflineWorker)
    params.py           运行时参数单例 (Params)
  models/
    hubert.py           HuBERT 特征提取器
    rmvpe.py            RMVPE F0 提取器
    cache.py            InferenceCache（线程安全）
  nn/
    attentions.py       Transformer 编码器
    modules.py          WaveNet / ResBlock
    commons.py          工具函数
    discriminator.py    判别器（训练用）
  synthesizer/
    encoder.py          TextEncoder + PosteriorEncoder
    decoder.py          Generator + GeneratorNSF
    flow.py             ResidualCouplingBlock
    model.py            Synthesizer 模型变体
  train/
    trainer.py          GAN 训练循环
    train_worker.py     训练后台线程
    preprocess.py       音频预处理（切片、去静音、归一化）
    extract_f0.py       F0 提取（RMVPE）
    extract_feature.py  HuBERT 特征提取
    data_utils.py       数据加载器
    losses.py           损失函数
    mel_processing.py   频谱处理
    ckpt_utils.py       checkpoint 工具（保存/恢复/导出/合并/查看）
configs/
  config.py             设备配置 (Config 单例)
  state/                运行时状态（gui.json, models.json, train.json）
  train/                训练超参数配置（32k.json, 48k.json）
assets/
  hubert/               HuBERT 模型
  rmvpe/                RMVPE 模型
  pretrained_v2/        预训练权重（G/D）
  weights/              导出的推理模型
  ffmpeg/               ffmpeg 二进制
```

## 技术特性

### 核心技术

- **辅音保护机制** — 使用 F0 contour 作为掩码，将清音/辅音区域混回原始特征，防止齿音失真
- **频域 EQ** — 基于 FFT 的无状态 EQ，避免 IIR 滤波器在 block 边界产生相位跳变
- **SOLA 算法** — 重叠相加实现无缝音频拼接，支持变速不变调
- **动态精度管理** — 自动处理 half/float 模型，确保推理稳定性
- **双采样率模式** — 可选模型原生采样率或设备采样率，适配不同场景

### UI 设计系统

- **集中管理** — `gui/styles.py` 统一定义颜色、布局、按钮样式
- **语义化设计** — Primary (绿) / Secondary (蓝) / Danger (红) / Muted (灰)
- **响应式组件** — 统一的状态指示、hover 效果、禁用样式
- **零内联样式** — 所有样式通过 ButtonStyles/LabelStyles/CardStyles 管理
- **易扩展** — 一处修改全局生效，支持主题定制

## 项目结构

```
app.py                  统一入口 (--infer / --train)
app/
  infer/
    window.py           推理主窗口 (MainWindow)
    widgets.py          ModelCard, ModelListData, LoadThread
    tabs/               推理 GUI Tab 构建器
      model_tab.py      模型管理
      settings_tab.py   设备和参数设置
      audio_tab.py      声学效果（5段EQ + 混响）
      offline_tab.py    离线推理
  train/
    window.py           训练主窗口 (TrainWindow)
    widgets.py          ToolThread, browse helpers
    tabs/               训练 GUI Tab 构建器
rvc/
  audio_io.py           音频 I/O 管理 (RealtimeEngine)
  vc_pipeline.py        实时推理管线 (VCPipeline, protect_blend)
  audio_loader.py       统一音频加载（librosa + ffmpeg fallback）
  offline_worker.py     离线推理线程
  hubert.py             HuBERT 特征提取器
  rmvpe.py              RMVPE F0 提取器
  params.py             运行时参数单例
  audio_utils.py        音频工具（相位声码器、设备枚举、预设）
  denoise/              TorchGate 降噪
  nn/
    attentions.py       Transformer 编码器
    modules.py          WaveNet / ResBlock
    commons.py          工具函数
    discriminator.py    判别器（训练用）
  synthesizer/
    encoder.py          TextEncoder + PosteriorEncoder
    decoder.py          Generator + GeneratorNSF
    flow.py             ResidualCouplingBlock
    model.py            Synthesizer 模型变体
  train/
    trainer.py          GAN 训练循环
    train_worker.py     训练后台线程
    preprocess.py       音频预处理
    extract_f0.py       F0 提取
    extract_feature.py  HuBERT 特征提取
    data_utils.py       数据加载器
    losses.py           损失函数
    mel_processing.py   频谱处理
    ckpt_utils.py       checkpoint 工具（保存/恢复/导出/合并/查看）
configs/
  config.py             设备配置 (Config 单例)
  models.json           推理模型列表
  v2/                   训练超参数
  inuse/                运行时配置（自动生成）
assets/
  hubert/               HuBERT 模型
  rmvpe/                RMVPE 模型
  pretrained_v2/        预训练权重
  weights/              导出的推理模型
```

## 训练数据要求

- 单人干净人声
- 背景噪声尽量少
- 建议总时长 10 分钟以上效果更好
- 音频会被自动切成 ~3.7 秒片段
- 支持任意常见音频格式（通过 ffmpeg 解码）

## 模型格式

推理模型 (.pth) 结构：

```python
{
    "weight": OrderedDict({...}),  # 权重（half 精度，无 enc_q）
    "config": [18 个模型参数],
    "info": "2000epoch",
    "sr": "48k",
    "f0": 1,
    "version": "v2"
}
```

## 开发文档

- **CLAUDE.md** — 完整的代码架构和开发指南
- **gui/styles.py** — UI 设计系统和样式规范
- **README.md** — 用户使用文档（本文件）

## 更新日志

### 2026-06-13 - UI 统一化改革

**重大更新**：建立完整的 UI 设计系统

- ✅ 新增 `gui/styles.py` 统一样式系统
- ✅ 消除 40+ 处内联样式
- ✅ 统一颜色系统（4 种语义色）
- ✅ 标准化布局参数（8px 边距、6px 间距）
- ✅ 规范按钮尺寸（60x28 / 48 / 28x24）
- ✅ 统一 Padding（5px 12px / 4px 8px）
- ✅ 修复滚动条宽度抖动
- ✅ 修复按钮文字显示问题
- ✅ 紧凑化界面（357x333 / 360x253）

详见桌面文档：`RVC_UI_Audit.md` 和 `RVC_UI_Reform_Summary.md`

### 之前的更新

- 移除 TorchGate 降噪功能
- 更新 requirements.txt
- EQ 预设重构（全女声向）
- 修复副输出初始化逻辑
- 修复 ffmpeg 路径计算

## 技术亮点

- **辅音保护机制**：使用 F0 contour 作为掩码，将清音/辅音区域混回原始特征，防止齿音失真
- **频域 EQ**：基于 FFT 的无状态 EQ，避免 IIR 滤波器在 block 边界产生相位跳变
- **SOLA 算法**：重叠相加实现无缝音频拼接，支持变速不变调
- **动态精度管理**：自动处理 half/float 模型，确保推理稳定性
- **双采样率模式**：可选模型原生采样率或设备采样率，适配不同场景

## 常见问题

**Q: 转换后声音有"嘟嘟嘟"的周期性失真？**  
A: 检查是否开启了音效。EQ 必须使用频域处理（已修复），如果使用旧版 IIR 滤波器会产生此问题。

**Q: 辅音保护（protect）如何调节？**  
A: 0.0 = 完全转换（音色纯净但可能糊掉），1.0 = 完全保留原音辅音（清晰但音色不纯）。建议从 0.5 开始调整。

**Q: 模型采样率和设备采样率选哪个？**  
A: 推荐"模型采样率"，保持模型原生质量。选"设备采样率"可降低重采样开销，但可能影响音质。

**Q: 快速切换模型时 UI 卡死？**  
A: 已修复。新版会自动取消旧的加载线程。

**Q: half 精度模型报 dtype 错误？**  
A: 已修复。确保使用最新版本。

## 性能优化建议

- 使用 CUDA GPU（RTX 5060 或更高）
- 输入延迟（block）：0.09~0.18 秒平衡延迟和稳定性
- 交叉淡化（crossfade）：0.04~0.08 秒，过大会增加延迟
- 额外推理（extra）：2.0~2.5 秒，提供足够上下文
- 关闭不需要的音效可降低 CPU 占用

## 许可证

本项目基于 RVC (Retrieval-based Voice Conversion) 开源项目。仅供学习研究使用。
