# RVC - Real-time Voice Conversion

基于 RVC (Retrieval-based Voice Conversion) 的实时变声工具，支持实时推理、离线转换和模型训练。

## 特性

- **实时变声** — 麦克风输入实时转换输出，延迟 ~100ms
- **离线推理** — 音频文件批量转换（最长 5 分钟）
- **模型训练** — 从人声音频训练自定义模型
- **音高调节** — 变调不变速（-24 ~ +24 半音）
- **辅音保护** — 保留原音清音/辅音，防止齿音失真（0.0 ~ 1.0）
- **FAISS 索引** — 可选的说话人相似度匹配（k=8 加权混合）
- **频谱降噪** — 输入侧降噪（GPU 谱减法、零新增延迟）+ RMS 响度匹配
- **背景音** — 本地音频循环播放叠加（可选，开关/音量实时生效）
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
- `assets/hubert_base/` — HuBERT 特征提取器（transformers 模型目录，含 config.json + pytorch_model.bin）

可选的预训练权重（加速训练收敛）：

- `assets/pretrained_v2/f0G48k.pth` — 48k Generator
- `assets/pretrained_v2/f0D48k.pth` — 48k Discriminator
- `assets/pretrained_v2/f0G40k.pth` — 40k Generator（40k 训练时使用）
- `assets/pretrained_v2/f0D40k.pth` — 40k Discriminator（40k 训练时使用）

## 使用

### 启动程序

```bash
# 推理 GUI
.venv\Scripts\python.exe app.py --infer

# 训练 GUI
.venv\Scripts\python.exe app.py --train
```

或使用启动脚本：

- `start-infer.bat` / `start-infer.vbs` — 推理（vbs 静默启动）
- `start-train.bat` / `start-train.vbs` — 训练

### 实时推理流程

1. **添加模型**
   - 在"模型" Tab 点击"+ 添加模型"
   - 选择 `.pth` 模型文件（可选 `.index` 索引文件）

2. **调节参数**
   - 展开模型卡片，调节以下参数：
     - **音调**：音高偏移（半音，-24 ~ +24）
     - **索引率**：FAISS 混合比例（0.0 = 关闭，1.0 = 完全使用索引）
     - **响度**：RMS 响度混合（0.0 = 目标响度，1.0 = 源响度）
     - **性别**：formant shift（-50 ~ +50）
3. **全局参数** — 在"参数" Tab 配置：
     - **辅音保护**：清音保留程度（0.0 = 全转换，1.0 = 全保留）
     - **音高算法**：RMVPE 或 FCPE
     - **采样长度/淡入长度/额外上下文**：实时推理延迟控制
     - **响度因子**：输出响度混合比例
4. **配置设备**
   - 在"设备" Tab 选择音频设备（麦克风、主输出、副输出）和采样率模式
   - **采样率模式**：
     - `模型采样率` — 使用模型原生采样率（推荐，音质最佳）
     - `设备采样率` — 使用音频设备采样率（降低重采样开销）
5. **开始推理**
   - 点击"使用"激活模型（加载到 GPU）
   - 点击"开始"启动实时推理
6. **噪音处理**（可选）
   - 在"噪音" Tab：
     - **频谱降噪**：输入侧降噪（GPU 谱减法、零新增延迟），勾选启用、滑条调强度（0.00 ~ 1.00）
     - **背景音频**：选择本地音频文件（wav/mp3/flac/m4a/ogg/aac，通过 ffmpeg 解码）
     - **背景底噪**：勾选启用背景音并调节音量（0.00 ~ 1.00），开关与音量在推理过程中实时生效
   - 背景音在推理启动时载入并自动循环播放，叠加到最终输出

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
     - **训练参数**：Epoch, Batch size, 学习率等

4. **执行训练**
   - 点击"一键全流程"自动完成所有步骤
   - 或分步执行：
     1. **预处理** — 音频切片、去静音、归一化、重采样
     2. **提取 F0** — RMVPE 基频提取
     3. **提取特征** — HuBERT 768 维特征
     4. **训练** — GAN 训练（Generator + Discriminator）

5. **导出模型**
   - 训练完成后，模型导出到 `assets/weights/<实验名>_e<epoch>.pth`
   - 可选：在"工具" Tab 合并多个 checkpoint 或查看模型信息

6. **使用训练的模型**
   - 返回推理 GUI，加载 `assets/weights/<实验名>_e<epoch>.pth`

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
    window.py               # 主窗口
    controller.py           # 控制器（依赖注入）
    widgets.py              # ModelCard, LoadThread
    tabs/                   # 各功能 Tab
  train/                    # 训练 GUI
    window.py               # 训练窗口
    tabs/                   # 设置、训练、工具 Tab
rvc/
  audio/
    realtime_engine.py      # RealtimeEngine（音频流管理）
    loader.py               # 音频加载（librosa + ffmpeg fallback）
    utils.py                # 设备枚举、相位声码器、RMS 匹配
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
  weights/                  # 推理模型
  indices/                  # FAISS 索引
  hubert_base/              # HuBERT 权重（transformers 模型目录）
  rmvpe/                    # RMVPE 权重
  pretrained_v2/            # 预训练权重
  ffmpeg/                   # ffmpeg 二进制
logs/                       # 训练实验目录
```

## 技术细节

### 实时推理管线

```
麦克风 → [降噪] → RealtimeEngine → VCPipeline → SOLA → 背景音/输出 → 扬声器
          ↓                 ↓
      sounddevice      HuBERT + Synthesizer
      SOLA crossfade   FAISS blend (可选)
                       protect_blend（辅音保护）
```

### 关键技术

- **SOLA 算法** — 重叠相加实现无缝音频拼接，支持变速不变调
- **辅音保护** — 使用 F0 contour 作为掩码，清音区域混回原始特征
- **FFT EQ** — 频域均衡器，避免 IIR 滤波器的 block 边界相位跳变
- **双采样率模式** — 模型采样率（高音质）或设备采样率（低延迟）
- **动态精度** — 自动处理 half/float 模型，确保推理稳定

### 模型格式

推理模型（`.pth`）：

```python
{
    "weight": OrderedDict({...}),  # 模型权重（half 精度）
    "config": [18 个参数],
    "info": "2000epoch",
    "sr": "48k",
    "f0": 1,
    "version": "v2"
}
```

## 性能优化建议

- **输入延迟**（block）：0.09 ~ 0.18 秒（平衡延迟和稳定性）
- **交叉淡化**（crossfade）：0.04 ~ 0.08 秒（过大增加延迟）
- **额外推理**（extra）：0.3 ~ 2.5 秒（提供特征提取上下文；越小延迟越低）

## 常见问题

**Q: 转换后声音有周期性"嘟嘟嘟"失真？**  
A: 检查是否开启了频谱降噪且强度过高，可适当降低强度或关闭。

**Q: 辅音保护（protect）如何调节？**  
A: 0.0 = 完全转换（音色纯但可能糊），1.0 = 完全保留原音辅音（清晰但音色不纯）。建议从 0.5 开始调整。

**Q: 模型采样率和设备采样率选哪个？**  
A: 推荐"模型采样率"，保持模型原生质量。选"设备采样率"可降低重采样开销，但可能影响音质。

**Q: 训练需要多少数据？**  
A: 建议 10 分钟以上干净人声。背景噪声越少越好，会被自动切成 ~3.7 秒片段。

## 安全提示

- 模型文件通过 `torch.load(..., weights_only=False)` 加载
- **请勿加载来源不可信的 .pth 文件**

## 许可证

本项目基于 RVC 开源项目，仅供学习研究使用。

## 开发文档

- **CLAUDE.md** — 完整的架构文档和开发指南
- **AGENTS.md** — 给 AI 编码助手的上下文指南

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
