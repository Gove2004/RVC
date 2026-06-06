# RVC 实时语音转换工具

基于 RVC (Retrieval-based Voice Conversion) 的实时变声器，支持实时变声、离线推理和模型训练。

## 功能

- **实时变声** — 麦克风输入实时变声输出，低延迟
- **离线推理** — 音频文件转换
- **模型训练** — 从干净音频训练自定义 voice 模型
- **音高调节** — 变调不变速
- **性别因子** — 声音性别特征调节
- **FAISS 索引** — 说话人相似度匹配
- **声学效果** — EQ / 饱和 / 压限 / 混响

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

### 启动推理 GUI

```bash
.venv\Scripts\python.exe app.py
```

或双击 `start.bat`（有黑窗）/ `start.vbs`（静默启动）

### 启动训练 GUI

```bash
.venv\Scripts\python.exe train_app.py
```

或双击 `start-train.bat` / `start-train.vbs`

### 推理流程

1. 在"模型" Tab 添加模型（.pth 文件）
2. 展开模型卡片，调节参数（音调/性别/索引率/响度）
3. 在"设置" Tab 选择音频设备
4. 点击"开始"，实时变声
5. 在"声学" Tab 调节音效

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

## 项目结构

```
app.py                  推理 GUI
train_app.py            训练 GUI
configs/
  config.py             设备配置
  models.json           推理模型列表
  v2/48k.json           48k 训练超参数
  v2/32k.json           32k 训练超参数
  inuse/                运行时配置（自动生成）
rvc/
  realtime_engine.py    实时推理引擎 (RealtimeVC)
  engine.py             音频 I/O 管理 (RealtimeEngine)
  synthesizer.py        神经网络模型（Generator）
  hubert.py             HuBERT 特征提取器
  rmvpe.py              RMVPE F0 提取器
  params.py             运行时参数单例
  audio_utils.py        音频工具（相位声码器）
  offline_worker.py     离线推理线程
  nn/
    attentions.py       Transformer 编码器
    modules.py          WaveNet / ResBlock
    commons.py          工具函数
    discriminator.py    判别器（训练用）
  train/
    trainer.py          GAN 训练循环
    train_worker.py     训练后台线程
    preprocess.py       音频预处理
    extract_f0.py       F0 提取
    extract_feature.py  HuBERT 特征提取
    data_utils.py       数据加载器
    losses.py           损失函数
    mel_processing.py   频谱处理
    ckpt_utils.py       checkpoint 工具
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

## 许可证

本项目基于 RVC (Retrieval-based Voice Conversion) 开源项目。仅供学习研究使用。
