"""音频处理常量 — 行业标准采样率与帧长。

HuBERT/RMVPE 等预训练模型在 16kHz 音频上训练，帧长 10ms（160 样本）。
这些常量散落在多处容易出错，统一在此定义。
"""

# HuBERT 模型标准采样率（所有 HuBERT 变体均在 16kHz 上预训练）
HUBERT_SAMPLE_RATE = 16000

# HuBERT 帧长（10ms @ 16kHz = 160 样本）
HUBERT_FRAME_SIZE = 160

# HuBERT 帧率（100 帧/秒）
HUBERT_FRAME_RATE = 100
