"""音频常量 — HuBERT 采样率与帧长、RMVPE 阈值，跨层共享。"""

HUBERT_SAMPLE_RATE = 16000
HUBERT_FRAME_SIZE = 160

# RMVPE 置信度阈值（低于此值判为静音/UV）。
# 训练与推理数值不同是有意为之（历史口径），单源定义防止双处漂移：
# - 训练侧 0.03：RMVPE 官方推荐值，批量切片底噪干净；
# - 推理侧 0.05：实时链路底噪更脏，用户可经 InferenceParams.f0.rmvpe_threshold 调整。
RMVPE_THRESHOLD_TRAIN = 0.03
