"""跨层共享常量 — 音频基础参数、F0 阈值、训练预处理默认值。

放在 `rvc/core` 是因为下面三处都要用到，而它们分属不同层：
`rvc/train/preprocess.py`（切片实现）、`rvc/train/trainer.py`（DataLoader）、
`rvc/train/cli/`（训练向导的提问默认值）。默认值只在这里声明一次，
各处引用常量而非字面量 —— 改一处即全链路生效。
"""

HUBERT_SAMPLE_RATE = 16000
HUBERT_FRAME_SIZE = 160

# F0 置信度阈值（低于此值判为静音/UV）。
# 训练与推理数值不同是有意为之（历史口径），单源定义防止双处漂移：
# - 训练侧 0.03：RMVPE 官方推荐值，批量切片底噪干净；
# - 推理侧 0.05：实时链路底噪更脏，用户可经 InferenceParams.f0.* 调整。
RMVPE_THRESHOLD_TRAIN = 0.03
RMVPE_THRESHOLD_INFER = 0.05
FCPE_CONFIDENCE_THRESHOLD_INFER = 0.05

# ── 训练预处理默认值（PreProcessor / Slicer 的签名默认值同时引用这里）──
# 切片时长（秒）：单条训练切片的目标长度
SLICE_PERIOD_DEFAULT = 3.7
# 相邻切片重叠（秒）：保证切片边界处的音素不被切碎
SLICE_OVERLAP_DEFAULT = 0.3
# 切片前高通截止（Hz）：滤掉素材里的直流与低频隆隆声
PREPROCESS_HIGHPASS_DEFAULT = 48.0
# 静音切分（Slicer）：低于阈值(dBFS)判为静音，据此在静音中间下刀
SLICE_THRESHOLD_DEFAULT = -42.0
# 最短切片（毫秒）：短于它的段落不会被单独切出来
SLICE_MIN_LENGTH_DEFAULT = 1500
# 最短静音间隔（毫秒）：短于此长度的静音不当作切点
SLICE_MIN_INTERVAL_DEFAULT = 400
# RMS 分析的帧步长（毫秒）：越小对静音边界越敏感（帧长 = 2 倍步长）
SLICE_HOP_SIZE_DEFAULT = 15
# 单个切点最多往切片里保留的静音（毫秒）
SLICE_MAX_SIL_KEPT_DEFAULT = 500

# DataLoader 取数进程数（0 = 主进程单线程取数，排查卡死时可用）
TRAIN_NUM_WORKERS_DEFAULT = 2
