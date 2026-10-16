"""跨层共享常量 — 音频基础参数、F0 阈值、训练预处理默认值。

放在 `rvc/core` 是因为下面几处都要用到，而它们分属不同层：
`rvc/train/preprocess.py`（切片/滤波实现）、`rvc/train/trainer.py`（DataLoader 进程数）、
`rvc/train/cli/`（切片时长默认值）。默认值只在这里声明一次，
各处引用常量而非字面量 —— 改一处即全链路生效。
"""

HUBERT_SAMPLE_RATE = 16000
HUBERT_FRAME_SIZE = 160

# 训练样本有效帧数范围（phone 帧数 = 特征帧数 ×2）。
# 单源定义防止双处漂移：_filter 用它过滤样本，BucketSampler 用它生成分桶边界——
# 上限改了但边界表没跟上时，末段样本会进数据集却全部被 sampler 跳过并刷警告。
TRAIN_FRAME_MIN = 100
TRAIN_FRAME_MAX = 900

# 训练/预处理扫描的音频扩展名单源：预处理指纹核对（preprocess.py）与向导素材
# 扫描（train/cli）必须同一口径，新增格式只改这里。
AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"})

# F0 置信度阈值（低于此值判为静音/UV）。
# 训练与推理数值不同是有意为之（历史口径），单源定义防止双处漂移：
# - 训练侧 0.03：RMVPE 官方推荐值，批量切片底噪干净；
# - 推理侧 0.05：实时链路底噪更脏，用户可经 InferenceParams.f0.* 调整。
RMVPE_THRESHOLD_TRAIN = 0.03
RMVPE_THRESHOLD_INFER = 0.05
FCPE_CONFIDENCE_THRESHOLD_INFER = 0.05

# F0 提取的尾部窗口（秒）。F0 的参考窗与 HuBERT 的上下文窗（extra_time）无关：
# 最终只有 p_len 末尾的 return_length 帧进合成器，且 F0 网络是局部感受野 + 因果
# 中值滤波，不需要整段上下文。
# 实测口径（真实 5.16s 缓冲、0.1s 块）——尾窗 vs 整段逐帧比较：
#   1.0s  半音Δ中位 -0.0030 / 最大 0.019 / 清浊 0 不一致
#   0.6s  半音Δ中位 -0.0035 / 最大 0.032 / 清浊 0 不一致   ← 取值
#   0.5s  半音Δ中位 -0.0036 / 最大 0.034 / 清浊 0 不一致
#   0.4s                                         最大 0.052（开始劣化）
# 0.6s 相对 5.16s 省约 2.2ms/块。**改动前必须确认 `_sync_f0_parallel` 的
# 右对齐分支生效**：F0 帧数少于 p_len 时要左补零，否则 F0 与特征错位。
FCPE_TAIL_WINDOW_SEC = 0.6

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

# ── 切片归一化（preprocess.normalize_audio）──
# 目标峰值（归一化到满幅的 90%，留 10% 余量防爆音）
NORMALIZE_PEAK_DEFAULT = 0.9
# 归一化分量占比（剩余 1-MIX 混回原始动态，避免过度压缩动态范围）
NORMALIZE_DYNAMIC_MIX = 0.75
# 峰值超过此值视为坏样本（削波失真），直接丢弃
BAD_SAMPLE_PEAK = 2.5
