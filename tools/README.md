# tools/ — 离线小工具

不参与构建、不被任何代码 import，纯粹为调试/教学用。

## `pitch_map_demo.html`

音域映射计算器的可视化演示：拖动源音域 / 目标音域两端，实时看
`apply_pitch_map` 的映射曲线与转换后的音高分布。

- 打开方式：直接用浏览器打开本文件（无需起服务）
- 曲线图依赖 CDN 的 ECharts，离线环境图会空白，滑块与数值仍可用
- 算法口径与 `rvc/pipeline/pitch/postprocess.py:apply_pitch_map` 一致：
  在 MIDI 半音尺度上做线性映射（`k = 目标音域宽度 / 原声音域宽度`），
  映射结果钳制在 `[dst_min, dst_max]`（钳制而非静音）。页面上的 `slope`
  即代码里的 `k`，折线只画原声音域内的映射段。

## `convert_hubert_fp16.py`

把 HuBERT 权重转成 fp16，去掉"假的 fp32 精度"。

本项目**全程 fp16**：推理侧 `rvc/runtime/device.py` 把 `is_half` 硬编码为 True，
HuBERT 加载后立刻 `.to(dtype=torch.float16)`。所以权重文件里的 fp32 低位在加载的
第一瞬间就被丢掉，只白占一倍磁盘，`torch.load` 期间还多占一倍宿主内存。

- 用法：`.\.venv\Scripts\python.exe tools\convert_hubert_fp16.py [变体...] [--dry-run]`
- 幂等：已是全 fp16 会直接跳过，可安全重复运行
- 实测效果：`chinese` 377,552,987 → 188,813,420 字节（省 180.0 MB / 50.0%），
  加载后特征与转换前**逐位相同**（最大绝对差 0.000e+00）
- 权重在 `.gitignore` 内（`assets/hubert/**`），换机器或重新下载会拿回 fp32 版，
  届时重跑本工具即可；`tests/test_half_precision_only.py` 会在权重非 fp16 时报错提醒
- 安全：原文件先改名为 `pytorch_model.fp32.bak`，只有两道验证关卡（逐张量等价、
  端到端特征逐位相等）全部通过才删除；任一失败即回滚
- 不要误删训练侧的 fp32 回退（`rvc/train/cli/envcheck.py`，算力 < 7.0 时启用）：
  合成器权重本来是 fp32，那条路径对它提供的是**真实** fp32 精度，不是死代码
