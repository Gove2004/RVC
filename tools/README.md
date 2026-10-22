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
