"""dsp — 纯信号处理数学，无 torch 模型、无编排逻辑。

只允许依赖：numpy / torch / 标准库 / rvc.core。禁止依赖任何上层包
（models/io/pipeline/streaming/train/gui/autodl），由 tests/test_architecture.py 守护。
"""
