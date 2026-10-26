"""离线文件推理的纯步骤 — 读音频 / 逐块喂实时管线 / 写结果。

本模块只做「文件 ↔ 块流」的搬运，不含任何调度与状态：
- 谁创建 runner、什么时机重置效果器、运行时参数怎么更新，全在
  VoiceEngine.process_file 里（那是引擎的生命周期，不属于文件 IO）。
- 逐块走的是同一条 InferenceRunner.process_block，因此离线与实时音质一致。

分成模块而不是留在 engine 里的原因：engine 是门面（实时流 + 离线 + 模型
生命周期三件事），文件 IO 与 pad/切块循环属于可独立测试的纯逻辑，
放一起会让门面里混进 np.pad / 写 wav 这类与设备流无关的细节。
"""
import logging

import numpy as np

from rvc.core.errors import CancelledError

logger = logging.getLogger(__name__)


def load_audio_at_sr(input_path: str, sr: int) -> np.ndarray:
    """加载音频并重采样到目标采样率（float32, 单声道, C 连续）。"""
    from rvc.io.audio_file import load_audio

    wav, _ = load_audio(input_path, sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def run_stream(wav: np.ndarray, runner, block: int, pad: int,
               progress_cb=None, cancel_cb=None) -> np.ndarray:
    """把整段音频按块喂给实时推理，返回裁剪掉 pad 的输出。

    Args:
        wav: 输入音频（单声道 float32）
        runner: InferenceRunner（已 init_processing）
        block: 块样本数
        pad: 两端补零样本数（给 SOLA/上下文留边）
        progress_cb: 可选 (done, total) 回调
        cancel_cb: 可选回调，返回 True 时抛 CancelledError 中止
    """
    # lookahead 是输出时间轴上的固定延迟；右端需要额外喂入一段尾部数据，
    # 否则 pad 较小时，最后一个输入块虽然完成推理，裁剪时却拿不到完整结果。
    try:
        lookahead = max(
            0, int(getattr(getattr(runner, "state", None), "lookahead_samples", 0)))
    except (TypeError, ValueError):
        lookahead = 0
    padded = np.pad(
        wav, (pad, pad),
        # reflect 要求 pad < len；超短音频（<3s pad）退回常数 pad
        mode="reflect" if len(wav) > pad else "constant",
    )
    # 右侧已有 pad 样本也可承担一部分未来上下文，只补不足的部分，避免默认
    # 3 秒 pad 场景额外多跑一个完整块。
    extra_tail = max(lookahead - pad, 0)
    if extra_tail:
        padded = np.concatenate(
            [padded, np.zeros(extra_tail, dtype=np.float32)])
    if len(padded) % block:
        padded = np.concatenate(
            [padded, np.zeros(block - len(padded) % block, dtype=np.float32)])

    total_blocks = len(padded) // block
    out_blocks = []
    for i in range(total_blocks):
        if cancel_cb and cancel_cb():
            raise CancelledError("离线推理已取消")
        seg = padded[i * block: (i + 1) * block]
        outdata = np.zeros((block, 1), dtype=np.float32)
        runner.process_block(seg, outdata, block)
        out_blocks.append(outdata[:, 0])
        if progress_cb:
            progress_cb(i + 1, total_blocks)
    rendered = np.concatenate(out_blocks)
    # 实时链路为稳定 HuBERT 未来上下文会延迟一个输入块；离线结果需要移除
    # 这段算法延迟，才能与原始文件时间轴对齐。测试用的轻量 runner 没有该
    # 字段时保持旧行为。
    start = pad + lookahead
    return rendered[start: start + len(wav)]


def write_output(result: np.ndarray, output_path: str, sr: int) -> None:
    """写出 wav（与实时推理完全一致：process_block 内已硬钳位，不做峰值归一化）。"""
    from rvc.io.wav_file import write_wav

    write_wav(output_path, result, sr, subtype="FLOAT")
