"""缓冲区几何与 formant 档位量程 — 纯算术，不碰设备、不分配张量。

从 InferenceRunner.init_processing / prewarm_formant 抽出：这两处的数值契约
（10ms 对齐、SOLA 重叠上限、HuBERT 帧数换算、预捕获档位覆盖）此前只能靠开流
实跑验证，抽出后无需 GPU 即可锁住。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rvc.core.constants import HUBERT_FRAME_SIZE


@dataclass(frozen=True)
class BufferGeometry:
    """一次 init_processing 算出的全部长度参数，单位见各字段名。"""

    samples_per_centis: int
    block_samples: int
    crossfade_samples: int
    sola_buffer_samples: int
    sola_search_samples: int
    extra_samples: int
    block_samples_16k: int
    skip_head: int
    return_length: int
    work_buffer_samples: int
    buffer_16k_samples: int


def buffer_geometry(work_sr: int, block_time: float, crossfade_time: float,
                    extra_time: float) -> BufferGeometry:
    """把时长参数换算成样本数与帧数，全部对齐到 10ms 边界。

    Args:
        work_sr: 工作采样率（流实际跑的采样率，不必是模型率）
        block_time / crossfade_time / extra_time: 秒
    """
    zc = work_sr // 100
    block_samples = int(np.round(block_time * work_sr / zc)) * zc
    crossfade_samples = int(np.round(crossfade_time * work_sr / zc)) * zc
    # 重叠上限 100ms。上限只决定参数"能调到多长"，实际取多长由用户自己的
    # crossfade_time 决定。重叠区是旧块尾部与新块头部按 sola_offset 错位相加，
    # 错位相加会周期性相消；重叠越长，被相消区替换掉的原始音频越多。
    sola_buffer_samples = min(crossfade_samples, 10 * zc)
    sola_search_samples = zc
    extra_samples = int(np.round(extra_time * work_sr / zc)) * zc
    return BufferGeometry(
        samples_per_centis=zc,
        block_samples=block_samples,
        crossfade_samples=crossfade_samples,
        sola_buffer_samples=sola_buffer_samples,
        sola_search_samples=sola_search_samples,
        extra_samples=extra_samples,
        block_samples_16k=HUBERT_FRAME_SIZE * block_samples // zc,
        skip_head=extra_samples // zc,
        # 传给合成器的帧数 = 当前块 + SOLA 缓冲 + SOLA 搜索。尾窗只有
        # return_length 帧进合成器；输出侧按同样三项之和截齐。
        return_length=(block_samples + sola_buffer_samples + sola_search_samples) // zc,
        work_buffer_samples=extra_samples + crossfade_samples + sola_search_samples + block_samples,
        buffer_16k_samples=HUBERT_FRAME_SIZE
        * (extra_samples + crossfade_samples + sola_search_samples + block_samples) // zc,
    )


def formant_frame_range(return_length: int, formant_min: float,
                        formant_max: float) -> range:
    """formant 量程内会出现的全部合成器帧数（含两端）。

    必须与 `rvc.pipeline.synthesis.formant_frames` 的取整口径一致（ceil 后
    max(1, …)）：预捕获少覆盖一档，运行中拖到那一档就会在音频回调线程里现场
    捕获，对听者就是一次断音。
    """
    lo = max(1, int(math.ceil(return_length * 2.0 ** (formant_min / 12.0))))
    hi = int(math.ceil(return_length * 2.0 ** (formant_max / 12.0)))
    return range(lo, hi + 1)