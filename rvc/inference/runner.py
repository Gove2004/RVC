"""推理运行器 — 实时/离线统一入口。

历史上实时推理走 RealtimeEngine._run_inference → VCPipeline.infer，
离线走 RealtimeEngine.process_file（8 个可选参数，pitch 缓存跨文件污染）。
本模块提供唯一推理入口 InferenceRunner.process_block，实时与离线共用，
消除双份推理逻辑和参数漂移。

用法:
    runner = InferenceRunner(pipeline)
    output = runner.process_block(input_wav, config, block_16k, skip_head, ret_len)
    runner.reset()  # 切换文件/模型时重置 pitch 缓存
"""
import logging

import torch

from rvc.config import InferenceConfig
from rvc.inference.pipeline import VCPipeline

logger = logging.getLogger(__name__)


class InferenceRunner:
    """统一推理运行器 — 持有 VCPipeline，提供 process_block 唯一入口。"""

    def __init__(self, pipeline: VCPipeline):
        self.pipeline = pipeline

    @property
    def target_sr(self) -> int:
        return self.pipeline.target_sr

    @property
    def use_f0(self) -> int:
        return self.pipeline.use_f0

    def process_block(self, input_wav: torch.Tensor, config: InferenceConfig,
                      block_frame_16k: int, skip_head: int, return_length: int) -> torch.Tensor:
        """推理一个音频块（实时/离线共用）。

        Args:
            input_wav: 滚动缓冲区 (16kHz, GPU)
            config: 推理参数
            block_frame_16k: 本块新增的 16kHz 采样数
            skip_head: 跳过的 10ms 帧数（上下文）
            return_length: 需要返回的 10ms 帧数

        Returns:
            合成音频 (target_sr 采样率)
        """
        return self.pipeline.infer(input_wav, config, block_frame_16k, skip_head, return_length)

    def reset(self) -> None:
        """重置 pitch 缓存（切换文件/模型时调用，避免跨上下文污染）。"""
        self.pipeline.reset_pitch_cache()
