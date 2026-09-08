"""实时语音转换管线 — HuBERT + 合成器 + F0 提取。

无状态设计：推理参数通过 InferenceConfig 每次传入，pipeline 只持有缓存状态
（pitch_cache / resample_kernel / long_tensor_cache）。消除历史上 configure()
同步链导致的参数双份源问题。
"""
from rvc.audio.constants import HUBERT_FRAME_SIZE
import logging
from types import SimpleNamespace

import numpy as np
import torch

from rvc.core.config import InferenceConfig
from rvc.inference.inference_cache import default_inference_cache
from rvc.inference.feature_processing import clone_protect_source, extract_hubert_features, upsample_features
from rvc.inference.model_session import ModelSessionManager
from rvc.inference.pitch_tracker import create_pitch_cache, update_realtime_pitch_cache
from rvc.inference.synthesis import apply_formant_resample, cached_long_tensor, infer_synth_audio

logger = logging.getLogger(__name__)


class InferencePipeline:
    """实时语音转换管线。

    用法:
        pipeline = InferencePipeline(device_config, pth_path, hubert="chinese")
        pipeline.load()
        output = pipeline.infer(input_wav, inference_config, block_16k, skip_head, ret_len)
    """

    def __init__(self, device_config, pth_path, inference_cache=None, hubert: str = "chinese"):
        """初始化推理管线。

        Args:
            device_config: 设备配置（device/is_half）
            pth_path: 模型权重文件路径
            inference_cache: 推理缓存（默认使用全局缓存）
            hubert: HuBERT 模型变体（chinese/base/japanese 等）
        """
        self.device = device_config.device
        self.is_half = device_config.is_half
        self.inference_cache = inference_cache or default_inference_cache
        self.pth_path = pth_path
        self.hubert_variant = hubert

        # 仅缓存状态，不持有推理参数（参数每次 infer 从 config 传入）
        self.pitch_cache, self.pitchf_cache = create_pitch_cache(self.device)
        self.resample_kernel = {}
        self._long_tensor_cache = {}

        # 模型引用（load 后填充）
        self.hubert_model = None
        self.synthesizer = None
        self.target_sr = None
        self.use_f0 = 1

    def load(self) -> None:
        """加载模型（HuBERT + 合成器），通过 ModelSessionManager 缓存复用。"""
        manager = ModelSessionManager(
            SimpleNamespace(device=self.device, is_half=self.is_half),
            self.inference_cache,
        )
        session = manager.load(self.pth_path, hubert_variant=self.hubert_variant)
        self.hubert_model = session.hubert
        self.synthesizer = session.synthesizer
        self.target_sr = session.target_sr
        self.use_f0 = session.use_f0

    def reset_pitch_cache(self) -> None:
        """重置音高缓存（切换模型/文件时调用，避免跨上下文污染）。"""
        self.pitch_cache.zero_()
        self.pitchf_cache.zero_()

    @torch.no_grad()
    def infer(self, input_wav: torch.Tensor, config: InferenceConfig,
              block_frame_16k: int, skip_head: int, return_length: int) -> torch.Tensor:
        """实时推理一个音频块。

        Args:
            input_wav: 滚动缓冲区 (16kHz, GPU)
            config: 推理参数（音高/音色/保护/降噪/破音）
            block_frame_16k: 本块新增的 16kHz 采样数
            skip_head: 跳过的 10ms 帧数（上下文）
            return_length: 需要返回的 10ms 帧数

        Returns:
            合成音频 (target_sr 采样率)
        """
        return self._infer_impl(input_wav, config, block_frame_16k, skip_head, return_length)

    def _infer_impl(self, input_wav, config: InferenceConfig, block_frame_16k, skip_head, return_length):
        """推理实现：特征提取 → F0 跟踪 → 特征上采样 → 合成 → 后处理。

        Args:
            input_wav: 16kHz 滚动缓冲区（GPU tensor）
            config: 推理参数配置
            block_frame_16k: 本块新增的 16kHz 采样数
            skip_head: 跳过的 10ms 帧数（上下文前缀）
            return_length: 需要返回的 10ms 帧数

        Returns:
            合成音频张量（target_sr 采样率）
        """
        p_len = input_wav.shape[0] // HUBERT_FRAME_SIZE
        formant_factor = config.formant
        factor = pow(2, formant_factor / 12)
        return_length2_val = int(np.ceil(return_length * factor))

        # f0_proc = (开关, 源临界Hz, 压缩比, 膝宽) — 全部从 config 读取
        bp = config.break_protect
        f0_proc = (bp.enable, bp.src_hz, bp.ratio, bp.knee)

        # 特征提取：HuBERT → 辅音保护克隆
        feats = extract_hubert_features(self.hubert_model, input_wav, self.device, self.is_half)
        feats0 = clone_protect_source(feats, self.use_f0, config.protect)

        # 音高（F0）缓存更新
        if self.use_f0 == 1:
            cache_pitch, cache_pitchf = update_realtime_pitch_cache(
                input_wav, block_frame_16k, p_len,
                return_length, return_length2_val,
                config.pitch - formant_factor,  # f0_up_key
                config.f0_method,
                self.pitch_cache, self.pitchf_cache,
                self.device, self.is_half,
                self.inference_cache, f0_proc,
            )
        else:
            cache_pitch = cache_pitchf = None

        # 特征上采样（含辅音保护混合）
        feats = upsample_features(feats, p_len, self.is_half, feats0, cache_pitchf, config.protect)

        # 合成 + 后处理（formant 重采样）
        infered_audio = self._synthesize_realtime(
            feats, p_len, cache_pitch, cache_pitchf,
            skip_head, return_length, return_length2_val,
        )
        return self._postprocess_realtime(infered_audio, factor, return_length)

    def _synthesize_realtime(self, feats, p_len, cache_pitch, cache_pitchf, skip_head, return_length, return_length2_val):
        """实时合成：调用合成器生成音频。

        Args:
            feats: 上采样后的特征张量
            p_len: 音高帧数
            cache_pitch: 离散 F0 缓存
            cache_pitchf: 连续 F0 缓存
            skip_head: 跳过的帧数
            return_length: 返回帧数
            return_length2_val: formant 调整后的返回帧数

        Returns:
            合成音频张量
        """
        p_len_t = cached_long_tensor(self._long_tensor_cache, p_len, self.device)
        sid = cached_long_tensor(self._long_tensor_cache, 0, self.device)
        infered_audio, _, _ = infer_synth_audio(
            self.synthesizer, feats, p_len_t,
            cache_pitch, cache_pitchf, sid,
            self.use_f0, self.is_half,
            skip_head=skip_head, return_length=return_length, return_length2=return_length2_val,
        )
        return infered_audio.squeeze(1).float()

    def _postprocess_realtime(self, infered_audio, factor, return_length):
        """实时后处理：formant 重采样（如果需要）。

        Args:
            infered_audio: 合成器输出的音频
            factor: formant 缩放因子（2^(formant/12)）
            return_length: 返回帧数

        Returns:
            后处理后的音频张量
        """
        upp_res = int(np.floor(factor * self.target_sr // 100))
        if upp_res != self.target_sr // 100:
            infered_audio = apply_formant_resample(
                infered_audio[:, : return_length * upp_res],
                factor, self.target_sr, self.resample_kernel, self.device,
            )
        return infered_audio.squeeze()
