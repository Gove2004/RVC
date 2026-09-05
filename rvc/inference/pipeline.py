"""实时语音转换管线 — HuBERT + 合成器 + F0 提取"""
import logging

import numpy as np
import torch

from rvc.models.inference_cache import default_inference_cache
from rvc.inference.feature_processing import clone_protect_source, extract_hubert_features, upsample_features
from rvc.inference.model_session import load_model_session
from rvc.inference.pitch_tracker import create_pitch_cache, prepare_offline_pitch, update_realtime_pitch_cache
from rvc.inference.synthesis import apply_formant_resample, cached_long_tensor, infer_synth_audio

logger = logging.getLogger(__name__)

class VCPipeline:
    """实时语音转换管线。

    用法:
        pipeline = VCPipeline(config, pth_path)
        pipeline.load()  # 加载所有模型
        output = pipeline.infer(input_wav_tensor, block_samples_16k, skip_head, ret_len, "fcpe")
    """

    def __init__(self, config, pth_path, inference_cache=None,
                 hubert: str = "base"):
        self.config = config
        self.device = config.device
        self.is_half = config.is_half
        self.inference_cache = inference_cache or default_inference_cache
        self.pth_path = pth_path
        self.hubert_variant = hubert

        self.f0_semitones = 0
        self.formant_factor = 0.0
        self.formant_factor_pow = 1.0  # pow(2, formant/12)，change_formant 时缓存，避免每块重算
        # 破音保护（用户核心瑕疵）：f0_proc=(开关, 破音临界[源Hz])，
        # extractor 内按 key 换算成变声后临界。源 ≤临界 → ×2 天然安全，原样保留。
        # 压缩比/膝宽为内部固定默认值，用户只调开关 + 临界 Hz。
        self.break_enable = True
        self.break_src_hz = 300.0
        self._f0_proc = (self.break_enable, self.break_src_hz)

        self.pitch_cache, self.pitchf_cache = create_pitch_cache(self.device)

        self.hubert_model = None       # HuBERT
        self.synthesizer = None       # Synthesizer model
        self.target_sr = None
        self.use_f0 = 1
        self.resample_kernel = {}
        self.model_rmvpe = None
        self.model_fcpe = None
        self._long_tensor_cache = {}

    def _cached_long_tensor(self, value: int) -> torch.Tensor:
        return cached_long_tensor(self._long_tensor_cache, value, self.device)

    def load(self) -> None:
        session = load_model_session(self.config, self.pth_path,
                                     self.inference_cache, hubert_variant=self.hubert_variant)
        self.hubert_model = session.hubert
        self.synthesizer = session.synthesizer
        self.target_sr = session.target_sr
        self.use_f0 = session.use_f0

    def configure(self, pitch=None, gender=None, break_enable=None, break_src_hz=None) -> None:
        """统一参数应用入口（实时/离线共用，避免多处手写 change_* 漏改）。

        只更新传入的非 None 字段；f0method/protect 是每次推理的入参，不在此缓存。
        以后新增音质参数只需在 configure 里加一行，实时与离线自动同步。
        """
        if pitch is not None:
            self.f0_semitones = pitch
        if gender is not None:
            self._set_formant(gender)
        if break_enable is not None or break_src_hz is not None:
            self._set_break(
                break_enable if break_enable is not None else self.break_enable,
                break_src_hz if break_src_hz is not None else self.break_src_hz,
            )

    def _set_formant(self, shift: float) -> None:
        self.formant_factor = shift
        self.formant_factor_pow = pow(2, shift / 12)

    def _set_break(self, enable: bool, break_src_hz: float) -> None:
        """更新破音保护参数（破音临界用源赫兹，extractor 内换算变声后）。

        方案 A（压缩比 + 平滑膝，内部固定默认值）：高音区按固定压缩比收窄但仍保留旋律，
        膝宽固定。用户只需调「临界 Hz」（超过开始收敛）与开关。
        """
        self.break_enable = bool(enable)
        self.break_src_hz = max(50.0, float(break_src_hz))
        self._f0_proc = (self.break_enable, self.break_src_hz)

    def _extract_hubert_features(self, input_wav):
        return extract_hubert_features(self.hubert_model, input_wav, self.device, self.is_half)

    def _clone_protect_source(self, feats, protect):
        return clone_protect_source(feats, self.use_f0, protect)

    def _upsample_features(self, feats, p_len, feats0=None, pitchf=None, protect=0.0):
        return upsample_features(feats, p_len, self.is_half, feats0, pitchf, protect)

    @torch.no_grad()
    def infer_offline(self, input_wav, f0method="rmvpe", protect=0.0):
        """离线推理（完整音频）。

        Args:
            input_wav: np.ndarray or torch.Tensor, 输入音频 (16kHz)
            f0method: str, F0 提取方法
            protect: float, 辅音保护强度 [0, 1.0]

        Returns:
            np.ndarray: 合成音频 (target_sr 采样率)
        """
        if not torch.is_tensor(input_wav):
            input_wav = torch.from_numpy(input_wav).float()
        p_len = input_wav.shape[0] // 160

        # 计算 formant 因子
        factor = self.formant_factor_pow

        pitch = pitchf = None
        if self.use_f0 == 1:
            pitch, pitchf = prepare_offline_pitch(
                input_wav,
                p_len,
                self.f0_semitones - self.formant_factor,
                f0method,
                self.device,
                self.is_half,
                self.inference_cache,
                self._f0_proc,
            )
        feats = self._extract_hubert_features(input_wav)
        feats0 = self._clone_protect_source(feats, protect)
        feats = self._upsample_features(feats, p_len, feats0, pitchf, protect)

        p_len_t = self._cached_long_tensor(p_len)
        sid = self._cached_long_tensor(0)
        with torch.no_grad():
            result = infer_synth_audio(self.synthesizer, feats, p_len_t, pitch, pitchf, sid, self.use_f0, self.is_half)

        audio = result[0][0, 0].data.float()
        audio = apply_formant_resample(audio, factor, self.target_sr, self.resample_kernel, self.device)

        return audio.cpu().numpy()

    def infer(self, input_wav: torch.Tensor, block_frame_16k: int, skip_head: int, return_length: int, f0method: str = "rmvpe", protect: float = 0.0) -> torch.Tensor:
        """实时推理一个音频块。

        Args:
            input_wav: torch.Tensor, 滚动缓冲区 (16kHz, GPU)
            block_frame_16k: int, 本块新增的16kHz采样数
            skip_head: int, 跳过的10ms帧数（上下文）
            return_length: int, 需要返回的10ms帧数
            f0method: str, "rmvpe" 或 "fcpe"
            protect: float, 辅音保护强度 [0, 1.0]，值越大保护越强

        Returns:
            torch.Tensor: 合成音频 (target_sr 采样率)
        """
        with torch.no_grad():
            return self._infer_impl(input_wav, block_frame_16k, skip_head, return_length, f0method, protect)

    def _infer_impl(self, input_wav, block_frame_16k, skip_head, return_length, f0method, protect):
        p_len = input_wav.shape[0] // 160
        factor = self.formant_factor_pow
        return_length2_val = int(np.ceil(return_length * factor))

        # 特征提取：HuBERT → 辅音保护克隆
        feats = self._extract_hubert_features(input_wav)
        feats0 = self._clone_protect_source(feats, protect)

        # 音高（F0）缓存更新
        if self.use_f0 == 1:
            cache_pitch, cache_pitchf = self._update_realtime_pitch(
                input_wav, block_frame_16k, p_len,
                return_length, return_length2_val, f0method,
            )
        else:
            cache_pitch = cache_pitchf = None

        # 特征上采样（含辅音保护混合）
        feats = self._upsample_features(feats, p_len, feats0, cache_pitchf, protect)

        # 合成 + 后处理（formant 重采样）
        infered_audio = self._synthesize_realtime(
            feats, p_len, cache_pitch, cache_pitchf,
            skip_head, return_length, return_length2_val,
        )
        return self._postprocess_realtime(infered_audio, factor, return_length)

    def _update_realtime_pitch(self, input_wav, block_frame_16k, p_len, return_length, return_length2_val, f0method):
        """更新实时音高缓存（滑动窗口 + 尾部重算）"""
        return update_realtime_pitch_cache(
            input_wav,
            block_frame_16k,
            p_len,
            return_length,
            return_length2_val,
            self.f0_semitones - self.formant_factor,
            f0method,
            self.pitch_cache,
            self.pitchf_cache,
            self.device,
            self.is_half,
            self.inference_cache,
            self._f0_proc,
        )

    def _synthesize_realtime(self, feats, p_len, cache_pitch, cache_pitchf, skip_head, return_length, return_length2_val):
        """Synthesizer 实时推理，返回 [channels, frames] 音频"""
        p_len_t = self._cached_long_tensor(p_len)
        sid = self._cached_long_tensor(0)
        infered_audio, _, _ = infer_synth_audio(
            self.synthesizer,
            feats,
            p_len_t,
            cache_pitch,
            cache_pitchf,
            sid,
            self.use_f0,
            self.is_half,
            skip_head=skip_head,
            return_length=return_length,
            return_length2=return_length2_val,
        )
        return infered_audio.squeeze(1).float()

    def _postprocess_realtime(self, infered_audio, factor, return_length):
        """formant 重采样（factor≠1 时）"""
        upp_res = int(np.floor(factor * self.target_sr // 100))
        if upp_res != self.target_sr // 100:
            infered_audio = apply_formant_resample(
                infered_audio[:, : return_length * upp_res],
                factor,
                self.target_sr,
                self.resample_kernel,
                self.device,
            )
        return infered_audio.squeeze()
