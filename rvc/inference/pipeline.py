"""实时语音转换管线 — HuBERT + 合成器 + FAISS + F0 提取"""
import logging

import numpy as np
import torch

from rvc.models.inference_cache import default_inference_cache
from rvc.inference.f0_extractor import F0_MIN, F0_MAX
from rvc.inference.feature_processing import clone_protect_source, extract_hubert_features, upsample_features
from rvc.inference.index_retrieval import apply_faiss_index, load_index
from rvc.inference.model_session import load_model_session
from rvc.inference.pitch_tracker import create_pitch_cache, prepare_offline_pitch, update_realtime_pitch_cache
from rvc.inference.synthesis import apply_formant_resample, cached_long_tensor, cast_pitch_tensors, infer_offline_audio, infer_realtime_audio

logger = logging.getLogger(__name__)

class VCPipeline:
    """实时语音转换管线。

    用法:
        pipeline = VCPipeline(config, pth_path, index_path, index_rate)
        pipeline.load()  # 加载所有模型
        output = pipeline.infer(input_wav_tensor, block_samples_16k, skip_head, ret_len, "fcpe")
    """

    def __init__(self, config, pth_path, index_path="", index_rate=0.0, inference_cache=None):
        self.config = config
        self.device = config.device
        self.is_half = config.is_half
        self.inference_cache = inference_cache or default_inference_cache
        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate

        self.f0_semitones = 0
        self.formant_factor = 0.0
        self.f0_min = F0_MIN
        self.f0_max = F0_MAX
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.pitch_cache, self.pitchf_cache = create_pitch_cache(self.device)

        self.hubert_model = None       # HuBERT
        self.synthesizer = None       # Synthesizer model
        self.target_sr = None
        self.use_f0 = 1
        self.index = None
        self.index_vectors = None
        self.resample_kernel = {}
        self.model_rmvpe = None
        self.model_fcpe = None
        self._long_tensor_cache = {}
        self._padding_mask_cache = {}

    def _cached_long_tensor(self, value: int) -> torch.Tensor:
        return cached_long_tensor(self._long_tensor_cache, value, self.device)

    def load(self) -> None:
        session = load_model_session(self.config, self.pth_path, self.index_path, self.index_rate, self.inference_cache)
        self.hubert_model = session["hubert"]
        self.synthesizer = session["synthesizer"]
        self.target_sr = session["target_sr"]
        self.use_f0 = session["use_f0"]
        self.ckpt_version = session["ckpt_version"]
        self.index = session["index"]
        self.index_vectors = session["index_vectors"]

    def change_key(self, key: int) -> None:
        self.f0_semitones = key

    def change_formant(self, shift: float) -> None:
        self.formant_factor = shift

    def change_index_rate(self, rate: float) -> None:
        if rate > 0 and self.index is None:
            self.index, self.index_vectors = load_index(self.index_path, self.inference_cache)
        self.index_rate = rate

    def _extract_hubert_features(self, input_wav):
        return extract_hubert_features(self.hubert_model, input_wav, self.device, self.is_half, self._padding_mask_cache)

    def _clone_protect_source(self, feats, protect):
        return clone_protect_source(feats, self.use_f0, protect)

    def _apply_faiss_index(self, feats, skip_head=0):
        return apply_faiss_index(feats, self.index, self.index_vectors, self.index_rate, self.is_half, self.device, skip_head)

    def _upsample_features(self, feats, p_len, feats0=None, pitchf=None, protect=0.0):
        return upsample_features(feats, p_len, self.is_half, feats0, pitchf, protect)

    def _cast_pitch_tensors(self, pitch, pitchf):
        return cast_pitch_tensors(pitch, pitchf, self.is_half)

    def infer_offline(self, input_wav, f0method="fcpe", protect=0.0):
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
        factor = pow(2, self.formant_factor / 12)

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
            )
        feats = self._extract_hubert_features(input_wav)
        feats0 = self._clone_protect_source(feats, protect)
        feats = self._apply_faiss_index(feats, 0)
        feats = self._upsample_features(feats, p_len, feats0, pitchf, protect)

        p_len_t = self._cached_long_tensor(p_len)
        sid = self._cached_long_tensor(0)
        with torch.no_grad():
            result = infer_offline_audio(self.synthesizer, feats, p_len_t, pitch, pitchf, sid, self.use_f0, self.is_half)

        audio = result[0][0, 0].data.float()
        audio = apply_formant_resample(audio, factor, self.target_sr, self.resample_kernel, self.device)

        return audio.cpu().numpy()

    def infer(self, input_wav: torch.Tensor, block_frame_16k: int, skip_head: int, return_length: int, f0method: str = "fcpe", protect: float = 0.0) -> torch.Tensor:
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
        feats = self._extract_hubert_features(input_wav)

        feats0 = self._clone_protect_source(feats, protect)
        feats = self._apply_faiss_index(feats, skip_head)

        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_factor / 12)
        return_length2_val = int(np.ceil(return_length * factor))
        if self.use_f0 == 1:
            cache_pitch, cache_pitchf = update_realtime_pitch_cache(
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
            )
        else:
            cache_pitch = cache_pitchf = None

        feats = self._upsample_features(
            feats,
            p_len,
            feats0,
            cache_pitchf.clone() if feats0 is not None else None,
            protect,
        )

        p_len_t = self._cached_long_tensor(p_len)
        sid = self._cached_long_tensor(0)
        skip_head_t = self._cached_long_tensor(skip_head)
        return_length_t = self._cached_long_tensor(return_length)
        return_length2 = self._cached_long_tensor(return_length2_val)

        infered_audio, _, _ = infer_realtime_audio(
            self.synthesizer,
            feats,
            p_len_t,
            cache_pitch,
            cache_pitchf,
            sid,
            skip_head_t,
            return_length_t,
            return_length2,
            self.use_f0,
            self.is_half,
        )

        infered_audio = infered_audio.squeeze(1).float()

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
