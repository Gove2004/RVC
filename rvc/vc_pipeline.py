"""实时语音转换管线 — HuBERT + 合成器 + FAISS + F0 提取"""
import logging
import os
import traceback

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample as TatResample

logger = logging.getLogger(__name__)


def faiss_blend(feats_npy, index, big_npy, index_rate, is_half):
    """FAISS k=8 加权混合 — 提取为共享函数供实时/离线推理复用。

    Args:
        feats_npy: np.ndarray, shape (T, D), float32
        index: faiss.Index
        big_npy: np.ndarray, index 全量特征
        index_rate: float, 混合比例
        is_half: bool

    Returns:
        np.ndarray: 混合后的特征 (float16 if is_half else float32)
    """
    k = min(8, index.ntotal)
    score, ix = index.search(feats_npy, k=k)
    valid = (ix >= 0).all()
    if not valid:
        return feats_npy
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    blended = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
    result = blended * index_rate + feats_npy * (1 - index_rate)
    if is_half:
        result = result.astype("float16")
    return result

# F0 提取器模块级缓存（跨 VCPipeline 实例复用）
_rmvpe_cache = {}
_fcpe_cache = {}


class VCPipeline:
    """实时语音转换管线。

    用法:
        pipeline = VCPipeline(config, pth_path, index_path, index_rate)
        pipeline.load()  # 加载所有模型
        output = pipeline.infer(input_wav_tensor, block_frame_16k, skip_head, return_length, "fcpe")
    """

    def __init__(self, config, pth_path, index_path="", index_rate=0.0):
        self.config = config
        self.device = config.device
        self.is_half = config.is_half
        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate

        self.f0_up_key = 0
        self.formant_shift = 0.0
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)

        self.model = None       # HuBERT
        self.net_g = None       # Synthesizer
        self.tgt_sr = None
        self.if_f0 = 1
        self.index = None
        self.big_npy = None
        self.resample_kernel = {}
        self.model_rmvpe = None
        self.model_fcpe = None

    def load(self):
        """加载所有模型（HuBERT + Synthesizer + 可选Index）"""
        logger.info("加载 %s", os.path.basename(self.pth_path))

        # HuBERT
        from rvc.hubert import load_hubert
        self.model = load_hubert(self.config)

        # Synthesizer
        self._load_synthesizer()

        # 移除 weight_norm
        try:
            self.net_g.remove_weight_norm()
        except Exception:
            pass

        # FAISS Index
        if self.index_rate > 0 and self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            logger.info("加载 %s", os.path.basename(self.index_path))

    def _load_synthesizer(self):
        logger.info("加载 Synthesizer")
        ckpt = torch.load(self.pth_path, map_location="cpu", weights_only=False)
        self.tgt_sr = ckpt["config"][-1]
        self.if_f0 = ckpt.get("f0", 1)
        self.version = ckpt.get("version", "v2")
        n_spk = ckpt["config"][-3] = ckpt["weight"]["emb_g.weight"].shape[0]

        from rvc.synthesizer import SynthesizerTrnMsNSFsid, SynthesizerTrnMsNSFsid_nono
        if self.if_f0 == 1:
            self.net_g = SynthesizerTrnMsNSFsid(
                *ckpt["config"],
                is_half=self.is_half,
            )
        else:
            self.net_g = SynthesizerTrnMsNSFsid_nono(
                *ckpt["config"],
                is_half=self.is_half,
            )

        self.net_g.load_state_dict(ckpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        if self.is_half:
            self.net_g.half()

    def change_key(self, key):
        self.f0_up_key = key

    def change_formant(self, shift):
        self.formant_shift = shift

    def change_index_rate(self, rate):
        if rate > 0 and self.index is None and self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        self.index_rate = rate

    def _get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def _get_f0_rmvpe(self, x, f0_up_key):
        cache_key = (self.device, self.is_half)
        if cache_key not in _rmvpe_cache:
            from rvc.rmvpe import RMVPE
            logger.info("加载 RMVPE（一次性）")
            _rmvpe_cache[cache_key] = RMVPE(
                "assets/rmvpe/rmvpe.pt", is_half=self.is_half, device=self.device
            )
        self.model_rmvpe = _rmvpe_cache[cache_key]
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self._get_f0_post(f0)

    def _get_f0_fcpe(self, x, f0_up_key):
        cache_key = self.device
        if cache_key not in _fcpe_cache:
            from torchfcpe import spawn_bundled_infer_model
            logger.info("加载 FCPE...")
            fcpe_logger = logging.getLogger("torchfcpe")
            saved_level = fcpe_logger.level
            fcpe_logger.setLevel(logging.ERROR)
            try:
                _fcpe_cache[cache_key] = spawn_bundled_infer_model(self.device)
            finally:
                fcpe_logger.setLevel(saved_level)
        self.model_fcpe = _fcpe_cache[cache_key]
        f0 = self.model_fcpe.infer(
            x.to(self.device).unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
        f0 *= pow(2, f0_up_key / 12)
        return self._get_f0_post(f0)

    def _get_f0(self, x, f0_up_key, method="fcpe"):
        if method == "rmvpe":
            return self._get_f0_rmvpe(x, f0_up_key)
        return self._get_f0_fcpe(x, f0_up_key)

    def infer(self, input_wav, block_frame_16k, skip_head, return_length, f0method="fcpe"):
        """实时推理一个音频块。

        Args:
            input_wav: torch.Tensor, 滚动缓冲区 (16kHz, GPU)
            block_frame_16k: int, 本块新增的16kHz采样数
            skip_head: int, 跳过的10ms帧数（上下文）
            return_length: int, 需要返回的10ms帧数
            f0method: str, "rmvpe" 或 "fcpe"

        Returns:
            torch.Tensor: 合成音频 (tgt_sr 采样率)
        """
        with torch.no_grad():
            return self._infer_impl(input_wav, block_frame_16k, skip_head, return_length, f0method)

    def _infer_impl(self, input_wav, block_frame_16k, skip_head, return_length, f0method):
        # HuBERT 特征提取
        if self.is_half:
            feats = input_wav.half().view(1, -1)
        else:
            feats = input_wav.float().view(1, -1)
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self.device)
        inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 12}
        logits = self.model.extract_features(**inputs)
        feats = logits[0]
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        # FAISS 索引匹配
        if self.index is not None and self.index_rate > 0:
            try:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                blended = faiss_blend(npy, self.index, self.big_npy, self.index_rate, self.is_half)
                feats[0][skip_head // 2 :] = torch.from_numpy(blended).to(self.device)
            except Exception:
                logger.debug("索引匹配失败: %s", traceback.format_exc())

        # F0 提取
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2_val = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self._get_f0(
                input_wav[-f0_extractor_frame:], self.f0_up_key - self.formant_shift, f0method
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2_val / return_length

        # 上采样特征
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_t = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_t = torch.LongTensor([skip_head]).to(self.device)
        return_length_t = torch.LongTensor([return_length]).to(self.device)
        return_length2 = torch.LongTensor([return_length2_val])

        # 合成
        if self.if_f0 == 1:
            infered_audio, _, _ = self.net_g.infer(
                feats, p_len_t, cache_pitch, cache_pitchf, sid,
                skip_head_t, return_length_t, return_length2,
            )
        else:
            infered_audio, _, _ = self.net_g.infer(
                feats, p_len_t, sid, skip_head_t, return_length_t, return_length2
            )

        infered_audio = infered_audio.squeeze(1).float()

        # 性别因子: 重采样回目标采样率
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                if len(self.resample_kernel) >= 16:
                    self.resample_kernel.clear()
                self.resample_kernel[upp_res] = TatResample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )

        return infered_audio.squeeze()
