"""实时语音转换管线 — 阶段3-6（HuBERT → F0 → 合成预处理 → 合成）。

架构重构后：
- 持有 EngineState（跨块持续状态：模型引用、F0缓存、合成器缓存等）
- 每块复用 InferenceContext（单块临时状态）
- infer() 按阶段执行，去掉了清辅音保护（uv_prob / protect_blend / 清音保护）

对外接口保持不变：__init__(device_config, pth_path, inference_cache, hubert) / load() / infer()
"""
import logging
import math
from types import SimpleNamespace

import torch

from rvc.audio.constants import HUBERT_FRAME_SIZE
from rvc.core.config import InferenceConfig
from rvc.inference.engine_state import EngineState
from rvc.inference.inference_context import InferenceContext
from rvc.inference.inference_cache import default_inference_cache
from rvc.inference.feature_processing import extract_hubert_features, upsample_features
from rvc.inference.f0_extractor import postprocess_f0
from rvc.inference.model_session import ModelSessionManager
from rvc.inference.pitch_tracker import create_pitch_cache, update_realtime_pitch_cache_raw
from rvc.inference.synthesis import cached_long_tensor, infer_synth_audio

logger = logging.getLogger(__name__)


class InferencePipeline:
    """实时语音转换管线 — 阶段3-6。

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
        # 创建跨块持续状态
        self.state = EngineState()
        self.state.device = device_config.device
        self.state.is_half = device_config.is_half
        self.state.inference_cache = inference_cache or default_inference_cache

        self.pth_path = pth_path
        self.hubert_variant = hubert

        # 初始化 F0 滚动缓存
        self.state.pitch_cache, self.state.pitchf_cache, self.state.confidence_cache = create_pitch_cache(self.state.device)

        # 单块上下文（复用实例，每块 reset()）
        self.ctx = InferenceContext()

        # 呼吸感动态调制：因果 EMA 上一帧值（None 表示未初始化）

    # ── 便捷属性（对外接口保持不变） ──

    @property
    def device(self) -> str:
        return self.state.device

    @property
    def is_half(self) -> bool:
        return self.state.is_half

    @property
    def target_sr(self) -> int:
        return self.state.target_sr

    @property
    def use_f0(self) -> int:
        return self.state.use_f0

    @property
    def hubert_model(self):
        return self.state.hubert_model

    @property
    def synthesizer(self):
        return self.state.synthesizer

    # ── 模型加载 ──

    def load(self) -> None:
        """加载模型（HuBERT + 合成器），通过 ModelSessionManager 缓存复用。"""
        manager = ModelSessionManager(
            SimpleNamespace(device=self.state.device, is_half=self.state.is_half),
            self.state.inference_cache,
        )
        session = manager.load(self.pth_path, hubert_variant=self.hubert_variant)
        self.state.hubert_model = session.hubert
        self.state.synthesizer = session.synthesizer
        self.state.target_sr = session.target_sr
        self.state.use_f0 = session.use_f0

    def reset_pitch_cache(self) -> None:
        """重置音高缓存（切换模型/文件时调用，避免跨上下文污染）。"""
        self.state.reset_pitch_cache()

    # ── 主推理入口 ──

    @torch.no_grad()
    def infer(self, input_wav: torch.Tensor, config: InferenceConfig,
              block_frame_16k: int, skip_head: int, return_length: int) -> torch.Tensor:
        """实时推理一个音频块（阶段3-6）。

        Args:
            input_wav: 滚动缓冲区 (16kHz, GPU)
            config: 推理参数（音高/音色/formant/pitch_map 等）
            block_frame_16k: 本块新增的 16kHz 采样数
            skip_head: 跳过的 10ms 帧数（上下文）
            return_length: 需要返回的 10ms 帧数

        Returns:
            合成音频 (target_sr 采样率)
        """
        state = self.state
        ctx = self.ctx

        # 重置单块上下文
        ctx.reset(config, block_frame_16k, 0)
        ctx.p_len = input_wav.shape[0] // HUBERT_FRAME_SIZE

        # 把参数存到 state（runner 层也可能设置，这里确保一致）
        state.input_wav_16k = input_wav
        state.skip_head = skip_head
        state.return_length = return_length

        # formant 因子计算
        formant_factor = config.formant
        factor = pow(2, formant_factor / 12)
        state.return_length2 = int(math.ceil(return_length * factor))

        # 阶段3：HuBERT 特征提取（50fps）
        feats = self._stage_extract_hubert(ctx)

        # 阶段4：F0 原始提取（只提取原始 F0 + confidence，更新缓存）
        self._stage_extract_f0_raw(ctx)

        # 阶段5：合成预处理（音域映射 + 中值滤波 + 离散化）
        pitch, pitchf = self._stage_postprocess_f0(ctx)

        # 阶段5：特征上采样（50fps → 100fps，截取 p_len 帧）+ 清辅音保护
        feats = upsample_features(
            feats, ctx.p_len, state.is_half,
            feats0=feats, pitchf=pitchf, protect=config.protect,
            protect_threshold_hz=config.protect_threshold_hz,
        )
        ctx.features_upsampled = feats

        # 阶段6：合成（formant 后处理移到引擎层阶段7a）
        return self._stage_synthesize(ctx, feats, pitch, pitchf)

    # ── 阶段3：HuBERT 特征提取 ──

    def _stage_extract_hubert(self, ctx: InferenceContext) -> torch.Tensor:
        """阶段3：从 16k 滚动缓冲区提取 HuBERT 特征（50fps）。"""
        feats = extract_hubert_features(
            self.state.hubert_model,
            self.state.input_wav_16k,
            self.state.device,
            self.state.is_half,
        )
        ctx.hubert_features = feats
        return feats

    # ── 阶段4：F0 原始提取 ──

    def _stage_extract_f0_raw(self, ctx: InferenceContext) -> None:
        """阶段4：F0 原始提取 + 缓存更新。

        只调用 extract_raw 获取原始 (f0, confidence)，写入缓存。
        不做后处理（音域映射/中值滤波/离散化），后处理在阶段5完成。
        结构上与阶段3（HuBERT）独立，预留 CUDA Stream 并行接口。

        use_f0=0 时直接返回，不做任何操作。
        """
        state = self.state
        if state.use_f0 == 0:
            return

        f0_raw, confidence_raw = update_realtime_pitch_cache_raw(
            state.input_wav_16k,
            ctx.block_frame_16k,
            ctx.p_len,
            ctx.config.f0_method,
            state.pitch_cache,
            state.pitchf_cache,
            state.confidence_cache,
            state.device,
            state.is_half,
            state.inference_cache,
            config=ctx.config,
        )
        ctx.f0_raw = f0_raw
        ctx.confidence_raw = confidence_raw

    # ── 阶段5：F0 后处理（音域映射 + 中值滤波 + 离散化）──

    def _stage_postprocess_f0(self, ctx: InferenceContext):
        """阶段5：F0 后处理（音域映射 + 中值滤波 + 离散化）。

        从 ctx 读取阶段4提取的原始 F0，调用 postprocess_f0 做后处理。
        后处理后的离散 F0 写入 cache_pitch 的最后 p_len 帧。
        pitchf 乘以 return_length2/return_length 做 formant 缩放。

        Returns:
            (pitch, pitchf): 离散 F0 (1, p_len)、连续 F0 (1, p_len)
            use_f0=0 时返回 (None, None)
        """
        state = self.state
        if state.use_f0 == 0:
            return None, None

        pitch, pitchf, confidence = postprocess_f0(
            ctx.f0_raw.squeeze(0),
            state.device,
            confidence=ctx.confidence_raw.squeeze(0),
            config=ctx.config,
        )
        # formant 缩放（pitchf 需要乘以 return_length2 / return_length）
        pitchf = pitchf * state.return_length2 / state.return_length

        # 把后处理后的离散 F0 写入 cache_pitch 的最后 p_len 帧
        state.pitch_cache[-ctx.p_len:] = pitch

        ctx.f0_mapped = pitchf
        ctx.pitch_discrete = pitch[None, :]
        ctx.pitchf_continuous = pitchf[None, :]
        return pitch[None, :], pitchf[None, :]

    def _compute_noise_mod(self, confidence: torch.Tensor, pitchf: torch.Tensor, breathiness: float) -> torch.Tensor:
        """计算逐帧噪声调制系数（呼吸感动态建模）。

        用 F0 置信度作为呼吸感代理：confidence 低 = 周期性弱 = 气声多 → 增大噪声。
        breathiness 控制整体调制强度（0=关闭，1=完全启用）。
        对所有帧应用调制（清音帧 confidence 低，噪声自然增大）。

        Args:
            confidence: 置信度 (1, T)，0-1
            pitchf: 连续 F0 (1, T)，保留参数兼容性
            breathiness: 呼吸感强度 (0-1)

        Returns:
            noise_mod: 噪声调制系数 (1, T)，1.0=默认
        """
        if breathiness <= 0:
            return torch.ones_like(confidence)

        # 映射：confidence=1.0 → base=0.05（几乎无噪声），confidence=0.0 → base=20.0（噪声增20倍，超级夸张）
        base = 20.0 - 19.95 * confidence
        base = base.clamp(0.05, 20.0)

        # 按 breathiness 强度混合：breathiness=0 → 全1.0，breathiness=1 → 完全启用
        noise_mod = 1.0 + breathiness * (base - 1.0)

        return noise_mod

    # ── 阶段6：合成 ──

    def _stage_synthesize(
        self,
        ctx: InferenceContext,
        feats: torch.Tensor,
        pitch,
        pitchf,
    ) -> torch.Tensor:
        """阶段6：合成器推理。

        formant 重采样已移到引擎层阶段7a，这里只返回合成器原始输出。

        Args:
            feats: 上采样后的特征 (1, p_len, C)，100fps
            pitch: 离散 F0 (1, p_len)，use_f0=0 时为 None
            pitchf: 连续 F0 (1, p_len)，use_f0=0 时为 None

        Returns:
            合成音频张量（target_sr 采样率，未做 formant 后处理）
        """
        state = self.state

        # 呼吸感动态调制：用 confidence 计算逐帧噪声调制系数
        noise_mod = None
        if state.use_f0 == 1 and state.confidence_cache is not None and pitchf is not None:
            conf = state.confidence_cache[-ctx.p_len:][None, :]
            pf = pitchf if pitchf.dim() == 2 else pitchf[None, :]
            noise_mod = self._compute_noise_mod(conf, pf, ctx.config.breathiness)

        # 合成器推理
        p_len_t = cached_long_tensor(state.long_tensor_cache, ctx.p_len, state.device)
        sid = cached_long_tensor(state.long_tensor_cache, state.sid, state.device)
        infered_audio, _, _ = infer_synth_audio(
            state.synthesizer, feats, p_len_t,
            pitch, pitchf, sid,
            state.use_f0, state.is_half,
            skip_head=state.skip_head,
            return_length=state.return_length,
            return_length2=state.return_length2,
            noise_mod=noise_mod,
        )
        infered_audio = infered_audio.squeeze(1).float()
        ctx.synthesized_audio = infered_audio

        return infered_audio.squeeze()
