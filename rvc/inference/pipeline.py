"""实时语音转换管线 — 阶段3-6（HuBERT → F0 → 合成预处理 → 合成）。

架构重构后：
- 持有 EngineState（跨块持续状态：模型引用、F0缓存、合成器缓存等）
- 每块复用 InferenceContext（单块临时状态）
- infer() 按阶段执行，清辅音保护用简化版（F0阈值判断 + 原始特征混合）

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


def _is_cuda(device) -> bool:
    """判断设备是否为 CUDA（用于决定是否启用 Stream 并行）。"""
    return str(device).startswith("cuda")

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

        # 阶段3+4 并行：HuBERT 特征提取和 F0 原始提取互不依赖，用 CUDA Stream 并行
        if _is_cuda(state.device):
            s_hubert = torch.cuda.Stream()
            s_f0 = torch.cuda.Stream()
            with torch.cuda.stream(s_hubert):
                feats = self._stage_extract_hubert(ctx)
            with torch.cuda.stream(s_f0):
                self._stage_extract_f0_raw(ctx)
            torch.cuda.current_stream().wait_stream(s_hubert)
            torch.cuda.current_stream().wait_stream(s_f0)
        else:
            feats = self._stage_extract_hubert(ctx)
            self._stage_extract_f0_raw(ctx)

        # 阶段5a+5b 并行：F0 后处理和特征上采样互不依赖，用 CUDA Stream 并行
        # 阶段5a: F0 后处理（音域映射 + 中值滤波 + 离散化），结果存 ctx.pitchf_block 供输出侧清辅音保护用
        # 阶段5b: 特征上采样（50fps → 100fps）
        # 清辅音保护已移到输出侧（effects.py AudioProcessor），特征侧只做上采样
        if _is_cuda(state.device):
            s_post = torch.cuda.Stream()
            s_up = torch.cuda.Stream()
            with torch.cuda.stream(s_post):
                pitch, pitchf = self._stage_5a_postprocess_f0(ctx)
            with torch.cuda.stream(s_up):
                feats_up = self._stage_5b_upsample_features(ctx, feats)
            torch.cuda.current_stream().wait_stream(s_post)
            torch.cuda.current_stream().wait_stream(s_up)
            feats = feats_up
        else:
            pitch, pitchf = self._stage_5a_postprocess_f0(ctx)
            feats = self._stage_5b_upsample_features(ctx, feats)
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

        # F0 提取器缓存：method 变化时重建，否则跨块复用
        method = ctx.config.f0_method
        if state.f0_extractor is None or state.f0_extractor_method != method:
            from rvc.inference.f0_extractor import create_f0_extractor
            state.f0_extractor = create_f0_extractor(
                method, state.device, state.is_half, state.inference_cache, config=ctx.config,
            )
            state.f0_extractor_method = method

        f0_raw, confidence_raw = update_realtime_pitch_cache_raw(
            state.input_wav_16k,
            ctx.block_frame_16k,
            ctx.p_len,
            method,
            state.pitch_cache,
            state.pitchf_cache,
            state.confidence_cache,
            state.device,
            state.is_half,
            state.inference_cache,
            config=ctx.config,
            extractor=state.f0_extractor,
        )
        ctx.f0_raw = f0_raw
        ctx.confidence_raw = confidence_raw

    # ── 阶段5：F0 后处理（音域映射 + 中值滤波 + 离散化）──

    def _stage_5a_postprocess_f0(self, ctx: InferenceContext):
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

        # 计算原始输入音高（音域映射之前，非零帧平均，用于 GUI 显示）
        f0_raw_flat = ctx.f0_raw.squeeze(0)
        nonzero = f0_raw_flat[f0_raw_flat > 0]
        if nonzero.numel() > 0:
            state.last_input_pitch = float(nonzero.mean().item())

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
        # 存到 ctx 供输出侧清辅音保护用（100fps，连续 F0）
        ctx.pitchf_block = pitchf.clone()
        return pitch[None, :], pitchf[None, :]


    def _stage_5b_upsample_features(self, ctx: InferenceContext, feats: torch.Tensor) -> torch.Tensor:
        """阶段5b：特征上采样（50fps → 100fps，截取 p_len 帧）。

        清辅音保护已移到输出侧，这里只做纯上采样。
        """
        return upsample_features(feats, ctx.p_len, self.state.is_half)

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
        )
        infered_audio = infered_audio.squeeze(1).float()
        ctx.synthesized_audio = infered_audio

        return infered_audio.squeeze()
