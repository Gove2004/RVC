"""实时语音转换管线 — stage_features / stage_f0 / stage_synthesis 三个推理阶段。

架构：
- 持有 EngineState（跨块持续状态：模型引用、缓冲区、合成器缓存等）
- 每块复用 InferenceContext（单块临时状态）
- extract_features（stage_features：HuBERT 特征提取）
  → postprocess_features（F0 提取 + 后处理 + 特征上采样）
  → synthesize_audio（stage_synthesis）
- formant 因子在 extract_features 中计算并存到 ctx.formant_factor，输出侧复用避免重复计算

对外接口：__init__ / load() / extract_features() / postprocess_features() / synthesize_audio()
"""
import logging
import math
from types import SimpleNamespace

import torch

from rvc.core.constants import HUBERT_FRAME_SIZE
from rvc.core.config import InferenceParams
from rvc.pipeline.state import EngineState
from rvc.pipeline.context import InferenceContext
from rvc.pipeline.cache import default_inference_cache
from rvc.pipeline.features import extract_hubert_features, upsample_features


from rvc.pipeline.pitch.extractor import postprocess_f0, create_f0_extractor
from rvc.pipeline.sessions import ModelSessions
from rvc.pipeline.synthesis import cached_long_tensor, infer_synth_audio
from rvc.core.constants import HUBERT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class InferencePipeline:
    """实时语音转换管线 — stage_features / stage_f0 / stage_synthesis。

    用法:
        pipeline = InferencePipeline(device_config, pth_path, hubert="chinese")
        pipeline.load()
        feats = pipeline.extract_features(input_wav, config, block_16k, skip_head, ret_len)
        pitch, pitchf, feats = pipeline.postprocess_features(feats)
        output = pipeline.synthesize_audio(feats, pitch, pitchf)
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
        return self.state.model_sr

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
        """加载模型（HuBERT + 合成器），通过 ModelSessions 缓存复用。"""
        sessions = ModelSessions(
            SimpleNamespace(device=self.state.device, is_half=self.state.is_half),
            self.state.inference_cache,
        )
        session = sessions.load(self.pth_path, hubert_variant=self.hubert_variant)
        self.state.hubert_model = session.hubert
        self.state.synthesizer = session.synthesizer
        self.state.model_sr = session.target_sr
        self.state.use_f0 = session.use_f0

    # ── stage_features / stage_f0：HuBERT 特征 + F0 原始提取 ──

    def extract_features(self, input_wav: torch.Tensor, config: InferenceParams,
                         block_frame_16k: int, skip_head: int, return_length: int) -> torch.Tensor:
        """stage_features：formant 因子 + HuBERT 特征提取 + F0 原始提取。

        注意：ctx.reset() 由 inference_runner 在调用本方法前执行（runner 管理 ctx 生命周期），
        本方法不再 reset ctx，只填充本阶段字段。

        formant 因子计算后存到 ctx.formant_factor，供输出侧 formant 重采样复用，
        避免 inference_runner 重复计算 pow(2, formant/12)。

        Args:
            input_wav: 滚动缓冲区 (16kHz, GPU)
            config: 推理参数
            block_frame_16k: 本块新增的 16kHz 采样数
            skip_head: 跳过的 10ms 帧数（上下文）
            return_length: 需要返回的 10ms 帧数

        Returns:
            HuBERT 特征 (1, T, 768)，50fps
        """
        state = self.state
        ctx = self.ctx

        # ctx 已由 inference_runner 在调用前重置（ctx.reset）
        # p_len 不再预计算，而是等 HuBERT upsample 后用实际长度

        # skip_head / return_length 由 runner 传入，同步到 state 供后续阶段使用
        state.skip_head = skip_head
        state.return_length = return_length

        # formant 因子计算（存到 ctx，输出侧复用，避免重复计算）
        formant_factor = config.voice.formant
        factor = pow(2, formant_factor / 12)
        ctx.formant_factor = factor
        ctx.return_length2 = int(math.ceil(return_length * factor))

        # 只提取 HuBERT 特征，F0 提取移到 postprocess_features
        # 等 upsample 得到实际长度后，再让 F0 对齐到这个实际长度
        feats = self._stage_extract_hubert(ctx)

        # 并行启动 F0 提取（side stream），与后续 upsample 重叠
        # F0 提取不需要 p_len，只在最后截断时需要，所以可以提前启动
        self._start_f0_parallel(ctx)

        return feats

    # ── stage_f0 后处理（F0 后处理 + 特征上采样） ──

    def postprocess_features(self, feats: torch.Tensor):
        """postprocess：F0 提取 + 后处理 + 特征上采样。

        先 upsample HuBERT 特征得到实际长度，再让 F0 对齐到这个长度。
        F0 后处理（音域映射 + 中值滤波 + 离散化）。

        Args:
            feats: HuBERT 特征 (1, T, 768)，50fps

        Returns:
            (pitch, pitchf, feats_up): 离散F0、连续F0、上采样后的特征
        """
        state = self.state
        ctx = self.ctx

        # 先 upsample HuBERT 特征，得到实际长度 actual_len
        # 然后 F0 对齐到 actual_len，确保两者长度完全一致
        feats = self._stage_upsample_features(ctx, feats)
        ctx.p_len = feats.shape[1]  # 用 upsample 后的实际长度作为 p_len

        # 同步 F0 side stream，截断到 p_len
        self._sync_f0_parallel(ctx)

        # F0 后处理（音域映射 + 中值滤波 + 离散化）
        pitch, pitchf = self._stage_postprocess_f0(ctx)

        return pitch, pitchf, feats

    # ── stage_synthesis：合成 ──

    def synthesize_audio(self, feats: torch.Tensor, pitch, pitchf) -> torch.Tensor:
        """stage_synthesis：合成器推理。

        formant 重采样已移到引擎层 stage_synthesis，这里只返回合成器原始输出。

        Args:
            feats: 上采样后的特征 (1, p_len, C)，100fps
            pitch: 离散 F0 (1, p_len)，use_f0=0 时为 None
            pitchf: 连续 F0 (1, p_len)，use_f0=0 时为 None

        Returns:
            合成音频张量（target_sr 采样率，未做 formant 后处理）
        """
        return self._stage_synthesize(self.ctx, feats, pitch, pitchf)

    # ── 内部阶段方法 ──

    def _stage_extract_hubert(self, ctx: InferenceContext) -> torch.Tensor:
        """从 16k 滚动缓冲区提取 HuBERT 特征（50fps）。"""
        return extract_hubert_features(
            self.state.hubert_model,
            self.state.input_wav_16k,
            self.state.device,
            self.state.is_half,
        )

    def _stage_extract_f0_raw(self, ctx: InferenceContext) -> None:
        """F0 原始提取 — 已由 _start_f0_parallel / _sync_f0_parallel 替代。

        保留为空方法以兼容可能的外部调用；实际提取在 side stream 上并行完成。
        """
        pass

    def _start_f0_parallel(self, ctx: InferenceContext) -> None:
        """在 side stream 上并行启动 F0 原始提取（不截断，不后处理）。

        与 HuBERT 在默认 stream 上并行执行。F0 提取不需要 p_len，
        只在 _sync_f0_parallel 中截断到 p_len。

        use_f0=0 时直接返回。
        """
        state = self.state
        if state.use_f0 == 0:
            ctx.f0_raw = None
            ctx.confidence_raw = None
            return

        # 确保 F0 提取器存在（懒加载，与原逻辑一致）
        method = ctx.config.f0.method
        if state.f0_extractor is None or state.f0_extractor_method != method:
            state.f0_extractor = create_f0_extractor(
                method, state.device, state.is_half, state.inference_cache, config=ctx.config,
            )
            state.f0_extractor_method = method

        # 确保 side stream 存在
        if state.f0_stream is None:
            state.f0_stream = torch.cuda.Stream(device=state.device)

        # 在 side stream 上启动 F0 提取（原始值，未截断）
        # extract_raw 内部使用 run_cuda_graph，会在当前 stream（f0_stream）上 capture/replay
        with torch.cuda.stream(state.f0_stream):
            f0_raw, confidence_raw = state.f0_extractor.extract_raw(
                state.input_wav_16k, HUBERT_SAMPLE_RATE,
            )
        # 存储原始（未截断）结果引用；实际 GPU 计算在 f0_stream 上异步执行
        ctx._f0_raw_untruncated = f0_raw
        ctx._conf_raw_untruncated = confidence_raw

    def _sync_f0_parallel(self, ctx: InferenceContext) -> None:
        """同步 F0 side stream，截断到 p_len，存入 ctx.f0_raw / ctx.confidence_raw。

        use_f0=0 时直接返回。
        """
        state = self.state
        if state.use_f0 == 0:
            return

        # 等待 side stream 上的 F0 提取完成
        if state.f0_stream is not None:
            torch.cuda.current_stream(state.device).wait_stream(state.f0_stream)

        f0_raw = ctx._f0_raw_untruncated
        confidence_raw = ctx._conf_raw_untruncated
        p_len = ctx.p_len

        # 对齐到 p_len 帧（与 extract_f0_for_block 的截断逻辑一致）
        n = f0_raw.shape[0]
        if n >= p_len:
            f0 = f0_raw[:p_len]
            conf = confidence_raw[:p_len]
        else:
            pad_len = p_len - n
            f0 = torch.nn.functional.pad(f0_raw, (0, pad_len))
            conf = torch.nn.functional.pad(confidence_raw, (0, pad_len))

        ctx.f0_raw = f0[None, :]
        ctx.confidence_raw = conf[None, :]

    def _stage_postprocess_f0(self, ctx: InferenceContext):
        """F0 后处理（音域映射 + 中值滤波 + 离散化）。

        从 ctx 读取 stage_f0 提取的原始 F0，调用 postprocess_f0 做后处理。
        pitchf 乘以 return_length2/return_length 做 formant 缩放。

        Returns:
            (pitch, pitchf): 离散 F0 (1, p_len)、连续 F0 (1, p_len)
            use_f0=0 时返回 (None, None)
        """
        state = self.state
        if state.use_f0 == 0:
            return None, None

        # 输入音高统计（音域映射之前，非零帧平均，用于 GUI 显示）。
        # 全程定形 GPU 算子（掩码乘法 + 求和），不做布尔索引 / .item()——
        # 那会输出数据依赖形状，每块强制同步 GPU 队列 2~3 次打断流水线；
        # GUI 低频读取 VoiceEngine.input_pitch 时才做唯一一次同步取值。
        f0_raw_flat = ctx.f0_raw.squeeze(0)
        mask = f0_raw_flat > 0
        state.last_input_pitch_gpu = ((f0_raw_flat * mask).sum(), mask.sum())

        pitch, pitchf, confidence = postprocess_f0(
            f0_raw_flat,
            state.device,
            confidence=ctx.confidence_raw.squeeze(0),
            config=ctx.config,
        )
        # formant 缩放（pitchf 需要乘以 return_length2 / return_length）
        pitchf = pitchf * ctx.return_length2 / state.return_length

        return pitch[None, :], pitchf[None, :]

    def _stage_upsample_features(self, ctx: InferenceContext, feats: torch.Tensor) -> torch.Tensor:
        """特征上采样（50fps → 100fps，最近邻插值 + 末帧 padding）。"""
        return upsample_features(feats, self.state.is_half)

    def _stage_synthesize(
        self,
        ctx: InferenceContext,
        feats: torch.Tensor,
        pitch,
        pitchf,
    ) -> torch.Tensor:
        """合成器推理。

        formant 重采样已移到引擎层阶段4，这里只返回合成器原始输出。

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
            return_length2=ctx.return_length2,
        )
        infered_audio = infered_audio.squeeze(1).float()

        return infered_audio.squeeze()

