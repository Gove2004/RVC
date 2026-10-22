"""实时语音转换管线 — stage_features / stage_f0 / stage_synthesis 三个推理阶段。

架构：
- 持有 EngineState（跨块持续状态：模型引用、缓冲区、合成器缓存等）
- 每块复用 InferenceContext（单块临时状态）
- extract_features（stage_features：HuBERT 特征提取；F0 原始提取也在本方法末尾启动）
  → postprocess_features（stage_f0：F0 同步/截断 + 后处理 + 特征上采样）
  → synthesize_audio（stage_synthesis）
- formant 因子在 init_processing 时快照进 state.formant_factor（会话级参数，见 EngineState）

对外接口：__init__ / load() / extract_features() / postprocess_features() / synthesize_audio()
"""
import logging
import math

import torch

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
        feats = pipeline.extract_features(config)
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

    # ── 便捷属性 ──

    @property
    def target_sr(self) -> int:
        return self.state.model_sr

    # ── 模型加载 ──

    def load(self) -> None:
        """加载模型（HuBERT + 合成器），通过 ModelSessions 缓存复用。"""
        sessions = ModelSessions(
            self.state.device, self.state.is_half, self.state.inference_cache,
        )
        session = sessions.load(self.pth_path, hubert_variant=self.hubert_variant)
        self.state.hubert_model = session.hubert
        self.state.synthesizer = session.synthesizer
        self.state.model_sr = session.target_sr
        self.state.use_f0 = session.use_f0

    # ── stage_features / stage_f0：HuBERT 特征 + F0 原始提取 ──

    def extract_features(self) -> torch.Tensor:
        """stage_features：HuBERT 特征提取 + F0 原始提取。

        注意：ctx.reset() 由 inference_runner 在调用本方法前执行（runner 管理 ctx 生命周期），
        本方法不再 reset ctx，只填充本阶段字段。

        formant 因子只从 state.formant_factor 读（init_processing 时快照）：
        它参与 return_length2 的算术，而 return_length2 会进合成器图的 cache key，
        运行中变化等于逼音频回调线程重捕获（实测 585ms）。本方法不再接受
        config 参数——那会和 ctx.config 构成两个真相源。
        这里仍把 return_length2 算好放进 ctx，供合成阶段直接使用。

        输入音频固定取自 state.input_wav_16k（16kHz 滚动缓冲区），它是本阶段
        唯一的数据来源；`skip_head` / `return_length` 由 setup 阶段算好写入
        state（init_processing），本方法只读——不再作为参数每块重复回写。

        Returns:
            HuBERT 特征 (1, T, 768)，50fps
        """
        state = self.state
        ctx = self.ctx

        ctx.return_length2 = int(math.ceil(state.return_length * state.formant_factor))

        # 只提取 HuBERT 特征（50fps）。
        #
        # **不要改成「F0 先启动、再跑 HuBERT」**：那样会让 F0 与 HuBERT 真正并发
        # （实测整块只省 0.61ms/2.7%——GPU 已饱和，并行不减少总工作量），代价是
        # **F0 side stream 的 CUDA Graph 回放与主流的 HuBERT / 合成器图回放并发**。
        # 每个 CUDA Graph 各持私有显存池，PyTorch 明确要求同一张图必须「同流、同序」
        # 回放；跨流并发回放属未定义行为。
        # 实测后果（真实实时链路，RTX 5060 Laptop）：数秒~数十秒后音频回调卡在
        # F0 后处理的 GPU→CPU 同步上（当时表现为 `apply_pitch_map`，那里有个拿
        # 0 维张量做 if 判断的守卫；该同步点现已改成 Python float，症状会转移到
        # 下一个真正同步的位置），设备再也排不空 → 回调不再被调用、GUI 冻结、
        # GPU 被占满。**与 mel 前端是否图化无关**——mel 已改回 eager 后
        # 仅恢复 F0 先启动，~6s 即复现（fcpe-core replay=56 后挂死）。
        # 旧顺序（HuBERT 先入队、F0 等它算完）虽然名义上没并行，但它保证了侧流图
        # 回放不与主流图回放重叠——这个"免费"的串行约束必须保留。
        feats = self._stage_extract_hubert()

        # F0 提取（side stream）必须等主流上的 HuBERT 入队之后再启动（见上）。
        self._start_f0_parallel(ctx)

        return feats

    # ── stage_f0 后处理（F0 后处理 + 特征上采样） ──

    def postprocess_features(self, feats: torch.Tensor):
        """stage_f0：F0 同步/截断 + 后处理 + 特征上采样。

        先 upsample HuBERT 特征得到实际长度，再让 F0 对齐到这个长度（F0 的
        原始提取已在 extract_features 末尾于 side stream 上启动，这里只等它完成）。
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
        feats = self._stage_upsample_features(feats)
        ctx.p_len = feats.shape[1]  # 用 upsample 后的实际长度作为 p_len

        # 同步 F0 side stream，截断到 p_len
        self._sync_f0_parallel(ctx)

        # F0 后处理（音域映射 + 中值滤波 + 离散化）
        pitch, pitchf = self._stage_postprocess_f0(ctx)

        return pitch, pitchf, feats

    # ── stage_synthesis：合成 ──

    def synthesize_audio(self, feats: torch.Tensor, pitch, pitchf) -> torch.Tensor:
        """stage_synthesis：合成器推理。

        formant 重采样由引擎层负责，这里只返回合成器原始输出。

        Args:
            feats: 上采样后的特征 (1, p_len, C)，100fps
            pitch: 离散 F0 (1, p_len)，use_f0=False 时为 None
            pitchf: 连续 F0 (1, p_len)，use_f0=False 时为 None

        Returns:
            合成音频张量（target_sr 采样率，未做 formant 后处理）
        """
        return self._stage_synthesize(self.ctx, feats, pitch, pitchf)

    # ── 内部阶段方法 ──

    def _stage_extract_hubert(self) -> torch.Tensor:
        """从 16k 滚动缓冲区提取 HuBERT 特征（50fps）。"""
        return extract_hubert_features(
            self.state.hubert_model,
            self.state.input_wav_16k,
            self.state.device,
            self.state.is_half,
        )

    def _start_f0_parallel(self, ctx: InferenceContext) -> None:
        """在 side stream 上启动 F0 原始提取（不截断，不后处理）。

        名字里的 "parallel" 指「F0 计算与后续的 upsample/后处理/合成错开在另一条流上」，
        **不是**「与 HuBERT 并发」——与本方法的调用时机契约一起看，它实际是串行于 HuBERT 的
        （原因见 extract_features 与下方 wait_stream 处的注释）。F0 提取不需要 p_len，
        只在 _sync_f0_parallel 中截断到 p_len。

        use_f0=False 时直接返回。
        """
        state = self.state
        if not state.use_f0:
            ctx.f0_raw = None
            ctx._f0_raw_untruncated = None
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

        # 关键：side stream 必须等待主 stream 完成 input_wav_16k 的写入，
        # 否则 F0 可能读到半写入的缓冲区（数据竞争 → 偶发 NaN）。
        # 反向同步在 _sync_f0_parallel 中通过 wait_stream(f0_stream) 完成。
        #
        # **调用时机契约**：本方法必须在 _stage_extract_hubert **之后**调用。
        # 这看着"浪费"——wait_stream 会连带等上 HuBERT 的图回放，F0 名义并行、实际串行——
        # 但这正是必要的约束：一旦 F0 真的与 HuBERT 并发，side stream 上的 CUDA Graph
        # 回放就会与主流的图回放重叠，跨流并发回放未定义（见 extract_features 注释），
        # 实测会卡死音频回调。代价只有 0.61ms/2.7%（GPU 已饱和）。
        state.f0_stream.wait_stream(torch.cuda.current_stream(state.device))

        # 在 side stream 上启动 F0 提取（原始值，未截断）
        # extract_raw 内部使用 run_cuda_graph，会在当前 stream（f0_stream）上 capture/replay
        # 第二个返回值（confidence）在本链路不参与任何计算，故不接收：
        # 见 _stage_postprocess_f0 的说明。
        with torch.cuda.stream(state.f0_stream):
            f0_raw, _ = state.f0_extractor.extract_raw(
                state.input_wav_16k, HUBERT_SAMPLE_RATE,
            )
        # 存储原始（未截断）结果引用；实际 GPU 计算在 f0_stream 上异步执行
        ctx._f0_raw_untruncated = f0_raw

    def _sync_f0_parallel(self, ctx: InferenceContext) -> None:
        """同步 F0 side stream，截断到 p_len，存入 ctx.f0_raw。

        use_f0=False 时直接返回。
        """
        state = self.state
        if not state.use_f0:
            ctx._f0_raw_untruncated = None
            return

        # 等待 side stream 上的 F0 提取完成
        if state.f0_stream is not None:
            torch.cuda.current_stream(state.device).wait_stream(state.f0_stream)

        f0_raw = ctx._f0_raw_untruncated
        p_len = ctx.p_len

        # 对齐到 p_len 帧，分两种情形（由 F0 提取器是否用尾部窗口决定）：
        # - 整段窗口（f0 帧数 >= p_len）：取**前** p_len 帧——F0 与特征缓冲左对齐，
        #   这是原有的对齐口径，不能改成取尾部，否则整体平移若干帧。
        # - 尾窗（f0 帧数 < p_len，见 FCPE_TAIL_WINDOW_SEC）：F0 只覆盖缓冲末端，
        #   必须**右对齐**——左补零到 p_len，使 F0 的末帧落在下标 p_len-1。
        #   中值滤波是因果的，前段补零不会影响合成器实际取用的尾部帧。
        n = f0_raw.shape[0]
        if n >= p_len:
            f0 = f0_raw[:p_len]
        else:
            f0 = torch.nn.functional.pad(f0_raw, (p_len - n, 0))

        ctx.f0_raw = f0[None, :]
        # 未截断的原始值是「单块临时状态」，用完即弃：留着会让 ctx 的
        # 生命周期契约（reset 清除一切）不成立，调试/统计代码会读到陈旧 F0，
        # 而且它是一条跨块活着的 GPU 张量引用。
        ctx._f0_raw_untruncated = None

    def _stage_postprocess_f0(self, ctx: InferenceContext):
        """F0 后处理（音域映射 + 中值滤波 + 离散化）。

        从 ctx 读取 stage_f0 提取的原始 F0，调用 postprocess_f0 做后处理。
        pitchf 乘以 return_length2/return_length 做 formant 缩放。

        Returns:
            (pitch, pitchf): 离散 F0 (1, p_len)、连续 F0 (1, p_len)
            use_f0=False 时返回 (None, None)
        """
        state = self.state
        if not state.use_f0:
            # 无 F0 模型：本阶段不产出音高，必须清掉上一次（有 F0 模型时）的统计，
            # 否则 GUI 会永久显示切换前的陈旧音高。engine.input_pitch 对 None 返回 0。
            state.last_input_pitch_gpu = None
            return None, None

        # 输入音高统计（音域映射之前，非零帧平均，用于 GUI 显示）。
        # 全程定形 GPU 算子（掩码乘法 + 求和），不做布尔索引 / .item()——
        # 那会输出数据依赖形状，每块强制同步 GPU 队列 2~3 次打断流水线。
        # 取回 Python float 的唯一点在 runner 写完主输出之后（队列已排空，零等待）。
        f0_raw_flat = ctx.f0_raw.squeeze(0)
        mask = f0_raw_flat > 0
        state.last_input_pitch_gpu = ((f0_raw_flat * mask).sum(), mask.sum())

        # confidence 刻意不传入：postprocess_f0 不消费它（pitchf 路径只有音域映射 +
        # 中值滤波 + 离散化）。旧链路把 confidence 一路透传进后处理再丢弃，
        # 既无收益，又在 FCPE 非 CUDA Graph 路径（confidence=None）触发过 None 下标崩溃。
        pitch, pitchf = postprocess_f0(
            f0_raw_flat,
            state.device,
            config=ctx.config,
        )
        # formant 缩放（pitchf 需要乘以 return_length2 / return_length）
        pitchf = pitchf * ctx.return_length2 / state.return_length

        return pitch[None, :], pitchf[None, :]

    def _stage_upsample_features(self, feats: torch.Tensor) -> torch.Tensor:
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

        formant 重采样由引擎层负责，这里只返回合成器原始输出。

        Args:
            feats: 上采样后的特征 (1, p_len, C)，100fps
            pitch: 离散 F0 (1, p_len)，use_f0=False 时为 None
            pitchf: 连续 F0 (1, p_len)，use_f0=False 时为 None

        Returns:
            合成音频张量（target_sr 采样率，未做 formant 后处理）
        """
        state = self.state

        p_len_t = cached_long_tensor(state.long_tensor_cache, ctx.p_len, state.device)
        sid = cached_long_tensor(state.long_tensor_cache, state.sid, state.device)

        # 合成器推理（只回传音频：中间量在图回放时没必要 clone，见 synthesis）
        inferred_audio = infer_synth_audio(
            state.synthesizer, feats, p_len_t,
            pitch, pitchf, sid,
            state.use_f0, state.is_half,
            skip_head=state.skip_head,
            return_length=state.return_length,
            return_length2=ctx.return_length2,
        )
        return inferred_audio.squeeze(1).float().squeeze()

