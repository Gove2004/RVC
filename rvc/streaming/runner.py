"""推理运行器 — 封装实时推理的缓冲区管理、推理调用、效果器链。

架构：
- 所有跨块持续状态统一放在 EngineState（与 InferencePipeline 共享）
- InferenceRunner 只持有 pipeline/runtime_params 引用，不重复持有状态字段
- 每块复用 InferenceContext（单块临时状态）
- process_block 按阶段执行：
    stage_input：硬件输入 + 缓冲区滚动 + work_sr→16k 重采样
    stage_features：HuBERT 特征提取（pipeline.extract_features，F0 原始提取也在此启动）
    stage_f0：F0 同步/截断 + 后处理 + 特征上采样（pipeline.postprocess_features）
    stage_synthesis：合成器推理 + 模型→工作采样率重采样 + formant + 长度对齐
    stage_output：RMS 混合 + SOLA + 硬件输出
- formant 因子可运行中调整（state.formant_factor → formant_ratio），各档位的
  合成器图在启动时预捕获（prewarm_formant），切换即纯回放
"""
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample as TatResample

from rvc.core.config import VOLUME_MAX, VOLUME_MIN
from rvc.core.constants import HUBERT_FRAME_SIZE, HUBERT_SAMPLE_RATE
from rvc.pipeline.synthesis import apply_formant_resample, formant_resample_ratio
from rvc.streaming.effects import AudioProcessor
from rvc.streaming.output import write_main_output

logger = logging.getLogger(__name__)


class InferenceRunner:
    """推理运行器 — 管理实时推理的缓冲区与处理流程。

    所有跨块持续状态放在 EngineState（与 pipeline 共享），本类不重复持有。
    process_block 按角色名阶段闭环执行：stage_input → stage_features/stage_f0
    → stage_synthesis → stage_output。
    """

    def __init__(self, pipeline, runtime_params):
        self.pipeline = pipeline
        self.runtime_params = runtime_params

        # 共享 EngineState（pipeline 创建并持有，device/is_half 已在 pipeline 构造时设置）
        self.state = pipeline.state

        # 单块上下文（复用实例，每块 reset()）
        self.ctx = pipeline.ctx

        # 输出效果器（RMS 混合 + SOLA）由 runner 直接持有：
        # 它们是输出侧效果，不属于推理状态（pipeline 产物止于合成音频块）
        self.effects = AudioProcessor()

    def init_processing(self, work_sr: int, block_time: float, crossfade_time: float,
                        extra_time: float, channels: int, model_sr: int) -> None:
        """初始化所有缓冲区和计算参数，直接写入 EngineState。

        Args:
            work_sr: 工作采样率（流实际跑的采样率，模型率或设备率之一）
            block_time: 块时长（秒）
            crossfade_time: 交叉淡化时长（秒）
            extra_time: 额外上下文时长（秒）
            channels: 通道数
            model_sr: 模型原生采样率
        """
        state = self.state
        state.work_sr = work_sr
        state.model_sr = model_sr
        state.channels = channels
        zc = work_sr // 100
        state.samples_per_centis = zc

        # 计算参数（对齐到 10ms 边界）
        state.block_samples = int(np.round(block_time * work_sr / zc)) * zc
        state.crossfade_samples = int(np.round(crossfade_time * work_sr / zc)) * zc
        state.sola_buffer_samples = min(state.crossfade_samples, 5 * zc)
        state.sola_search_samples = zc
        state.extra_samples = int(np.round(extra_time * work_sr / zc)) * zc

        state.block_samples_16k = HUBERT_FRAME_SIZE * state.block_samples // zc
        state.skip_head = state.extra_samples // zc
        # 传给合成器的帧数 = 当前块 + SOLA 缓冲 + SOLA 搜索（10ms 帧对齐）。
        # 尾窗只有 return_length 帧进合成器；SOLA 拼接期望固定块长，输出侧
        # 按 block_samples + sola_buffer + sola_search 截齐（见 _stage_formant）。
        state.return_length = (state.block_samples + state.sola_buffer_samples + state.sola_search_samples) // zc

        # 启动时以配置值初始化；运行中由 set_formant_factor 改写（滑条实时生效）。
        # 每换一个 formant 档位就是另一张合成器图（输出形状随 return_length2 变），
        # 这些图在 prewarm_formant 里按量程整段预捕获。
        state.formant_factor = 2.0 ** (float(self.runtime_params.voice.formant) / 12.0)
        state.formant_ratio = 1.0

        # work_sr 滚动缓冲区
        n = state.extra_samples + state.crossfade_samples + state.sola_search_samples + state.block_samples
        state.input_wav_work = torch.zeros(n, device=state.device)
        state.input_wav_work_swap = torch.empty_like(state.input_wav_work)

        # 16k 滚动缓冲区（HuBERT / F0 用）
        state.input_wav_16k = torch.zeros(HUBERT_FRAME_SIZE * n // zc, device=state.device)
        state.input_wav_16k_swap = torch.empty_like(state.input_wav_16k)

        # pinned memory 双缓冲（异步 DMA 交替使用，避免下一块覆写时上一块 DMA 未完成）
        state.in_pin = [
            torch.empty(state.block_samples, dtype=torch.float32, pin_memory=True),
            torch.empty(state.block_samples, dtype=torch.float32, pin_memory=True),
        ]
        # 对应的 numpy 视图：每块单声道化直接写进 pinned，省掉一次分配与一次拷贝
        # （pinned 张量尺寸在会话内固定，视图可以一直复用）
        state.in_pin_np = [buf.numpy() for buf in state.in_pin]
        state.in_pin_idx = 0

        # pinned memory 用于快速 GPU→CPU 传输（输出侧）。单缓冲即可：D2H 是
        # 同步的，拷完立刻被复制进设备缓冲，不存在两块同时在飞。
        # SOLA 输出恒为 block_samples（对齐契约），长度异常块由 write_main_output
        # 回退 chunk.cpu().numpy() 兜底——该分支现在不可达，属防御而非正常路径。
        state.out_pin = torch.empty(state.block_samples, dtype=torch.float32, pin_memory=True)

        # 重采样器（work_sr → 16k）
        state.resampler_work_to_16k = TatResample(work_sr, HUBERT_SAMPLE_RATE, dtype=torch.float32).to(state.device)

        # 重采样器（模型采样率 → work_sr）。两者相同（sr_mode="model"）时不需要
        if model_sr != work_sr:
            state.resampler_model_to_work = TatResample(model_sr, work_sr, dtype=torch.float32).to(state.device)
        else:
            state.resampler_model_to_work = None

        # 效果器（RMS + SOLA）；sola_buffer_samples 已在上面统一计算（单源）
        self.effects.setup(
            work_sr, state.block_samples, state.sola_buffer_samples,
            state.sola_search_samples, state.device,
        )

    def warmup(self, n: int = 2) -> None:
        """预热推理引擎：用静音数据跑 n 个完整块。

        职能是 GPU 时钟爬坡 + cuDNN/L2 缓存热身。CUDA Graph 的**捕获**也发生
        在这里——在第一个「签名未命中」的块上，与 n 的取值无关（n 只决定随后
        还能回放几次）。失败只警告不影响运行。
        """
        state = self.state
        if self.pipeline is None or state.input_wav_16k is None:
            return
        frames = state.block_samples
        indata = np.zeros((frames, state.channels), dtype=np.float32)
        outdata = np.zeros((frames, state.channels), dtype=np.float32)
        try:
            with torch.inference_mode():
                for _ in range(n):
                    self.process_block(indata, outdata, frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("预热失败，不影响运行: %s", exc)

    def prewarm_formant(self, formant_min: float, formant_max: float) -> None:
        """预捕获 formant 全量程各档位的合成器图（必须在音频流启动前调用）。

        formant 经 return_length2 决定合成器输出形状，每换一个整数帧数就是另一
        张图。缺图时捕获发生在音频回调线程里，单张实测 0.19~0.30s（RTX 5060
        Laptop，真实链路），对听者就是一次断音。而全量程的整数帧数只有
        `ceil(rl*f_max) - ceil(rl*f_min) + 1` 个（250ms 块 / 40kHz → 10 个），
        一次性把这段跑完约 2s，换来运行中拖动滑条的零开销切换。

        只在流启动前调用：此刻没有在途回放，捕获不会与回放互相干扰。各档位用
        静音输入，调用方随后 reset_buffers() 清掉静音上下文。

        Args:
            formant_min / formant_max: 量程（半音），取 FORMANT_MIN/MAX
        """
        state = self.state
        if self.pipeline is None or state.input_wav_16k is None or state.return_length <= 0:
            return

        ratio = state.return_length
        lo = int(np.ceil(ratio * 2.0 ** (formant_min / 12.0)))
        hi = int(np.ceil(ratio * 2.0 ** (formant_max / 12.0)))
        saved = state.formant_factor
        frames = state.block_samples
        indata = np.zeros((frames, state.channels), dtype=np.float32)
        outdata = np.zeros((frames, state.channels), dtype=np.float32)
        try:
            with torch.inference_mode():
                for frames_rl2 in range(max(1, lo), hi + 1):
                    # 该档位的代表因子：ceil(rl * (rl2/rl)) 恰为 rl2
                    state.formant_factor = frames_rl2 / ratio
                    self.process_block(indata, outdata, frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("formant 图预捕获失败，运行中切档会在回调线程补捕获: %s", exc)
        finally:
            state.formant_factor = saved

    def set_formant_factor(self, factor: float) -> None:
        """更新性别因子（共振峰比例），下一块即生效。

        单次属性赋值：音频回调线程每块读一次 state.formant_factor，Python 赋值
        原子，不需要锁。目标档位的图已由 prewarm_formant 捕获，切换是纯回放。
        """
        self.state.formant_factor = float(factor)

    def reset_buffers(self) -> None:
        """重置所有缓冲区（warmup 后调用，避免静音数据污染）。"""
        state = self.state
        self.effects.reset()
        if state.input_wav_work is not None:
            state.input_wav_work.zero_()
        if state.input_wav_16k is not None:
            state.input_wav_16k.zero_()

    def reset_error_state(self) -> None:
        self.state.reset_error_state()

    def acknowledge_success(self) -> None:
        """每块推理成功后清零错误计数（连续 N 次错误才停止）。"""
        self.state.error_count = 0

    def handle_error(self, error: Exception) -> bool:
        return self.state.handle_error(error)

    # ── 主处理入口（五阶段闭环） ──

    def process_block(self, indata: np.ndarray, outdata: np.ndarray, frames: int) -> None:
        """处理一块音频，执行完整的五阶段闭环，直接写入 outdata。

        stage_input：硬件输入 + 缓冲区滚动 + work_sr→16k 重采样
        stage_features / stage_f0：HuBERT 特征 + F0 原始提取
        （F0 后处理 + 特征上采样在 postprocess_features）
        stage_synthesis：合成器推理 + 模型→工作采样率重采样 + formant + 长度对齐
        stage_output：RMS 混合 + SOLA + 硬件输出

        Args:
            indata: 输入音频（numpy array，shape=(frames, channels)）
            outdata: 输出音频缓冲区（numpy array，shape=(frames, channels)）
            frames: 帧数
        """
        state = self.state
        ctx = self.ctx
        t0 = time.perf_counter()
        # 热读 runtime_params：GUI 线程可能正在写，但 CPython float 赋值原子，
        # 读到旧值或新值都安全（最多影响当前块 RMS 混合比例，不会崩溃）。
        p_rms_mix = self.runtime_params.rms_mix

        with torch.inference_mode():
            # ── stage_input：硬件输入 + 缓冲区滚动 + work_sr→16k 重采样 ──
            mono = self._stage_input(indata)
            self._stage_preprocess(mono)

            # ── stage_features / stage_f0：HuBERT 特征 + F0 原始提取 ──
            # self.pipeline 由 VoiceEngine._make_runner 构造时注入，恒非 None
            # （离线/实时都必经 load_model → pipeline 已就绪）；若为 None 会在这里
            # 立刻 AttributeError 暴露，不做「静默直通」降级——那会把配置错误
            # 伪装成「能出声但没变声」，掩盖真正的故障。
            ctx.reset(self.runtime_params)
            feats = self.pipeline.extract_features()

            # ── stage_f0 后处理（F0 后处理 + 特征上采样） ──
            pitch, pitchf, feats = self.pipeline.postprocess_features(feats)

            # ── stage_synthesis：合成器推理 + 模型→工作采样率重采样 ──
            infer = self.pipeline.synthesize_audio(feats, pitch, pitchf)
            # 模型采样率 → 工作采样率 重采样（如果需要）
            if state.resampler_model_to_work:
                infer = state.resampler_model_to_work(infer)

            # formant 重采样 + 长度对齐（因子取自 state.formant_factor 快照）
            infer = self._stage_formant(infer)

            # ── stage_output：RMS + SOLA + 硬件输出 ──
            ref = state.input_wav_work[state.extra_samples:]
            chunk = self.effects.process_output(infer, ref, p_rms_mix)

            # 输出音量增益（纯标量乘法，热路径上 in-place 避免分配）。
            # 范围钳制放在消费点：配置文件可被外部改写，滑块量程不构成保证。
            vol = min(max(self.runtime_params.volume, VOLUME_MIN), VOLUME_MAX)
            if vol != 1.0:
                chunk.mul_(vol)
            # 硬钳位到 [-1, 1]：写设备前的最后一道保护，**必须无条件执行**。
            # RMS 混合（rvc/dsp/rms.py）会乘上 (r1/r2)^(1-rms_mix) 这一独立于
            # volume 的增益源，参考音量更大时该增益 > 1；此时即使 vol <= 1，
            # chunk 仍可能越界。钳位是一次逐元素比较，相对块预算可忽略。
            chunk.clamp_(min=-1.0, max=1.0)

            # 硬件输出（写入 outdata；out_pin 走 pinned DMA，避免每块分配 pageable 临时内存）
            write_main_output(chunk, outdata, state.channels, state.out_pin)

            # 音高遥测取回成 Python float：放在同步 D2H 之后，本块的 GPU 队列
            # 已经排空，.item() 几乎零等待。反过来（让 GUI 线程自己 .item()）
            # 每一次刷新都是一次真实的队列排空——刷得越快，越会踩停音频管线。
            stat = state.last_input_pitch_gpu
            if stat is None:
                state.input_pitch_hz = 0.0
            else:
                pitch_sum, count = stat
                state.input_pitch_hz = float((pitch_sum / count.clamp(min=1)).item())

        state.infer_ms = (time.perf_counter() - t0) * 1000

    # ── stage_input：硬件输入 ──

    def _upload_input(self, indata: np.ndarray) -> torch.Tensor:
        """单声道化 + pinned 双缓冲 + CPU→GPU 异步上传。

        返回 GPU 上的单声道张量。

        双缓冲：块 N 用 pin_a 做异步 DMA，块 N+1 用 pin_b，互不干扰。
        异步 DMA（non_blocking=True）：CPU 不等待传输完成，立即提交后续
        GPU 算子（HuBERT 等），传输与计算在 GPU 侧自动重叠。
        后续 GPU 算子在同一 stream 上，会自动等待 DMA 完成，无需显式同步。
        """
        state = self.state
        n = indata.shape[0]
        # 交替使用双缓冲
        pin = state.in_pin[state.in_pin_idx]
        pin_np = state.in_pin_np[state.in_pin_idx]
        state.in_pin_idx ^= 1
        # 直接写进 pinned 内存：indata.mean(axis=1) 会为每块新分配一个 float32
        # 数组，再经 torch.from_numpy 拷进 pinned —— 双声道输入就是每块一次
        # 8KB 分配 + 一次多余的 CPU→CPU 拷贝，而回调线程最不该做的就是这个。
        if indata.ndim == 1:
            np.copyto(pin_np[:n], indata)
        elif indata.shape[1] == 1:
            np.copyto(pin_np[:n], indata[:, 0])
        else:
            np.mean(indata, axis=1, out=pin_np[:n])
        if state.input_gpu is None or state.input_gpu.shape[0] < n:
            state.input_gpu = torch.empty(n, device=state.device, dtype=torch.float32)
        state.input_gpu[:n].copy_(pin[:n], non_blocking=True)  # 异步 DMA，CPU 不等
        return state.input_gpu[:n]

    def _stage_input(self, indata: np.ndarray) -> torch.Tensor:
        """stage_input：硬件输入 → GPU（复用 _upload_input）。"""
        return self._upload_input(indata)

    # ── stage_input：输入预处理（缓冲区滚动 + work_sr→16k 重采样） ──

    def _stage_preprocess(self, mono: torch.Tensor) -> None:
        """缓冲区滚动 + work_sr→16k 重采样。

        使用双缓冲交换避免额外拷贝。16k 重采样只处理新块（带额外上下文抵消重采样延迟）。
        """
        state = self.state

        # 16k 缓冲区左移（双缓冲交换）— 末尾块随后被重采样输出覆盖，无需 zero_
        state.input_wav_16k_swap[:-state.block_samples_16k].copy_(state.input_wav_16k[state.block_samples_16k:])
        state.input_wav_16k, state.input_wav_16k_swap = state.input_wav_16k_swap, state.input_wav_16k

        # work_sr 缓冲区左移（双缓冲交换）— 末尾块随后被 mono 覆盖，无需 zero_
        state.input_wav_work_swap[:-state.block_samples].copy_(state.input_wav_work[state.block_samples:])
        state.input_wav_work, state.input_wav_work_swap = state.input_wav_work_swap, state.input_wav_work
        state.input_wav_work[-mono.shape[0]:] = mono

        # 只重采样新块（带额外 2 厘秒样本上下文抵消重采样延迟），输出跳过前 HUBERT_FRAME_SIZE 样本
        resampler_in = state.input_wav_work[-mono.shape[0] - 2 * state.samples_per_centis:]
        resampler_out = state.resampler_work_to_16k(resampler_in)[HUBERT_FRAME_SIZE:]
        # 只写当前块（最后 block_samples_16k 样本），不写修正帧（前 160 样本）。
        # 修正帧会覆盖上一块最后10ms，导致 HuBERT 看到的历史在每次推理中都被"修正"，
        # 连续累积后特征漂移（如"凤凰"连续读越读越飘成"废黄"）。
        target_len = state.block_samples_16k
        state.input_wav_16k[-target_len:] = resampler_out[-target_len:]

    # ── stage_synthesis：formant 重采样 + 长度对齐 ──

    def _stage_formant(self, infer: torch.Tensor) -> torch.Tensor:
        """formant 共振峰偏移（重采样实现）+ 长度对齐。

        比例取自 state.formant_ratio——它是 state.formant_factor 经合成帧数取整
        后的有效值，与 _stage_postprocess_f0 里 pitchf 的缩放严格同一个数。
        比例 1.0（formant=0）时跳过重采样，直接做长度对齐。

        Args:
            infer: 合成器输出音频（work_sr 采样率）

        Returns:
            formant 处理 + 长度对齐后的音频
        """
        state = self.state

        expected = state.block_samples + state.sola_buffer_samples + state.sola_search_samples

        factor = state.formant_ratio
        if factor != 1.0:
            target_sr = state.work_sr
            upp_res = formant_resample_ratio(factor, target_sr)
            if upp_res != target_sr // 100:
                infer = apply_formant_resample(
                    infer[: state.return_length * upp_res],
                    factor, target_sr, state.resample_kernel, state.device,
                ).squeeze()

        # 统一长度对齐（formant 重采样可能改变长度，SOLA 期望固定长度）
        if infer.shape[0] > expected:
            infer = infer[:expected]
        elif infer.shape[0] < expected:
            infer = F.pad(infer, (0, expected - infer.shape[0]))
        return infer
