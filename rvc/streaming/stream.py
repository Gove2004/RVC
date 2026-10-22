"""音频流管理器 — 封装 sounddevice 设备查询、流创建/启动/停止、副输出。

从 VoiceEngine 中拆分出的设备/流管理组件，负责：
- 设备查询与校验（输入/输出/副输出通道数）
- 主流创建与启动（sd.Stream，显式指定设备 + 错误回调）
- 副输出流创建与启动（sd.OutputStream + 块队列）
- 流安全中止与关闭、错误日志

术语：**主**流/主输出 = `main_*`，**副**输出 = `out2_*`——与配置字段
`audio.output2_device`、`audio.enable_out2` 同一套词表，不要另起 `secondary_*`。

VoiceEngine 保留对外接口，内部委托给本组件处理音频流。

刻意不在这里存「流统计」：sr / block_samples 的真相源是 EngineState，
错误数在 engine.error_count。另存一份本地副本就得配一把锁，而音频回调线程
每次 status 非空都要为它加锁、拼字符串。
"""
import logging

import sounddevice as sd

from rvc.core.errors import AudioDeviceError
from rvc.streaming.output import Out2Queue

logger = logging.getLogger(__name__)


class AudioStreams:
    """音频流管理器 — 管理 sounddevice 输入/输出/副输出流。"""

    def __init__(self):
        self.main_stream = None
        self.out2_stream = None
        self.out2_q = Out2Queue()
        self.enable_out2 = False
        self.channels = 1

    def query_device(self, dev_idx: int) -> dict:
        """查询设备信息。"""
        return sd.query_devices(dev_idx)

    def validate_and_log_devices(self, in_dev: int, out_dev: int, out2_dev_idx: int | None = None) -> int:
        """校验输入/输出设备通道数，设置 channels，打印设备日志。

        Returns:
            channels: 最小通道数（min(输入, 输出, 2)）
        """
        input_record = self.query_device(in_dev)
        output_record = self.query_device(out_dev)
        in_max = int(input_record["max_input_channels"])
        out_max = int(output_record["max_output_channels"])

        if in_max <= 0:
            raise AudioDeviceError(
                f"输入设备不支持录音（max_input_channels={in_max}）："
                f"索引 {in_dev}「{input_record.get('name', '?')}」。请在设备设置中选择支持输入的设备。"
            )
        if out_max <= 0:
            raise AudioDeviceError(
                f"输出设备不支持播放（max_output_channels={out_max}）："
                f"索引 {out_dev}「{output_record.get('name', '?')}」。请在设备设置中选择支持输出的设备。"
            )

        self.channels = min(in_max, out_max, 2)

        logger.info("音频设备：")
        logger.info("  · 麦克风：%s", input_record.get('name', '?'))
        logger.info("  · 主输出：%s", output_record.get('name', '?'))

        if out2_dev_idx is not None:
            out2_record = self.query_device(out2_dev_idx)
            logger.info("  · 副输出：%s", out2_record.get('name', f"#{out2_dev_idx}"))

        return self.channels

    def validate_out2_device(self, dev_idx: int, channels: int) -> None:
        """校验副输出设备通道数。"""
        out2_record = self.query_device(dev_idx)
        out2_max = int(out2_record["max_output_channels"])
        if out2_max < channels:
            raise AudioDeviceError(
                f"副输出设备「{out2_record.get('name', '?')}」只支持 {out2_max} 通道，"
                f"但主输出使用 {channels} 通道。请选择支持至少 {channels} 通道的副输出设备。"
            )

    def _handle_stream_error(self, status, prefix=""):
        """记录音频流错误（不计数、不上报：调用方只关心「有没有崩」，见模块 docstring）。

        Args:
            status: sounddevice 回调状态对象
            prefix: 错误来源前缀（如 "out2"），用于区分主流和副输出
        """
        # 只记录非频繁错误：input_overflow / output_underflow 在 WASAPI 共享
        # 模式高负载下几乎每块都有，逐条打日志等于在回调线程里刷盘。
        if status.input_overflow or status.output_underflow:
            return
        msg = f"{prefix}: {status}" if prefix else str(status)
        logger.warning("音频流错误: %s", msg)

    def start_main_stream(self, callback, sr: int, channels: int, block_samples: int,
                           in_dev: int | None = None, out_dev: int | None = None) -> None:
        """创建并启动主流。

        Args:
            callback: sounddevice 回调函数 (indata, outdata, frames, times, status)
            sr: 采样率
            channels: 通道数
            block_samples: 块大小（样本数）
            in_dev: 输入设备索引（None = 使用默认设备）
            out_dev: 输出设备索引（None = 使用默认设备）
        """
        self.channels = channels

        device = (in_dev, out_dev) if (in_dev is not None and out_dev is not None) else None

        def _wrapped_callback(indata, outdata, frames, audio_clock, status):
            if status:
                self._handle_stream_error(status)
            callback(indata, outdata, frames, audio_clock, status)

        # latency 显式取 "low"（设备 defaultLow*Latency，WASAPI 共享约 10-30ms）。
        # 不传时用设备默认建议延迟，可能高达 100-200ms：PortAudio 会按建议值
        # 铺输出环，我们写的块要排在环内已缓冲音频之后才出声，这部分直接
        # 叠加进端到端延迟（对拍实测比"块时长+推理"多 ~180ms 的主因）。
        self.main_stream = sd.Stream(
            callback=_wrapped_callback,
            blocksize=block_samples,
            samplerate=sr,
            channels=channels,
            dtype="float32",
            device=device,
            latency="low",
        )
        self.main_stream.start()

    def push_out2(self, outdata) -> None:
        """把已写进主输出的块投递给副输出（调用方已判断 enable_out2）。

        只拷一块进队列，不分配新内存——见 Out2Queue.push。
        """
        if self.out2_stream is not None:
            self.out2_q.push(outdata)

    def start_out2_stream(self, dev_idx: int, sr: int, channels: int, block_samples: int) -> None:
        """创建并启动副输出流。

        幂等：重复调用先关掉旧流。不能直接在 self.out2_stream 上覆盖新流——
        被覆盖的 OutputStream 再没人 abort/close，设备会一直处于占用状态
        （下次打开或其它程序占用同一虚拟设备时表现为「设备不可用」）。

        Args:
            dev_idx: 副输出设备索引
            sr: 采样率
            channels: 通道数
            block_samples: 块大小（样本数）
        """
        self._safe_close_stream(self.out2_stream)
        self.out2_stream = None
        self.enable_out2 = False

        self.validate_out2_device(dev_idx, channels)

        # 清空上次残留的数据（在启动前做：启动后回调线程会开始消费）
        self.out2_q.clear()

        def out2_callback(outdata, frames, audio_clock, status):
            if status:
                self._handle_stream_error(status, prefix="out2")
            if not self.out2_q.take_into(outdata, frames):
                outdata[:] = 0

        self.out2_stream = sd.OutputStream(
            device=dev_idx,
            samplerate=sr,
            channels=channels,
            dtype="float32",
            blocksize=block_samples,
            callback=out2_callback,
            latency="low",
        )
        self.out2_stream.start()
        self.enable_out2 = True

    def stop_all(self) -> None:
        """安全中止并关闭所有流（主流 + 副输出）。"""
        self._safe_close_stream(self.out2_stream)
        self._safe_close_stream(self.main_stream)
        self.main_stream = None
        self.out2_stream = None
        self.enable_out2 = False

    @staticmethod
    def _safe_close_stream(stream) -> None:
        """安全中止并关闭 sounddevice 流，忽略所有异常。"""
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
